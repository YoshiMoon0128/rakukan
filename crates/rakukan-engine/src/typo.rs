//! 誤入力補正: ローマ字の打鍵列から「打ちたかったはずの読み」の候補を作る（第 3 段の spec、段取り 2）。
//!
//! 間違いは指で起きるので、かなではなくローマ字の列で見る。`input_log` のユニット（1 かなぶんの `typed`）を材料に、
//! 編集距離 1 の操作（ユニット入れ替え、文字入れ替え、隣キー置換、二重打ち削除、抜けの挿入）で T' を作り、
//! `RomajiConverter` に通し直して読み R' を得る。R' は辞書の読みで分割できるものだけ残し、分割数が少ない順
//! （= 長い語で説明できる順）→ 編集コスト順に並べて上位だけ返す。
//!
//! 審判（どれが本命か）は LM のリランカーがやる。ここの仕事は「数を絞って無意味を落とす」まで。

use std::cell::RefCell;
use std::collections::HashMap;

use crate::romaji::RomajiConverter;

/// 補正の規則。編集コストの重みは規則ごとの定数から始め、typo ログの実測で置き換える。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Rule {
    /// 隣接ユニットの入れ替え（"kan ji ni" → "kan ni ji"）
    TransposeUnit,
    /// 隣接文字の入れ替え（"kanjini" → "kanijni"）
    TransposeChar,
    /// 隣のキーへの置換（"kanji" → "kanhi"）
    AdjacentKey,
    /// 二重打ちの削除（"kannji" → "kanji"）
    Double,
    /// 抜けの挿入（"kaji" → "kanji"）
    Drop,
    /// 余計な 1 字の削除（"kakarurmitai" → "kakarumitai"）。隣のキーを一緒に押した打ち間違いで、読みに英字が残りやすい
    Extra,
}

impl Rule {
    pub const ALL: [Rule; 6] = [Rule::TransposeUnit, Rule::TransposeChar, Rule::AdjacentKey, Rule::Double, Rule::Drop, Rule::Extra];

    pub fn cost(self) -> f32 {
        match self {
            Rule::TransposeUnit => 1.0,
            Rule::TransposeChar => 1.0,
            Rule::AdjacentKey => 1.0,
            Rule::Double => 0.7,
            Rule::Drop => 1.3,
            Rule::Extra => 1.0,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Rule::TransposeUnit => "transpose_unit",
            Rule::TransposeChar => "transpose_char",
            Rule::AdjacentKey => "adjacent_key",
            Rule::Double => "double",
            Rule::Drop => "drop",
            Rule::Extra => "extra",
        }
    }
}

/// 補正後の読み 1 つ。
#[derive(Debug, Clone, PartialEq)]
pub struct Alt {
    pub reading: String,
    pub romaji: String,
    pub rule: Rule,
    pub cost: f32,
    /// 辞書の読みで分割したときの最少の区切り数
    pub segments: usize,
}

/// QWERTY の英字の並び。JIS / US で同じ。
const ROWS: [&str; 3] = ["qwertyuiop", "asdfghjkl", "zxcvbnm"];

/// 数字の段の `0` を押したときに狙っていたキー。`-`（長音）と `o` `p` は `0` の右と下。
/// 数字の段を `ROWS` に足すと、全部の英字で数字への置き換えを試すことになるので `0` だけ持つ
const ZERO_SLIPS: [char; 3] = ['-', 'o', 'p'];

/// 隣のキー（同じ行の左右と、上下の行の同じ列 ±1）。
pub fn adjacent_keys(c: char) -> Vec<char> {
    if c == '0' {
        return ZERO_SLIPS.to_vec();
    }
    let mut out = Vec::new();
    for (r, row) in ROWS.iter().enumerate() {
        let Some(i) = row.find(c) else { continue };
        let bytes = row.as_bytes();
        if i > 0 {
            out.push(bytes[i - 1] as char);
        }
        if i + 1 < bytes.len() {
            out.push(bytes[i + 1] as char);
        }
        for rr in [r.wrapping_sub(1), r + 1] {
            let Some(other) = ROWS.get(rr) else { continue };
            for j in i.saturating_sub(1)..=(i + 1) {
                if let Some(&b) = other.as_bytes().get(j) {
                    out.push(b as char);
                }
            }
        }
    }
    out
}

/// 抜けの挿入で試す文字。子音の前の `n`、母音、拗音の `y`。全部試すと数が爆発するので絞る。
const INSERTS: [char; 7] = ['n', 'a', 'i', 'u', 'e', 'o', 'y'];

/// ローマ字列を読みに変換する。未変換の文字が残る（解析が途中で止まる）ものは None。
pub fn romaji_to_reading(romaji: &str) -> Option<String> {
    let mut conv = RomajiConverter::new();
    for c in romaji.chars() {
        conv.push(c);
    }
    conv.flush();
    let out = conv.output().to_string();
    if out.is_empty() || !out.chars().all(is_kana) {
        return None;
    }
    Some(out)
}

pub(crate) fn is_kana(c: char) -> bool {
    matches!(c, '\u{3041}'..='\u{3096}' | 'ー')
}

/// 編集距離 1 のローマ字列を作る。返すのは（T', 規則）。重複はここでは消さない。
pub fn generate(units: &[String], rules: &[Rule]) -> Vec<(String, Rule)> {
    let joined: String = units.concat();
    let chars: Vec<char> = joined.chars().collect();
    let mut out = Vec::new();
    for &rule in rules {
        match rule {
            Rule::TransposeUnit => {
                for i in 0..units.len().saturating_sub(1) {
                    if units[i] == units[i + 1] {
                        continue;
                    }
                    let mut u = units.to_vec();
                    u.swap(i, i + 1);
                    out.push((u.concat(), rule));
                }
            }
            Rule::TransposeChar => {
                for i in 0..chars.len().saturating_sub(1) {
                    if chars[i] == chars[i + 1] {
                        continue;
                    }
                    let mut c = chars.clone();
                    c.swap(i, i + 1);
                    out.push((c.iter().collect(), rule));
                }
            }
            Rule::AdjacentKey => {
                for i in 0..chars.len() {
                    for k in adjacent_keys(chars[i]) {
                        let mut c = chars.clone();
                        c[i] = k;
                        out.push((c.iter().collect(), rule));
                    }
                }
            }
            Rule::Double => {
                for i in 1..chars.len() {
                    if chars[i] == chars[i - 1] {
                        let mut c = chars.clone();
                        c.remove(i);
                        out.push((c.iter().collect(), rule));
                    }
                }
            }
            Rule::Drop => {
                for i in 0..=chars.len() {
                    for k in INSERTS {
                        let mut c = chars.clone();
                        c.insert(i, k);
                        out.push((c.iter().collect(), rule));
                    }
                }
            }
            Rule::Extra => {
                for i in 0..chars.len() {
                    // 同じ字の連続は Double が受け持つ
                    if i > 0 && chars[i] == chars[i - 1] {
                        continue;
                    }
                    let mut c = chars.clone();
                    c.remove(i);
                    out.push((c.iter().collect(), rule));
                }
            }
        }
    }
    out
}

/// 1 かなで区切ってよい読み（助詞）。それ以外の 1 かなの区切りは認めない。
/// mozc 辞書は「じ」「や」のような 1 かなの読みも持つので、これを認めると何でも分割できてしまう。
const PARTICLES: [&str; 12] = ["は", "が", "を", "に", "で", "と", "も", "の", "へ", "や", "か", "ね"];

/// 読みを辞書の読みだけで分割したときの最少の区切り数。分割できなければ None。
/// `has(r)` は読み r が辞書にあるか。
pub fn min_segments(reading: &str, has: &dyn Fn(&str) -> bool) -> Option<usize> {
    let chars: Vec<char> = reading.chars().collect();
    let n = chars.len();
    let mut best: Vec<Option<usize>> = vec![None; n + 1];
    best[0] = Some(0);
    for end in 1..=n {
        for start in 0..end {
            let Some(prev) = best[start] else { continue };
            let seg: String = chars[start..end].iter().collect();
            if end - start == 1 && !PARTICLES.contains(&seg.as_str()) {
                continue;
            }
            if !has(&seg) {
                continue;
            }
            let cand = prev + 1;
            if best[end].is_none_or(|b| cand < b) {
                best[end] = Some(cand);
            }
        }
    }
    best[n]
}

/// 補正後の読みを作る。`units` は `input_log` の romaji ユニット、`original` は元の読み。
/// 辞書で分割できるものだけ残し、（区切り数, 編集コスト, `rank`, 読み）の順に並べて `max` 個返す。同じ読みは最小コストの 1 つに畳む。
/// `rank` は読みの頻度の代わり（mozc の cost。小さいほど頻出）。同点を読みの文字順で切ると
/// 「きあき」の補正で「きかい」が「いかき」に負けて落ちるので、頻度で切る
pub fn alternatives(units: &[String], original: &str, rules: &[Rule], max: usize, has: &dyn Fn(&str) -> bool, rank: &dyn Fn(&str) -> u16) -> Vec<Alt> {
    if units.is_empty() || max == 0 {
        return Vec::new();
    }
    let mut alts: Vec<Alt> = Vec::new();
    for (romaji, rule) in generate(units, rules) {
        let Some(reading) = romaji_to_reading(&romaji) else { continue };
        if reading == original {
            continue;
        }
        let Some(segments) = min_segments(&reading, has) else { continue };
        let cost = rule.cost();
        match alts.iter_mut().find(|a| a.reading == reading) {
            Some(a) => {
                if cost < a.cost {
                    a.cost = cost;
                    a.rule = rule;
                    a.romaji = romaji;
                }
            }
            None => alts.push(Alt { reading, romaji, rule, cost, segments }),
        }
    }
    alts.sort_by(|a, b| {
        a.segments
            .cmp(&b.segments)
            .then(a.cost.partial_cmp(&b.cost).unwrap_or(std::cmp::Ordering::Equal))
            .then(rank(&a.reading).cmp(&rank(&b.reading)))
            .then(a.reading.cmp(&b.reading))
    });
    alts.truncate(max);
    alts
}

/// 文の打鍵列の区間。直すのはローマ字の区間だけで、記号・数字・直接入力は固定のまま読みに入る。
#[derive(Debug, Clone, PartialEq)]
pub enum Part {
    /// ローマ字の打鍵（`input_log` の 1 かなぶんの `typed`）と、この区間が読みに足した文字列
    Romaji { units: Vec<String>, output: String },
    Fixed(String),
}

impl Part {
    pub fn output(&self) -> &str {
        match self {
            Part::Romaji { output, .. } => output,
            Part::Fixed(s) => s,
        }
    }
}

/// 文の中の 1 か所を直した読み。
#[derive(Debug, Clone, PartialEq)]
pub struct SentenceAlt {
    pub reading: String,
    pub rule: Rule,
    pub cost: f32,
    /// 直した場所の前後で `cover` の点がどれだけ下がったか
    pub gain: u32,
    /// 直した場所の前後を覆った語の mozc コストの合計（小さいほど頻出の語で説明できた）
    pub word_cost: u32,
    /// 読みに残った英字の近くで余計な 1 字を消した直し。typo.log の実測では、英字が残る打ち間違いの多くが
    /// 隣のキーを一緒に押したもの（`kakarurmitai` `houkjou` `wqo`）なので、並べるときに先頭へ置く
    pub latin_extra: bool,
}

/// 辞書のどの語にも入らない文字 1 字の点。語 1 つは 1 点。打ち間違いは辞書に無い切れ端を作るので重く数える
const UNCOVERED: u32 = 3;
/// 辞書の読みとして探す最長のかな数
const MAX_WORD_CHARS: usize = 12;
/// 直した場所の前後で点を比べる幅（かな数）
const REGION_PAD: usize = 6;
/// 点がこれ以上下がった直しだけ残す。語の区切りが 1 つ減るだけ（1 点）の直しは残さない
const MIN_GAIN: u32 = 2;
/// 並べるときの下がり幅の頭打ち。語に入らない文字を 1 つ消せば十分で、それ以上の差は mozc が持つ切れ端
/// （「てく」「うる」）の当たり外れで決まり当てにならない（「でてくyる」で「でてくうる」が「でてくる」を上回った）
const GAIN_CAP: u32 = UNCOVERED;

/// 読みを辞書の語で覆ったときの最小の点と、その覆い方で語に入らなかった文字の位置。
/// 語 1 つは 1 点、句読点などの記号 1 字は 1 点、語に入らないかな・英字 1 字は `UNCOVERED` 点。
/// 1 かなの語は助詞だけ認める（`min_segments` と同じ理由）。点が同じ覆い方が複数あれば、語の mozc コストの合計が小さい方。
/// `word_cost(r)` は読み r の語の mozc コスト（辞書に無ければ None）。戻りは（点, 語のコストの合計, 語に入らなかった位置）。
fn cover(chars: &[char], word_cost: &dyn Fn(&str) -> Option<u16>) -> (u32, u32, Vec<usize>) {
    let n = chars.len();
    // best[i] = 先頭 i 字を覆う最小の（点, コスト）、from[i] = (直前の区切り, 語か記号で覆ったか)
    let mut best = vec![(u32::MAX, u32::MAX); n + 1];
    let mut from = vec![(0usize, false); n + 1];
    best[0] = (0, 0);
    for end in 1..=n {
        let c = chars[end - 1];
        let symbol = !(is_kana(c) || c.is_alphanumeric());
        let (p, w) = best[end - 1];
        best[end] = (p + if symbol { 1 } else { UNCOVERED }, w);
        from[end] = (end - 1, symbol);
        for start in end.saturating_sub(MAX_WORD_CHARS)..end {
            let (p, w) = best[start];
            if p == u32::MAX || p + 1 > best[end].0 {
                continue;
            }
            let seg: String = chars[start..end].iter().collect();
            if end - start == 1 && !PARTICLES.contains(&seg.as_str()) {
                continue;
            }
            if let Some(cost) = word_cost(&seg) {
                let cand = (p + 1, w + cost as u32);
                if cand < best[end] {
                    best[end] = cand;
                    from[end] = (start, true);
                }
            }
        }
    }
    let mut uncovered = Vec::new();
    let mut i = n;
    while i > 0 {
        let (start, covered) = from[i];
        if !covered {
            uncovered.push(start);
        }
        i = start;
    }
    uncovered.reverse();
    (best[n].0, best[n].1, uncovered)
}

/// 文（長い読み）の打ち間違いを 1 か所直した読みを作る。`parts` は打鍵列をローマ字の区間と固定の区間に分けたもの。
///
/// 読み全体を辞書の語で覆い、語に入らない文字が無ければ何も返さない（正しく打った文はここで終わる）。
/// 語に入らない文字があれば、その近くを直す編集距離 1 の打鍵列を作り、直した場所の前後 `REGION_PAD` 字で
/// 点を比べて `MIN_GAIN` 以上下がったものを残す。（英字の近くの余計な 1 字の削除か, `GAIN_CAP` で頭打ちにした下がり幅,
/// 直した場所の語のコスト, 編集コスト）の順に `max` 個返す。
/// 審判（LM のリランカー）が文ごと採点するので、ここの仕事は数を絞ることだけ。
pub fn sentence_alternatives(parts: &[Part], rules: &[Rule], max: usize, word_cost: &dyn Fn(&str) -> Option<u16>) -> Vec<SentenceAlt> {
    if max == 0 {
        return Vec::new();
    }
    let memo: RefCell<HashMap<String, Option<u16>>> = RefCell::new(HashMap::new());
    let has = |r: &str| {
        if let Some(&v) = memo.borrow().get(r) {
            return v;
        }
        let v = word_cost(r);
        memo.borrow_mut().insert(r.to_string(), v);
        v
    };
    let original: String = parts.iter().map(Part::output).collect();
    let orig: Vec<char> = original.chars().collect();
    let (_, _, uncovered) = cover(&orig, &has);
    if uncovered.is_empty() {
        return Vec::new();
    }
    // 範囲 [lo, hi) の前後 REGION_PAD 字以内に、語に入らない文字があるか
    let near = |lo: usize, hi: usize| uncovered.iter().any(|&u| u + REGION_PAD >= lo && u < hi + REGION_PAD);

    let mut out: Vec<SentenceAlt> = Vec::new();
    let mut offset = 0usize;
    for (pi, part) in parts.iter().enumerate() {
        let len = part.output().chars().count();
        let Part::Romaji { units, output } = part else {
            offset += len;
            continue;
        };
        if !near(offset, offset + len) {
            offset += len;
            continue;
        }
        let prefix: String = parts[..pi].iter().map(Part::output).collect();
        let suffix: String = parts[pi + 1..].iter().map(Part::output).collect();
        for (romaji, rule) in generate(units, rules) {
            let Some(r) = romaji_to_reading(&romaji) else { continue };
            if r == *output {
                continue;
            }
            let alt: Vec<char> = prefix.chars().chain(r.chars()).chain(suffix.chars()).collect();
            // 直した場所 = 元と共通の先頭・末尾を除いた範囲
            let p = orig.iter().zip(&alt).take_while(|(a, b)| a == b).count();
            let max_s = orig.len().min(alt.len()) - p;
            let s = orig.iter().rev().zip(alt.iter().rev()).take(max_s).take_while(|(a, b)| a == b).count();
            let (o_end, a_end) = (orig.len() - s, alt.len() - s);
            if !near(p, o_end) {
                continue;
            }
            let lo = p.saturating_sub(REGION_PAD);
            let (so, _, _) = cover(&orig[lo..(o_end + REGION_PAD).min(orig.len())], &has);
            let (sa, word_cost, _) = cover(&alt[lo..(a_end + REGION_PAD).min(alt.len())], &has);
            if so < sa + MIN_GAIN {
                continue;
            }
            let reading: String = alt.iter().collect();
            let latin_extra = rule == Rule::Extra
                && orig[lo..(o_end + REGION_PAD).min(orig.len())].iter().any(|c| c.is_ascii_alphabetic());
            let cand = SentenceAlt { reading, rule, cost: rule.cost(), gain: so - sa, word_cost, latin_extra };
            match out.iter_mut().find(|a| a.reading == cand.reading) {
                Some(a) => {
                    if cand.cost < a.cost {
                        *a = cand;
                    }
                }
                None => out.push(cand),
            }
        }
        offset += len;
    }
    out.sort_by(|a, b| {
        b.latin_extra
            .cmp(&a.latin_extra)
            .then(b.gain.min(GAIN_CAP).cmp(&a.gain.min(GAIN_CAP)))
            .then(a.word_cost.cmp(&b.word_cost))
            .then(a.cost.partial_cmp(&b.cost).unwrap_or(std::cmp::Ordering::Equal))
            .then(a.reading.cmp(&b.reading))
    });
    out.truncate(max);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn units(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    /// 偽の辞書: この読みだけがある
    fn dict(words: &'static [&'static str]) -> impl Fn(&str) -> bool {
        move |r| words.contains(&r)
    }

    /// 頻度の情報なし（全部同点）
    fn flat(_: &str) -> u16 {
        0
    }

    #[test]
    fn 同じ区切り数と編集コストなら_rank_が小さい_頻出の_読みが先に来る() {
        // きあき → 入れ替えで かいき と きかい（どちらも 1 区切り・コスト 1）。文字順なら かいき が先
        let d = dict(&["かいき", "きかい"]);
        let rank = |r: &str| if r == "きかい" { 100u16 } else { 5000u16 };
        let alts = alternatives(&units(&["ki", "a", "ki"]), "きあき", &[Rule::TransposeChar], 1, &d, &rank);
        assert_eq!(alts.iter().map(|a| a.reading.as_str()).collect::<Vec<_>>(), vec!["きかい"]);
    }

    #[test]
    fn ローマ字列は読みに戻り_未変換が残るものは捨てる() {
        assert_eq!(romaji_to_reading("shinjinimo").as_deref(), Some("しんじにも"));
        assert_eq!(romaji_to_reading("shinnnijimo").as_deref(), Some("しんにじも"));
        assert_eq!(romaji_to_reading("kanj"), None); // j が残る
        assert_eq!(romaji_to_reading("kaqji"), None); // q が残る
    }

    #[test]
    fn 隣接ユニットの入れ替えで_じ_と_に_の取り違えが直る() {
        let u = units(&["shi", "n", "ni", "ji", "mo"]);
        let d = dict(&["しんじ", "に", "も", "しん"]);
        let alts = alternatives(&u, "しんにじも", &Rule::ALL, 4, &d, &flat);
        // 余計な 1 字の削除でできる「かんじや」（2 区切り）が先に並ぶことがある。どちらが本命かは審判が決める
        let fixed = alts.iter().find(|a| a.reading == "しんじにも").expect("candidate");
        assert_eq!(fixed.rule, Rule::TransposeUnit);
        assert_eq!(fixed.segments, 3); // しんじ / に / も
    }

    #[test]
    fn 二重打ちと抜けと隣キーが編集距離1で直る() {
        let d = dict(&["かんじ"]);
        let double = alternatives(&units(&["ka", "n", "n", "ji"]), "かんんじ", &[Rule::Double], 4, &d, &flat);
        assert_eq!(double[0].reading, "かんじ");
        let drop = alternatives(&units(&["ka", "ji"]), "かじ", &[Rule::Drop], 4, &d, &flat);
        assert_eq!(drop[0].reading, "かんじ");
        // kanhi → kanji（h の隣は j）
        let adj = alternatives(&units(&["ka", "n", "hi"]), "かんひ", &[Rule::AdjacentKey], 4, &d, &flat);
        assert_eq!(adj[0].reading, "かんじ");
    }

    #[test]
    fn 元の読みと辞書で分割できない読みは出ない() {
        let d = dict(&["しんじ", "に", "も"]);
        let alts = alternatives(&units(&["shi", "n", "ji", "ni", "mo"]), "しんじにも", &Rule::ALL, 10, &d, &flat);
        assert!(alts.iter().all(|a| a.reading != "しんじにも"));
        // 「しんにじも」は しん が無く、1 かなの じ は助詞でないので分割できない
        assert!(alts.iter().all(|a| a.reading != "しんにじも"), "{alts:?}");
    }

    #[test]
    fn 分割数が少ない読みが先に来て_上限で切れる() {
        let d = dict(&["きかい", "き", "かい", "か", "い"]);
        // 「きかい」自体は元。隣キーの候補が多数出るが、辞書で分割できるものだけ残る
        let alts = alternatives(&units(&["ki", "ka", "i"]), "きかい", &Rule::ALL, 2, &d, &flat);
        assert!(alts.len() <= 2);
        for w in alts.windows(2) {
            assert!(w[0].segments <= w[1].segments);
        }
    }

    #[test]
    fn min_segments_は1かなの区切りを助詞にだけ許す() {
        let d = dict(&["しんじ", "に", "じ", "も"]);
        assert_eq!(min_segments("しんじにも", &d), Some(3));
        assert_eq!(min_segments("じ", &d), None);
        assert_eq!(min_segments("に", &d), Some(1));
    }

    /// 偽の mozc: この読みだけがあり、コストは並びの順（先ほど頻出）
    fn costs(words: &'static [&'static str]) -> impl Fn(&str) -> Option<u16> {
        move |r| words.iter().position(|w| *w == r).map(|i| 100 * (i as u16 + 1))
    }

    fn romaji(units_: &[&str], output: &str) -> Part {
        Part::Romaji { units: units(units_), output: output.to_string() }
    }

    #[test]
    fn 文の中の二重打ちが_辞書に無い切れ端を消す直しとして出る() {
        // ちなみに っこの（k の二重打ち）→ ちなみに この
        let d = costs(&["ちなみに", "この", "しすてむ"]);
        let parts = [romaji(&["ti", "na", "mi", "ni", "k", "ko", "no", "si", "su", "te", "mu"], "ちなみにっこのしすてむ")];
        let alts = sentence_alternatives(&parts, &Rule::ALL, 4, &d);
        let top = alts.first().expect("candidate");
        assert_eq!(top.reading, "ちなみにこのしすてむ");
        assert_eq!(top.rule, Rule::Double);
    }

    #[test]
    fn 読みに英字が残る余計なキーは_1字消す直しで出る() {
        // jikanngakakarurmitai（r が余計）→ じかんがかかるみたい
        let d = costs(&["じかん", "が", "かかる", "みたい"]);
        let u = ["ji", "ka", "nn", "ga", "ka", "ka", "ru", "r", "mi", "ta", "i"];
        let parts = [romaji(&u, "じかんがかかるrみたい")];
        let alts = sentence_alternatives(&parts, &Rule::ALL, 4, &d);
        let top = alts.first().expect("candidate");
        assert_eq!(top.reading, "じかんがかかるみたい");
        assert_eq!(top.rule, Rule::Extra);
    }

    #[test]
    fn 辞書の語で覆い切れる文には直しを出さない() {
        let d = costs(&["ちなみに", "この", "しすてむ"]);
        let parts = [romaji(&["ti", "na", "mi", "ni", "ko", "no", "si", "su", "te", "mu"], "ちなみにこのしすてむ")];
        assert!(sentence_alternatives(&parts, &Rule::ALL, 4, &d).is_empty());
    }

    #[test]
    fn 固定の区間は直さずに読みへ残す() {
        // するから、 のあとの区間で隣キー（tensai の s を a と打って tenaai = てなあい）
        let d = costs(&["するから", "として", "てんさい", "なあ"]);
        let parts = [
            romaji(&["su", "ru", "ka", "ra"], "するから"),
            Part::Fixed("、".to_string()),
            romaji(&["te", "na", "a", "i", "to", "si", "te"], "てなあいとして"),
        ];
        let alts = sentence_alternatives(&parts, &Rule::ALL, 4, &d);
        assert!(alts.iter().any(|a| a.reading == "するから、てんさいとして"), "{alts:?}");
        assert!(alts.iter().all(|a| a.reading.starts_with("するから、")), "{alts:?}");
    }

    #[test]
    fn cover_は語に入らない文字の位置を返し_句読点は罰しない() {
        let d = costs(&["この", "しすてむ"]);
        let chars: Vec<char> = "っこの、しすてむ".chars().collect();
        let (score, _, uncovered) = cover(&chars, &d);
        assert_eq!(uncovered, vec![0]);
        assert_eq!(score, UNCOVERED + 1 + 1 + 1);
    }

    #[test]
    fn 隣のキーは同じ行の左右と上下の行() {
        let g = adjacent_keys('g');
        for k in ['f', 'h', 't', 'y', 'v', 'b'] {
            assert!(g.contains(&k), "{k} not in {g:?}");
        }
        assert!(adjacent_keys('q').contains(&'w'));
        assert!(adjacent_keys('x').is_empty() == false);
    }
}
