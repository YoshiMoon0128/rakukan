//! 誤入力補正: ローマ字の打鍵列から「打ちたかったはずの読み」の候補を作る（第 3 段の spec、段取り 2）。
//!
//! 間違いは指で起きるので、かなではなくローマ字の列で見る。`input_log` のユニット（1 かなぶんの `typed`）を材料に、
//! 編集距離 1 の操作（ユニット入れ替え、文字入れ替え、隣キー置換、二重打ち削除、抜けの挿入）で T' を作り、
//! `RomajiConverter` に通し直して読み R' を得る。R' は辞書の読みで分割できるものだけ残し、分割数が少ない順
//! （= 長い語で説明できる順）→ 編集コスト順に並べて上位だけ返す。
//!
//! 審判（どれが本命か）は LM のリランカーがやる。ここの仕事は「数を絞って無意味を落とす」まで。

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
}

impl Rule {
    pub const ALL: [Rule; 5] = [Rule::TransposeUnit, Rule::TransposeChar, Rule::AdjacentKey, Rule::Double, Rule::Drop];

    pub fn cost(self) -> f32 {
        match self {
            Rule::TransposeUnit => 1.0,
            Rule::TransposeChar => 1.0,
            Rule::AdjacentKey => 1.0,
            Rule::Double => 0.7,
            Rule::Drop => 1.3,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Rule::TransposeUnit => "transpose_unit",
            Rule::TransposeChar => "transpose_char",
            Rule::AdjacentKey => "adjacent_key",
            Rule::Double => "double",
            Rule::Drop => "drop",
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

/// 隣のキー（同じ行の左右と、上下の行の同じ列 ±1）。
pub fn adjacent_keys(c: char) -> Vec<char> {
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

fn is_kana(c: char) -> bool {
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
        assert_eq!(romaji_to_reading("kanjiniya").as_deref(), Some("かんじにや"));
        assert_eq!(romaji_to_reading("kannnijiya").as_deref(), Some("かんにじや"));
        assert_eq!(romaji_to_reading("kanj"), None); // j が残る
        assert_eq!(romaji_to_reading("kaqji"), None); // q が残る
    }

    #[test]
    fn 隣接ユニットの入れ替えで_じ_と_に_の取り違えが直る() {
        let u = units(&["ka", "n", "ni", "ji", "ya"]);
        let d = dict(&["かんじ", "に", "や", "かん"]);
        let alts = alternatives(&u, "かんにじや", &Rule::ALL, 4, &d, &flat);
        let top = alts.first().expect("candidate");
        assert_eq!(top.reading, "かんじにや");
        assert_eq!(top.rule, Rule::TransposeUnit);
        assert_eq!(top.segments, 3); // かんじ / に / や
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
        let d = dict(&["かんじ", "に", "や"]);
        let alts = alternatives(&units(&["ka", "n", "ji", "ni", "ya"]), "かんじにや", &Rule::ALL, 10, &d, &flat);
        assert!(alts.iter().all(|a| a.reading != "かんじにや"));
        // 「かんにじや」は かん が無く、1 かなの じ は助詞でないので分割できない
        assert!(alts.iter().all(|a| a.reading != "かんにじや"), "{alts:?}");
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
        let d = dict(&["かんじ", "に", "じ", "や"]);
        assert_eq!(min_segments("かんじにや", &d), Some(3));
        assert_eq!(min_segments("じ", &d), None);
        assert_eq!(min_segments("に", &d), Some(1));
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
