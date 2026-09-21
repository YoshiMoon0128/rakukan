//! 打ち間違いの読み → 正しい語 の辞書を作る。
//!
//! mozc 辞書（rakukan.dict）の頻出語を種にして、rakukan の誤入力補正と同じ規則（隣キー・二重打ち・抜け・
//! 入れ替え、ローマ字の編集距離 1）で崩した読みを生成し、**崩した読みが辞書に無いものだけ**を残す。
//! 辞書にある読み（例: かいご）を潰すと正しく打った語が出なくなるので、そこは触らない。
//! 出力は MS-IME ユーザー辞書ツール「テキストファイルからの登録」の形式（読み TAB 語句 TAB 品詞、UTF-16LE、CRLF）。
//! ユーザー辞書は実質 16,384 件が上限なので、頻度（mozc の cost）の順に切る。
use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use rakukan_dict::cost_band;
use rakukan_dict::mozc_dict::MozcDict;
use rakukan_engine::typo::{self, Rule};

#[derive(Parser, Debug)]
#[command(about = "打ち間違いの読みを MS-IME のユーザー辞書テキストにする")]
struct Args {
    /// rakukan.dict（mozc バイナリ辞書）
    #[arg(long)]
    dict: PathBuf,
    /// 出力（UTF-16LE、タブ区切り）
    #[arg(long)]
    out: PathBuf,
    /// 種にする頻出語の数（読みごとに最頻の表層 1 つ）
    #[arg(long, default_value_t = 3000)]
    words: usize,
    /// 出力する行数の上限（MS-IME のユーザー辞書は実質 16,384）
    #[arg(long, default_value_t = 16384)]
    max_entries: usize,
    /// 種にする読みの長さ（かな数）
    #[arg(long, default_value_t = 3)]
    min_kana: usize,
    #[arg(long, default_value_t = 8)]
    max_kana: usize,
    /// 品詞の列に書く語
    #[arg(long, default_value = "名詞")]
    pos: String,
    /// 種にする語の cost の下限。mozc の cost 0 は「かえよ」「させれ」のような活用形なので既定で外す
    #[arg(long, default_value_t = 1)]
    min_cost: u16,
    /// 種は表層に漢字（CJK 統合漢字）を含む語だけにする。ひらがな・カタカナだけの語は外す
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    kanji_only: bool,
    /// 種は表層にひらがなを含まない語（名詞相当）だけにする。mozc の cost 1 には「回りゃ」「高かっ」のような活用形が
    /// 混ざるので、名詞の辞書を作るときは既定で on
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    noun_only: bool,
    /// 種の語を先頭からこの数だけ標準出力に出す（選び方の確認用）
    #[arg(long, default_value_t = 0)]
    dump_words: usize,
    /// 1 語あたりに登録する崩れの上限。形の自然なもの → 起きやすい規則 の順に取る。
    /// 16,384 の枠を隣キーだけで使い切らず、語の数を増やすため
    #[arg(long, default_value_t = 6)]
    per_word: usize,
    /// 種の語を頻度順に並べたファイル（1 行 1 表層、UTF-8）。mozc の cost は品詞内の重みで頻度ではないので、
    /// 実際の頻度表（wordfreq など）から与えると「機械」「天気」のような日常語が入る。無ければ cost 順
    #[arg(long)]
    seed_file: Option<PathBuf>,
    /// 種の候補（読み TAB 表層 TAB cost）を全部このファイルに書いて終わる。読みの選び方を外（wordfreq）で決めるため
    #[arg(long)]
    dump_pairs: Option<PathBuf>,
    /// 同じ読みに頻出の表層が複数あるとき（きかい → 機会・機械）、崩れ 1 つに登録する表層の数。
    /// MS-IME は同じ読みで複数の語を持てるので、候補窓に両方出る
    #[arg(long, default_value_t = 2)]
    surfaces_per_reading: usize,
}

/// 崩れ方の自然さ。かな数が変わらず小書き文字（ぁぃぅぇぉ ゃゅょ の単独）を含まないもの（かてい → かとい）を先に、
/// 母音が増えて形が崩れたもの（けっか → けおか、かてい → ぁてい）を後に
fn shape_rank(mis: &str, reading: &str) -> u8 {
    let small = mis.chars().any(|c| matches!(c, 'ぁ' | 'ぃ' | 'ぅ' | 'ぇ' | 'ぉ'));
    let same_len = mis.chars().count() == reading.chars().count();
    match (same_len, small) {
        (true, false) => 0,
        (false, false) => 1,
        _ => 2,
    }
}

/// 規則の優先順位。ユーザー辞書の枠（16,384）を配るとき、起きやすい間違いから埋める
fn rule_rank(r: Rule) -> u8 {
    match r {
        Rule::AdjacentKey => 0,
        Rule::TransposeChar => 1,
        Rule::TransposeUnit => 2,
        Rule::Double => 3,
        Rule::Drop => 4,
    }
}

fn has_kanji(s: &str) -> bool {
    s.chars().any(|c| matches!(c, '\u{4E00}'..='\u{9FFF}' | '\u{3400}'..='\u{4DBF}'))
}

/// ひらがな 1〜2 文字 → ローマ字ユニット（打つときの綴り。MS-IME / rakukan とも受け付ける Hepburn 寄り）
fn kana_unit(k: &str) -> Option<&'static str> {
    Some(match k {
        "あ" => "a", "い" => "i", "う" => "u", "え" => "e", "お" => "o",
        "か" => "ka", "き" => "ki", "く" => "ku", "け" => "ke", "こ" => "ko",
        "さ" => "sa", "し" => "shi", "す" => "su", "せ" => "se", "そ" => "so",
        "た" => "ta", "ち" => "chi", "つ" => "tsu", "て" => "te", "と" => "to",
        "な" => "na", "に" => "ni", "ぬ" => "nu", "ね" => "ne", "の" => "no",
        "は" => "ha", "ひ" => "hi", "ふ" => "fu", "へ" => "he", "ほ" => "ho",
        "ま" => "ma", "み" => "mi", "む" => "mu", "め" => "me", "も" => "mo",
        "や" => "ya", "ゆ" => "yu", "よ" => "yo",
        "ら" => "ra", "り" => "ri", "る" => "ru", "れ" => "re", "ろ" => "ro",
        "わ" => "wa", "を" => "wo",
        "が" => "ga", "ぎ" => "gi", "ぐ" => "gu", "げ" => "ge", "ご" => "go",
        "ざ" => "za", "じ" => "ji", "ず" => "zu", "ぜ" => "ze", "ぞ" => "zo",
        "だ" => "da", "ぢ" => "di", "づ" => "du", "で" => "de", "ど" => "do",
        "ば" => "ba", "び" => "bi", "ぶ" => "bu", "べ" => "be", "ぼ" => "bo",
        "ぱ" => "pa", "ぴ" => "pi", "ぷ" => "pu", "ぺ" => "pe", "ぽ" => "po",
        "きゃ" => "kya", "きゅ" => "kyu", "きょ" => "kyo",
        "しゃ" => "sha", "しゅ" => "shu", "しょ" => "sho",
        "ちゃ" => "cha", "ちゅ" => "chu", "ちょ" => "cho",
        "にゃ" => "nya", "にゅ" => "nyu", "にょ" => "nyo",
        "ひゃ" => "hya", "ひゅ" => "hyu", "ひょ" => "hyo",
        "みゃ" => "mya", "みゅ" => "myu", "みょ" => "myo",
        "りゃ" => "rya", "りゅ" => "ryu", "りょ" => "ryo",
        "ぎゃ" => "gya", "ぎゅ" => "gyu", "ぎょ" => "gyo",
        "じゃ" => "ja", "じゅ" => "ju", "じょ" => "jo",
        "びゃ" => "bya", "びゅ" => "byu", "びょ" => "byo",
        "ぴゃ" => "pya", "ぴゅ" => "pyu", "ぴょ" => "pyo",
        "ふぁ" => "fa", "ふぃ" => "fi", "ふぇ" => "fe", "ふぉ" => "fo",
        "てぃ" => "thi", "でぃ" => "dhi", "うぃ" => "wi", "うぇ" => "we",
        "ー" => "-",
        _ => return None,
    })
}

/// 読み（ひらがな）→ 打鍵のユニット列。ん は次が子音なら n、それ以外は nn。っ は次のユニットの子音を重ねる。
/// 表に無い字があれば None。
fn reading_to_units(reading: &str) -> Option<Vec<String>> {
    let chars: Vec<char> = reading.chars().collect();
    let mut units: Vec<String> = Vec::new();
    let mut i = 0;
    let mut pending_sokuon = false;
    while i < chars.len() {
        let c = chars[i];
        if c == 'っ' {
            pending_sokuon = true;
            i += 1;
            continue;
        }
        if c == 'ん' {
            let next = chars.get(i + 1).copied();
            let nn = match next {
                None => true,
                Some(n) => {
                    let unit = kana_unit(&n.to_string()).or_else(|| kana_unit(&chars[i + 1..(i + 2).min(chars.len())].iter().collect::<String>()));
                    match unit {
                        Some(u) => matches!(u.chars().next(), Some('a' | 'i' | 'u' | 'e' | 'o' | 'y' | 'n')),
                        None => true,
                    }
                }
            };
            units.push(if nn { "nn".into() } else { "n".into() });
            i += 1;
            continue;
        }
        // 2 文字（拗音）を先に試す
        let two: Option<&str> = if i + 1 < chars.len() {
            let s: String = chars[i..i + 2].iter().collect();
            kana_unit(&s).map(|u| { i += 1; u })
        } else {
            None
        };
        let unit = match two.or_else(|| kana_unit(&c.to_string())) {
            Some(u) => u,
            None => return None,
        };
        i += 1;
        let mut u = unit.to_string();
        if pending_sokuon {
            let first = u.chars().next()?;
            if !first.is_ascii_alphabetic() || matches!(first, 'a' | 'i' | 'u' | 'e' | 'o' | 'n') {
                return None; // っ + 母音 / ん は打ち方が割れるので種にしない
            }
            u.insert(0, first);
            pending_sokuon = false;
        }
        units.push(u);
    }
    if pending_sokuon {
        return None;
    }
    Some(units)
}

fn is_hiragana(s: &str) -> bool {
    s.chars().all(|c| matches!(c, '\u{3041}'..='\u{3096}' | 'ー'))
}

fn main() -> Result<()> {
    let args = Args::parse();
    let dict = MozcDict::open(&args.dict).with_context(|| format!("open {}", args.dict.display()))?;

    // 読みごとに最頻（cost 最小）の通常語の表層
    let mut best: HashMap<String, (String, u16)> = HashMap::new();
    let mut pairs: Vec<(String, String, u16)> = Vec::new();
    dict.for_each_entry(|reading, surface, cost| {
        if cost_band::classify(cost) != cost_band::Class::Normal {
            return;
        }
        let n = reading.chars().count();
        if n < args.min_kana || n > args.max_kana || !is_hiragana(reading) {
            return;
        }
        if cost < args.min_cost || (args.kanji_only && !has_kanji(surface)) {
            return;
        }
        if args.noun_only && surface.chars().any(|c| matches!(c, '\u{3041}'..='\u{3096}')) {
            return;
        }
        if args.dump_pairs.is_some() {
            pairs.push((reading.to_string(), surface.to_string(), cost));
        }
        match best.get(reading) {
            Some((_, c)) if *c <= cost => {}
            _ => {
                best.insert(reading.to_string(), (surface.to_string(), cost));
            }
        }
    });
    if let Some(p) = &args.dump_pairs {
        let mut s = String::new();
        for (r, sf, c) in &pairs {
            s.push_str(&format!("{r}\t{sf}\t{c}\n"));
        }
        std::fs::write(p, s)?;
        println!("dumped {} pairs to {}", pairs.len(), p.display());
        return Ok(());
    }
    let mut words: Vec<(String, String, u16)> = best.into_iter().map(|(r, (s, c))| (r, s, c)).collect();
    words.sort_by(|a, b| a.2.cmp(&b.2).then(a.0.cmp(&b.0)));
    let total_readings = words.len();
    if let Some(seed_path) = &args.seed_file {
        // 表層 → 読み（複数あれば cost 最小）。頻度表の順に並べ直し、表に無い語は捨てる。
        // 出力の並び順にも頻度を使いたいので、cost の列に頻度表の順位を入れる
        let seeds = std::fs::read_to_string(seed_path).with_context(|| format!("read {}", seed_path.display()))?;
        let mut by_surface: HashMap<&str, (&str, u16)> = HashMap::new();
        for (r, s, c) in &words {
            match by_surface.get(s.as_str()) {
                Some((_, c0)) if *c0 <= *c => {}
                _ => {
                    by_surface.insert(s.as_str(), (r.as_str(), *c));
                }
            }
        }
        let mut ordered: Vec<(String, String, u16)> = Vec::new();
        // 同じ（読み, 表層）は 1 回、同じ読みは surfaces_per_reading 個まで
        let mut seen: std::collections::HashSet<(String, String)> = std::collections::HashSet::new();
        let mut per_reading: HashMap<String, usize> = HashMap::new();
        let mut push = |ordered: &mut Vec<(String, String, u16)>, r: &str, s: &str, rank: u16| {
            if !seen.insert((r.to_string(), s.to_string())) {
                return;
            }
            let n = per_reading.entry(r.to_string()).or_default();
            if *n >= args.surfaces_per_reading {
                return;
            }
            *n += 1;
            ordered.push((r.to_string(), s.to_string(), rank));
        };
        for (rank, line) in seeds.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let rank = rank.min(u16::MAX as usize) as u16;
            // 「読み TAB 表層」なら読みはそのまま使う（外で頻度から選んだもの）。表層だけなら辞書の cost 最小の読み
            if let Some((r, s)) = line.split_once('\t') {
                if !dict.lookup(r, 64).iter().any(|(sf, _)| sf == s) {
                    continue;
                }
                push(&mut ordered, r, s, rank);
            } else if let Some((r, _)) = by_surface.get(line) {
                push(&mut ordered, r, line, rank);
            }
        }
        println!("seed file: {} lines → {} words found in dict", seeds.lines().count(), ordered.len());
        words = ordered;
    }
    for (r, s, c) in words.iter().take(args.dump_words) {
        println!("seed {c:>5} {r} -> {s}");
    }

    // 生成。崩した読み → (語句, 種の cost, 規則, 種の読み)。同じ崩れが複数の語から出たら頻出の方
    let mut out: BTreeMap<String, (Vec<String>, u16, Rule, String)> = BTreeMap::new();
    let mut used_words = 0usize;
    let mut skipped_units = 0usize;
    let mut generated = 0usize;
    let mut collided = 0usize;
    let mut per_rule: BTreeMap<&'static str, usize> = BTreeMap::new();
    for (reading, surface, cost) in words.iter().take(args.words) {
        let Some(units) = reading_to_units(reading) else {
            skipped_units += 1;
            continue;
        };
        // 表が正しいことの確認: ユニットを rakukan のローマ字変換に通すと元の読みに戻る
        if typo::romaji_to_reading(&units.concat()).as_deref() != Some(reading.as_str()) {
            skipped_units += 1;
            continue;
        }
        used_words += 1;
        // この語の崩れを集めて、形の自然さ → 規則の起きやすさ の順に per_word 件だけ採る
        let mut mine: Vec<(String, Rule)> = Vec::new();
        for (romaji, rule) in typo::generate(&units, &Rule::ALL) {
            generated += 1;
            let Some(mis) = typo::romaji_to_reading(&romaji) else { continue };
            if mis == *reading || !is_hiragana(&mis) || mine.iter().any(|(m, _)| *m == mis) {
                continue;
            }
            // 崩した読みが辞書にある（別の正しい語）なら登録しない。文脈が無い辞書登録では区別できない
            if !dict.lookup(&mis, 1).is_empty() {
                collided += 1;
                continue;
            }
            mine.push((mis, rule));
        }
        mine.sort_by(|a, b| {
            shape_rank(&a.0, reading)
                .cmp(&shape_rank(&b.0, reading))
                .then(rule_rank(a.1).cmp(&rule_rank(b.1)))
                .then(a.0.cmp(&b.0))
        });
        for (mis, rule) in mine.into_iter().take(args.per_word) {
            match out.get_mut(&mis) {
                // 同じ読みの別の表層（機会 の次に 機械）は同じ崩れに足す
                Some(e) if e.3 == *reading => {
                    if !e.0.contains(surface) {
                        e.0.push(surface.clone());
                    }
                }
                // 別の語から先に出た崩れは、頻出の方（先に来た方）に譲る
                Some(_) => {}
                None => {
                    *per_rule.entry(rule.name()).or_default() += 1;
                    out.insert(mis, (vec![surface.clone()], *cost, rule, reading.clone()));
                }
            }
        }
    }

    // 規則の起きやすさ → 種の頻度 の順に上限で切る（抜け規則は母音を差し込むだけで数が膨らむので最後）
    let mut rows: Vec<(String, String, u16, Rule, String)> = out
        .into_iter()
        .flat_map(|(mis, (ss, c, r, src))| ss.into_iter().map(move |s| (mis.clone(), s, c, r, src.clone())))
        .collect();
    // 語ごとの上限で絞った後は、種の頻度の順に切る
    rows.sort_by(|a, b| a.2.cmp(&b.2).then(a.4.cmp(&b.4)).then(a.0.cmp(&b.0)));
    let kept = rows.len().min(args.max_entries);
    rows.truncate(kept);

    // UTF-16LE + BOM + CRLF。MS-IME のツールは UTF-8 を受け付けない
    let mut text = String::new();
    text.push_str("!Microsoft IME Dictionary Tool\r\n");
    text.push_str("!rakukan-typo-dict: 打ち間違いの読み → 正しい語（崩した読みが辞書に無いものだけ）\r\n");
    for (mis, surface, _, _, _) in &rows {
        text.push_str(&format!("{mis}\t{surface}\t{}\r\n", args.pos));
    }
    let mut bytes: Vec<u8> = vec![0xFF, 0xFE];
    for u in text.encode_utf16() {
        bytes.extend_from_slice(&u.to_le_bytes());
    }
    std::fs::write(&args.out, &bytes).with_context(|| format!("write {}", args.out.display()))?;

    // 人が読む報告（UTF-8）
    let report = args.out.with_extension("report.txt");
    let mut rep = String::new();
    rep.push_str(&format!(
        "readings in dict (normal, {}..={} kana): {total_readings}\nseed words: {} (skipped by romaji table: {skipped_units})\ngenerated misreadings: {generated}\ncollided with real readings (dropped): {collided}\nunique safe misreadings: {}\nwritten: {kept}\nper rule: {per_rule:?}\n\nsamples (misreading -> surface [rule, from reading, cost]):\n",
        args.min_kana, args.max_kana, used_words, rows.len().max(kept)
    ));
    let mut written_per_rule: BTreeMap<&'static str, usize> = BTreeMap::new();
    for row in &rows {
        *written_per_rule.entry(row.3.name()).or_default() += 1;
    }
    rep.push_str(&format!("written per rule: {written_per_rule:?}\n"));
    for (mis, surface, cost, rule, src) in rows.iter().take(40) {
        rep.push_str(&format!("  {mis} -> {surface}  [{} <- {src}, cost {cost}]\n", rule.name()));
    }
    rep.push_str("\nlast rows (the cut line):\n");
    for (mis, surface, cost, rule, src) in rows.iter().rev().take(10) {
        rep.push_str(&format!("  {mis} -> {surface}  [{} <- {src}, cost {cost}]\n", rule.name()));
    }
    std::fs::write(&report, rep.as_bytes())?;
    println!("wrote {} rows to {} (report: {})", kept, args.out.display(), report.display());
    Ok(())
}
