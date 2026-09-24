//! 文の誤入力補正を端から端まで測る。本物の mozc 辞書・jinen・リランカーを載せ、打鍵を 1 字ずつ流して
//! BG 変換 → 文脈付きマージの 1 位を見る（TSF のライブ変換 = beam 1、Space = beam 6 と同じ呼び方）。
//!
//! - 直った: 打ち間違えた打鍵の 1 位が、正しい打鍵の 1 位と同じ
//! - 壊した: 正しい打鍵の 1 位が、jinen 自身の 1 位から変わった
//!
//! cargo run -p rakukan-engine --features rerank --example eval_sentence_typo --release -- <rakukan.dict> <reranker.gguf> [beam]

use std::time::{Duration, Instant};

use rakukan_dict::DictStore;
use rakukan_engine::{EngineConfig, RakunEngine, conv_cache};

/// （ラベル, 打ち間違えた打鍵, 正しい打鍵）。打ち間違いの型は本人の typo.log で多かったもの（余計なキー 1 つ）と、4 種の誤り
const TYPOS: &[(&str, &str, &str)] = &[
    ("余計 r", "gojuppunnkakarurmitaidesu", "gojuppunnkakarumitaidesu"),
    ("余計 j", "konohoukjoudesusumemasu", "konohoukoudesusumemasu"),
    ("余計 y", "sorekaramondaigadetekuyru", "sorekaramondaigadetekuru"),
    ("余計 q", "sorewqomitekudasai", "sorewomitekudasai"),
    ("余計 m", "korehadokomniarunodesuka", "korehadokoniarunodesuka"),
    ("余計 u", "wasurenaiyouunisitekudasai", "wasurenaiyounisitekudasai"),
    ("余計 n", "sagyouwosusunmetekudasai", "sagyouwosusumetekudasai"),
    ("余計 h", "asitanokaigidehjananiwohanasimasuka", "asitanokaigidehananiwohanasimasuka"),
    ("抜け i", "asitanokaignosiryouwojunbisiteokimasu", "asitanokaiginosiryouwojunbisiteokimasu"),
    ("隣 e→r", "kyounotenkihaharenotiamrdesu", "kyounotenkihaharenotiamedesu"),
    ("隣 o→p", "kaigisitunoyoyakuwpsitekudasai", "kaigisitunoyoyakuwositekudasai"),
    ("二重 k", "tinaminikkonohonnhaomosiroidesu", "tinaminikonohonnhaomosiroidesu"),
    ("読点のあと 余計 j", "raisyuunoteireikaigideha,konnkinoshinntyokutokadaiwomatometehjappyousuruyoteidesu", "raisyuunoteireikaigideha,konnkinoshinntyokutokadaiwomatometehappyousuruyoteidesu"),
];

/// 正しく打った文（余計な直しを出さないか）
const CLEAN: &[(&str, &str)] = &[
    ("会議", "asitanokaiginosiryouwojunbisiteokimasu"),
    ("天気", "kyounotenkihaharenotiamedesu"),
    ("予約", "kaigisitunoyoyakuwositekudasai"),
    ("方向", "jasonohoukoudesusumemasu"),
    ("読点", "sirabetemitakeredo,geninnhamadawakarimasenndesita"),
    ("手順", "saishonitesutowohasirasetekara,kekkawohoukokusimasu"),
    ("外来語", "konosa-ba-noroguwokakuninnsite,era-nobasyowosagasimasu"),
    ("長文", "raisyuunoteireikaigidehakonnkinoshinntyokutokadaiwomatometehappyousuruyoteidesu"),
    ("口語", "sorettehontoniimanoyarikatadeiinndakke"),
    ("数字", "tugiha3jikarakaisisimasu"),
];

const LEFT: &str = "今日は朝から作業している。";

fn run(e: &mut RakunEngine, romaji: &str, beam: usize) -> (String, Option<String>, u128) {
    e.reset_all();
    e.commit(LEFT);
    for c in romaji.chars() {
        e.push_char(c);
    }
    e.flush_pending_n();
    let reading = e.hiragana_text().to_string();
    let t = Instant::now();
    e.bg_start(beam);
    conv_cache::wait_done_timeout(Duration::from_secs(30));
    let llm = e.bg_take_candidates(&reading).unwrap_or_default();
    let jinen_top = llm.first().cloned();
    let merged = e.merge_candidates_for_reading_with_context(&reading, llm, 40, Some(LEFT), None);
    let ms = t.elapsed().as_millis();
    e.reset_preedit();
    (merged.first().cloned().unwrap_or_default(), jinen_top, ms)
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(std::env::var("RUST_LOG").unwrap_or_else(|_| "rakukan_engine=info".into()))
        .with_writer(std::io::stderr)
        .init();
    let mut args = std::env::args().skip(1);
    let dict = args.next().expect("path to rakukan.dict");
    let reranker = args.next().expect("path to reranker gguf");
    let beam: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(1);

    let mut config = EngineConfig::default();
    config.model_variant = Some("jinen-v1-xsmall-q5".into());
    config.typo.enabled = true;
    config.rerank.enabled = true;
    config.rerank.model_path = Some(reranker);
    config.rerank.timeout_ms = 30_000;
    let mut e = RakunEngine::new(config);
    e.set_dict_store(DictStore::load(None, Some(std::path::Path::new(&dict)), None)?);
    e.init_kanji()?;
    let t = Instant::now();
    while !e.is_reranker_ready() {
        anyhow::ensure!(t.elapsed() < Duration::from_secs(120), "reranker not ready");
        std::thread::sleep(Duration::from_millis(200));
    }
    println!("beam={beam} reranker ready in {} ms", t.elapsed().as_millis());

    let mut fixed = 0;
    let only = std::env::var("ONLY").unwrap_or_default();
    for (label, typo, clean) in TYPOS.iter().filter(|t| t.0.contains(only.as_str())) {
        let (want, _, _) = run(&mut e, clean, beam);
        let (got, jinen, ms) = run(&mut e, typo, beam);
        let ok = got == want;
        fixed += ok as usize;
        println!("{} [{label}] {ms} ms\n    got ={got}\n    want={want}\n    jinen={}", if ok { "FIX " } else { "miss" }, jinen.unwrap_or_default());
    }
    let mut harmed = 0;
    for (label, clean) in CLEAN.iter().filter(|c| only.is_empty() || c.0.contains(only.as_str())) {
        let (got, jinen, ms) = run(&mut e, clean, beam);
        let jinen = jinen.unwrap_or_default();
        let bad = got != jinen;
        harmed += bad as usize;
        println!("{} [{label}] {ms} ms {got}{}", if bad { "HARM" } else { "ok  " }, if bad { format!(" (jinen={jinen})") } else { String::new() });
    }
    println!("=== beam={beam}: fixed {fixed}/{}, harmed {harmed}/{}", TYPOS.len(), CLEAN.len());
    Ok(())
}
