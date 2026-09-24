//! 文の誤入力補正の候補を本物の mozc 辞書で出す。正しい文で余計な直しを出さないか、打ち間違いを拾うか、時間を見る。
//!
//! cargo run -p rakukan-engine --example probe_sentence_typo --release -- <rakukan.dict>

use rakukan_dict::DictStore;
use rakukan_engine::{EngineConfig, RakunEngine};

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt().with_env_filter("rakukan_engine=debug").with_writer(std::io::stderr).init();
    let dict = std::env::args().nth(1).expect("path to rakukan.dict");
    let store = DictStore::load(None, Some(std::path::Path::new(&dict)), None)?;
    let mut config = EngineConfig::default();
    config.typo.enabled = true;
    if std::env::var("FULLWIDTH").is_ok() {
        config.alpha_width = rakukan_engine::AlphaWidth::Fullwidth;
    }
    config.typo.max_alternatives = std::env::var("MAX_ALTS").ok().and_then(|s| s.parse().ok()).unwrap_or(4);
    // （ラベル, ローマ字）。打ち間違いは本人の typo.log で多かった型（余計なキー 1 つ）と、4 種の誤り
    let cases = [
        ("b1", "matometehjappyousuru"),
        ("b2", "kadaiwomatometehjappyousuruyoteidesu"),
        ("b3", "konnkinoshinntyokutokadaiwomatometehjappyousuru"),
        ("b4", "raisyuunoteireikaigideha,matometehjappyousuru"),
        ("余計 j 長文", "raisyuunoteireikaigideha,konnkinoshinntyokutokadaiwomatometehjappyousuruyoteidesu"),
        ("ok 読点", "sirabetemitakeredo,geninnhamadawakarimasenndesita"),
        ("ok 長文", "raisyuunoteireikaigidehakonnkinoshinntyokutokadaiwomatometehappyousuruyoteidesu"),
        ("ok 会議", "asitanokaiginosiryouwojunbisiteokimasu"),
        ("ok 天気", "kyounotenkihaharenotiamedesu"),
        ("余計 r", "gojuppunnkakarurmitaidesu"),
        ("余計 j", "konohoukjoudesusumemasu"),
        ("余計 y", "sorekaramondaigadetekuyru"),
        ("余計 m", "korehadokomniarunodesuka"),
        ("余計 h", "asitanokaigidehjananiwohanasimasuka"),
        ("抜け i", "asitanokaignosiryouwojunbisiteokimasu"),
        ("隣 kaigi", "asitanokaihinosiryouwojunbisiteokimasu"),
        ("入替 shiryou", "asitanokaiginosiryuowojunbisiteokimasu"),
        ("二重 junbi", "asitanokaiginosiryouwojunnbbisiteokimasu"),
        ("二重 k", "tinaminikkonohonnhaomosiroidesu"),
    ];
    for (label, romaji) in cases {
        let mut e = RakunEngine::new(config.clone());
        e.set_dict_store(store.clone());
        for c in romaji.chars() {
            e.push_char(c);
        }
        e.flush_pending_n();
        let t = std::time::Instant::now();
        let alts = e.sentence_typo_alternatives();
        println!("[{label}] {} ({} us)", e.hiragana_text(), t.elapsed().as_micros());
        for (r, c) in &alts {
            println!("    {c:.1} {r}");
        }
    }
    Ok(())
}
