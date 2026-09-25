//! 打ち間違いの計測ログ（`[typo] log = true` のときだけ）。
//!
//! Backspace で消して打ち直した打鍵列を「消す前の romaji」「確定時の romaji」の対で残す。
//! 誤入力補正の生成規則の重みを実データで決めるための材料で、確定した文そのものは書かない。
//! 1 行 1 JSON、`%LOCALAPPDATA%\rakukan\typo.log`。

use std::io::Write;
use std::path::{Path, PathBuf};

pub struct TypoLog {
    path: PathBuf,
}

impl TypoLog {
    pub fn new(path: impl AsRef<Path>) -> Self {
        Self { path: path.as_ref().to_path_buf() }
    }

    /// 既定の置き場: `%LOCALAPPDATA%\rakukan\typo.log`（engine DLL のログと同じディレクトリ）。
    pub fn at_default_path() -> Self {
        let dir = std::env::var("LOCALAPPDATA")
            .map(|d| PathBuf::from(d).join("rakukan"))
            .unwrap_or_else(|_| PathBuf::from("."));
        Self::new(dir.join("typo.log"))
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// 1 件書く。`before` は最初の Backspace の直前、`after` は確定時。それぞれ romaji とかな。
    ///
    /// romaji は `input_log` の連結で、Backspace のたびに「残った表示を再生できる列」に書き換わる
    /// （"kanji" の じ を消すと "kanj" が残る）。打ち直した部分の継ぎ目には再生用の文字が混ざるので、
    /// 打鍵そのものの記録ではなく材料として読む。かなは表示そのもの。
    /// 書けなくても IME は止めない（失敗は無視する）。
    /// `after_surface` は打ち直した composition を確定した文字列。崩れ → 正しい語 の対を MS-IME の辞書へ写すときの語
    /// （文の長さの読みには辞書の表層が無い）。
    pub fn record(&self, before: &str, after: &str, before_kana: &str, after_kana: &str, after_surface: &str) {
        let t = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let line = format!(
            "{{\"t\":{t},\"before\":{},\"after\":{},\"before_kana\":{},\"after_kana\":{},\"after_surface\":{}}}\n",
            json_str(before),
            json_str(after),
            json_str(before_kana),
            json_str(after_kana),
            json_str(after_surface)
        );
        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&self.path) {
            let _ = f.write_all(line.as_bytes());
        }
    }
}

pub(crate) fn json_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn 消す前と確定時の打鍵列が_1行のjsonで追記される() {
        let dir = tempfile::tempdir().unwrap();
        let log = TypoLog::new(dir.path().join("typo.log"));
        log.record("kannijiya", "kanjiniya", "かんにじや", "かんじにや", "感じにや");
        log.record("a\"b", "ab", "", "", "");
        let text = std::fs::read_to_string(log.path()).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 2);
        assert!(lines[0].contains(r#""before":"kannijiya","after":"kanjiniya","before_kana":"かんにじや","after_kana":"かんじにや","after_surface":"感じにや""#), "{}", lines[0]);
        assert!(lines[1].contains(r#""before":"a\"b""#), "{}", lines[1]);
    }
}
