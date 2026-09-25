//! 変換の記録（`[typo] log = true` のときだけ）。
//!
//! 確定のたびに、読み・打鍵・確定した文字列と、最後に並べた候補のうち何番目を確定したかを残す。
//! 1 位以外を確定したものは並べ替えの外れで、候補と文脈があればリランカーの評価に使える。
//! 1 行 1 JSON、`%LOCALAPPDATA%\rakukan\conv.log`。

use std::io::Write;
use std::path::{Path, PathBuf};

use crate::typo_log::json_str;

/// 記録に残す左文脈の長さ（末尾から）と右文脈の長さ（先頭から）。文字数
const LEFT_CHARS: usize = 40;
const RIGHT_CHARS: usize = 10;
/// 記録に残す候補の数
const MAX_CANDS: usize = 8;

pub struct ConvLog {
    path: PathBuf,
}

/// 確定 1 回ぶんの記録。
pub struct ConvRecord<'a> {
    /// 確定時の読み（`hiragana_buf`）
    pub reading: &'a str,
    /// 確定時の打鍵（`input_log` の連結。Backspace で書き換わる）
    pub keys: &'a str,
    pub committed: &'a str,
    /// 同じ読みで最後に並べた候補。並べていなければ空
    pub cands: &'a [String],
    /// 並べたときの候補数の上限。ライブ変換のプレビューは 1
    pub limit: usize,
    pub left: &'a str,
    pub right: &'a str,
}

impl ConvLog {
    pub fn new(path: impl AsRef<Path>) -> Self {
        Self { path: path.as_ref().to_path_buf() }
    }

    /// 既定の置き場: `%LOCALAPPDATA%\rakukan\conv.log`（typo.log と同じディレクトリ）。
    pub fn at_default_path() -> Self {
        let dir = std::env::var("LOCALAPPDATA")
            .map(|d| PathBuf::from(d).join("rakukan"))
            .unwrap_or_else(|_| PathBuf::from("."));
        Self::new(dir.join("conv.log"))
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// 1 件書く。`rank` は確定した文字列が候補の何番目か（0 が 1 位）。候補に無ければ null。
    /// 書けなくても IME は止めない（失敗は無視する）。
    pub fn record(&self, r: &ConvRecord) {
        let t = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let rank = r
            .cands
            .iter()
            .position(|c| c == r.committed)
            .map_or_else(|| "null".to_string(), |i| i.to_string());
        let cands: Vec<String> = r.cands.iter().take(MAX_CANDS).map(|c| json_str(c)).collect();
        let left_skip = r.left.chars().count().saturating_sub(LEFT_CHARS);
        let left: String = r.left.chars().skip(left_skip).collect();
        let right: String = r.right.chars().take(RIGHT_CHARS).collect();
        let line = format!(
            "{{\"t\":{t},\"reading\":{},\"keys\":{},\"committed\":{},\"rank\":{rank},\"cands\":[{}],\"limit\":{},\"left\":{},\"right\":{}}}\n",
            json_str(r.reading),
            json_str(r.keys),
            json_str(r.committed),
            cands.join(","),
            r.limit,
            json_str(&left),
            json_str(&right),
        );
        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&self.path) {
            let _ = f.write_all(line.as_bytes());
        }
    }
}
