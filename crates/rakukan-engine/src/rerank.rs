//! 同音異義語リランカー: 辞書が出した候補を、左文脈を読む小型言語モデルの対数尤度で並べ替える。
//!
//! 候補は辞書由来に限るので、辞書に無い熟語は出ない。学習履歴・ユーザー辞書由来の候補は先頭に固定し、
//! システム辞書と LLM 由来の候補だけを並べ替える。確信不足で辞書順に戻すフォールバック（τ）は持たない
//! （spike の実測で τ=0 が全条件で最良だったため）。
//!
//! 採点は「prefix（左文脈）を seq 0 で decode → KV を候補数に複製 → 候補 + 右文脈の先頭を 1 batch で
//! decode → 候補トークンの log-softmax を合計」。prefix の KV は変換をまたいで持ち越し、差分だけ decode する。
//!
//! モデルのロードと採点は専用スレッド（`rerank-worker`）で行う。呼び元は `timeout_ms` だけ待ち、
//! 間に合わなければ辞書順をそのまま返す（並べ替えは常に任意）。遅れて届いた結果は次回の呼び出しで
//! 捨てる。ロード前・ロード失敗時も辞書順のまま。

use std::num::NonZeroU32;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender, TryRecvError};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::token::LlamaToken;

use crate::RerankSettings;

/// 並べ替えの設定（engine 内部）。`config.toml` の `[rerank]`（`RerankSettings`）から作る。
#[derive(Debug, Clone, PartialEq)]
pub struct RerankConfig {
    /// 辞書順の事前分布との混合比。1.0 で LM だけ。0.6B は 0.8、1.7B は 1.0 を出発点にする
    pub lambda: f32,
    /// 事前分布の減衰。`log_prior(r) = r · ln(rho)`。0.5 を既定にする
    pub rho: f32,
    /// 並べ替えの対象にする候補数の上限。tail の計算量はこれに比例する
    pub max_candidates: usize,
    /// 右文脈から候補の後ろに付ける文字数
    pub right_tail_chars: usize,
    /// 右文脈が無いときは並べ替えない（0.6B は右文脈なしで辞書に負けるため true にする）
    pub require_right_context: bool,
    /// 左文脈として使う末尾の文字数
    pub left_chars: usize,
}

impl Default for RerankConfig {
    fn default() -> Self {
        Self {
            lambda: 1.0,
            rho: 0.5,
            max_candidates: 6,
            right_tail_chars: 2,
            require_right_context: false,
            left_chars: 200,
        }
    }
}

impl From<&RerankSettings> for RerankConfig {
    fn from(s: &RerankSettings) -> Self {
        Self {
            lambda: s.lambda,
            rho: s.rho,
            max_candidates: s.max_candidates.max(1),
            right_tail_chars: s.right_tail_chars,
            require_right_context: s.require_right_context,
            left_chars: s.left_chars.max(1),
        }
    }
}

/// `[rerank].model` に書ける ID と、その GGUF の置き場（HuggingFace の repo とファイル名）。
/// ダウンロードと置き場は jinen と同じ `kanji::download_gguf` に乗せる。
pub const MODEL_IDS: &[(&str, &str, &str)] = &[
    ("qwen3-1.7b-q8_0", "Qwen/Qwen3-1.7B-GGUF", "Qwen3-1.7B-Q8_0.gguf"),
    ("qwen3-0.6b-q8_0", "Qwen/Qwen3-0.6B-GGUF", "Qwen3-0.6B-Q8_0.gguf"),
];

/// 設定からモデルの GGUF のパスを決める。`model_path` があればそれ、無ければ `model` の ID を
/// `MODEL_IDS` で引いてダウンロード（済みならキャッシュ）する。
pub fn resolve_model_path(settings: &RerankSettings) -> Result<PathBuf, String> {
    if let Some(p) = settings.model_path.as_deref().filter(|p| !p.trim().is_empty()) {
        return Ok(PathBuf::from(p));
    }
    let id = settings.model.trim();
    let (_, repo, file) = MODEL_IDS
        .iter()
        .find(|(known, _, _)| known.eq_ignore_ascii_case(id))
        .ok_or_else(|| {
            let known: Vec<&str> = MODEL_IDS.iter().map(|(k, _, _)| *k).collect();
            format!("unknown rerank model id {id:?} (known: {known:?}); set [rerank].model_path to use another GGUF")
        })?;
    crate::kanji::download_gguf(repo, file).map_err(|e| e.to_string())
}

/// 採点に使うスレッド数。0 は自動（論理コア数、上限 8。8 を超えても速くならない実測に合わせる）。
pub fn effective_threads(threads: u32) -> u32 {
    if threads > 0 {
        return threads;
    }
    std::thread::available_parallelism()
        .map(|n| n.get() as u32)
        .unwrap_or(4)
        .min(8)
}

fn log_softmax(xs: &[f32]) -> Vec<f32> {
    let m = xs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let lse = m + xs.iter().map(|x| (x - m).exp()).sum::<f32>().ln();
    xs.iter().map(|x| x - lse).collect()
}

/// LM スコアと辞書順の事前分布を混ぜ、候補のインデックスを並べ替えて返す。純関数。
///
/// `lm_scores[i]` は辞書順 i 番目の候補の log P(候補 | 左文脈)。同点は辞書順で安定させる。
pub fn rank_indices(lm_scores: &[f32], lambda: f32, rho: f32) -> Vec<usize> {
    let n = lm_scores.len();
    if n == 0 {
        return Vec::new();
    }
    let lm = log_softmax(lm_scores);
    let prior_raw: Vec<f32> = (0..n).map(|i| i as f32 * rho.ln()).collect();
    let prior = log_softmax(&prior_raw);
    let final_: Vec<f32> = lm.iter().zip(&prior).map(|(a, b)| lambda * a + (1.0 - lambda) * b).collect();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| final_[b].partial_cmp(&final_[a]).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(&b)));
    idx
}

/// `merged` の先頭 `pinned` 個（学習履歴・ユーザー辞書由来）を固定し、その後ろの最大 `max_candidates` 個を
/// `lm_scores` で並べ替える。`lm_scores.len()` は並べ替え対象の数と一致していること。
pub fn reorder(merged: Vec<String>, pinned: usize, lm_scores: &[f32], cfg: &RerankConfig) -> Vec<String> {
    let pinned = pinned.min(merged.len());
    let end = (pinned + lm_scores.len()).min(merged.len());
    if lm_scores.is_empty() || end <= pinned {
        return merged;
    }
    let order = rank_indices(lm_scores, cfg.lambda, cfg.rho);
    let mut out: Vec<String> = Vec::with_capacity(merged.len());
    out.extend_from_slice(&merged[..pinned]);
    let target = &merged[pinned..end];
    out.extend(order.into_iter().map(|i| target[i].clone()));
    out.extend_from_slice(&merged[end..]);
    out
}

/// 並べ替え対象の候補を `merged` から選ぶ。学習履歴・ユーザー辞書由来を先頭に固定し、
/// 残りのうち先頭 `max_candidates` 個を対象にする。戻りは（固定数, 対象の数）。
pub fn split_targets(merged: &[String], learn: &[String], user: &[String], max_candidates: usize) -> (usize, usize) {
    let pinned = merged
        .iter()
        .take_while(|c| learn.contains(c) || user.contains(c))
        .count();
    let rest = merged.len().saturating_sub(pinned);
    (pinned, rest.min(max_candidates))
}

/// 文字列の末尾 `n` 文字。
pub fn tail_chars(s: &str, n: usize) -> String {
    let total = s.chars().count();
    s.chars().skip(total.saturating_sub(n)).collect()
}

/// seq 0 に載っている prefix の状態。変換をまたいで持ち越す。
#[derive(Default)]
struct PrefixCache {
    tokens: Vec<LlamaToken>,
    last_logits: Vec<f32>,
}

/// 対数尤度で候補を採点する。jinen とは別のモデル（Qwen3 系 Q8_0）を 1 本持つ。
/// llama.cpp の backend はプロセスで 1 つなので jinen と共有する。
pub struct Scorer {
    backend: &'static LlamaBackend,
    model: LlamaModel,
    n_threads: i32,
    n_seq_max: u32,
}

/// `Scorer` から作る作業コンテキスト。KV cache と prefix の持ち越しを持つ。
pub struct ScorerSession<'a> {
    ctx: LlamaContext<'a>,
    model: &'a LlamaModel,
    cache: PrefixCache,
}

fn log_softmax_pick(logits: &[f32], target: LlamaToken) -> f32 {
    let m = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let lse = m + logits.iter().map(|x| (x - m).exp()).sum::<f32>().ln();
    logits[target.0 as usize] - lse
}

impl Scorer {
    pub fn load(path: impl AsRef<Path>, n_threads: u32, n_seq_max: u32) -> Result<Self, String> {
        let backend = crate::kanji::llamacpp::get_backend().map_err(|e| e.to_string())?;
        let mparams = LlamaModelParams::default().with_n_gpu_layers(0);
        let model = LlamaModel::load_from_file(backend, path.as_ref(), &mparams).map_err(|e| e.to_string())?;
        Ok(Self { backend, model, n_threads: n_threads as i32, n_seq_max })
    }

    pub fn session(&self) -> Result<ScorerSession<'_>, String> {
        let cparams = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(2048))
            .with_n_batch(2048)
            .with_n_ubatch(512)
            .with_n_seq_max(self.n_seq_max)
            .with_n_threads(self.n_threads)
            .with_n_threads_batch(self.n_threads);
        let ctx = self.model.new_context(self.backend, cparams).map_err(|e| e.to_string())?;
        Ok(ScorerSession { ctx, model: &self.model, cache: PrefixCache::default() })
    }
}

impl ScorerSession<'_> {
    fn tokens(&self, text: &str) -> Result<Vec<LlamaToken>, String> {
        self.model.str_to_token(text, AddBos::Never).map_err(|e| e.to_string())
    }

    /// seq 0 の KV を `prefix` に合わせる。前回と共通する先頭は残し、差分だけ decode する。
    fn ensure_prefix(&mut self, prefix: &[LlamaToken], n_seq: usize) -> Result<(), String> {
        let mut common = self.cache.tokens.iter().zip(prefix).take_while(|(a, b)| a == b).count();
        if common == prefix.len() && common == self.cache.tokens.len() && !self.cache.last_logits.is_empty() {
            return Ok(());
        }
        if common == prefix.len() {
            common -= 1; // 最終位置の logits を取り直すために 1 トークン戻す
        }
        if common == 0 {
            self.ctx.clear_kv_cache();
        } else if common < self.cache.tokens.len() {
            self.ctx.clear_kv_cache_seq(Some(0), Some(common as u32), None).map_err(|e| e.to_string())?;
        }
        let new = &prefix[common..];
        let mut batch = LlamaBatch::new(new.len().max(1), n_seq as i32);
        for (i, t) in new.iter().enumerate() {
            batch.add(*t, (common + i) as i32, &[0], i + 1 == new.len()).map_err(|e| e.to_string())?;
        }
        self.ctx.decode(&mut batch).map_err(|e| e.to_string())?;
        self.cache.last_logits = self.ctx.get_logits_ith((new.len() - 1) as i32).to_vec();
        self.cache.tokens = prefix.to_vec();
        Ok(())
    }

    /// 候補ごとの log P(候補 + 右文脈の先頭 | 左文脈) を返す。`candidates` は空でないこと。
    pub fn score(&mut self, left: &str, right: Option<&str>, candidates: &[String], right_tail_chars: usize) -> Result<Vec<f32>, String> {
        let left_text = if left.is_empty() { "\n" } else { left };
        let prefix = self.tokens(left_text)?;
        let head: String = right.unwrap_or("").chars().take(right_tail_chars).collect();
        let tails = candidates
            .iter()
            .map(|c| self.tokens(&format!("{c}{head}")))
            .collect::<Result<Vec<_>, _>>()?;
        let n = tails.len();
        self.ensure_prefix(&prefix, n)?;
        let plen = prefix.len();
        for s in 1..n {
            self.ctx.copy_kv_cache_seq(0, s as i32, None, None).map_err(|e| e.to_string())?;
        }
        let total: usize = tails.iter().map(Vec::len).sum();
        let mut batch = LlamaBatch::new(total.max(1), n as i32);
        let mut index: Vec<(usize, usize, i32)> = Vec::new();
        let mut bi = 0i32;
        for (s, tail) in tails.iter().enumerate() {
            for (j, t) in tail.iter().enumerate() {
                batch.add(*t, (plen + j) as i32, &[s as i32], true).map_err(|e| e.to_string())?;
                index.push((s, j, bi));
                bi += 1;
            }
        }
        self.ctx.decode(&mut batch).map_err(|e| e.to_string())?;
        let mut scores = vec![0.0f32; n];
        for (s, tail) in tails.iter().enumerate() {
            scores[s] += log_softmax_pick(&self.cache.last_logits, tail[0]);
        }
        for (s, j, bi) in index {
            if j + 1 < tails[s].len() {
                scores[s] += log_softmax_pick(self.ctx.get_logits_ith(bi), tails[s][j + 1]);
            }
        }
        for s in 1..n {
            self.ctx.clear_kv_cache_seq(Some(s as u32), None, None).map_err(|e| e.to_string())?;
        }
        self.ctx.clear_kv_cache_seq(Some(0), Some(plen as u32), None).map_err(|e| e.to_string())?;
        Ok(scores)
    }
}

/// 採点器の差し替え口。実物は `ScorerSession`、テストは偽物を入れる。
/// worker スレッドの中で作られ、そこから出ないので `Send` は要らない（`LlamaContext` は Send でない）。
pub trait ScoreBackend {
    fn score(&mut self, left: &str, right: Option<&str>, candidates: &[String]) -> Result<Vec<f32>, String>;
}

/// `Scorer` を worker スレッドで所有し、`ScoreBackend` として見せる。
struct LlamaScoreBackend {
    // `session` は `scorer` を借りる。フィールドは宣言順に drop されるので session を先に置く
    session: ScorerSession<'static>,
    _scorer: Box<Scorer>,
    right_tail_chars: usize,
}

impl ScoreBackend for LlamaScoreBackend {
    fn score(&mut self, left: &str, right: Option<&str>, candidates: &[String]) -> Result<Vec<f32>, String> {
        self.session.score(left, right, candidates, self.right_tail_chars)
    }
}

fn load_llama_backend(path: &Path, n_threads: u32, n_seq_max: u32, right_tail_chars: usize) -> Result<Box<dyn ScoreBackend>, String> {
    let scorer = Box::new(Scorer::load(path, n_threads, n_seq_max)?);
    // Box の中身は動かないので、session の借用を 'static に延ばしても指す先は変わらない。
    // 両方を同じ構造体に入れ、session → scorer の順で drop する。
    let scorer_ref: &'static Scorer = unsafe { &*(scorer.as_ref() as *const Scorer) };
    let session = scorer_ref.session()?;
    Ok(Box::new(LlamaScoreBackend { session, _scorer: scorer, right_tail_chars }))
}

struct Job {
    id: u64,
    left: String,
    right: Option<String>,
    candidates: Vec<String>,
}

struct Reply {
    id: u64,
    scores: Result<Vec<f32>, String>,
    elapsed: Duration,
}

struct Channel {
    tx: Sender<Job>,
    rx: Receiver<Reply>,
    next_id: u64,
    /// タイムアウトで見捨てた採点。結果が届くまで次の採点を出さない（worker は 1 本なので順番待ちになるだけ）
    inflight: Option<u64>,
}

impl Channel {
    /// 見捨てた採点の結果が届いていれば片付ける。まだなら false（採点をスキップする）。
    fn reclaim_inflight(&mut self) -> bool {
        let Some(id) = self.inflight else { return true };
        loop {
            match self.rx.try_recv() {
                Ok(reply) if reply.id == id => {
                    self.inflight = None;
                    return true;
                }
                Ok(_) => continue,
                Err(TryRecvError::Empty) | Err(TryRecvError::Disconnected) => return false,
            }
        }
    }
}

const STATUS_LOADING: u8 = 0;
const STATUS_READY: u8 = 1;
const STATUS_FAILED: u8 = 2;

/// Engine が持つリランカー本体。モデルと KV cache は専用スレッドが所有し、ここはチャネルの口だけを持つ。
pub struct Reranker {
    cfg: RerankConfig,
    timeout: Duration,
    status: Arc<AtomicU8>,
    chan: Mutex<Channel>,
}

impl Reranker {
    /// `config.toml` の `[rerank]` からリランカーを作る。`enabled = false` なら None。
    /// モデルの取得とロードは別スレッドで進み、この関数はすぐ返る。
    pub fn from_settings(settings: &RerankSettings) -> Option<Self> {
        if !settings.enabled {
            return None;
        }
        let s = settings.clone();
        let cfg = RerankConfig::from(settings);
        let n_threads = effective_threads(settings.threads);
        let timeout = Duration::from_millis(settings.timeout_ms.max(1));
        let n_seq_max = cfg.max_candidates as u32 + 1;
        let tail = cfg.right_tail_chars;
        tracing::info!(
            "rerank: enabled model={} lambda={} threads={} max_candidates={} timeout_ms={}",
            s.model,
            cfg.lambda,
            n_threads,
            cfg.max_candidates,
            settings.timeout_ms
        );
        Some(Self::spawn(cfg, timeout, move || {
            let path = resolve_model_path(&s)?;
            tracing::info!("rerank: loading {}", path.display());
            load_llama_backend(&path, n_threads, n_seq_max, tail)
        }))
    }

    /// GGUF のパスを直接指定して作る（テストと CLI 用）。ロードが終わるまで待つ。
    pub fn load(model_path: impl AsRef<Path>, cfg: RerankConfig, n_threads: u32) -> Result<Self, String> {
        let path = model_path.as_ref().to_path_buf();
        let n_seq_max = cfg.max_candidates.max(1) as u32 + 1;
        let tail = cfg.right_tail_chars;
        let rr = Self::spawn(cfg, Duration::from_secs(30), move || load_llama_backend(&path, n_threads, n_seq_max, tail));
        if rr.wait_ready(Duration::from_secs(600)) {
            Ok(rr)
        } else {
            Err("rerank model load failed".into())
        }
    }

    /// 採点器の作り方を差し込んで起動する。`make` は worker スレッドで 1 回呼ばれる。
    pub fn spawn(
        cfg: RerankConfig,
        timeout: Duration,
        make: impl FnOnce() -> Result<Box<dyn ScoreBackend>, String> + Send + 'static,
    ) -> Self {
        let (job_tx, job_rx) = mpsc::channel::<Job>();
        let (reply_tx, reply_rx) = mpsc::channel::<Reply>();
        let status = Arc::new(AtomicU8::new(STATUS_LOADING));
        let st = Arc::clone(&status);
        let spawned = std::thread::Builder::new().name("rerank-worker".into()).spawn(move || {
            let mut backend = match make() {
                Ok(b) => b,
                Err(e) => {
                    tracing::warn!("rerank: model load failed, reranking disabled: {e}");
                    st.store(STATUS_FAILED, Ordering::Release);
                    return;
                }
            };
            st.store(STATUS_READY, Ordering::Release);
            tracing::info!("rerank: model ready");
            while let Ok(job) = job_rx.recv() {
                let t = Instant::now();
                let scores = backend.score(&job.left, job.right.as_deref(), &job.candidates);
                let reply = Reply { id: job.id, scores, elapsed: t.elapsed() };
                if reply_tx.send(reply).is_err() {
                    break;
                }
            }
            tracing::info!("rerank: worker exit");
        });
        if let Err(e) = spawned {
            tracing::warn!("rerank: worker thread spawn failed: {e}");
            status.store(STATUS_FAILED, Ordering::Release);
        }
        Self {
            cfg,
            timeout,
            status,
            chan: Mutex::new(Channel { tx: job_tx, rx: reply_rx, next_id: 1, inflight: None }),
        }
    }

    pub fn config(&self) -> &RerankConfig {
        &self.cfg
    }

    /// モデルがロード済みで採点できる状態か。
    pub fn is_ready(&self) -> bool {
        self.status.load(Ordering::Acquire) == STATUS_READY
    }

    /// ロードの完了を最長 `max_wait` 待つ。true = ready、false = 失敗または時間切れ。
    pub fn wait_ready(&self, max_wait: Duration) -> bool {
        let deadline = Instant::now() + max_wait;
        loop {
            match self.status.load(Ordering::Acquire) {
                STATUS_READY => return true,
                STATUS_FAILED => return false,
                _ if Instant::now() >= deadline => return false,
                _ => std::thread::sleep(Duration::from_millis(20)),
            }
        }
    }

    /// `merged` を並べ替えて返す。ロード前・失敗・タイムアウト・採点エラーのどれでも `merged` をそのまま返す。
    pub fn rerank(&self, left: &str, right: Option<&str>, merged: Vec<String>, learn: &[String], user: &[String]) -> Vec<String> {
        if !self.is_ready() {
            return merged;
        }
        if self.cfg.require_right_context && right.map_or(true, |r| r.trim().is_empty()) {
            return merged;
        }
        // 文脈が左右どちらも無いとき（文書の先頭で確定文も無い）は並べ替えない。
        // 文脈ゼロの LM の好みは辞書順より当たらない（実機で「きかい」に「奇怪」が先頭に来た）
        if left.trim().is_empty() && right.map_or(true, |r| r.trim().is_empty()) {
            return merged;
        }
        let (pinned, n) = split_targets(&merged, learn, user, self.cfg.max_candidates);
        if n < 2 {
            return merged;
        }
        let left_tail = tail_chars(left, self.cfg.left_chars);
        let targets = merged[pinned..pinned + n].to_vec();

        let mut chan = match self.chan.lock() {
            Ok(g) => g,
            Err(_) => return merged,
        };
        if !chan.reclaim_inflight() {
            tracing::debug!("rerank: previous scoring still running, keeping dictionary order");
            return merged;
        }
        let id = chan.next_id;
        chan.next_id += 1;
        let job = Job { id, left: left_tail, right: right.map(str::to_owned), candidates: targets };
        if chan.tx.send(job).is_err() {
            return merged;
        }
        let deadline = Instant::now() + self.timeout;
        loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
            match chan.rx.recv_timeout(remaining) {
                Ok(reply) if reply.id == id => {
                    return match reply.scores {
                        Ok(sc) => {
                            tracing::debug!("rerank: scored {} candidates in {} ms", sc.len(), reply.elapsed.as_millis());
                            reorder(merged, pinned, &sc, &self.cfg)
                        }
                        Err(e) => {
                            tracing::warn!("rerank: scoring failed, keeping dictionary order: {e}");
                            merged
                        }
                    };
                }
                Ok(_stale) => continue,
                Err(RecvTimeoutError::Timeout) => {
                    chan.inflight = Some(id);
                    tracing::warn!("rerank: scoring exceeded {} ms, keeping dictionary order", self.timeout.as_millis());
                    return merged;
                }
                Err(RecvTimeoutError::Disconnected) => return merged,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(v: &[&str]) -> Vec<String> {
        v.iter().map(|x| x.to_string()).collect()
    }

    #[test]
    fn lm_が圧勝する候補が先頭に上がる() {
        let cfg = RerankConfig::default();
        let out = reorder(s(&["機械", "機会", "器械"]), 0, &[-20.0, -2.0, -25.0], &cfg);
        assert_eq!(out, s(&["機会", "機械", "器械"]));
    }

    #[test]
    fn 学習履歴由来の候補は動かない() {
        let cfg = RerankConfig::default();
        let out = reorder(s(&["キカイ", "機械", "機会"]), 1, &[-20.0, -2.0], &cfg);
        assert_eq!(out, s(&["キカイ", "機会", "機械"]));
    }

    #[test]
    fn 対象外の末尾はそのまま残る() {
        let cfg = RerankConfig { max_candidates: 2, ..RerankConfig::default() };
        let out = reorder(s(&["機械", "機会", "器械", "奇怪"]), 0, &[-20.0, -2.0], &cfg);
        assert_eq!(out, s(&["機会", "機械", "器械", "奇怪"]));
    }

    #[test]
    fn lambda_が0なら辞書順のまま() {
        let cfg = RerankConfig { lambda: 0.0, ..RerankConfig::default() };
        let out = reorder(s(&["機械", "機会"]), 0, &[-20.0, -2.0], &cfg);
        assert_eq!(out, s(&["機械", "機会"]));
    }

    #[test]
    fn split_targets_は学習とユーザー辞書を先頭に固定し上限で切る() {
        let merged = s(&["キカイ", "機械", "機会", "器械", "奇怪"]);
        let (pinned, n) = split_targets(&merged, &s(&["キカイ"]), &[], 3);
        assert_eq!((pinned, n), (1, 3));
    }

    #[test]
    fn rank_indices_は同点を辞書順で安定させる() {
        assert_eq!(rank_indices(&[-5.0, -5.0, -5.0], 1.0, 0.5), vec![0, 1, 2]);
    }

    #[test]
    fn 既知のモデルidはhfの置き場に解決され未知のidはエラーになる() {
        let known = RerankSettings { model: "Qwen3-0.6B-Q8_0".into(), ..RerankSettings::default() };
        // ダウンロードは走らせない: 解決先の (repo, file) が表にあることだけ確かめる
        assert!(MODEL_IDS.iter().any(|(id, _, _)| id.eq_ignore_ascii_case(known.model.trim())));
        let unknown = RerankSettings { model: "llama-99b".into(), ..RerankSettings::default() };
        assert!(resolve_model_path(&unknown).unwrap_err().contains("llama-99b"));
        let explicit = RerankSettings { model_path: Some("/tmp/x.gguf".into()), ..RerankSettings::default() };
        assert_eq!(resolve_model_path(&explicit).unwrap(), PathBuf::from("/tmp/x.gguf"));
    }

    /// 2 番目の候補を常に最良にする偽の採点器。`gate` があると、1 回分の合図が届くまで採点を止める。
    struct Fake {
        gate: Option<std::sync::mpsc::Receiver<()>>,
    }

    impl ScoreBackend for Fake {
        fn score(&mut self, _left: &str, _right: Option<&str>, candidates: &[String]) -> Result<Vec<f32>, String> {
            if let Some(g) = &self.gate {
                let _ = g.recv();
            }
            Ok((0..candidates.len()).map(|i| if i == 1 { -1.0 } else { -10.0 }).collect())
        }
    }

    fn fake(gate: Option<std::sync::mpsc::Receiver<()>>) -> Result<Box<dyn ScoreBackend>, String> {
        Ok(Box::new(Fake { gate }))
    }

    #[test]
    fn 採点器のロードに失敗しても辞書順のまま返る() {
        let rr = Reranker::spawn(RerankConfig::default(), Duration::from_secs(1), || Err("no model".into()));
        assert!(!rr.wait_ready(Duration::from_secs(5)));
        assert_eq!(rr.rerank("左", None, s(&["機械", "機会"]), &[], &[]), s(&["機械", "機会"]));
    }

    #[test]
    fn 文脈が左右どちらも無ければ並べ替えない() {
        let rr = Reranker::spawn(RerankConfig::default(), Duration::from_secs(1), || fake(None));
        assert!(rr.wait_ready(Duration::from_secs(5)));
        assert_eq!(rr.rerank("", None, s(&["機械", "機会"]), &[], &[]), s(&["機械", "機会"]));
        assert_eq!(rr.rerank("次の", None, s(&["機械", "機会"]), &[], &[]), s(&["機会", "機械"]));
        assert_eq!(rr.rerank("", Some("を待つ"), s(&["機械", "機会"]), &[], &[]), s(&["機会", "機械"]));
    }

    #[test]
    fn 右文脈が必須の設定では右文脈が無いと並べ替えない() {
        let cfg = RerankConfig { require_right_context: true, ..RerankConfig::default() };
        let rr = Reranker::spawn(cfg, Duration::from_secs(1), || fake(None));
        assert!(rr.wait_ready(Duration::from_secs(5)));
        assert_eq!(rr.rerank("左", None, s(&["機械", "機会"]), &[], &[]), s(&["機械", "機会"]));
        assert_eq!(rr.rerank("左", Some("を"), s(&["機械", "機会"]), &[], &[]), s(&["機会", "機械"]));
    }

    #[test]
    fn 採点がタイムアウトしたら辞書順のまま返し結果が届いた後は採点に戻る() {
        let (open, gate) = std::sync::mpsc::channel::<()>();
        let rr = Reranker::spawn(RerankConfig::default(), Duration::from_millis(20), move || fake(Some(gate)));
        assert!(rr.wait_ready(Duration::from_secs(5)));
        // 1 回目: 採点器が止まっているのでタイムアウト → 辞書順
        assert_eq!(rr.rerank("左", None, s(&["機械", "機会"]), &[], &[]), s(&["機械", "機会"]));
        // 2 回目: 見捨てた採点がまだ終わっていないのでスキップ → 辞書順
        assert_eq!(rr.rerank("左", None, s(&["機械", "機会"]), &[], &[]), s(&["機械", "機会"]));
        // 採点器を進める。見捨てた 1 回目の結果が届いた後は普通に採点される
        open.send(()).unwrap();
        for _ in 0..20 {
            open.send(()).unwrap();
        }
        let deadline = Instant::now() + Duration::from_secs(5);
        let mut out = s(&["機械", "機会"]);
        while out[0] != "機会" && Instant::now() < deadline {
            out = rr.rerank("左", None, s(&["機械", "機会"]), &[], &[]);
        }
        assert_eq!(out, s(&["機会", "機械"]));
    }

    /// 実モデルを使う。`RERANK_TEST_MODEL` に GGUF のパスが無ければ何もしない。
    #[test]
    fn 実モデルで左右の文脈から機会を先頭に上げる() {
        let Ok(path) = std::env::var("RERANK_TEST_MODEL") else { return };
        let rr = Reranker::load(path, RerankConfig::default(), 8).expect("model");
        let merged = s(&["機械", "機会", "器械", "奇怪"]);
        let out = rr.rerank("来週の打ち合わせは先方の都合で流れた。次の", Some("を待つしかない。"), merged, &[], &[]);
        assert_eq!(out[0], "機会");
        // 2 回目は prefix の KV を持ち越す経路。結果が変わらないこと
        let out2 = rr.rerank("来週の打ち合わせは先方の都合で流れた。次の", Some("を待つしかない。"), s(&["機械", "機会", "器械", "奇怪"]), &[], &[]);
        assert_eq!(out, out2);
        // 学習履歴由来は固定される
        let out3 = rr.rerank("工場の3号ラインで", Some("が停止した"), s(&["キカイ", "機会", "機械"]), &s(&["キカイ"]), &[]);
        assert_eq!(out3[0], "キカイ");
        assert_eq!(out3[1], "機械");
    }
}
