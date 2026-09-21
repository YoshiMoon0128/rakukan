//! composition 周辺のテキストを読む（同音異義語リランカーの左右の文脈）。
//!
//! 変換キーを押した時点で、composition の直前 `left_chars` 文字と直後 `right_chars` 文字を
//! `ITfRange::GetText` で読み、`RpcEngine::set_surrounding_context` に預ける。以降の
//! `merge_candidates_for_reading` はこの文脈を添えてホストへ送り、engine 側のリランカーが
//! 辞書候補を並べ替える。確定・リセットで文脈は捨てる（`RpcEngine` 側）。
//!
//! # なぜ `TF_ES_SYNC` を使うか
//! factory.rs の原則は「`OnKeyDown` をブロックしない・`TF_ES_SYNC` を使わない」だが、
//! Space の `on_convert[new]` は LLM 変換の完了まで TSF スレッドをブロックする特例経路で、
//! 文脈は最初の RPC より前に手元に要る。`TF_ES_READ | TF_ES_SYNC` はキーイベントの
//! シンク内からなら TSF が同期実行を認めており（他の場面では `TF_E_SYNCHRONOUS` で
//! 拒否される）、読むだけなので edit session は数十マイクロ秒で終わる。拒否されたら
//! 文脈なし（None）で進み、engine は確定文を左文脈に使う。
//!
//! # 取れない場面
//! `RequestEditSession` が拒否される、`GetText` が失敗する、アプリが周辺テキストを
//! 公開しない（空文字が返る）。回数を数えてログに出し、「読めない場面の割合」を
//! 実機で測れるようにする。

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

use windows::Win32::UI::TextServices::{
    ITfContext, ITfRange, TF_ANCHOR_END, TF_ANCHOR_START, TF_CONTEXT_EDIT_CONTEXT_FLAGS,
    TF_ES_READ, TF_ES_SYNC, TF_HALTCOND,
};

use crate::engine::state::{DynEngine, composition_clone};
use crate::tsf::edit_session::EditSession;

static READ_OK: AtomicU32 = AtomicU32::new(0);
static READ_FAIL: AtomicU32 = AtomicU32::new(0);

/// 読めた・読めなかった回数（診断用）。
pub fn read_counts() -> (u32, u32) {
    (READ_OK.load(Ordering::Relaxed), READ_FAIL.load(Ordering::Relaxed))
}

/// `[rerank].enabled` のときだけ composition 周辺のテキストを読み、engine ハンドルに預ける。
/// 読めなかったときも預け直す（前回の Space の文脈を残さないため）。
pub(super) fn refresh_surrounding_context(ctx: &ITfContext, tid: u32, engine: &DynEngine) {
    let cfg = crate::engine::config::current_config().rerank;
    if !cfg.enabled {
        return;
    }
    match read_surrounding_text(ctx, tid, cfg.left_chars, cfg.right_chars) {
        Some((left, right)) => {
            READ_OK.fetch_add(1, Ordering::Relaxed);
            let (ok, fail) = read_counts();
            tracing::debug!(
                "context_read: left={} chars right={} chars (ok={ok} fail={fail})",
                left.as_deref().map_or(0, |s| s.chars().count()),
                right.as_deref().map_or(0, |s| s.chars().count()),
            );
            tracing::trace!("context_read: left={left:?} right={right:?}");
            engine.set_surrounding_context(left, right);
        }
        None => {
            READ_FAIL.fetch_add(1, Ordering::Relaxed);
            let (ok, fail) = read_counts();
            tracing::debug!("context_read: unavailable (ok={ok} fail={fail})");
            engine.set_surrounding_context(None, None);
        }
    }
}

/// composition（無ければ現在の選択範囲）の直前 `left_chars` 文字と直後 `right_chars` 文字を読む。
///
/// 戻り値の外側 `None` は edit session が取れなかった。内側の `None` はその側だけ読めなかった。
/// 文書の端で文字が無いときは `Some("")`（読めたが何も無い）。
pub(super) fn read_surrounding_text(
    ctx: &ITfContext,
    tid: u32,
    left_chars: usize,
    right_chars: usize,
) -> Option<(Option<String>, Option<String>)> {
    let result: Arc<Mutex<Option<(Option<String>, Option<String>)>>> = Arc::new(Mutex::new(None));
    let slot = Arc::clone(&result);
    let ctx_in = ctx.clone();
    let comp = composition_clone().ok().flatten();
    let session = EditSession::new(move |ec| {
        let anchor: ITfRange = match comp.as_ref().and_then(|c| unsafe { c.GetRange() }.ok()) {
            Some(r) => r,
            None => match unsafe { super::on_compose::get_cursor_range(&ctx_in, ec) } {
                Some(r) => r,
                None => return Ok(()),
            },
        };
        let left = unsafe { read_side(&anchor, ec, Side::Left, left_chars) };
        let right = unsafe { read_side(&anchor, ec, Side::Right, right_chars) };
        if let Ok(mut g) = slot.lock() {
            *g = Some((left, right));
        }
        Ok(())
    });
    let flags = TF_CONTEXT_EDIT_CONTEXT_FLAGS(TF_ES_READ.0 | TF_ES_SYNC.0);
    match unsafe { ctx.RequestEditSession(tid, &session, flags) } {
        Ok(hr) if hr.is_ok() => {}
        Ok(hr) => {
            tracing::debug!("context_read: RequestEditSession refused: {hr:?}");
            return None;
        }
        Err(e) => {
            tracing::debug!("context_read: RequestEditSession failed: {e}");
            return None;
        }
    }
    // 同期実行なら閉じたクロージャがもう走っている。非同期に落とされた（TF_S_ASYNC）なら
    // まだ空なので None を返す。
    result.lock().ok().and_then(|g| g.clone())
}

#[derive(Clone, Copy)]
enum Side {
    Left,
    Right,
}

/// `anchor` の片側に `n` 文字（UTF-16 単位）伸ばした range のテキストを読む。
unsafe fn read_side(anchor: &ITfRange, ec: u32, side: Side, n: usize) -> Option<String> {
    if n == 0 {
        return None;
    }
    let want = i32::try_from(n).ok()?;
    let r = unsafe { anchor.Clone() }.ok()?;
    let mut shifted: i32 = 0;
    let halt = std::ptr::null::<TF_HALTCOND>();
    match side {
        Side::Left => {
            unsafe { r.Collapse(ec, TF_ANCHOR_START) }.ok()?;
            unsafe { r.ShiftStart(ec, -want, &mut shifted, halt) }.ok()?;
        }
        Side::Right => {
            unsafe { r.Collapse(ec, TF_ANCHOR_END) }.ok()?;
            unsafe { r.ShiftEnd(ec, want, &mut shifted, halt) }.ok()?;
        }
    }
    let units = shifted.unsigned_abs() as usize;
    if units == 0 {
        return Some(String::new());
    }
    let mut buf = vec![0u16; units + 1];
    let mut got: u32 = 0;
    unsafe { r.GetText(ec, 0, &mut buf, &mut got) }.ok()?;
    let got = (got as usize).min(buf.len());
    Some(String::from_utf16_lossy(&buf[..got]))
}
