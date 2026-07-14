//! Browser-only mic capture + WebSocket streaming, and the paced-release pump.
//!
//! Everything that touches `web_sys` lives behind `hydrate`; the `ssr` build
//! gets no-op stubs with the same signatures so `app.rs` compiles either way.

use leptos::prelude::RwSignal;
use std::collections::VecDeque;

#[cfg(feature = "hydrate")]
mod imp {
    use super::*;
    use leptos::prelude::*;
    use std::cell::RefCell;
    use std::rc::Rc;
    use wasm_bindgen::prelude::*;
    use wasm_bindgen_futures::{JsFuture, spawn_local};
    use web_sys::{
        AudioContext, AudioContextOptions, AudioProcessingEvent, BinaryType, GainNode, MediaStream,
        MediaStreamAudioSourceNode, MediaStreamConstraints, MediaStreamTrack, MessageEvent,
        ScriptProcessorNode, WebSocket,
    };

    use crate::app::SttMsg;

    /// Live-capture handle. Owning it keeps the mic graph, WebSocket, and JS
    /// closures alive; dropping it (via [`stop`]) tears everything down.
    pub struct Capture {
        ws: WebSocket,
        ctx: AudioContext,
        stream: Rc<RefCell<Option<MediaStream>>>,
        _source: Rc<RefCell<Option<MediaStreamAudioSourceNode>>>,
        _processor: ScriptProcessorNode,
        _gain: GainNode,
        _on_audio: Closure<dyn FnMut(AudioProcessingEvent)>,
        _on_msg: Closure<dyn FnMut(MessageEvent)>,
    }

    /// Start capturing. Returns `None` on any failure, after setting `status`
    /// with a user-facing (Chinese) message — it never panics.
    pub fn start(
        partial: RwSignal<String>,
        queue: RwSignal<VecDeque<String>>,
        running: RwSignal<bool>,
        status: RwSignal<String>,
    ) -> Option<Capture> {
        match build(partial, queue, running, status) {
            Ok(cap) => Some(cap),
            Err(msg) => {
                status.set(msg);
                None
            }
        }
    }

    fn build(
        partial: RwSignal<String>,
        queue: RwSignal<VecDeque<String>>,
        running: RwSignal<bool>,
        status: RwSignal<String>,
    ) -> Result<Capture, String> {
        let window = web_sys::window().ok_or_else(|| "无法访问浏览器窗口".to_string())?;
        let navigator = window.navigator();

        // `navigator.mediaDevices` is undefined outside a secure context (i.e.
        // plain HTTP on anything but localhost). Detect that up front and
        // explain it instead of throwing deep in getUserMedia.
        let nav_js: &JsValue = navigator.as_ref();
        let has_media = js_sys::Reflect::get(nav_js, &JsValue::from_str("mediaDevices"))
            .map(|v| !v.is_undefined() && !v.is_null())
            .unwrap_or(false);
        if !has_media {
            return Err(
                "无法访问麦克风：请通过 HTTPS 访问（在前面加反向代理，如 Pingora/Cloudflare），或在本机用 http://localhost 打开。"
                    .to_string(),
            );
        }

        // Same-origin WebSocket (wss when the page is https).
        let loc = window.location();
        let proto = if loc.protocol().unwrap_or_default() == "https:" { "wss" } else { "ws" };
        let host = loc.host().unwrap_or_default();
        let ws = WebSocket::new(&format!("{proto}://{host}/ws"))
            .map_err(|e| format!("无法连接服务器：{e:?}"))?;
        ws.set_binary_type(BinaryType::Arraybuffer);

        // Server → browser: partial/final text.
        let on_msg = Closure::<dyn FnMut(MessageEvent)>::new(move |e: MessageEvent| {
            let Some(txt) = e.data().as_string() else { return };
            let Ok(msg) = serde_json::from_str::<SttMsg>(&txt) else { return };
            match msg {
                SttMsg::Partial { text } => partial.set(text),
                SttMsg::Final { text } => {
                    partial.set(String::new());
                    queue.update(|q| q.push_back(text));
                }
                SttMsg::Status { .. } => {}
            }
        });
        ws.set_onmessage(Some(on_msg.as_ref().unchecked_ref()));

        // Force a 16 kHz context so the browser resamples for us; the
        // ScriptProcessor then hands us 16 kHz mono f32 directly.
        let opts = AudioContextOptions::new();
        opts.set_sample_rate(16_000.0);
        let ctx = AudioContext::new_with_context_options(&opts)
            .map_err(|e| format!("无法创建音频上下文：{e:?}"))?;
        let processor = ctx
            .create_script_processor_with_buffer_size_and_number_of_input_channels_and_number_of_output_channels(4096, 1, 1)
            .map_err(|e| format!("无法创建音频处理器：{e:?}"))?;
        let gain = ctx.create_gain().map_err(|e| format!("无法创建增益节点：{e:?}"))?;
        gain.gain().set_value(0.0); // silent sink so onaudioprocess fires without feedback.

        // Each audio block → raw f32 little-endian bytes over the socket.
        let ws_send = ws.clone();
        let on_audio = Closure::<dyn FnMut(AudioProcessingEvent)>::new(move |e: AudioProcessingEvent| {
            if ws_send.ready_state() != WebSocket::OPEN {
                return;
            }
            let Ok(buf) = e.input_buffer() else { return };
            let Ok(data) = buf.get_channel_data(0) else { return };
            // wasm is little-endian, matching the server's `f32::from_le_bytes`.
            let bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(&data[..]))
            };
            let _ = ws_send.send_with_u8_array(bytes);
        });
        processor.set_onaudioprocess(Some(on_audio.as_ref().unchecked_ref()));
        let _ = processor.connect_with_audio_node(&gain);
        let _ = gain.connect_with_audio_node(&ctx.destination());

        // Ask for the mic, then wire it into the graph once granted. Failures
        // here (permission denied, no device) arrive asynchronously.
        let devices =
            navigator.media_devices().map_err(|e| format!("无法访问麦克风设备：{e:?}"))?;
        let constraints = MediaStreamConstraints::new();
        constraints.set_audio(&JsValue::TRUE);
        let promise = devices
            .get_user_media_with_constraints(&constraints)
            .map_err(|e| format!("无法启动麦克风：{e:?}"))?;

        let stream = Rc::new(RefCell::new(None));
        let source = Rc::new(RefCell::new(None));
        {
            let ctx = ctx.clone();
            let processor = processor.clone();
            let stream = stream.clone();
            let source = source.clone();
            spawn_local(async move {
                match JsFuture::from(promise).await {
                    Ok(js) => match js.dyn_into::<MediaStream>() {
                        Ok(ms) => match ctx.create_media_stream_source(&ms) {
                            Ok(src) => {
                                let _ = src.connect_with_audio_node(&processor);
                                *stream.borrow_mut() = Some(ms);
                                *source.borrow_mut() = Some(src);
                            }
                            Err(e) => fail(running, status, format!("无法连接麦克风：{e:?}")),
                        },
                        Err(_) => fail(running, status, "麦克风返回了非预期的数据".to_string()),
                    },
                    Err(e) => fail(
                        running,
                        status,
                        format!("麦克风被拒绝或不可用：请允许麦克风权限后重试。({e:?})"),
                    ),
                }
            });
        }

        Ok(Capture {
            ws,
            ctx,
            stream,
            _source: source,
            _processor: processor,
            _gain: gain,
            _on_audio: on_audio,
            _on_msg: on_msg,
        })
    }

    /// Report an async capture failure: flip the button back off and show why.
    fn fail(running: RwSignal<bool>, status: RwSignal<String>, msg: String) {
        leptos::logging::error!("{msg}");
        status.set(msg);
        running.set(false);
    }

    pub fn stop(cap: Capture) {
        let _ = cap.ws.close();
        if let Some(ms) = cap.stream.borrow().as_ref() {
            for track in ms.get_tracks().iter() {
                if let Ok(t) = track.dyn_into::<MediaStreamTrack>() {
                    t.stop();
                }
            }
        }
        let _ = cap.ctx.close();
        // Dropping `cap` frees the retained closures.
    }

    /// Interval-driven pump that releases one queued line per `dwell` seconds
    /// into `visible`, keeping only the most recent handful on screen.
    pub fn start_pump(
        queue: RwSignal<VecDeque<String>>,
        visible: RwSignal<Vec<String>>,
        dwell: RwSignal<f64>,
    ) {
        const KEEP: usize = 6;
        let Some(window) = web_sys::window() else { return };
        let Some(perf) = window.performance() else { return };
        let last = Rc::new(RefCell::new(perf.now()));

        let perf_cb = perf.clone();
        let cb = Closure::<dyn FnMut()>::new(move || {
            let empty = queue.with_untracked(|q| q.is_empty());
            if empty {
                return;
            }
            let now = perf_cb.now();
            if now - *last.borrow() < dwell.get_untracked() * 1000.0 {
                return;
            }
            *last.borrow_mut() = now;
            let mut line = None;
            queue.update(|q| line = q.pop_front());
            if let Some(l) = line {
                visible.update(|v| {
                    v.push(l);
                    let n = v.len();
                    if n > KEEP {
                        v.drain(0..n - KEEP);
                    }
                });
            }
        });
        let _ = window.set_interval_with_callback_and_timeout_and_arguments_0(
            cb.as_ref().unchecked_ref(),
            150,
        );
        cb.forget(); // pump lives for the page's lifetime.
    }
}

#[cfg(not(feature = "hydrate"))]
mod imp {
    use super::*;

    pub struct Capture;

    pub fn start(
        _partial: RwSignal<String>,
        _queue: RwSignal<VecDeque<String>>,
        _running: RwSignal<bool>,
        _status: RwSignal<String>,
    ) -> Option<Capture> {
        None
    }
    pub fn stop(_cap: Capture) {}
    pub fn start_pump(
        _queue: RwSignal<VecDeque<String>>,
        _visible: RwSignal<Vec<String>>,
        _dwell: RwSignal<f64>,
    ) {
    }
}

pub use imp::{Capture, start, start_pump, stop};
