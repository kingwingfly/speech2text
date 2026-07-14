//! Leptos UI (compiles for both ssr and hydrate) plus the shared WebSocket
//! message type. The browser-only mic/WebSocket plumbing lives in `capture`.

use leptos::prelude::*;
use leptos_meta::{MetaTags, Stylesheet, Title, provide_meta_context};
use leptos_router::components::{Route, Router, Routes};
use leptos_router::path;
use std::cell::RefCell;
use std::collections::VecDeque;
use std::rc::Rc;

use crate::capture;

/// Messages the server streams to the browser over `/ws`.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum SttMsg {
    /// The current, still-in-progress utterance (updates in place).
    Partial { text: String },
    /// A completed line — the browser enqueues it into the paced reader.
    Final { text: String },
    /// Connection/engine status.
    Status { state: String },
}

/// SSR document shell.
pub fn shell(options: LeptosOptions) -> impl IntoView {
    view! {
        <!DOCTYPE html>
        <html lang="zh">
            <head>
                <meta charset="utf-8"/>
                <meta name="viewport" content="width=device-width, initial-scale=1"/>
                <AutoReload options=options.clone()/>
                <HydrationScripts options/>
                <MetaTags/>
            </head>
            <body>
                <App/>
            </body>
        </html>
    }
}

#[component]
pub fn App() -> impl IntoView {
    provide_meta_context();
    view! {
        <Stylesheet id="leptos" href="/pkg/stt.css"/>
        <Title text="实时语音转文字"/>
        <Router>
            <Routes fallback=|| view! { <p>"Not found"</p> }>
                <Route path=path!("/") view=Realtime/>
            </Routes>
        </Router>
    }
}

#[component]
fn Realtime() -> impl IntoView {
    let running = RwSignal::new(false);
    // Reader controls.
    let font_px = RwSignal::new(48u32);
    let dwell = RwSignal::new(4.0f64); // seconds each finalized line stays before the next.
    // Transcript state.
    let partial = RwSignal::new(String::new());
    let visible = RwSignal::new(Vec::<String>::new());
    let queue = RwSignal::new(VecDeque::<String>::new());
    // User-facing error/status (e.g. mic permission or insecure-context hints).
    let status = RwSignal::new(String::new());

    // Paced release pump: moves queued finalized lines into view one at a time,
    // honoring the dwell time. Runs on the client only.
    Effect::new(move |_| {
        capture::start_pump(queue, visible, dwell);
    });

    // Live capture handle (browser only). Rc<RefCell<..>> because it holds
    // non-Send web_sys objects; wasm is single-threaded so that's fine.
    let handle: Rc<RefCell<Option<capture::Capture>>> = Rc::new(RefCell::new(None));

    let toggle = {
        let handle = handle.clone();
        move |_| {
            let mut h = handle.borrow_mut();
            if running.get() {
                if let Some(c) = h.take() {
                    capture::stop(c);
                }
                partial.set(String::new());
                running.set(false);
            } else {
                status.set(String::new());
                // On failure `status` already carries a user-facing message.
                if let Some(c) = capture::start(partial, queue, running, status) {
                    *h = Some(c);
                    running.set(true);
                }
            }
        }
    };

    view! {
        <div class="min-h-screen bg-black text-white flex flex-col">
            // ── Controls ─────────────────────────────────────────────
            <div class="flex flex-wrap items-center gap-6 p-4 bg-neutral-900 border-b border-neutral-700">
                <button
                    class=move || format!(
                        "px-8 py-3 rounded-xl text-2xl font-bold transition-colors {}",
                        if running.get() { "bg-red-600 hover:bg-red-500" } else { "bg-green-600 hover:bg-green-500" },
                    )
                    on:click=toggle
                >
                    {move || if running.get() { "停止" } else { "开始" }}
                </button>

                <label class="flex items-center gap-3 text-xl">
                    "字号"
                    <input
                        type="range" min="24" max="96" step="2"
                        class="w-40"
                        prop:value=move || font_px.get().to_string()
                        on:input=move |ev| {
                            if let Ok(v) = event_target_value(&ev).parse::<u32>() { font_px.set(v); }
                        }
                    />
                    <span class="tabular-nums w-16">{move || format!("{}px", font_px.get())}</span>
                </label>

                <label class="flex items-center gap-3 text-xl">
                    "停留"
                    <input
                        type="range" min="1" max="15" step="0.5"
                        class="w-40"
                        prop:value=move || dwell.get().to_string()
                        on:input=move |ev| {
                            if let Ok(v) = event_target_value(&ev).parse::<f64>() { dwell.set(v); }
                        }
                    />
                    <span class="tabular-nums w-20">{move || format!("{:.1}秒", dwell.get())}</span>
                </label>

                <Show when=move || !queue.get().is_empty()>
                    <span class="ml-auto text-2xl text-amber-400 tabular-nums">
                        {move || format!("⏳ {} 等待中", queue.get().len())}
                    </span>
                </Show>
            </div>

            // ── Error / status banner ────────────────────────────────
            <Show when=move || !status.get().is_empty()>
                <div class="p-3 bg-red-900 text-red-100 text-lg">{move || status.get()}</div>
            </Show>

            // ── Paced reading pane ───────────────────────────────────
            <div
                class="flex-1 overflow-y-auto flex flex-col justify-end p-8 gap-4 leading-relaxed"
                style:font-size=move || format!("{}px", font_px.get())
            >
                {move || {
                    visible
                        .get()
                        .into_iter()
                        .enumerate()
                        .map(|(i, line)| {
                            let last = i + 1 == visible.get().len();
                            let cls = if last { "text-white" } else { "text-neutral-400" };
                            view! { <p class=cls>{line}</p> }
                        })
                        .collect_view()
                }}
            </div>

            // ── Live "listening…" preview ────────────────────────────
            <div class="p-4 bg-neutral-900 border-t border-neutral-700 text-neutral-400 italic text-xl min-h-[3.5rem]">
                {move || {
                    let p = partial.get();
                    if running.get() {
                        if p.is_empty() { "聆听中…".to_string() } else { p }
                    } else {
                        "已停止".to_string()
                    }
                }}
            </div>
        </div>
    }
}
