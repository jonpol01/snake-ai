#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::sync::Arc;
use tokio::sync::{mpsc, oneshot};

use snake_ai::protocol::ClientMsg;
use snake_ai::shared::{LogKind, SharedState};

fn main() {
    // Set working directory to project root (find Cargo.toml)
    if let Ok(exe) = std::env::current_exe() {
        let mut dir = exe.parent().map(|p| p.to_path_buf());
        while let Some(ref d) = dir {
            if d.join("Cargo.toml").exists() {
                let _ = std::env::set_current_dir(d);
                break;
            }
            dir = d.parent().map(|p| p.to_path_buf());
        }
    }

    let shared = Arc::new(SharedState::new());
    let (cmd_tx, cmd_rx) = mpsc::channel::<ClientMsg>(32);
    let (ready_tx, ready_rx) = oneshot::channel::<()>();

    // Spawn backend (axum + GPU + sim) in background thread
    let shared_bg = shared.clone();
    std::thread::spawn(move || {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async move {
            snake_ai::run_backend(shared_bg, cmd_tx, cmd_rx, Some(ready_tx)).await;
        });
    });

    shared.push_log("Starting Snake AI desktop app...".into(), LogKind::Info);

    // Wait for the axum server to be ready before opening the webview
    // (blocks main thread briefly — Tauri hasn't started its event loop yet)
    let _ = ready_rx.blocking_recv();

    // Build Tauri app — webview points at the running axum server
    tauri::Builder::default()
        .setup(|app| {
            use tauri::WebviewUrl;
            let url = WebviewUrl::External("http://localhost:3030".parse().unwrap());
            tauri::WebviewWindowBuilder::new(app, "main", url)
                .title("Snake AI")
                .inner_size(1200.0, 800.0)
                .min_inner_size(800.0, 600.0)
                .build()?;
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
