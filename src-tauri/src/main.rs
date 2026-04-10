#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::sync::Arc;
use tokio::sync::{mpsc, oneshot};

use snake_ai::protocol::ClientMsg;
use snake_ai::shared::{LogKind, SharedState};

fn main() {
    let shared = Arc::new(SharedState::new());
    let (cmd_tx, cmd_rx) = mpsc::channel::<ClientMsg>(32);
    let (ready_tx, ready_rx) = oneshot::channel::<()>();

    // Log data directory
    let data_dir = snake_ai::paths::data_dir();
    shared.push_log(format!("Data: {}", data_dir.display()), LogKind::Info);

    // Spawn backend in background thread
    let shared_bg = shared.clone();
    std::thread::spawn(move || {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async move {
            snake_ai::run_backend(shared_bg, cmd_tx, cmd_rx, Some(ready_tx)).await;
        });
    });

    // Wait for server readiness
    let _ = ready_rx.blocking_recv();

    // Tauri native window
    tauri::Builder::default()
        .setup(|app| {
            use tauri::WebviewUrl;
            let port = std::env::var("PORT").unwrap_or_else(|_| "3030".into());
            let url = WebviewUrl::External(format!("http://localhost:{}", port).parse().unwrap());
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
