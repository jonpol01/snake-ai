use std::sync::Arc;
use tokio::sync::mpsc;

use snake_ai::shared::{LogKind, SharedState};
use snake_ai::protocol::ClientMsg;

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
    let rt = tokio::runtime::Runtime::new().unwrap();
    let (cmd_tx, cmd_rx) = mpsc::channel::<ClientMsg>(32);

    // Start backend in background thread
    let shared_bg = shared.clone();
    std::thread::spawn(move || {
        rt.block_on(async move {
            snake_ai::run_backend(shared_bg, cmd_tx, cmd_rx, None).await;
        });
    });

    shared.push_log(
        "Running in headless mode — use browser at http://localhost:3030".into(),
        LogKind::Info,
    );

    // Block main thread forever (server runs in background)
    loop {
        std::thread::park();
    }
}
