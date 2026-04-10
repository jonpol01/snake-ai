use std::sync::Arc;
use tokio::sync::mpsc;

use snake_ai::protocol::ClientMsg;
use snake_ai::shared::{LogKind, SharedState};

fn main() {
    let shared = Arc::new(SharedState::new());
    let rt = tokio::runtime::Runtime::new().unwrap();
    let (cmd_tx, cmd_rx) = mpsc::channel::<ClientMsg>(32);

    // Log data directory on startup
    let data_dir = snake_ai::paths::data_dir();
    eprintln!("Data directory: {}", data_dir.display());

    let shared_bg = shared.clone();
    std::thread::spawn(move || {
        rt.block_on(async move {
            snake_ai::run_backend(shared_bg, cmd_tx, cmd_rx, None).await;
        });
    });

    shared.push_log(
        format!("Headless mode — data: {} — browser: http://localhost:3030", data_dir.display()),
        LogKind::Info,
    );

    loop {
        std::thread::park();
    }
}
