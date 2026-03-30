mod gpu;
mod gui;
mod neural_net;
mod population;
mod protocol;
mod shared;
mod snake;

use std::sync::Arc;
use std::time::Instant;

use axum::extract::ws::{Message, WebSocket};
use axum::extract::WebSocketUpgrade;
use axum::response::IntoResponse;
use axum::routing::get;
use axum::Router;
use futures::SinkExt;
use futures::StreamExt;
use tokio::sync::{broadcast, mpsc};
use tower_http::services::ServeDir;

use gpu::GpuCompute;
use population::Population;
use protocol::{ClientMsg, ServerMsg};
use shared::{LogKind, SharedState};

fn main() {
    let shared = Arc::new(SharedState::new());

    // Build tokio runtime in a background thread (egui needs the main thread on macOS)
    let rt = tokio::runtime::Runtime::new().unwrap();
    let handle = rt.handle().clone();

    // Command channel: GUI + WebSocket -> sim loop
    let (cmd_tx, cmd_rx) = mpsc::channel::<ClientMsg>(32);

    // Start server + sim in background
    let shared_bg = shared.clone();
    let cmd_tx_bg = cmd_tx.clone();
    std::thread::spawn(move || {
        rt.block_on(async move {
            run_backend(shared_bg, cmd_tx_bg, cmd_rx).await;
        });
    });

    // Run GUI on main thread
    gui::run(shared, cmd_tx, handle);
}

async fn run_backend(
    shared: Arc<SharedState>,
    cmd_tx: mpsc::Sender<ClientMsg>,
    cmd_rx: mpsc::Receiver<ClientMsg>,
) {
    // Init GPU
    let gpu = GpuCompute::new().await;
    if let Some(ref g) = gpu {
        let info = format!("Metal — {}", g.adapter_name());
        shared.push_log(format!("GPU initialized: {}", info), LogKind::Done);
        if let Ok(mut b) = shared.gpu_backend.lock() {
            *b = info;
        }
    } else {
        shared.push_log("GPU not available, using CPU fallback".into(), LogKind::Error);
        if let Ok(mut b) = shared.gpu_backend.lock() {
            *b = "CPU fallback".into();
        }
    }
    let gpu = Arc::new(gpu);

    // Broadcast channel for WebSocket state updates
    let (state_tx, _) = broadcast::channel::<String>(16);
    let state_tx = Arc::new(state_tx);

    // Start sim loop
    let sim_shared = shared.clone();
    let sim_state_tx = state_tx.clone();
    let sim_gpu = gpu.clone();
    tokio::spawn(sim_loop(sim_shared, sim_state_tx, cmd_rx, sim_gpu));

    // Start HTTP server
    let app = Router::new()
        .route(
            "/ws",
            get({
                let state_tx = state_tx.clone();
                let cmd_tx = cmd_tx.clone();
                move |ws: WebSocketUpgrade| {
                    let state_tx = state_tx.clone();
                    let cmd_tx = cmd_tx.clone();
                    async move { ws_handler(ws, state_tx, Arc::new(cmd_tx)).await }
                }
            }),
        )
        .route(
            "/logs",
            get({
                let shared = shared.clone();
                move || {
                    let shared = shared.clone();
                    async move { get_logs(shared).await }
                }
            }),
        )
        .fallback_service(ServeDir::new("static"));

    let addr = "0.0.0.0:3030";
    shared.push_log(format!("Server listening on http://{}", addr), LogKind::Info);

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

async fn get_logs(shared: Arc<SharedState>) -> axum::Json<serde_json::Value> {
    let lines = shared.log_lines.lock().unwrap();
    let entries: Vec<serde_json::Value> = lines
        .iter()
        .map(|e| {
            serde_json::json!({
                "timestamp": e.timestamp,
                "message": e.message,
                "kind": match e.kind {
                    LogKind::Info => "info",
                    LogKind::Phase => "phase",
                    LogKind::Error => "error",
                    LogKind::Done => "done",
                    LogKind::Stat => "stat",
                }
            })
        })
        .collect();
    axum::Json(serde_json::json!(entries))
}

async fn ws_handler(
    ws: WebSocketUpgrade,
    state_tx: Arc<broadcast::Sender<String>>,
    cmd_tx: Arc<mpsc::Sender<ClientMsg>>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_socket(socket, state_tx, cmd_tx))
}

async fn handle_socket(
    socket: WebSocket,
    state_tx: Arc<broadcast::Sender<String>>,
    cmd_tx: Arc<mpsc::Sender<ClientMsg>>,
) {
    let (mut ws_tx, mut ws_rx) = socket.split();
    let mut state_rx = state_tx.subscribe();

    let send_task = tokio::spawn(async move {
        while let Ok(msg) = state_rx.recv().await {
            if ws_tx.send(Message::Text(msg.into())).await.is_err() {
                break;
            }
        }
    });

    while let Some(Ok(msg)) = ws_rx.next().await {
        if let Message::Text(text) = msg {
            if let Ok(cmd) = serde_json::from_str::<ClientMsg>(&text) {
                let _ = cmd_tx.send(cmd).await;
            }
        }
    }

    send_task.abort();
}

async fn sim_loop(
    shared: Arc<SharedState>,
    state_tx: Arc<broadcast::Sender<String>>,
    mut cmd_rx: mpsc::Receiver<ClientMsg>,
    gpu: Arc<Option<GpuCompute>>,
) {
    let mut population: Option<Population> = None;
    let mut running = false;
    let mut speed: u32 = 10;
    let mut frame_interval = tokio::time::interval(std::time::Duration::from_millis(33));

    // Helper: push log to both SharedState and WebSocket
    let log = |shared: &SharedState, state_tx: &broadcast::Sender<String>, msg: String, kind: LogKind| {
        shared.push_log(msg.clone(), kind);
        let kind_str = match kind {
            LogKind::Info => "info",
            LogKind::Phase => "phase",
            LogKind::Error => "error",
            LogKind::Done => "done",
            LogKind::Stat => "stat",
        };
        let log_msg = ServerMsg::Log {
            timestamp: chrono::Local::now().format("%H:%M:%S").to_string(),
            message: msg,
            kind: kind_str.into(),
        };
        if let Ok(json) = serde_json::to_string(&log_msg) {
            let _ = state_tx.send(json);
        }
    };

    loop {
        tokio::select! {
            _ = frame_interval.tick() => {
                if !running || population.is_none() {
                    continue;
                }

                let pop = population.as_mut().unwrap();

                for _ in 0..speed {
                    if pop.done() {
                        pop.calculate_fitness();

                        let gen_best = pop.snakes.iter().map(|s| s.score).max().unwrap_or(0);
                        let all_time_best = pop.best_scores.iter().copied().max().unwrap_or(0);

                        // Log generation completion
                        log(&shared, &state_tx,
                            format!(
                                "Gen {} complete — best: {}, all-time: {}, compute: {:.2}ms",
                                pop.gen, gen_best, all_time_best, pop.last_compute_ms
                            ),
                            LogKind::Stat,
                        );

                        if gen_best > all_time_best {
                            log(&shared, &state_tx,
                                format!("New all-time best score: {}!", gen_best),
                                LogKind::Done,
                            );
                        }

                        pop.natural_selection();

                        // Auto-save checkpoint every 10 generations
                        if pop.gen % 10 == 0 {
                            pop.save_checkpoint();
                            log(&shared, &state_tx,
                                format!("Checkpoint saved (gen {})", pop.gen),
                                LogKind::Info,
                            );
                        }

                        let graph_msg = ServerMsg::Graph {
                            scores: pop.best_scores.clone(),
                        };
                        if let Ok(json) = serde_json::to_string(&graph_msg) {
                            let _ = state_tx.send(json);
                        }
                    }

                    pop.look_all();

                    let t0 = Instant::now();
                    if let Some(ref g) = *gpu {
                        g.forward_pass(&mut pop.snakes).await;
                    } else {
                        gpu::cpu_forward_pass(&mut pop.snakes);
                    }
                    pop.last_compute_ms = t0.elapsed().as_secs_f64() * 1000.0;

                    pop.move_all();
                }

                // Update shared stats
                let alive_count = pop.alive_count();
                if let Ok(mut g) = shared.gen.lock() { *g = pop.gen; }
                if let Ok(mut b) = shared.best_score.lock() { *b = pop.best_scores.iter().copied().max().unwrap_or(0); }
                if let Ok(mut a) = shared.alive.lock() { *a = format!("{}/{}", alive_count, population::POP_SIZE); }
                if let Ok(mut g) = shared.gpu_ms.lock() { *g = pop.last_compute_ms; }

                // Find best alive snake for browser
                let gpu_ms = pop.last_compute_ms;
                let best = pop
                    .snakes
                    .iter()
                    .filter(|s| !s.dead)
                    .max_by(|a, b| a.score.cmp(&b.score).then(a.lifetime.cmp(&b.lifetime)));

                if let Some(best) = best {
                    let msg = ServerMsg::State {
                        gen: pop.gen,
                        best_score: pop.best_scores.iter().copied().max().unwrap_or(0),
                        alive: alive_count,
                        total: population::POP_SIZE,
                        gpu_ms,
                        snake_body: best.body.clone(),
                        food: (best.food_x, best.food_y),
                        score: best.score,
                        vision: best.vision.to_vec(),
                        decision: best.decision.to_vec(),
                        nn_weights: best.brain.clone(),
                    };
                    if let Ok(json) = serde_json::to_string(&msg) {
                        let _ = state_tx.send(json);
                    }
                }
            }

            Some(cmd) = cmd_rx.recv() => {
                match cmd {
                    ClientMsg::Start => {
                        if let Some(restored) = Population::from_checkpoint() {
                            let gen = restored.gen;
                            let best = restored.best_scores.iter().copied().max().unwrap_or(0);
                            // Send graph data immediately
                            let graph_msg = ServerMsg::Graph { scores: restored.best_scores.clone() };
                            if let Ok(json) = serde_json::to_string(&graph_msg) {
                                let _ = state_tx.send(json);
                            }
                            population = Some(restored);
                            log(&shared, &state_tx,
                                format!("Resumed from checkpoint — gen {}, best score: {}", gen, best),
                                LogKind::Done);
                        } else {
                            population = Some(Population::new());
                            log(&shared, &state_tx,"Evolution started — population: 2000".into(), LogKind::Phase);
                        }
                        running = true;
                        if let Ok(mut r) = shared.running.lock() { *r = true; }
                    }
                    ClientMsg::Pause => {
                        running = false;
                        if let Ok(mut r) = shared.running.lock() { *r = false; }
                        log(&shared, &state_tx,"Simulation paused".into(), LogKind::Info);
                    }
                    ClientMsg::Resume => {
                        running = true;
                        if let Ok(mut r) = shared.running.lock() { *r = true; }
                        log(&shared, &state_tx,"Simulation resumed".into(), LogKind::Info);
                    }
                    ClientMsg::Speed { value } => {
                        speed = value.max(1).min(500);
                        if let Ok(mut s) = shared.speed.lock() { *s = speed; }
                    }
                }
            }
        }
    }
}
