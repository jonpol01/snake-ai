mod gpu;
mod gui;
mod leaderboard;
mod llm;
mod neural_net;
mod population;
mod protocol;
mod shared;
mod snake;
mod stage;

use std::sync::Arc;
use std::time::Instant;

use axum::extract::ws::{Message, WebSocket};
use axum::extract::WebSocketUpgrade;
use axum::response::{Html, IntoResponse};
use axum::routing::{get, post};
use axum::Router;
use futures::SinkExt;
use futures::StreamExt;
use tokio::sync::{broadcast, mpsc};

use gpu::GpuCompute;
use population::Population;
use protocol::{ClientMsg, ServerMsg};
use shared::{LogKind, SharedState};
use stage::StageKind;

fn main() {
    // Set working directory to the project root so checkpoint.json / leaderboard.json
    // are always found regardless of how the app is launched (double-click, terminal, etc.)
    // Walk up from the binary to find the directory containing Cargo.toml.
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
    let (state_tx, _) = broadcast::channel::<String>(64);
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
        .route(
            "/info",
            get({
                let shared = shared.clone();
                move || {
                    let shared = shared.clone();
                    async move { get_info(shared).await }
                }
            }),
        )
        .route("/leaderboard", get(get_leaderboard))
        .route("/api/llm/chat", post(llm::llm_chat))
        .route("/api/llm/models", post(llm::llm_models))
        .route("/", get(serve_index));

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

async fn get_info(shared: Arc<SharedState>) -> axum::Json<serde_json::Value> {
    let backend = shared.gpu_backend.lock().unwrap().clone();
    axum::Json(serde_json::json!({
        "backend": backend,
    }))
}

async fn get_leaderboard() -> axum::Json<serde_json::Value> {
    let lb = leaderboard::Leaderboard::load();
    axum::Json(serde_json::json!(lb.entries))
}

async fn serve_index() -> Html<&'static str> {
    Html(include_str!("../static/index.html"))
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
    let mut lb = leaderboard::Leaderboard::load();
    let mut speed: u32 = 10;
    let mut stage_kind = StageKind::Classic;
    let mut coach_interval: u32 = 0; // 0 = disabled, N = every N generations
    let mut last_coach_gen: u32 = 0;
    let mut live_best_score: u32 = 0; // Track best score seen mid-generation
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
                        // Send death frame: show the best snake from this gen
                        let best_dead = pop.snakes.iter().enumerate()
                            .max_by(|(_, a), (_, b)| a.score.cmp(&b.score).then(a.lifetime.cmp(&b.lifetime)));
                        if let Some((idx, best)) = best_dead {
                            // Include this gen's best in the all-time display
                            let current_best = pop.best_scores.iter().copied().max().unwrap_or(0).max(best.score);
                            let msg = ServerMsg::State {
                                gen: pop.gen,
                                best_score: current_best,
                                alive: 0,
                                total: population::POP_SIZE,
                                gpu_ms: pop.last_compute_ms,
                                snake_body: best.body.clone(),
                                food: (best.food_x, best.food_y),
                                score: best.score,
                                vision: best.vision.to_vec(),
                                decision: best.decision.to_vec(),
                                nn_weights: best.brain.clone(),
                                snake_id: idx,
                                dead: true,
                                obstacles: pop.stage.obstacle_list(),
                            };
                            if let Ok(json) = serde_json::to_string(&msg) {
                                let _ = state_tx.send(json);
                            }
                        }

                        // Reset live tracker for next gen
                        live_best_score = 0;

                        // Brief pause so the death frame is visible
                        tokio::time::sleep(std::time::Duration::from_millis(400)).await;

                        pop.calculate_fitness();

                        let gen_best = pop.snakes.iter().map(|s| s.score).max().unwrap_or(0);
                        // Use the true all-time best including this generation
                        let all_time_best = pop.best_scores.iter().copied().max().unwrap_or(0).max(gen_best);

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

                            // Record to leaderboard
                            let champion = pop.snakes.iter().enumerate()
                                .max_by(|(_, a), (_, b)| a.score.cmp(&b.score).then(a.lifetime.cmp(&b.lifetime)));
                            if let Some((champ_idx, champ)) = champion {
                                let stage_name = match pop.stage.kind {
                                    stage::StageKind::Classic => "Classic",
                                    stage::StageKind::Warehouse => "Warehouse",
                                    stage::StageKind::Mixed => "Mixed",
                                };
                                // Use agent ID as the player name
                                let agent_name = match pop.stage.kind {
                                    stage::StageKind::Warehouse => format!("AMR #{}", champ_idx + 1),
                                    _ => format!("Snake #{}", champ_idx + 1),
                                };
                                let entry = leaderboard::LeaderboardEntry {
                                    rank: 0,
                                    player: agent_name,
                                    score: gen_best,
                                    gen: pop.gen,
                                    stage: stage_name.to_string(),
                                    lifetime: champ.lifetime,
                                    fitness: champ.fitness,
                                    mutation_rate: neural_net::MUTATION_RATE,
                                    population_size: population::POP_SIZE,
                                    timestamp: chrono::Local::now().format("%Y-%m-%d %H:%M:%S").to_string(),
                                };
                                lb.add_entry(entry);
                                lb.save();

                                // Broadcast to browser
                                let lb_msg = ServerMsg::Leaderboard { entries: lb.entries.clone() };
                                if let Ok(json) = serde_json::to_string(&lb_msg) {
                                    let _ = state_tx.send(json);
                                }
                            }
                        }

                        let is_new_record = gen_best > pop.best_scores.iter().copied().max().unwrap_or(0);
                        pop.natural_selection();

                        // Upload new generation's weights to GPU once (not every step)
                        if let Some(ref g) = *gpu {
                            g.upload_weights(&pop.snakes);
                        }

                        // Force checkpoint on new record so we never lose a best score
                        if is_new_record {
                            pop.save_checkpoint();
                            log(&shared, &state_tx,
                                format!("Record checkpoint saved (score {})", gen_best),
                                LogKind::Done,
                            );
                        }
                        // Auto-save checkpoint every 10 generations
                        else if pop.gen % 10 == 0 && pop.gen >= 10 {
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

                        // Send coach stats every N generations (browser-side calls LLM)
                        if coach_interval > 0 && pop.gen >= last_coach_gen + coach_interval {
                            last_coach_gen = pop.gen;

                            // Build stats summary for LLM analysis
                            let recent_scores: Vec<u32> = pop.best_scores.iter()
                                .rev().take(20).copied().collect();
                            let avg_recent = if recent_scores.is_empty() { 0.0 }
                                else { recent_scores.iter().sum::<u32>() as f64 / recent_scores.len() as f64 };
                            let all_time = pop.best_scores.iter().copied().max().unwrap_or(0);
                            let alive = pop.alive_count();
                            let avg_fitness = pop.snakes.iter().map(|s| s.fitness).sum::<f64>()
                                / pop.snakes.len() as f64;
                            let max_fitness = pop.snakes.iter().map(|s| s.fitness)
                                .fold(0.0_f64, f64::max);
                            let stage_label = match pop.stage.kind {
                                stage::StageKind::Classic => "Classic",
                                stage::StageKind::Warehouse => "Warehouse",
                                stage::StageKind::Mixed => "Mixed",
                            };

                            let summary = format!(
                                "Generation: {}\n\
                                 Stage: {}\n\
                                 Population: {}\n\
                                 Current gen best score: {}\n\
                                 All-time best score: {}\n\
                                 Average score (last 20 gens): {:.1}\n\
                                 Recent scores (last 20 gens): {:?}\n\
                                 Alive this gen: {}/{}\n\
                                 Average fitness: {:.1}\n\
                                 Max fitness: {:.1}\n\
                                 GPU compute: {:.2}ms\n\
                                 Mutation rate: {:.1}%\n\
                                 Total generations trained: {}",
                                pop.gen, stage_label, population::POP_SIZE,
                                gen_best, all_time,
                                avg_recent, recent_scores,
                                alive, population::POP_SIZE,
                                avg_fitness, max_fitness,
                                pop.last_compute_ms,
                                neural_net::MUTATION_RATE * 100.0,
                                pop.best_scores.len(),
                            );

                            let coach_msg = ServerMsg::CoachStats {
                                summary,
                                gen: pop.gen,
                            };
                            if let Ok(json) = serde_json::to_string(&coach_msg) {
                                let _ = state_tx.send(json);
                            }
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

                    // Track best score mid-generation — save immediately so it's never lost
                    let current_max = pop.snakes.iter().map(|s| s.score).max().unwrap_or(0);
                    let all_time = pop.best_scores.iter().copied().max().unwrap_or(0);
                    if current_max > all_time && current_max > live_best_score {
                        live_best_score = current_max;

                        // Find the champion snake (may be dead or alive)
                        let champion = pop.snakes.iter().enumerate()
                            .max_by(|(_, a), (_, b)| a.score.cmp(&b.score).then(a.lifetime.cmp(&b.lifetime)));
                        if let Some((champ_idx, champ)) = champion {
                            // Compute fitness inline (calc_fitness hasn't run yet mid-gen)
                            let lt = champ.lifetime as f64;
                            let live_fitness = if champ.score < 10 {
                                lt * 2.0f64.powi(champ.score as i32)
                            } else {
                                lt * 2.0f64.powi(10) * (champ.score as f64 - 9.0)
                            };

                            // Update best brain for elitism
                            pop.best_brain = Some(champ.brain.clone());
                            pop.best_fitness = live_fitness;

                            // Record to leaderboard immediately
                            let stage_name = match pop.stage.kind {
                                stage::StageKind::Classic => "Classic",
                                stage::StageKind::Warehouse => "Warehouse",
                                stage::StageKind::Mixed => "Mixed",
                            };
                            let agent_name = match pop.stage.kind {
                                stage::StageKind::Warehouse => format!("AMR #{}", champ_idx + 1),
                                _ => format!("Snake #{}", champ_idx + 1),
                            };
                            let entry = leaderboard::LeaderboardEntry {
                                rank: 0,
                                player: agent_name,
                                score: current_max,
                                gen: pop.gen,
                                stage: stage_name.to_string(),
                                lifetime: champ.lifetime,
                                fitness: live_fitness,
                                mutation_rate: neural_net::MUTATION_RATE,
                                population_size: population::POP_SIZE,
                                timestamp: chrono::Local::now().format("%Y-%m-%d %H:%M:%S").to_string(),
                            };
                            lb.add_entry(entry);
                            lb.save();

                            // Push score and save checkpoint immediately
                            pop.best_scores.push(current_max);
                            pop.save_checkpoint();

                            log(&shared, &state_tx,
                                format!("LIVE RECORD! Score {} — checkpoint saved", current_max),
                                LogKind::Done);

                            // Broadcast leaderboard update
                            let lb_msg = ServerMsg::Leaderboard { entries: lb.entries.clone() };
                            if let Ok(json) = serde_json::to_string(&lb_msg) {
                                let _ = state_tx.send(json);
                            }
                        }
                    }
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
                    .enumerate()
                    .filter(|(_, s)| !s.dead)
                    .max_by(|(_, a), (_, b)| a.score.cmp(&b.score).then(a.lifetime.cmp(&b.lifetime)));

                if let Some((idx, best)) = best {
                    let historical_best = pop.best_scores.iter().copied().max().unwrap_or(0);
                    let msg = ServerMsg::State {
                        gen: pop.gen,
                        best_score: historical_best.max(live_best_score),
                        alive: alive_count,
                        total: population::POP_SIZE,
                        gpu_ms,
                        snake_body: best.body.clone(),
                        food: (best.food_x, best.food_y),
                        score: best.score,
                        vision: best.vision.to_vec(),
                        decision: best.decision.to_vec(),
                        nn_weights: best.brain.clone(),
                        snake_id: idx,
                        dead: false,
                        obstacles: pop.stage.obstacle_list(),
                    };
                    if let Ok(json) = serde_json::to_string(&msg) {
                        let _ = state_tx.send(json);
                    }
                }
            }

            Some(cmd) = cmd_rx.recv() => {
                match cmd {
                    ClientMsg::Start => {
                        if let Some(mut restored) = Population::from_checkpoint(stage_kind) {
                            // Reconcile best_scores with leaderboard (checkpoint may miss scores between saves)
                            let lb_max = lb.entries.iter().map(|e| e.score).max().unwrap_or(0);
                            let cp_max = restored.best_scores.iter().copied().max().unwrap_or(0);
                            if lb_max > cp_max {
                                restored.best_scores.push(lb_max);
                            }

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
                            population = Some(Population::new(stage_kind));
                            log(&shared, &state_tx,"Evolution started — population: 2000".into(), LogKind::Phase);
                        }
                        // Upload initial weights to GPU
                        if let (Some(ref pop), Some(ref g)) = (&population, &*gpu) {
                            g.upload_weights(&pop.snakes);
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
                    ClientMsg::Stage { value } => {
                        stage_kind = match value.as_str() {
                            "warehouse" => StageKind::Warehouse,
                            "mixed" => StageKind::Mixed,
                            _ => StageKind::Classic,
                        };
                        let label = match stage_kind {
                            StageKind::Classic => "Classic",
                            StageKind::Warehouse => "Warehouse",
                            StageKind::Mixed => "Mixed",
                        };
                        // Switch stage but KEEP trained brains
                        if let Some(ref mut pop) = population {
                            pop.stage = stage::Stage::new(stage_kind);
                            // Respawn snakes with same brains on new stage
                            for snake in &mut pop.snakes {
                                let brain = snake.brain.clone();
                                *snake = crate::snake::Snake::new_with_stage(&pop.stage);
                                snake.brain = brain;
                            }
                        } else {
                            population = Some(Population::new(stage_kind));
                        }
                        // Upload weights after stage switch
                        if let (Some(ref pop), Some(ref g)) = (&population, &*gpu) {
                            g.upload_weights(&pop.snakes);
                        }
                        running = true;
                        if let Ok(mut r) = shared.running.lock() { *r = true; }
                        log(&shared, &state_tx,
                            format!("Stage changed to {} — brains preserved", label),
                            LogKind::Phase);
                    }
                    ClientMsg::Regenerate => {
                        if let Some(ref mut pop) = population {
                            pop.regenerate_stage();
                            log(&shared, &state_tx,
                                "Stage layout regenerated".into(),
                                LogKind::Phase);
                        }
                    }
                    ClientMsg::SetPlayer { .. } => {
                        // Player naming is now automatic (AMR #N / Snake #N)
                    }
                    ClientMsg::CoachInterval { gens } => {
                        coach_interval = gens;
                        last_coach_gen = population.as_ref().map_or(0, |p| p.gen);
                        let label = if gens == 0 { "disabled".into() } else { format!("every {} gens", gens) };
                        log(&shared, &state_tx,
                            format!("AI Coach: {}", label),
                            LogKind::Info);
                    }
                }
            }
        }
    }
}
