use crate::protocol::ClientMsg;
use crate::shared::{LogKind, SharedState};
use crate::stage::StageKind;
use eframe::egui;
use std::sync::Arc;
use tokio::sync::mpsc;

pub struct SnakeApp {
    state: Arc<SharedState>,
    cmd_tx: mpsc::Sender<ClientMsg>,
    speed_str: String,
    auto_scroll: bool,
    rt: tokio::runtime::Handle,
    stage_idx: usize,
}

const STAGE_OPTIONS: &[(StageKind, &str)] = &[
    (StageKind::Classic, "Classic (Empty)"),
    (StageKind::Warehouse, "Warehouse (Racks)"),
    (StageKind::Mixed, "Mixed (Randomized)"),
];

impl SnakeApp {
    pub fn new(
        state: Arc<SharedState>,
        cmd_tx: mpsc::Sender<ClientMsg>,
        rt: tokio::runtime::Handle,
    ) -> Self {
        Self {
            state,
            cmd_tx,
            speed_str: "10".into(),
            auto_scroll: true,
            rt,
            stage_idx: 0,
        }
    }
}

pub fn run(
    state: Arc<SharedState>,
    cmd_tx: mpsc::Sender<ClientMsg>,
    rt: tokio::runtime::Handle,
) {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([620.0, 700.0])
            .with_min_inner_size([500.0, 400.0])
            .with_title("Snake AI"),
        ..Default::default()
    };

    let _ = eframe::run_native(
        "Snake AI",
        options,
        Box::new(move |cc| {
            configure_fonts(&cc.egui_ctx);
            Ok(Box::new(SnakeApp::new(state, cmd_tx, rt)))
        }),
    );
}

fn configure_fonts(ctx: &egui::Context) {
    let mut style = (*ctx.style()).clone();
    style.spacing.item_spacing = egui::vec2(8.0, 6.0);
    ctx.set_style(style);
}

impl eframe::App for SnakeApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        ctx.request_repaint_after(std::time::Duration::from_millis(100));

        let is_running = self.state.running.lock().map(|r| *r).unwrap_or(false);
        let gen = self.state.gen.lock().map(|g| *g).unwrap_or(0);
        let best = self.state.best_score.lock().map(|b| *b).unwrap_or(0);
        let alive = self
            .state
            .alive
            .lock()
            .map(|a| a.clone())
            .unwrap_or_default();
        let gpu_ms = self.state.gpu_ms.lock().map(|g| *g).unwrap_or(0.0);
        let backend = self
            .state
            .gpu_backend
            .lock()
            .map(|b| b.clone())
            .unwrap_or_default();

        // ── Header ──
        egui::TopBottomPanel::top("header").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("Snake AI");
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if is_running {
                        ui.colored_label(
                            egui::Color32::from_rgb(0, 255, 136),
                            format!("Gen {} — Score {}", gen, best),
                        );
                        ui.spinner();
                    } else {
                        ui.colored_label(egui::Color32::from_rgb(100, 116, 139), "● Idle");
                    }
                    ui.colored_label(egui::Color32::from_rgb(100, 116, 139), "|");
                    ui.colored_label(egui::Color32::from_rgb(100, 116, 139), &backend);
                });
            });
        });

        // ── Main ──
        egui::CentralPanel::default().show(ctx, |ui| {
            // Settings
            egui::CollapsingHeader::new("Settings")
                .default_open(true)
                .show(ui, |ui| {
                    egui::Grid::new("settings_grid")
                        .num_columns(2)
                        .spacing([12.0, 8.0])
                        .show(ui, |ui| {
                            ui.label("Population:");
                            ui.label("2000");
                            ui.end_row();

                            ui.label("Mutation Rate:");
                            ui.label("5%");
                            ui.end_row();

                            ui.label("Network:");
                            ui.label("24 → 16 → 16 → 4 (sigmoid)");
                            ui.end_row();

                            ui.label("Stage:");
                            let prev_idx = self.stage_idx;
                            egui::ComboBox::from_id_salt("stage_combo")
                                .selected_text(STAGE_OPTIONS[self.stage_idx].1)
                                .show_ui(ui, |ui| {
                                    for (i, (_, label)) in STAGE_OPTIONS.iter().enumerate() {
                                        ui.selectable_value(&mut self.stage_idx, i, *label);
                                    }
                                });
                            if self.stage_idx != prev_idx {
                                let (kind, _) = STAGE_OPTIONS[self.stage_idx];
                                let value = match kind {
                                    StageKind::Classic => "classic",
                                    StageKind::Warehouse => "warehouse",
                                    StageKind::Mixed => "mixed",
                                }.to_string();
                                let tx = self.cmd_tx.clone();
                                let _ = self.rt.spawn(async move {
                                    let _ = tx.send(ClientMsg::Stage { value }).await;
                                });
                            }
                            ui.end_row();

                            ui.label("");
                            if ui.button("🔀 Shuffle Layout").clicked() {
                                let tx = self.cmd_tx.clone();
                                let _ = self.rt.spawn(async move {
                                    let _ = tx.send(ClientMsg::Regenerate).await;
                                });
                            }
                            ui.end_row();

                            ui.label("Speed:");
                            let slider = ui.add(
                                egui::Slider::new(
                                    &mut {
                                        let v: u32 =
                                            self.speed_str.parse().unwrap_or(10);
                                        v
                                    },
                                    1..=200,
                                )
                                .text("steps/frame"),
                            );
                            if slider.changed() {
                                // Re-read from slider
                            }
                            ui.end_row();
                        });

                    // Speed slider (simpler approach)
                    ui.add_space(4.0);
                    let mut speed_val: u32 = self.speed_str.parse().unwrap_or(10);
                    let response =
                        ui.add(egui::Slider::new(&mut speed_val, 1..=200).text("Speed (steps/frame)"));
                    if response.changed() {
                        self.speed_str = speed_val.to_string();
                        if let Ok(mut s) = self.state.speed.lock() {
                            *s = speed_val;
                        }
                        let tx = self.cmd_tx.clone();
                        let _ = self
                            .rt
                            .spawn(async move { let _ = tx.send(ClientMsg::Speed { value: speed_val }).await; });
                    }

                    ui.add_space(6.0);

                    ui.horizontal(|ui| {
                        if !is_running {
                            let start_btn = ui.add(
                                egui::Button::new("▶ Start")
                                    .fill(egui::Color32::from_rgb(34, 197, 94))
                                    .min_size(egui::vec2(100.0, 28.0)),
                            );
                            if start_btn.clicked() {
                                self.state.push_log(
                                    "Starting evolution...".into(),
                                    LogKind::Phase,
                                );
                                let tx = self.cmd_tx.clone();
                                let _ = self.rt.spawn(async move {
                                    let _ = tx.send(ClientMsg::Start).await;
                                });
                            }
                        } else {
                            let pause_btn = ui.add(
                                egui::Button::new("⏸ Pause")
                                    .fill(egui::Color32::from_rgb(239, 68, 68))
                                    .min_size(egui::vec2(100.0, 28.0)),
                            );
                            if pause_btn.clicked() {
                                self.state
                                    .push_log("Simulation paused".into(), LogKind::Info);
                                let tx = self.cmd_tx.clone();
                                let _ = self.rt.spawn(async move {
                                    let _ = tx.send(ClientMsg::Pause).await;
                                });
                            }

                            let resume_btn = ui.button("▶ Resume");
                            if resume_btn.clicked() {
                                self.state
                                    .push_log("Simulation resumed".into(), LogKind::Info);
                                let tx = self.cmd_tx.clone();
                                let _ = self.rt.spawn(async move {
                                    let _ = tx.send(ClientMsg::Resume).await;
                                });
                            }
                        }

                        ui.with_layout(
                            egui::Layout::right_to_left(egui::Align::Center),
                            |ui| {
                                if ui.button("Clear Log").clicked() {
                                    if let Ok(mut lines) = self.state.log_lines.lock() {
                                        lines.clear();
                                    }
                                }
                            },
                        );
                    });
                });

            ui.add_space(4.0);
            ui.separator();

            // ── Stats Row ──
            ui.horizontal(|ui| {
                stat_label(ui, "GEN", &gen.to_string());
                ui.separator();
                stat_label(ui, "BEST", &best.to_string());
                ui.separator();
                stat_label(ui, "ALIVE", &alive);
                ui.separator();
                stat_label(ui, "COMPUTE", &format!("{:.2}ms", gpu_ms));
            });

            ui.separator();
            ui.add_space(4.0);

            // ── Log Terminal ──
            ui.label("Log");
            let log_frame = egui::Frame::default()
                .fill(egui::Color32::from_rgb(10, 10, 10))
                .stroke(egui::Stroke::new(
                    1.0,
                    egui::Color32::from_rgb(31, 41, 55),
                ))
                .corner_radius(6.0)
                .inner_margin(10.0);

            log_frame.show(ui, |ui| {
                egui::ScrollArea::vertical()
                    .stick_to_bottom(self.auto_scroll)
                    .max_height(ui.available_height() - 10.0)
                    .show(ui, |ui| {
                        if let Ok(lines) = self.state.log_lines.lock() {
                            if lines.is_empty() {
                                ui.colored_label(
                                    egui::Color32::from_rgb(100, 116, 139),
                                    "Ready. Click ▶ Start to begin evolution. Open localhost:3030 in browser for game view.",
                                );
                            }
                            for entry in lines.iter() {
                                let color = match entry.kind {
                                    LogKind::Phase => egui::Color32::from_rgb(167, 139, 250),
                                    LogKind::Error => egui::Color32::from_rgb(239, 68, 68),
                                    LogKind::Done => egui::Color32::from_rgb(34, 197, 94),
                                    LogKind::Info => egui::Color32::from_rgb(74, 222, 128),
                                    LogKind::Stat => egui::Color32::from_rgb(56, 189, 248),
                                };
                                ui.horizontal(|ui| {
                                    ui.colored_label(
                                        egui::Color32::from_rgb(100, 116, 139),
                                        format!("[{}]", entry.timestamp),
                                    );
                                    ui.colored_label(color, &entry.message);
                                });
                            }
                        }
                    });
            });
        });
    }
}

fn stat_label(ui: &mut egui::Ui, label: &str, value: &str) {
    ui.horizontal(|ui| {
        ui.colored_label(egui::Color32::from_rgb(100, 116, 139), label);
        ui.colored_label(egui::Color32::from_rgb(0, 255, 136), value);
    });
}
