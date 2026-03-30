use std::sync::Mutex;

use chrono::Local;

#[derive(Clone, Copy, PartialEq)]
pub enum LogKind {
    Info,
    Phase,
    Error,
    Done,
    Stat,
}

#[derive(Clone)]
pub struct LogEntry {
    pub timestamp: String,
    pub message: String,
    pub kind: LogKind,
}

const MAX_LOG_LINES: usize = 2000;

pub struct SharedState {
    pub log_lines: Mutex<Vec<LogEntry>>,
    pub running: Mutex<bool>,
    pub gen: Mutex<u32>,
    pub best_score: Mutex<u32>,
    pub alive: Mutex<String>,
    pub gpu_ms: Mutex<f64>,
    pub gpu_backend: Mutex<String>,
    pub speed: Mutex<u32>,
}

impl SharedState {
    pub fn new() -> Self {
        Self {
            log_lines: Mutex::new(Vec::new()),
            running: Mutex::new(false),
            gen: Mutex::new(0),
            best_score: Mutex::new(0),
            alive: Mutex::new("0/2000".into()),
            gpu_ms: Mutex::new(0.0),
            gpu_backend: Mutex::new("Initializing...".into()),
            speed: Mutex::new(10),
        }
    }

    pub fn push_log(&self, message: String, kind: LogKind) {
        let entry = LogEntry {
            timestamp: Local::now().format("%H:%M:%S").to_string(),
            message,
            kind,
        };
        if let Ok(mut lines) = self.log_lines.lock() {
            lines.push(entry);
            if lines.len() > MAX_LOG_LINES {
                let excess = lines.len() - MAX_LOG_LINES;
                lines.drain(0..excess);
            }
        }
    }
}
