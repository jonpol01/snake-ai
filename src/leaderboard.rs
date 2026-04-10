use serde::{Deserialize, Serialize};

use crate::paths;

const LEADERBOARD_FILE: &str = "leaderboard.json";
const SCREENSHOTS_DIR: &str = "screenshots";

#[derive(Clone, Serialize, Deserialize)]
pub struct LeaderboardEntry {
    pub rank: u32,
    pub player: String,
    pub score: u32,
    pub gen: u32,
    pub stage: String,
    pub lifetime: u32,
    pub fitness: f64,
    pub mutation_rate: f32,
    pub population_size: usize,
    pub timestamp: String,
    /// Filename of the death screenshot (e.g. "score78-gen450.png")
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub screenshot: Option<String>,
}

/// Save a screenshot PNG and return the filename.
pub fn save_screenshot(score: u32, gen: u32, png_data: &[u8]) -> Option<String> {
    let dir = paths::data_path(SCREENSHOTS_DIR);
    if !dir.exists() {
        let _ = std::fs::create_dir_all(&dir);
    }
    let filename = format!("score{}-gen{}.png", score, gen);
    let path = dir.join(&filename);
    match std::fs::write(&path, png_data) {
        Ok(_) => Some(filename),
        Err(_) => None,
    }
}

/// Get the full path to a screenshot file.
pub fn screenshot_path(filename: &str) -> Option<std::path::PathBuf> {
    let path = paths::data_path(SCREENSHOTS_DIR).join(filename);
    if path.exists() { Some(path) } else { None }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Leaderboard {
    pub entries: Vec<LeaderboardEntry>,
}

impl Leaderboard {
    pub fn load() -> Self {
        let lb_path = paths::data_path(LEADERBOARD_FILE);
        if !lb_path.exists() {
            return Self { entries: Vec::new() };
        }
        let data = match std::fs::read_to_string(&lb_path) {
            Ok(d) => d,
            Err(_) => return Self { entries: Vec::new() },
        };
        serde_json::from_str(&data).unwrap_or(Self { entries: Vec::new() })
    }

    pub fn add_entry(&mut self, entry: LeaderboardEntry) {
        self.entries.push(entry);
        self.entries
            .sort_by(|a, b| b.score.cmp(&a.score).then(a.gen.cmp(&b.gen)));
        for (i, e) in self.entries.iter_mut().enumerate() {
            e.rank = (i + 1) as u32;
        }
    }

    pub fn save(&self) {
        if let Ok(json) = serde_json::to_string_pretty(self) {
            let _ = std::fs::write(paths::data_path(LEADERBOARD_FILE), json);
        }
    }
}
