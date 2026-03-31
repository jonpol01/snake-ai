use serde::{Deserialize, Serialize};
use std::path::Path;

const LEADERBOARD_PATH: &str = "leaderboard.json";

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
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Leaderboard {
    pub entries: Vec<LeaderboardEntry>,
}

impl Leaderboard {
    pub fn load() -> Self {
        if !Path::new(LEADERBOARD_PATH).exists() {
            return Self {
                entries: Vec::new(),
            };
        }
        let data = match std::fs::read_to_string(LEADERBOARD_PATH) {
            Ok(d) => d,
            Err(_) => {
                return Self {
                    entries: Vec::new(),
                }
            }
        };
        serde_json::from_str(&data).unwrap_or(Self {
            entries: Vec::new(),
        })
    }

    pub fn add_entry(&mut self, entry: LeaderboardEntry) {
        self.entries.push(entry);
        // Sort by score descending, then by gen ascending (earlier = more impressive)
        self.entries
            .sort_by(|a, b| b.score.cmp(&a.score).then(a.gen.cmp(&b.gen)));
        // Reassign ranks
        for (i, e) in self.entries.iter_mut().enumerate() {
            e.rank = (i + 1) as u32;
        }
    }

    pub fn save(&self) {
        if let Ok(json) = serde_json::to_string_pretty(self) {
            let _ = std::fs::write(LEADERBOARD_PATH, json);
        }
    }
}
