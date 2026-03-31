use serde::{Deserialize, Serialize};

use crate::leaderboard::LeaderboardEntry;
use crate::neural_net::NeuralNet;
use crate::snake::Pos;

/// Server -> Client
#[derive(Serialize)]
#[serde(tag = "type")]
pub enum ServerMsg {
    #[serde(rename = "state")]
    State {
        gen: u32,
        best_score: u32,
        alive: usize,
        total: usize,
        gpu_ms: f64,
        // Best alive snake data
        snake_body: Vec<Pos>,
        food: (i32, i32),
        score: u32,
        vision: Vec<f32>,
        decision: Vec<f32>,
        nn_weights: NeuralNet,
        snake_id: usize,
        #[serde(skip_serializing_if = "std::ops::Not::not")]
        dead: bool,
        #[serde(skip_serializing_if = "Vec::is_empty")]
        obstacles: Vec<(i32, i32)>,
    },
    #[serde(rename = "graph")]
    Graph { scores: Vec<u32> },
    #[serde(rename = "log")]
    Log {
        timestamp: String,
        message: String,
        kind: String,
    },
    #[serde(rename = "leaderboard")]
    Leaderboard {
        entries: Vec<LeaderboardEntry>,
    },
    /// Evolution stats summary sent to browser for LLM analysis
    #[serde(rename = "coach_stats")]
    CoachStats { summary: String, gen: u32 },
}


/// Client -> Server
#[derive(Deserialize)]
#[serde(tag = "type")]
pub enum ClientMsg {
    #[serde(rename = "start")]
    Start,
    #[serde(rename = "pause")]
    Pause,
    #[serde(rename = "resume")]
    Resume,
    #[serde(rename = "speed")]
    Speed { value: u32 },
    #[serde(rename = "stage")]
    Stage { value: String },
    #[serde(rename = "regenerate")]
    Regenerate,
    #[serde(rename = "set_player")]
    SetPlayer { name: String },
    #[serde(rename = "coach_interval")]
    CoachInterval { gens: u32 },
}
