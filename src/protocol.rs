use serde::{Deserialize, Serialize};

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
    },
    #[serde(rename = "graph")]
    Graph { scores: Vec<u32> },
    #[serde(rename = "log")]
    Log {
        timestamp: String,
        message: String,
        kind: String,
    },
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
}
