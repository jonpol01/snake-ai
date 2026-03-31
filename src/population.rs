use rand::Rng;
use serde::{Deserialize, Serialize};
use std::path::Path;

use crate::neural_net::NeuralNet;
use crate::snake::Snake;
use crate::stage::{Stage, StageKind};

pub const POP_SIZE: usize = 2000;
const CHECKPOINT_PATH: &str = "checkpoint.json";

/// Bump this when the Checkpoint struct or neural net architecture changes.
/// Old checkpoints (version 0 or missing) are treated as compatible with v1.
const CHECKPOINT_VERSION: u32 = 1;

fn default_version() -> u32 { 0 }

#[derive(Serialize, Deserialize)]
struct Checkpoint {
    /// Format version — missing in old checkpoints, defaults to 0
    #[serde(default = "default_version")]
    version: u32,
    gen: u32,
    best_scores: Vec<u32>,
    best_fitness: f64,
    best_brain: Option<NeuralNet>,
    /// All snake brains from the current generation (to resume mid-evolution)
    brains: Vec<NeuralNet>,
    /// Which stage was active (defaults to classic for old checkpoints)
    #[serde(default)]
    stage_kind: StageKind,
}

pub struct Population {
    pub snakes: Vec<Snake>,
    pub gen: u32,
    pub best_scores: Vec<u32>,
    pub best_fitness: f64,
    pub best_brain: Option<NeuralNet>,
    pub last_compute_ms: f64,
    pub stage: Stage,
}

impl Population {
    pub fn new(stage_kind: StageKind) -> Self {
        let stage = Stage::new(stage_kind);
        let snakes = (0..POP_SIZE).map(|_| Snake::new_with_stage(&stage)).collect();
        Self {
            snakes,
            last_compute_ms: 0.0,
            gen: 1,
            best_scores: Vec::new(),
            best_fitness: 0.0,
            best_brain: None,
            stage,
        }
    }

    pub fn done(&self) -> bool {
        self.snakes.iter().all(|s| s.dead)
    }

    pub fn alive_count(&self) -> usize {
        self.snakes.iter().filter(|s| !s.dead).count()
    }

    pub fn look_all(&mut self) {
        for snake in &mut self.snakes {
            if !snake.dead {
                snake.look(&self.stage);
            }
        }
    }

    pub fn move_all(&mut self) {
        for snake in &mut self.snakes {
            if !snake.dead {
                snake.move_snake(&self.stage);
            }
        }
    }

    pub fn calculate_fitness(&mut self) {
        for snake in &mut self.snakes {
            snake.calc_fitness();
        }

        let (max_idx, max_fitness) = self
            .snakes
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.fitness.partial_cmp(&b.fitness).unwrap())
            .map(|(i, s)| (i, s.fitness))
            .unwrap();

        if max_fitness > self.best_fitness {
            self.best_fitness = max_fitness;
            self.best_brain = Some(self.snakes[max_idx].brain.clone());
        }
    }

    fn select_parent(&self) -> &Snake {
        let mut rng = rand::thread_rng();
        let total: f64 = self.snakes.iter().map(|s| s.fitness).sum();
        let mut r = rng.gen::<f64>() * total;

        for snake in &self.snakes {
            r -= snake.fitness;
            if r <= 0.0 {
                return snake;
            }
        }

        self.snakes.last().unwrap()
    }

    pub fn natural_selection(&mut self) {
        let gen_best = self.snakes.iter().map(|s| s.score).max().unwrap_or(0);
        self.best_scores.push(gen_best);

        // Mixed mode: randomize stage each generation for generalization
        // Warehouse: keep fixed layout (user can shuffle manually)
        if self.stage.kind == StageKind::Mixed {
            self.stage = Stage::new(StageKind::Mixed);
        }

        let mut new_snakes = Vec::with_capacity(POP_SIZE);

        // Elitism: keep the best brain
        if let Some(ref best_brain) = self.best_brain {
            let mut elite = Snake::new_with_stage(&self.stage);
            elite.brain = best_brain.clone();
            new_snakes.push(elite);
        }

        while new_snakes.len() < POP_SIZE {
            let a = self.select_parent();
            let b = self.select_parent();
            let mut child = Snake::new_with_stage(&self.stage);
            child.brain = a.brain.crossover(&b.brain);
            child.brain.mutate();
            new_snakes.push(child);
        }

        self.snakes = new_snakes;
        self.gen += 1;
    }

    /// Regenerate the stage layout and restart the current generation
    pub fn regenerate_stage(&mut self) {
        self.stage = Stage::new(self.stage.kind);
        // Respawn all snakes but keep their trained brains
        for snake in &mut self.snakes {
            let brain = snake.brain.clone();
            *snake = Snake::new_with_stage(&self.stage);
            snake.brain = brain;
        }
    }

    /// Save checkpoint to disk
    pub fn save_checkpoint(&self) {
        let checkpoint = Checkpoint {
            version: CHECKPOINT_VERSION,
            gen: self.gen,
            best_scores: self.best_scores.clone(),
            best_fitness: self.best_fitness,
            best_brain: self.best_brain.clone(),
            brains: self.snakes.iter().map(|s| s.brain.clone()).collect(),
            stage_kind: self.stage.kind,
        };
        if let Ok(json) = serde_json::to_string(&checkpoint) {
            let _ = std::fs::write(CHECKPOINT_PATH, json);
        }
    }

    /// Load from checkpoint if it exists, otherwise create new
    pub fn from_checkpoint(stage_kind: StageKind) -> Option<Self> {
        if !Path::new(CHECKPOINT_PATH).exists() {
            return None;
        }
        let data = std::fs::read_to_string(CHECKPOINT_PATH).ok()?;
        let cp: Checkpoint = serde_json::from_str(&data).ok()?;

        // Version compatibility check
        if cp.version > CHECKPOINT_VERSION {
            eprintln!(
                "WARNING: Checkpoint is version {} but this build supports version {}. \
                 It may be incompatible — starting fresh.",
                cp.version, CHECKPOINT_VERSION
            );
            return None;
        }

        // Always load — single brain works across all stages
        let stage = Stage::new(stage_kind);

        let mut snakes: Vec<Snake> = cp
            .brains
            .into_iter()
            .map(|brain| {
                let mut s = Snake::new_with_stage(&stage);
                s.brain = brain;
                s
            })
            .collect();

        // Pad if checkpoint had fewer brains
        while snakes.len() < POP_SIZE {
            snakes.push(Snake::new_with_stage(&stage));
        }
        snakes.truncate(POP_SIZE);

        Some(Self {
            snakes,
            gen: cp.gen,
            best_scores: cp.best_scores,
            best_fitness: cp.best_fitness,
            best_brain: cp.best_brain,
            last_compute_ms: 0.0,
            stage,
        })
    }
}
