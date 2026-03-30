use rand::Rng;
use serde::{Deserialize, Serialize};
use std::path::Path;

use crate::neural_net::NeuralNet;
use crate::snake::Snake;

pub const POP_SIZE: usize = 2000;
const CHECKPOINT_PATH: &str = "checkpoint.json";

#[derive(Serialize, Deserialize)]
struct Checkpoint {
    gen: u32,
    best_scores: Vec<u32>,
    best_fitness: f64,
    best_brain: Option<NeuralNet>,
    /// All snake brains from the current generation (to resume mid-evolution)
    brains: Vec<NeuralNet>,
}

pub struct Population {
    pub snakes: Vec<Snake>,
    pub gen: u32,
    pub best_scores: Vec<u32>,
    pub best_fitness: f64,
    pub best_brain: Option<NeuralNet>,
    pub last_compute_ms: f64,
}

impl Population {
    pub fn new() -> Self {
        let snakes = (0..POP_SIZE).map(|_| Snake::new()).collect();
        Self {
            snakes,
            last_compute_ms: 0.0,
            gen: 1,
            best_scores: Vec::new(),
            best_fitness: 0.0,
            best_brain: None,
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
                snake.look();
            }
        }
    }

    pub fn move_all(&mut self) {
        for snake in &mut self.snakes {
            if !snake.dead {
                snake.move_snake();
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

        let mut new_snakes = Vec::with_capacity(POP_SIZE);

        // Elitism: keep the best brain
        if let Some(ref best_brain) = self.best_brain {
            let mut elite = Snake::new();
            elite.brain = best_brain.clone();
            new_snakes.push(elite);
        }

        while new_snakes.len() < POP_SIZE {
            let a = self.select_parent();
            let b = self.select_parent();
            let mut child = Snake::new();
            child.brain = a.brain.crossover(&b.brain);
            child.brain.mutate();
            new_snakes.push(child);
        }

        self.snakes = new_snakes;
        self.gen += 1;
    }

    /// Save checkpoint to disk
    pub fn save_checkpoint(&self) {
        let checkpoint = Checkpoint {
            gen: self.gen,
            best_scores: self.best_scores.clone(),
            best_fitness: self.best_fitness,
            best_brain: self.best_brain.clone(),
            brains: self.snakes.iter().map(|s| s.brain.clone()).collect(),
        };
        if let Ok(json) = serde_json::to_string(&checkpoint) {
            let _ = std::fs::write(CHECKPOINT_PATH, json);
        }
    }

    /// Load from checkpoint if it exists, otherwise create new
    pub fn from_checkpoint() -> Option<Self> {
        if !Path::new(CHECKPOINT_PATH).exists() {
            return None;
        }
        let data = std::fs::read_to_string(CHECKPOINT_PATH).ok()?;
        let cp: Checkpoint = serde_json::from_str(&data).ok()?;

        let mut snakes: Vec<Snake> = cp
            .brains
            .into_iter()
            .map(|brain| {
                let mut s = Snake::new();
                s.brain = brain;
                s
            })
            .collect();

        // Pad if checkpoint had fewer brains
        while snakes.len() < POP_SIZE {
            snakes.push(Snake::new());
        }
        snakes.truncate(POP_SIZE);

        Some(Self {
            snakes,
            gen: cp.gen,
            best_scores: cp.best_scores,
            best_fitness: cp.best_fitness,
            best_brain: cp.best_brain,
            last_compute_ms: 0.0,
        })
    }
}
