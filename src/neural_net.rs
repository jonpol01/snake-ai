use rand::Rng;
use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
}

impl Matrix {
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    pub fn randomize(&mut self) {
        let mut rng = rand::thread_rng();
        for v in &mut self.data {
            *v = rng.gen_range(-1.0..1.0);
        }
    }

    pub fn get(&self, row: usize, col: usize) -> f32 {
        self.data[row * self.cols + col]
    }

    pub fn crossover(&self, partner: &Matrix) -> Matrix {
        let mut child = Matrix::new(self.rows, self.cols);
        let mut rng = rand::thread_rng();
        let cut_row = rng.gen_range(0..self.rows);
        let cut_col = rng.gen_range(0..self.cols);

        for i in 0..self.rows {
            for j in 0..self.cols {
                let idx = i * self.cols + j;
                child.data[idx] = if i < cut_row || (i == cut_row && j <= cut_col) {
                    self.data[idx]
                } else {
                    partner.data[idx]
                };
            }
        }
        child
    }

    pub fn mutate(&mut self, rate: f32) {
        let mut rng = rand::thread_rng();
        for v in &mut self.data {
            if rng.gen::<f32>() < rate {
                *v = rng.gen_range(-1.0..1.0);
            }
        }
    }
}

pub const HIDDEN_NODES: usize = 16;
pub const INPUT_NODES: usize = 24;
pub const OUTPUT_NODES: usize = 4;
pub const MUTATION_RATE: f32 = 0.05;

// Weight counts per layer (including bias column):
// Layer 0: 16 × 25 = 400
// Layer 1: 16 × 17 = 272
// Layer 2: 4 × 17 = 68
// Total: 740
pub const WEIGHTS_PER_SNAKE: usize = 400 + 272 + 68;

#[derive(Clone, Serialize, Deserialize)]
pub struct NeuralNet {
    pub weights: Vec<Matrix>,
}

impl NeuralNet {
    pub fn new() -> Self {
        let mut w0 = Matrix::new(HIDDEN_NODES, INPUT_NODES + 1); // 16×25
        let mut w1 = Matrix::new(HIDDEN_NODES, HIDDEN_NODES + 1); // 16×17
        let mut w2 = Matrix::new(OUTPUT_NODES, HIDDEN_NODES + 1); // 4×17
        w0.randomize();
        w1.randomize();
        w2.randomize();
        Self {
            weights: vec![w0, w1, w2],
        }
    }

    pub fn crossover(&self, partner: &NeuralNet) -> NeuralNet {
        NeuralNet {
            weights: self
                .weights
                .iter()
                .zip(partner.weights.iter())
                .map(|(a, b)| a.crossover(b))
                .collect(),
        }
    }

    pub fn mutate(&mut self) {
        for w in &mut self.weights {
            w.mutate(MUTATION_RATE);
        }
    }

    /// CPU forward pass (fallback when GPU is unavailable)
    pub fn forward(&self, input: &[f32; 24]) -> [f32; 4] {
        // Layer 0: input(24) + bias -> hidden(16)
        let w0 = &self.weights[0];
        let mut hidden0 = [0.0f32; 16];
        for i in 0..16 {
            let mut sum = 0.0;
            for j in 0..24 {
                sum += w0.get(i, j) * input[j];
            }
            sum += w0.get(i, 24); // bias
            hidden0[i] = sigmoid(sum);
        }

        // Layer 1: hidden(16) + bias -> hidden(16)
        let w1 = &self.weights[1];
        let mut hidden1 = [0.0f32; 16];
        for i in 0..16 {
            let mut sum = 0.0;
            for j in 0..16 {
                sum += w1.get(i, j) * hidden0[j];
            }
            sum += w1.get(i, 16); // bias
            hidden1[i] = sigmoid(sum);
        }

        // Layer 2: hidden(16) + bias -> output(4)
        let w2 = &self.weights[2];
        let mut output = [0.0f32; 4];
        for i in 0..4 {
            let mut sum = 0.0;
            for j in 0..16 {
                sum += w2.get(i, j) * hidden1[j];
            }
            sum += w2.get(i, 16); // bias
            output[i] = sigmoid(sum);
        }

        output
    }

    /// Pack weights into a flat f32 slice (for GPU buffer)
    pub fn pack_weights(&self, buf: &mut [f32]) {
        let mut idx = 0;
        for w in &self.weights {
            for &v in &w.data {
                buf[idx] = v;
                idx += 1;
            }
        }
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
