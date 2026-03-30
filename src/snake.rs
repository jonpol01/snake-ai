use rand::Rng;
use serde::Serialize;

use crate::neural_net::NeuralNet;

pub const GRID_SIZE: usize = 20;

const DIRECTIONS: [(i32, i32); 8] = [
    (0, -1),  // up
    (0, 1),   // down
    (-1, 0),  // left
    (1, 0),   // right
    (-1, -1), // up-left
    (1, -1),  // up-right
    (-1, 1),  // down-left
    (1, 1),   // down-right
];

#[derive(Clone, Serialize)]
pub struct Pos {
    pub x: i32,
    pub y: i32,
}

#[derive(Clone)]
pub struct Snake {
    pub brain: NeuralNet,
    pub body: Vec<Pos>,
    pub food_x: i32,
    pub food_y: i32,
    pub score: u32,
    pub life_left: i32,
    pub lifetime: u32,
    pub dead: bool,
    pub x_vel: i32,
    pub y_vel: i32,
    pub fitness: f64,
    pub vision: [f32; 24],
    pub decision: [f32; 4],
}

impl Snake {
    pub fn new() -> Self {
        let cx = (GRID_SIZE / 2) as i32;
        let cy = (GRID_SIZE / 2) as i32;

        let mut snake = Self {
            brain: NeuralNet::new(),
            body: vec![
                Pos { x: cx, y: cy },
                Pos { x: cx, y: cy + 1 },
                Pos { x: cx, y: cy + 2 },
            ],
            food_x: 0,
            food_y: 0,
            score: 0,
            life_left: 200,
            lifetime: 0,
            dead: false,
            x_vel: 0,
            y_vel: -1,
            fitness: 0.0,
            vision: [0.0; 24],
            decision: [0.0; 4],
        };
        snake.place_food();
        snake
    }

    pub fn place_food(&mut self) {
        let mut rng = rand::thread_rng();
        let occupied: std::collections::HashSet<(i32, i32)> =
            self.body.iter().map(|p| (p.x, p.y)).collect();

        let mut free = Vec::new();
        for x in 0..GRID_SIZE as i32 {
            for y in 0..GRID_SIZE as i32 {
                if !occupied.contains(&(x, y)) {
                    free.push((x, y));
                }
            }
        }

        if free.is_empty() {
            self.dead = true;
            return;
        }

        let (fx, fy) = free[rng.gen_range(0..free.len())];
        self.food_x = fx;
        self.food_y = fy;
    }

    fn body_collide(&self, x: i32, y: i32) -> bool {
        self.body.iter().any(|p| p.x == x && p.y == y)
    }

    pub fn look(&mut self) {
        let head = &self.body[0];
        let size = GRID_SIZE as i32;

        for (d, &(dx, dy)) in DIRECTIONS.iter().enumerate() {
            let mut dist = 1i32;
            let mut food = 0.0f32;
            let mut body = 0.0f32;
            let mut px = head.x + dx;
            let mut py = head.y + dy;

            while px >= 0 && px < size && py >= 0 && py < size {
                if food == 0.0 && px == self.food_x && py == self.food_y {
                    food = 1.0;
                }
                if body == 0.0 && self.body_collide(px, py) {
                    body = 1.0;
                }
                px += dx;
                py += dy;
                dist += 1;
            }

            self.vision[d * 3] = food;
            self.vision[d * 3 + 1] = body;
            self.vision[d * 3 + 2] = 1.0 / dist as f32;
        }
    }

    pub fn move_snake(&mut self) {
        if self.dead {
            return;
        }

        self.lifetime += 1;
        self.life_left -= 1;

        if self.life_left <= 0 {
            self.dead = true;
            return;
        }

        let nx = self.body[0].x + self.x_vel;
        let ny = self.body[0].y + self.y_vel;
        let size = GRID_SIZE as i32;

        if nx < 0 || nx >= size || ny < 0 || ny >= size {
            self.dead = true;
            return;
        }

        if self.body_collide(nx, ny) {
            self.dead = true;
            return;
        }

        self.body.insert(0, Pos { x: nx, y: ny });

        if nx == self.food_x && ny == self.food_y {
            self.score += 1;
            self.life_left = (self.life_left + 100).min(500);
            self.place_food();
        } else {
            self.body.pop();
        }
    }

    pub fn calc_fitness(&mut self) {
        let lt = self.lifetime as f64;
        if self.score < 10 {
            self.fitness = lt * lt * 2.0f64.powi(self.score as i32);
        } else {
            self.fitness = lt * lt * 2.0f64.powi(10) * (self.score as f64 - 9.0);
        }
    }

    pub fn apply_decision(&mut self) {
        let max_idx = self
            .decision
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);

        match max_idx {
            0 => {
                if self.y_vel != 1 {
                    self.x_vel = 0;
                    self.y_vel = -1;
                }
            }
            1 => {
                if self.y_vel != -1 {
                    self.x_vel = 0;
                    self.y_vel = 1;
                }
            }
            2 => {
                if self.x_vel != 1 {
                    self.x_vel = -1;
                    self.y_vel = 0;
                }
            }
            3 => {
                if self.x_vel != -1 {
                    self.x_vel = 1;
                    self.y_vel = 0;
                }
            }
            _ => {}
        }
    }
}
