use std::collections::HashSet;

use rand::Rng;
use serde::{Deserialize, Serialize};

const GRID_SIZE: i32 = 20;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StageKind {
    Classic,
    Warehouse,
    Mixed,
}

impl Default for StageKind {
    fn default() -> Self {
        StageKind::Classic
    }
}

pub struct Stage {
    pub kind: StageKind,
    pub obstacles: HashSet<(i32, i32)>,
}

impl Default for Stage {
    fn default() -> Self {
        Self::new(StageKind::Classic)
    }
}

impl Stage {
    pub fn new(kind: StageKind) -> Self {
        let effective = match kind {
            StageKind::Mixed => {
                // Randomly pick Classic or Warehouse each time
                if rand::thread_rng().gen_bool(0.5) {
                    StageKind::Classic
                } else {
                    StageKind::Warehouse
                }
            }
            other => other,
        };
        let obstacles = match effective {
            StageKind::Classic | StageKind::Mixed => HashSet::new(),
            StageKind::Warehouse => generate_warehouse(),
        };
        // Store the original kind (Mixed), not the resolved one
        Self { kind, obstacles }
    }

    pub fn is_obstacle(&self, x: i32, y: i32) -> bool {
        self.obstacles.contains(&(x, y))
    }

    pub fn obstacle_list(&self) -> Vec<(i32, i32)> {
        let mut list: Vec<_> = self.obstacles.iter().copied().collect();
        list.sort();
        list
    }
}

/// Generate two horizontal warehouse racks (shelving pillars).
///
/// Tuned for learnability:
///   - Width: 3-5 cells (narrow enough to navigate around)
///   - Height: 1 cell (single row racks)
///   - Rack 1 in upper area (~y 5-7)
///   - Rack 2 in lower area (~y 12-14)
///   - Offset from center so the middle corridor stays open
///   - 3-cell clearance from grid edges (wide corridors)
///   - Avoids spawn zone (column 9-11, rows 8-12)
fn generate_warehouse() -> HashSet<(i32, i32)> {
    let mut rng = rand::thread_rng();
    let mut obstacles = HashSet::new();

    for rack_idx in 0..2 {
        let width = rng.gen_range(3..=5);

        // Vertical position: upper or lower area
        let y = if rack_idx == 0 {
            rng.gen_range(5..=7)
        } else {
            rng.gen_range(12..=14)
        };

        // Offset to one side — avoid blocking the center column
        let side = if rng.gen_bool(0.5) { -1 } else { 1 };
        let x_center = GRID_SIZE / 2 + side * rng.gen_range(2..=4);
        let x_start = (x_center - width / 2).max(3).min(GRID_SIZE - width - 3);

        for x in x_start..(x_start + width) {
            // Skip spawn zone
            if (9..=11).contains(&x) && (8..=12).contains(&y) {
                continue;
            }
            obstacles.insert((x, y));
        }
    }

    obstacles
}
