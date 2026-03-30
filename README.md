# Snake AI

[![Rust](https://img.shields.io/badge/Rust-1.85+-orange?logo=rust)](https://www.rust-lang.org/)
[![Metal](https://img.shields.io/badge/GPU-Metal%20%2F%20wgpu-blue?logo=apple)](https://wgpu.rs/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-macOS%20%7C%20Linux%20%7C%20Windows-lightgrey)]()

A neuroevolutionary system that trains neural networks to play Snake using a genetic algorithm, with GPU-accelerated inference via **wgpu** compute shaders.

The Rust backend runs the simulation and serves a browser-based dashboard over WebSocket. A native **egui** GUI provides a control panel with live logs.

## Architecture

```
 egui Window (native)          Browser (localhost:3030)
 - Controls + logs              - Game visualization
 - Stats display                - Neural network viz
        |                       - Fitness graph + logs
        |   SharedState (Arc)          |
        +------+-------+------+-------+
               |               |
         axum HTTP + WS     Simulation Loop
               |               |
            wgpu Metal      2000 snakes
          compute shader    per generation
```

## Features

- **GPU compute** -- all 2000 neural network forward passes run in parallel via wgpu (Metal on Mac, Vulkan on Linux)
- **CPU fallback** -- automatically detects when GPU is unavailable
- **Browser dashboard** -- real-time game view, neural network visualization, fitness graph, and color-coded log panel
- **Native GUI** -- egui control panel with stats, settings, and log terminal
- **Auto-checkpoint** -- saves progress every 10 generations to `checkpoint.json`, resumes on restart
- **Paper included** -- `paper.pdf` with full algorithm description in academic format

## Algorithm

| Component | Detail |
|-----------|--------|
| Population | 2000 snakes |
| Network | 24 inputs, 2 hidden layers (16 neurons each, sigmoid), 4 outputs |
| Vision | 8-direction raycasting (food, body, wall distance) |
| Fitness | `lifetime^2 * 2^min(score,10) * max(score-9, 1)` |
| Selection | Fitness-proportionate (roulette wheel) |
| Crossover | Single cut-point per weight matrix |
| Mutation | 5% random replacement in [-1, 1] |
| Elitism | Best brain preserved across generations |

## Quick Start

```bash
# Clone and build
git clone https://github.com/jonpol01/snake-ai.git
cd snake-ai
cargo build --release

# Run (opens native GUI + starts web server)
cargo run --release

# Open browser dashboard
open http://localhost:3030
```

Click **Start** from either the GUI or the browser to begin evolution. Scores typically reach 30+ within 20-50 generations.

## Requirements

- Rust 1.85+
- macOS with Apple Silicon (Metal) or Linux with Vulkan-capable GPU
- A modern browser (Chrome, Safari, Firefox)

## Project Structure

```
src/
  main.rs          -- axum server, sim loop, WebSocket handler
  gui.rs           -- egui native window
  gpu.rs           -- wgpu compute shader + CPU fallback
  neural_net.rs    -- Matrix, NeuralNet (crossover, mutation)
  snake.rs         -- Snake game logic, 8-direction vision
  population.rs    -- Genetic algorithm, checkpointing
  protocol.rs      -- WebSocket message types
  shared.rs        -- SharedState for GUI <-> sim communication
static/
  index.html       -- Browser dashboard (canvas + WebSocket)
```

## License

MIT
