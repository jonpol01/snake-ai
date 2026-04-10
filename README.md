# Snake AI

[![Rust](https://img.shields.io/badge/Rust-1.88+-orange?logo=rust)](https://www.rust-lang.org/)
[![Metal](https://img.shields.io/badge/GPU-Metal%20%2F%20wgpu-blue?logo=apple)](https://wgpu.rs/)
[![Tauri](https://img.shields.io/badge/Desktop-Tauri%202.x-yellow?logo=tauri)](https://v2.tauri.app/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-macOS%20%7C%20Linux%20%7C%20Windows%20%7C%20Docker-lightgrey)]()

A neuroevolutionary system that trains neural networks to play Snake using a genetic algorithm, with GPU-accelerated inference via **wgpu** compute shaders.

The Rust backend runs the simulation and serves a browser-based dashboard over WebSocket. A native **Tauri** desktop app wraps the dashboard in a proper macOS/Windows/Linux window. The frontend is embedded in the binary -- no external files needed.

## Architecture

```
Tauri native window (macOS .app / Windows .msi / Linux .AppImage)
  +-- WebKit/WebView2 loads http://localhost:3030
        |  WebSocket
Background thread (tokio)
  +-- axum HTTP server (port 3030)
  +-- Simulation loop (2000 snakes per generation)
  +-- wgpu Metal compute shader (parallel NN forward passes)
```

## Features

- **Tauri desktop app** -- native window on macOS (.app/.dmg), Windows (.msi/.exe), Linux (.AppImage/.deb)
- **GPU compute** -- all 2000 neural network forward passes run in parallel via wgpu (Metal on Mac, Vulkan on Linux)
- **CPU fallback** -- automatically detects when GPU is unavailable (Docker, Windows, etc.)
- **Stages** -- Classic (empty grid), Warehouse (obstacle racks with AMR robot visuals), Mixed (randomized per generation)
- **Browser dashboard** -- real-time game view, neural network visualization, fitness graph, hardware monitor, and color-coded log panel
- **Checkpoint export/import** -- share your trained brains with others or load someone else's checkpoint
- **Auto-checkpoint** -- saves progress every 10 generations, resumes on restart
- **Proper data directory** -- persistent files stored in platform-standard locations (not random CWD)
- **Headless mode** -- runs as a server for Docker or remote access, no GUI needed
- **Auto-versioning CI** -- pushes to main auto-bump version (Conventional Commits), build and release binaries for all platforms

## Data Storage

Persistent files (checkpoint, leaderboard, screenshots) are stored in platform-standard locations:

| Platform | Path |
|----------|------|
| macOS | `~/Library/Application Support/com.snake-ai.app/` |
| Linux | `~/.local/share/snake-ai/` |
| Windows | `%APPDATA%/snake-ai/` |
| Dev mode | Project root (if `Cargo.toml` exists in CWD) |
| Override | Set `SNAKE_AI_DATA_DIR=/path` env var |

## Stages

| Stage | Description |
|-------|-------------|
| **Classic** | Empty 20x20 grid. The default. Scores reach 50+ within 500 generations. |
| **Warehouse** | Two horizontal rack obstacles. Cartoon top-down warehouse floor with yellow safety lines, AMR robot head with lidar dome, trailing cargo carts. Food appears as cardboard packages. |
| **Mixed** | Randomly alternates between Classic and Warehouse each generation. Trains a single brain that generalizes across all stages. |

Switching stages preserves trained brains -- no progress is lost.

## Algorithm

| Component | Detail |
|-----------|--------|
| Population | 2000 snakes |
| Network | 24 inputs, 2 hidden layers (16 neurons each, sigmoid), 4 outputs |
| Vision | 8-direction raycasting (food, body/obstacle, wall distance) |
| Fitness | `lifetime * 2^min(score,10) * max(score-9, 1)` |
| Selection | Fitness-proportionate (roulette wheel) |
| Crossover | Single cut-point per weight matrix |
| Mutation | 8% Gaussian perturbation (std ~0.25, clamped to [-2, 2]) |
| Elitism | Best brain preserved across generations |

## Quick Start

### Download (easiest)

Grab the latest release for your platform from the [Releases page](https://github.com/jonpol01/snake-ai/releases):

- **macOS**: `Snake AI_x.x.x_aarch64.dmg` (Apple Silicon) or `x86_64.dmg` (Intel)
- **Windows**: `.msi` installer or `-setup.exe`
- **Linux**: `.AppImage` or `.deb`

### Build from source

```bash
# Desktop app (native window)
git clone https://github.com/jonpol01/snake-ai.git
cd snake-ai
cargo install tauri-cli --version "^2"
cargo tauri build

# The .app / .dmg is at:
# target/release/bundle/macos/Snake AI.app
# target/release/bundle/dmg/Snake AI_*.dmg
```

### Headless (server / Docker)

```bash
# Native
cargo run --release -p snake-ai
# Then open http://localhost:3030

# Docker
docker compose up --build
# Then open http://localhost:3030
```

## Checkpoint Sharing

Share your trained neural networks with others:

- **Export**: Click the blue EXPORT button in the control panel (or `GET /api/export-checkpoint`)
- **Import**: Click the purple IMPORT button and select a `.json` file (or `POST /api/import-checkpoint`)
- After importing, click START to load the checkpoint

## Requirements

### Desktop app
- macOS 11+ (Apple Silicon or Intel), Windows 10+, or Linux with WebKitGTK

### Build from source
- Rust 1.88+
- Tauri CLI v2 (`cargo install tauri-cli --version "^2"`)
- macOS: Xcode CLT, Linux: `libwebkit2gtk-4.1-dev libgtk-3-dev`

### Docker
- Docker and Docker Compose
- Any OS

## Project Structure

```
src/
  lib.rs           -- library crate: backend, sim loop, WebSocket, HTTP endpoints
  main.rs          -- headless binary entry point (Docker/server)
  gpu.rs           -- wgpu compute shader + CPU fallback
  neural_net.rs    -- Matrix, NeuralNet (crossover, Gaussian mutation)
  snake.rs         -- Snake game logic, 8-direction vision
  population.rs    -- Genetic algorithm, checkpointing
  stage.rs         -- Stage system (Classic, Warehouse, Mixed)
  paths.rs         -- Platform-aware data directory resolution
  protocol.rs      -- WebSocket message types
  shared.rs        -- SharedState for cross-thread communication
  leaderboard.rs   -- Persistent hall of fame
  screenshot.rs    -- Server-side PNG rendering for records
  llm.rs           -- AI Coach LLM proxy
src-tauri/
  src/main.rs      -- Tauri desktop entry point
  tauri.conf.json  -- Tauri window/bundle configuration
static/
  index.html       -- Browser dashboard (embedded into binary at compile time)
```

## CI/CD

Pushes to `main` automatically:
1. Analyze commits using Conventional Commits (`feat:` = minor, `fix:` = patch)
2. Bump version and create a git tag
3. Build headless binaries + Tauri desktop bundles for all platforms
4. Publish a GitHub Release with all artifacts

## License

MIT
