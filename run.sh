#!/usr/bin/env bash
# Snake AI — One-click launcher
# Works on macOS, Linux, and Windows (Git Bash / WSL)

set -e

PORT=3030
APP="target/release/snake-ai"
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo ""
echo -e "${GREEN}  ╔═══════════════════════════╗${NC}"
echo -e "${GREEN}  ║     SNAKE AI LAUNCHER     ║${NC}"
echo -e "${GREEN}  ╚═══════════════════════════╝${NC}"
echo ""

# --- Check if Docker is available (easiest path) ---
use_docker() {
    echo -e "${YELLOW}Using Docker (CPU mode)...${NC}"
    echo ""
    if command -v docker compose &>/dev/null; then
        docker compose up --build
    elif command -v docker-compose &>/dev/null; then
        docker-compose up --build
    else
        docker build -t snake-ai . && docker run -p ${PORT}:${PORT} --rm snake-ai
    fi
}

# --- Check if pre-built binary exists ---
use_binary() {
    echo -e "${GREEN}Found pre-built binary.${NC}"
    echo -e "Starting server on ${GREEN}http://localhost:${PORT}${NC}"
    echo -e "Press ${YELLOW}Ctrl+C${NC} to stop."
    echo ""
    ./${APP}
}

# --- Build from source ---
build_and_run() {
    echo -e "${YELLOW}Building from source (first time takes ~2 min)...${NC}"
    echo ""
    cargo build --release
    echo ""
    echo -e "${GREEN}Build complete.${NC}"
    use_binary
}

# --- Main ---

# Option 1: Binary already built
if [ -f "${APP}" ]; then
    use_binary
    exit 0
fi

# Option 2: Rust is installed — build it
if command -v cargo &>/dev/null; then
    echo -e "Rust toolchain detected ($(rustc --version 2>/dev/null || echo 'unknown'))."
    build_and_run
    exit 0
fi

# Option 3: Docker is installed — use container
if command -v docker &>/dev/null; then
    echo -e "No Rust toolchain found, but Docker is available."
    use_docker
    exit 0
fi

# Option 4: Nothing available
echo -e "${RED}ERROR: Neither Rust nor Docker found.${NC}"
echo ""
echo "Install one of the following:"
echo ""
echo "  Rust (recommended for GPU acceleration):"
echo "    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
echo ""
echo "  Docker (easiest, works everywhere, CPU only):"
echo "    https://www.docker.com/products/docker-desktop/"
echo ""
exit 1
