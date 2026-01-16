#!/bin/bash
set -euo pipefail

# =============================================================================
# vLLM Production Setup Script for RunPod
# =============================================================================
# Sets up vLLM with OpenAI-compatible API on a fresh RunPod GPU instance
# Optimized for RTX 3090/4090 (24GB VRAM)
# =============================================================================

PYTHON_VERSION="3.11"
PROJECT_DIR="${PROJECT_DIR:-$HOME/gpu-scripts}"
ENV_FILE="$PROJECT_DIR/.env"
LOG_DIR="$PROJECT_DIR/logs"
PID_FILE="$PROJECT_DIR/vllm.pid"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# =============================================================================
# Configuration
# =============================================================================

log_info "Starting vLLM server setup for RunPod..."

# Create necessary directories
mkdir -p "$PROJECT_DIR"
mkdir -p "$LOG_DIR"

# Load environment variables from .env if it exists
if [[ -f "$ENV_FILE" ]]; then
  set -a
  source "$ENV_FILE"
  set +a
  log_success "Loaded environment variables from $ENV_FILE"
else
  log_warn "No .env found at $ENV_FILE (using defaults)"
fi

# Configuration with defaults (can be overridden in .env)
MODEL_ID="${VLLM_MODEL_ID:-meta-llama/Llama-3.2-3B-Instruct}"
PORT="${VLLM_PORT:-8000}"
HOST="${VLLM_HOST:-0.0.0.0}"
GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
DTYPE="${VLLM_DTYPE:-auto}"
API_KEY="${VLLM_API_KEY:-}"
HF_TOKEN="${HF_TOKEN:-}"

log_info "Configuration:"
log_info "  Model: $MODEL_ID"
log_info "  Port: $PORT"
log_info "  GPU Memory: ${GPU_MEMORY_UTILIZATION}"
log_info "  Max Length: $MAX_MODEL_LEN"
log_info "  Data Type: $DTYPE"

# =============================================================================
# System Requirements Check
# =============================================================================

log_info "Checking system requirements..."

# Check for NVIDIA GPU
if ! command -v nvidia-smi &> /dev/null; then
    log_error "nvidia-smi not found. This script requires an NVIDIA GPU."
fi

GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)
log_success "GPU detected: $GPU_INFO"

# Check available GPU memory
GPU_MEMORY_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
GPU_MEMORY_GB=$((GPU_MEMORY_MB / 1024))
log_info "Available GPU memory: ${GPU_MEMORY_GB}GB"

if [[ $GPU_MEMORY_GB -lt 8 ]]; then
    log_warn "Less than 8GB GPU memory available. Large models may not fit."
fi

# =============================================================================
# Python Installation
# =============================================================================

log_info "Checking Python installation..."

if command -v python3.11 &> /dev/null; then
    PYTHON_BIN=$(which python3.11)
    log_success "Python 3.11 found: $PYTHON_BIN"
elif command -v python3 &> /dev/null; then
    PYTHON_VERSION_INSTALLED=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
    if [[ "$PYTHON_VERSION_INSTALLED" == "3.11" ]] || [[ "$PYTHON_VERSION_INSTALLED" == "3.10" ]]; then
        PYTHON_BIN=$(which python3)
        log_success "Python $PYTHON_VERSION_INSTALLED found: $PYTHON_BIN"
    else
        log_info "Installing Python 3.11..."
        apt-get update -qq
        apt-get install -y -qq python3.11 python3.11-venv python3-pip > /dev/null 2>&1
        PYTHON_BIN=$(which python3.11)
        log_success "Python 3.11 installed"
    fi
else
    log_info "Installing Python 3.11..."
    apt-get update -qq
    apt-get install -y -qq software-properties-common > /dev/null 2>&1
    add-apt-repository -y ppa:deadsnakes/ppa > /dev/null 2>&1
    apt-get update -qq
    apt-get install -y -qq python3.11 python3.11-venv python3-pip > /dev/null 2>&1
    PYTHON_BIN=$(which python3.11)
    log_success "Python 3.11 installed"
fi

# =============================================================================
# vLLM Installation
# =============================================================================

log_info "Installing vLLM..."

# Upgrade pip
"$PYTHON_BIN" -m pip install --upgrade pip -qq

# Install vLLM with CUDA support
log_info "Installing vLLM with CUDA support (this may take a few minutes)..."
"$PYTHON_BIN" -m pip install vllm -qq

# Install additional dependencies
"$PYTHON_BIN" -m pip install "huggingface_hub>=0.23.0" openai requests -qq

log_success "vLLM installed successfully"

# Verify installation
VLLM_VERSION=$("$PYTHON_BIN" -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo "unknown")
log_info "vLLM version: $VLLM_VERSION"

# =============================================================================
# HuggingFace Authentication
# =============================================================================

if [[ -n "$HF_TOKEN" ]]; then
    log_info "Configuring HuggingFace authentication..."
    "$PYTHON_BIN" -m pip install huggingface_hub -qq
    "$PYTHON_BIN" -c "from huggingface_hub import login; login('$HF_TOKEN')" 2>/dev/null || log_warn "HF login failed"
    log_success "HuggingFace authentication configured"
fi

# =============================================================================
# Stop Existing vLLM Process
# =============================================================================

if [[ -f "$PID_FILE" ]]; then
    OLD_PID=$(cat "$PID_FILE")
    if ps -p "$OLD_PID" > /dev/null 2>&1; then
        log_info "Stopping existing vLLM process (PID: $OLD_PID)..."
        kill "$OLD_PID" 2>/dev/null || true
        sleep 2
        if ps -p "$OLD_PID" > /dev/null 2>&1; then
            kill -9 "$OLD_PID" 2>/dev/null || true
        fi
        log_success "Existing process stopped"
    fi
    rm -f "$PID_FILE"
fi

# Also check for any vLLM processes on the port
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
    log_info "Killing process on port $PORT..."
    lsof -Pi :$PORT -sTCP:LISTEN -t | xargs kill -9 2>/dev/null || true
    sleep 1
fi

# =============================================================================
# Start vLLM Server
# =============================================================================

log_info "Starting vLLM server..."

cd "$PROJECT_DIR"

# Build the command
VLLM_CMD="$PYTHON_BIN -m vllm.entrypoints.openai.api_server \
  --model $MODEL_ID \
  --host $HOST \
  --port $PORT \
  --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
  --max-model-len $MAX_MODEL_LEN \
  --dtype $DTYPE"

# Add API key if provided
if [[ -n "$API_KEY" ]]; then
    VLLM_CMD="$VLLM_CMD --api-key $API_KEY"
    log_info "API key authentication enabled"
fi

# Start vLLM in background
LOG_FILE="$LOG_DIR/vllm.log"
nohup $VLLM_CMD > "$LOG_FILE" 2>&1 &
VLLM_PID=$!

# Save PID
echo "$VLLM_PID" > "$PID_FILE"

log_info "vLLM server starting (PID: $VLLM_PID)..."
log_info "Log file: $LOG_FILE"

# =============================================================================
# Health Check
# =============================================================================

log_info "Waiting for vLLM server to be ready..."

MAX_WAIT=180  # 3 minutes
ELAPSED=0
while [[ $ELAPSED -lt $MAX_WAIT ]]; do
    if curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
        log_success "vLLM server is ready!"
        break
    fi

    # Check if process is still running
    if ! ps -p "$VLLM_PID" > /dev/null 2>&1; then
        log_error "vLLM process died. Check logs: $LOG_FILE"
    fi

    if [[ $((ELAPSED % 10)) -eq 0 ]] && [[ $ELAPSED -gt 0 ]]; then
        log_info "Still waiting... (${ELAPSED}s elapsed)"
    fi

    sleep 2
    ELAPSED=$((ELAPSED + 2))
done

if [[ $ELAPSED -ge $MAX_WAIT ]]; then
    log_error "vLLM server failed to start within ${MAX_WAIT}s. Check logs: $LOG_FILE"
fi

# =============================================================================
# Verify Setup
# =============================================================================

log_info "Verifying server endpoints..."

# Check /v1/models endpoint
if curl -s "http://localhost:$PORT/v1/models" > /dev/null 2>&1; then
    AVAILABLE_MODELS=$(curl -s "http://localhost:$PORT/v1/models" | "$PYTHON_BIN" -c "import sys, json; data=json.load(sys.stdin); print(data['data'][0]['id'])" 2>/dev/null || echo "unknown")
    log_success "Models endpoint working: $AVAILABLE_MODELS"
else
    log_warn "Could not verify models endpoint"
fi

# =============================================================================
# Summary
# =============================================================================

echo ""
log_success "=========================================="
log_success "vLLM Server Setup Complete!"
log_success "=========================================="
echo ""
log_info "Server Details:"
log_info "  Endpoint: http://localhost:$PORT"
log_info "  OpenAI API: http://localhost:$PORT/v1"
log_info "  Health: http://localhost:$PORT/health"
log_info "  Models: http://localhost:$PORT/v1/models"
log_info "  Process ID: $VLLM_PID"
log_info "  Log file: $LOG_FILE"
echo ""
log_info "Management Commands:"
log_info "  View logs: tail -f $LOG_FILE"
log_info "  Stop server: kill $VLLM_PID"
log_info "  Check status: curl http://localhost:$PORT/health"
echo ""
log_info "Test the server:"
log_info "  python3 $PROJECT_DIR/test_vllm.py"
echo ""

# =============================================================================
# Create Test Script
# =============================================================================

TEST_SCRIPT="$PROJECT_DIR/test_vllm.py"
cat > "$TEST_SCRIPT" << 'EOFPYTHON'
#!/usr/bin/env python3
"""Test script for vLLM OpenAI-compatible API"""

import os
import sys
from openai import OpenAI

def test_vllm():
    # Configuration
    port = os.environ.get("VLLM_PORT", "8000")
    api_key = os.environ.get("VLLM_API_KEY", "dummy")  # vLLM requires a key but it can be anything if auth is disabled

    print(f"🚀 Testing vLLM server at http://localhost:{port}")

    # Initialize client
    client = OpenAI(
        base_url=f"http://localhost:{port}/v1",
        api_key=api_key,
    )

    try:
        # List available models
        print("\n📋 Available models:")
        models = client.models.list()
        for model in models.data:
            print(f"  - {model.id}")

        model_id = models.data[0].id

        # Test completion
        print(f"\n💬 Testing chat completion with {model_id}...")
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Explain what vLLM is in one sentence."}
            ],
            max_tokens=100,
            temperature=0.7,
        )

        print("\n📝 Response:")
        print(response.choices[0].message.content)

        # Print stats
        print(f"\n📊 Stats:")
        print(f"  Tokens used: {response.usage.total_tokens}")
        print(f"  Prompt tokens: {response.usage.prompt_tokens}")
        print(f"  Completion tokens: {response.usage.completion_tokens}")

        print("\n✅ Test successful!")
        return 0

    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        print("\nTroubleshooting:")
        print("  1. Check if vLLM is running: curl http://localhost:{port}/health")
        print("  2. Check logs: tail -f ~/gpu-scripts/logs/vllm.log")
        print("  3. Verify API key if authentication is enabled")
        return 1

if __name__ == "__main__":
    sys.exit(test_vllm())
EOFPYTHON

chmod +x "$TEST_SCRIPT"

log_success "Test script created: $TEST_SCRIPT"
