#!/bin/bash
set -euo pipefail

# =============================================================================
# TabbyAPI Setup Script for RunPod
# =============================================================================
# Sets up TabbyAPI with ExLlamaV2 backend for EXL2 quantized models
# Optimized for RTX 3090/4090 (24GB VRAM)
# Perfect for EXL2 models like Qwen2.5-Coder-32B-Instruct-exl2-4.0bpw
# =============================================================================

PYTHON_VERSION="3.11"
PROJECT_DIR="${PROJECT_DIR:-$HOME/gpu-scripts}"
TABBY_DIR="$PROJECT_DIR/tabbyAPI"
ENV_FILE="$PROJECT_DIR/.env"
LOG_DIR="$PROJECT_DIR/logs"
PID_FILE="$PROJECT_DIR/tabbyapi.pid"

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

log_info "Starting TabbyAPI setup for RunPod..."

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
MODEL_DIR="${TABBY_MODEL_DIR:-/workspace/models}"
MODEL_REPO="${TABBY_MODEL_REPO:-bartowski/Qwen2.5-Coder-32B-Instruct-exl2}"
MODEL_REVISION="${TABBY_MODEL_REVISION:-4.0bpw}"
PORT="${TABBY_PORT:-5000}"
HOST="${TABBY_HOST:-0.0.0.0}"
MAX_SEQ_LEN="${TABBY_MAX_SEQ_LEN:-8192}"
API_KEY="${TABBY_API_KEY:-}"
HF_TOKEN="${HF_TOKEN:-}"

# Use /workspace if available (RunPod), otherwise use home
if [[ -d "/workspace" ]] && [[ ! -d "$MODEL_DIR" ]]; then
    MODEL_DIR="/workspace/models"
fi

mkdir -p "$MODEL_DIR"

log_info "Configuration:"
log_info "  Model: $MODEL_REPO"
log_info "  Revision: $MODEL_REVISION"
log_info "  Model Dir: $MODEL_DIR"
log_info "  Port: $PORT"
log_info "  Max Seq Len: $MAX_SEQ_LEN"

# =============================================================================
# System Requirements Check
# =============================================================================

log_info "Checking system requirements..."

# Check for NVIDIA GPU
if ! command -v nvidia-smi &> /dev/null; then
    log_error "nvidia-smi not found. TabbyAPI requires an NVIDIA GPU."
fi

GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)
log_success "GPU detected: $GPU_INFO"

# Check available GPU memory
GPU_MEMORY_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
GPU_MEMORY_GB=$((GPU_MEMORY_MB / 1024))
log_info "Available GPU memory: ${GPU_MEMORY_GB}GB"

if [[ $GPU_MEMORY_GB -lt 16 ]]; then
    log_warn "Less than 16GB GPU memory available. 32B models may not fit."
fi

# Check GPU compute capability (TabbyAPI/ExLlama needs 6.0+)
COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
log_info "GPU Compute Capability: $COMPUTE_CAP"

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
    apt-get install -y -qq python3.11 python3.11-venv python3-pip git > /dev/null 2>&1
    PYTHON_BIN=$(which python3.11)
    log_success "Python 3.11 installed"
fi

# =============================================================================
# Clone/Update TabbyAPI
# =============================================================================

if [[ -d "$TABBY_DIR" ]]; then
    log_info "TabbyAPI directory exists, updating..."
    cd "$TABBY_DIR"
    git pull -q || log_warn "Failed to update TabbyAPI"
else
    log_info "Cloning TabbyAPI repository..."
    git clone -q https://github.com/theroyallab/tabbyAPI.git "$TABBY_DIR"
    cd "$TABBY_DIR"
    log_success "TabbyAPI cloned"
fi

# =============================================================================
# Create Virtual Environment and Install Dependencies
# =============================================================================

log_info "Setting up Python virtual environment..."

cd "$TABBY_DIR"

if [[ ! -d "venv" ]]; then
    "$PYTHON_BIN" -m venv venv
    log_success "Virtual environment created"
else
    log_info "Virtual environment already exists"
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
log_info "Upgrading pip..."
pip install --upgrade pip -qq

# Install TabbyAPI requirements
log_info "Installing TabbyAPI dependencies (this may take a few minutes)..."
pip install -r requirements.txt -qq

log_success "TabbyAPI dependencies installed"

# =============================================================================
# Configure TabbyAPI
# =============================================================================

log_info "Configuring TabbyAPI..."

# Create config.yml if it doesn't exist
CONFIG_FILE="$TABBY_DIR/config.yml"

if [[ -f "$CONFIG_FILE" ]]; then
    log_info "Backing up existing config.yml..."
    cp "$CONFIG_FILE" "$CONFIG_FILE.backup.$(date +%s)"
fi

# Create configuration
cat > "$CONFIG_FILE" << EOF
# TabbyAPI Configuration
# Generated by setup_tabbyapi.sh

# Network settings
network:
  host: $HOST
  port: $PORT

# Model settings
model:
  model_dir: $MODEL_DIR
  # Model name will be loaded from the directory
  max_seq_len: $MAX_SEQ_LEN
  # For EXL2 models, override max sequence length if needed
  override_base_seq_len: null

  # GPU split (automatic for single GPU)
  gpu_split_auto: true

  # Cache settings for better performance
  cache_mode: FP16
  cache_size: 4096

  # Prompt template (auto-detect from model)
  prompt_template: null

# Sampling settings (defaults, can be overridden per request)
sampling:
  max_tokens: 1000
  temperature: 0.7
  top_p: 0.9
  top_k: 40
  repetition_penalty: 1.15

# Logging
logging:
  log_level: INFO
  log_requests: true

# Developer settings
developer:
  unsafe_launch: false
  disable_auth: $([ -z "$API_KEY" ] && echo "true" || echo "false")
  cuda_malloc_backend: false

EOF

# Add API key if provided
if [[ -n "$API_KEY" ]]; then
    cat >> "$CONFIG_FILE" << EOF

# Authentication
auth:
  api_key: $API_KEY

EOF
    log_info "API key authentication enabled"
fi

log_success "Configuration created: $CONFIG_FILE"

# =============================================================================
# Download Model (if not exists)
# =============================================================================

log_info "Checking for model..."

# Expected model path
MODEL_PATH="$MODEL_DIR/$MODEL_REPO"

if [[ -d "$MODEL_PATH" ]]; then
    log_success "Model found: $MODEL_PATH"
else
    log_info "Model not found. Starting download..."
    log_warn "This will download a large model (~20-25GB for 4.0bpw). This may take a while."

    # Check disk space
    REQUIRED_SPACE_GB=30
    AVAILABLE_SPACE_KB=$(df "$MODEL_DIR" | tail -1 | awk '{print $4}')
    AVAILABLE_SPACE_GB=$((AVAILABLE_SPACE_KB / 1024 / 1024))

    if [[ $AVAILABLE_SPACE_GB -lt $REQUIRED_SPACE_GB ]]; then
        log_error "Insufficient disk space: ${AVAILABLE_SPACE_GB}GB available, ${REQUIRED_SPACE_GB}GB required"
    else
        log_info "Sufficient disk space: ${AVAILABLE_SPACE_GB}GB available"
    fi

    # Setup HuggingFace authentication if token provided
    if [[ -n "$HF_TOKEN" ]]; then
        log_info "Using HuggingFace authentication token"
        pip install -q huggingface_hub
        python3 -c "from huggingface_hub import login; login('$HF_TOKEN')" 2>/dev/null || true
    fi

    # Download using HuggingFace CLI
    log_info "Downloading model: $MODEL_REPO (revision: $MODEL_REVISION)"

    HF_TOKEN_ARG=""
    if [[ -n "$HF_TOKEN" ]]; then
        HF_TOKEN_ARG="--token $HF_TOKEN"
    fi

    if command -v huggingface-cli &> /dev/null || pip show huggingface_hub &> /dev/null; then
        huggingface-cli download \
            $HF_TOKEN_ARG \
            "$MODEL_REPO" \
            --revision "$MODEL_REVISION" \
            --local-dir "$MODEL_PATH" \
            --local-dir-use-symlinks False || log_error "Model download failed"

        log_success "Model downloaded successfully"
    else
        log_error "huggingface-cli not available. Install with: pip install huggingface_hub"
    fi
fi

# =============================================================================
# Stop Existing TabbyAPI Process
# =============================================================================

if [[ -f "$PID_FILE" ]]; then
    OLD_PID=$(cat "$PID_FILE")
    if ps -p "$OLD_PID" > /dev/null 2>&1; then
        log_info "Stopping existing TabbyAPI process (PID: $OLD_PID)..."
        kill "$OLD_PID" 2>/dev/null || true
        sleep 3
        if ps -p "$OLD_PID" > /dev/null 2>&1; then
            kill -9 "$OLD_PID" 2>/dev/null || true
        fi
        log_success "Existing process stopped"
    fi
    rm -f "$PID_FILE"
fi

# Also check for any TabbyAPI processes on the port
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
    log_info "Killing process on port $PORT..."
    lsof -Pi :$PORT -sTCP:LISTEN -t | xargs kill -9 2>/dev/null || true
    sleep 2
fi

# =============================================================================
# Start TabbyAPI Server
# =============================================================================

log_info "Starting TabbyAPI server..."

cd "$TABBY_DIR"
source venv/bin/activate

# Start TabbyAPI in background
LOG_FILE="$LOG_DIR/tabbyapi.log"

# Find the model directory name (last component of path)
MODEL_NAME=$(basename "$MODEL_PATH")

nohup python3 main.py \
    --model-name "$MODEL_NAME" \
    > "$LOG_FILE" 2>&1 &

TABBY_PID=$!

# Save PID
echo "$TABBY_PID" > "$PID_FILE"

log_info "TabbyAPI server starting (PID: $TABBY_PID)..."
log_info "Log file: $LOG_FILE"

# =============================================================================
# Health Check
# =============================================================================

log_info "Waiting for TabbyAPI server to be ready..."
log_warn "Initial startup may take 1-3 minutes while loading the model..."

MAX_WAIT=300  # 5 minutes for model loading
ELAPSED=0
while [[ $ELAPSED -lt $MAX_WAIT ]]; do
    # Check if process is still running
    if ! ps -p "$TABBY_PID" > /dev/null 2>&1; then
        log_error "TabbyAPI process died. Check logs: $LOG_FILE"
    fi

    # Check if API is responding
    if curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
        log_success "TabbyAPI server is ready!"
        break
    fi

    if [[ $((ELAPSED % 15)) -eq 0 ]] && [[ $ELAPSED -gt 0 ]]; then
        log_info "Still loading model... (${ELAPSED}s elapsed)"
        # Show last few lines of log
        if [[ -f "$LOG_FILE" ]]; then
            tail -n 2 "$LOG_FILE" | sed 's/^/  /' || true
        fi
    fi

    sleep 3
    ELAPSED=$((ELAPSED + 3))
done

if [[ $ELAPSED -ge $MAX_WAIT ]]; then
    log_error "TabbyAPI server failed to start within ${MAX_WAIT}s. Check logs: $LOG_FILE"
fi

# =============================================================================
# Verify Setup
# =============================================================================

log_info "Verifying server endpoints..."

# Check /v1/models endpoint
if curl -s "http://localhost:$PORT/v1/models" > /dev/null 2>&1; then
    log_success "Models endpoint working"
else
    log_warn "Could not verify models endpoint"
fi

# =============================================================================
# Summary
# =============================================================================

echo ""
log_success "=========================================="
log_success "TabbyAPI Server Setup Complete!"
log_success "=========================================="
echo ""
log_info "Server Details:"
log_info "  Endpoint: http://localhost:$PORT"
log_info "  OpenAI API: http://localhost:$PORT/v1"
log_info "  Health: http://localhost:$PORT/health"
log_info "  Models: http://localhost:$PORT/v1/models"
log_info "  Process ID: $TABBY_PID"
log_info "  Log file: $LOG_FILE"
log_info "  Model: $MODEL_REPO ($MODEL_REVISION)"
echo ""
log_info "Management Commands:"
log_info "  View logs: tail -f $LOG_FILE"
log_info "  Stop server: kill $TABBY_PID"
log_info "  Check status: curl http://localhost:$PORT/health"
log_info "  Control script: $PROJECT_DIR/tabbyapi_control.sh"
echo ""
log_info "Test the server:"
log_info "  python3 $PROJECT_DIR/test_tabbyapi.py"
echo ""
log_info "TabbyAPI Features:"
log_info "  ✓ EXL2/EXL3 quantization support"
log_info "  ✓ OpenAI-compatible API"
log_info "  ✓ Extremely fast inference (ExLlamaV2)"
log_info "  ✓ Low VRAM usage with EXL2"
log_info "  ✓ Built-in model downloader"
echo ""

# =============================================================================
# Create Test Script
# =============================================================================

TEST_SCRIPT="$PROJECT_DIR/test_tabbyapi.py"
cat > "$TEST_SCRIPT" << 'EOFPYTHON'
#!/usr/bin/env python3
"""Test script for TabbyAPI OpenAI-compatible API"""

import os
import sys
from openai import OpenAI

def test_tabbyapi():
    # Configuration
    port = os.environ.get("TABBY_PORT", "5000")
    api_key = os.environ.get("TABBY_API_KEY", "dummy")

    print(f"🚀 Testing TabbyAPI server at http://localhost:{port}")

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

        # Test completion with a coding task
        print(f"\n💬 Testing chat completion with {model_id}...")
        print("    (Testing Qwen2.5-Coder capabilities)")

        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": "You are a helpful coding assistant."},
                {"role": "user", "content": "Write a Python function to calculate fibonacci numbers using memoization."}
            ],
            max_tokens=300,
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
        print(f"  1. Check if TabbyAPI is running: curl http://localhost:{port}/health")
        print(f"  2. Check logs: tail -f ~/gpu-scripts/logs/tabbyapi.log")
        print("  3. Verify API key if authentication is enabled")
        print("  4. Ensure model is loaded (can take 1-3 minutes on first start)")
        return 1

if __name__ == "__main__":
    sys.exit(test_tabbyapi())
EOFPYTHON

chmod +x "$TEST_SCRIPT"

log_success "Test script created: $TEST_SCRIPT"

deactivate 2>/dev/null || true
