#!/bin/bash
set -euo pipefail

# --- Configuration ---
# Llama 3.3 70B (Q6_K - ~52GB, single file)
HF_REPO="bartowski/Llama-3.3-70B-Instruct-GGUF"
MODEL_FILE="Llama-3.3-70B-Instruct-Q6_K.gguf"
MODEL_NAME="llama3.3-70b-q6"

# Use /workspace if it exists (RunPod), otherwise use home directory
if [[ -d "/workspace" ]]; then
    MODEL_DIR="/workspace/models"
    LOG_DIR="/workspace/logs"
else
    MODEL_DIR="$HOME/models"
    LOG_DIR="$HOME/.ollama/logs"
fi

MODEL_PATH="$MODEL_DIR/$MODEL_FILE"

# --- Helper Functions ---
log() { echo -e "[\033[1;34mINFO\033[0m] $1"; }
warn() { echo -e "[\033[1;33mWARN\033[0m] $1"; }
error() { echo -e "[\033[1;31mERROR\033[0m] $1"; exit 1; }

# Check if running with sufficient permissions
check_sudo() {
    if [[ $EUID -ne 0 ]]; then
        if ! sudo -n true 2>/dev/null; then
            error "This script requires sudo privileges. Please run with sudo or ensure you have sudo access."
        fi
        SUDO="sudo"
    else
        SUDO=""
    fi
}

log "🚀 Starting Setup: Llama 3.3 70B (Q6_K) via Ollama"

# Check permissions first
check_sudo

# Create necessary directories
mkdir -p "$MODEL_DIR"
mkdir -p "$LOG_DIR"

# 1. Install Dependencies
if command -v apt-get &> /dev/null; then
    log "📦 Installing curl and git..."
    $SUDO apt-get update || warn "Failed to update apt cache"
    $SUDO apt-get install -y curl git python3-pip python3-venv || error "Failed to install system dependencies"
fi

# Install Python packages in user space to avoid permission issues
log "📦 Installing Python packages..."
pip install --user -q huggingface-hub openai || error "Failed to install Python packages"

# 2. Install Ollama
if ! command -v ollama &> /dev/null; then
    log "⚡ Installing Ollama (Pre-compiled with CUDA support)..."
    curl -fsSL https://ollama.com/install.sh | $SUDO sh || error "Failed to install Ollama"
else
    log "✓ Ollama is already installed"
fi

# 3. Start Ollama Server in the background
log "🔄 Starting Ollama background server..."
OLLAMA_LOG="$LOG_DIR/ollama.log"
ollama serve > "$OLLAMA_LOG" 2>&1 &
OLLAMA_PID=$!

# Wait for Ollama to be ready (with timeout)
log "⏳ Waiting for Ollama server to start..."
for i in {1..30}; do
    if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
        log "✓ Ollama server is running (PID: $OLLAMA_PID)"
        break
    fi
    if [[ $i -eq 30 ]]; then
        error "Ollama server failed to start. Check logs at: $OLLAMA_LOG"
    fi
    sleep 1
done

# 4. Download the Q6_K Model (single file - no splits!)
if [[ -f "$MODEL_PATH" ]]; then
    log "✓ Model file found: $MODEL_PATH"
    MODEL_SIZE=$(stat -c%s "$MODEL_PATH" 2>/dev/null || stat -f%z "$MODEL_PATH" 2>/dev/null)
    MODEL_SIZE_GB=$((MODEL_SIZE / 1024 / 1024 / 1024))
    log "   File size: ${MODEL_SIZE_GB}GB"
else
    log "📥 Downloading Llama 3.3 70B Q6_K (~52GB, single file)..."
    log "   (This uses high-speed connection, please wait)"

    # Check available disk space
    REQUIRED_SPACE_GB=55
    AVAILABLE_SPACE_KB=$(df "$MODEL_DIR" | tail -1 | awk '{print $4}')
    AVAILABLE_SPACE_GB=$((AVAILABLE_SPACE_KB / 1024 / 1024))
    
    if [[ $AVAILABLE_SPACE_GB -lt $REQUIRED_SPACE_GB ]]; then
        error "Insufficient disk space: ${AVAILABLE_SPACE_GB}GB available, ${REQUIRED_SPACE_GB}GB required"
    else
        log "✓ Sufficient disk space: ${AVAILABLE_SPACE_GB}GB available"
    fi

    # Check for HuggingFace token
    HF_TOKEN_ARG=""
    if [[ -n "${HF_TOKEN:-}" ]]; then
        log "✓ Using HuggingFace authentication token"
        HF_TOKEN_ARG="--token $HF_TOKEN"
    else
        warn "HF_TOKEN not set. You may encounter rate limits or issues with gated models."
        warn "Set HF_TOKEN environment variable if download fails."
    fi

    # Download the model file
    log "📥 Downloading model file (this will take a while)..."
    if ! huggingface-cli download \
        $HF_TOKEN_ARG \
        "$HF_REPO" \
        "$MODEL_FILE" \
        --local-dir "$MODEL_DIR" \
        --local-dir-use-symlinks False; then
        error "Failed to download model from HuggingFace. Check your internet connection and HF_TOKEN."
    fi

    # Verify download
    if [[ ! -f "$MODEL_PATH" ]]; then
        error "Model file not found after download. Expected at: $MODEL_PATH"
    fi

    MODEL_SIZE=$(stat -c%s "$MODEL_PATH" 2>/dev/null || stat -f%z "$MODEL_PATH" 2>/dev/null)
    MODEL_SIZE_GB=$((MODEL_SIZE / 1024 / 1024 / 1024))
    log "✓ Download complete! File size: ${MODEL_SIZE_GB}GB"
fi

# 5. Create the Custom Model in Ollama
log "⚙️  Registering Q6_K Model with Ollama..."

# Create Modelfile
MODELFILE_PATH="$MODEL_DIR/Modelfile"
cat > "$MODELFILE_PATH" <<EOF
FROM $MODEL_PATH
PARAMETER num_ctx 8192
PARAMETER temperature 0.7
PARAMETER top_p 0.9
EOF

log "📄 Modelfile created, referencing: $MODEL_PATH"

# Remove existing model if present
if ollama list | grep -q "$MODEL_NAME"; then
    log "🗑️  Removing existing model: $MODEL_NAME"
    ollama rm "$MODEL_NAME" || warn "Failed to remove existing model"
fi

# Create the model (this will import the GGUF file into Ollama)
log "🔨 Creating Ollama model '$MODEL_NAME' (this may take a few minutes)..."
if ! ollama create "$MODEL_NAME" -f "$MODELFILE_PATH"; then
    error "Failed to create Ollama model. Check if the model file is valid."
fi

log "✓ Model '$MODEL_NAME' registered successfully"

# 6. Python Client Script
TEST_SCRIPT="$MODEL_DIR/test_ollama.py"
log "📝 Creating test script at: $TEST_SCRIPT"

cat > "$TEST_SCRIPT" <<EOF
#!/usr/bin/env python3
from openai import OpenAI
import time
import sys

print("🤖 Connecting to Llama 3.3 70B (Q6_K)...")

try:
    # Ollama provides an OpenAI-compatible API
    client = OpenAI(
        base_url='http://localhost:11434/v1',
        api_key='ollama',  # required, but unused
    )

    prompt = "Q: Analyze the pros and cons of using Rust vs C++ for low-level systems programming. A: "
    print(f"\nPrompt: {prompt}")

    start = time.time()
    response = client.completions.create(
        model="$MODEL_NAME",
        prompt=prompt,
        max_tokens=500
    )
    end = time.time()

    print("\nGenerated Output:")
    print(response.choices[0].text)
    print(f"\n⏱️  Time taken: {end - start:.2f}s")

except Exception as e:
    print(f"\n❌ Error: {e}", file=sys.stderr)
    print("Make sure Ollama server is running and the model is loaded.", file=sys.stderr)
    sys.exit(1)
EOF

chmod +x "$TEST_SCRIPT"

echo ""
log "✅ Setup Complete!"
echo ""
log "📍 Model location: $MODEL_PATH"
log "📍 Ollama logs: $OLLAMA_LOG"
log "📍 Test script: $TEST_SCRIPT"
echo ""
log "🎯 Next steps:"
log "  1. Run test script: python3 $TEST_SCRIPT"
log "  2. Interactive chat: ollama run $MODEL_NAME"
log "  3. List models: ollama list"
echo ""
log "ℹ️  Note: Q6_K quantization provides excellent quality (~52GB)"
log "   Alternative: Q4_K_M is faster and smaller (~40GB) with good quality"
echo ""
