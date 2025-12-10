#!/bin/bash
set -euo pipefail

# Configuration
MODEL_FILENAME="mistral-7b-instruct-v0.2.Q8_0.gguf"
# Using TheBloke's repo as it is the standard for GGUF
HF_REPO="TheBloke/Mistral-7B-Instruct-v0.2-GGUF" 
MODEL_DIR="/workspace/models"
MODEL_PATH="$MODEL_DIR/$MODEL_FILENAME"

# Helper for formatted output
log() { echo -e "[\033[1;34mINFO\033[0m] $1"; }
warn() { echo -e "[\033[1;33mWARN\033[0m] $1"; }
err()  { echo -e "[\033[1;31mERR\033[0m]  $1"; }

log "🤖 LLM Showcase Setup (Q8 Quantization)"

# Ensure directories exist
mkdir -p "$MODEL_DIR"
mkdir -p /workspace/repos/llm-showcase

# Navigate to repo or create placeholder
cd /workspace/repos/llm-showcase || true

# 1. Install Dependencies
# We install huggingface-hub (for the CLI) and llama-cpp-python (for the Q8 model)
log "📦 Installing LLM dependencies..."
pip install -q transformers accelerate torch huggingface-hub bitsandbytes llama-cpp-python

# 2. Check/Download Model
if [[ -f "$MODEL_PATH" ]]; then
    log "✓ Model found at: $MODEL_PATH"
else
    log "Model not found locally. Checking download options..."

    # A. Try Azure Cache First
    DOWNLOAD_SUCCESS=false
    if [[ -n "${AZURE_STORAGE_ACCOUNT:-}" ]]; then
        log "📥 Checking Azure..."
        if az storage blob download \
            --account-name "${AZURE_STORAGE_ACCOUNT}" \
            --account-key "${AZURE_STORAGE_KEY}" \
            --container-name models \
            --name "llm/$MODEL_FILENAME" \
            --file "$MODEL_PATH" 2>/dev/null; then
            log "✓ Downloaded from Azure"
            DOWNLOAD_SUCCESS=true
        else
            warn "Model not in Azure"
        fi
    fi

    # B. Download from Hugging Face if Azure failed
    if [[ "$DOWNLOAD_SUCCESS" = false ]]; then
        log "📥 Downloading from Hugging Face..."
        
        # Check for HF_TOKEN
        if [[ -z "${HF_TOKEN:-}" ]]; then
            warn "HF_TOKEN is not set. You may hit rate limits or fail on gated models."
            TOKEN_ARG=""
        else
            log "Authenticated using HF_TOKEN"
            TOKEN_ARG="--token $HF_TOKEN"
        fi

        # Use huggingface-cli for robust download (handles redirects/LFS better than raw curl)
        huggingface-cli download \
            $TOKEN_ARG \
            "$HF_REPO" \
            "$MODEL_FILENAME" \
            --local-dir "$MODEL_DIR" \
            --local-dir-use-symlinks False

        if [[ -f "$MODEL_PATH" ]]; then
            log "✓ Download complete"
        else
            err "Download failed."
            exit 1
        fi
    fi
fi

echo ""
log "✅ LLM environment ready!"
echo ""
echo "Example usage (Q8 GGUF):"
echo "  cd /workspace/repos/llm-showcase"
echo "  python -c 'from llama_cpp import Llama; llm = Llama(model_path=\"$MODEL_PATH\", verbose=False); print(llm(\"Q: Hello! A: \", max_tokens=32))'"
echo ""
/bin/bash
