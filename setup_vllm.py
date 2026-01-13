#!/bin/bash
set -euo pipefail

PYTHON_VERSION="3.11.9"
PROJECT_DIR="$HOME/gpu-scripts"
PYTHON_BIN="/usr/local/bin/python3.11"
ENV_FILE="$PROJECT_DIR/.env"

echo "Starting vLLM server setup..."

# Load .env (must be KEY=VALUE lines, no spaces)
if [[ -f "$ENV_FILE" ]]; then
  set -a
  source "$ENV_FILE"
  set +a
  echo "Loaded env from $ENV_FILE"
else
  echo "No .env found at $ENV_FILE (continuing without it)"
fi

# Defaults (can be overridden in .env)
MODEL_ID="${VLLM_MODEL_ID:-${VLLM_LLM_MODEL:-${MODEL_ID:-NousResearch/Hermes-2-Pro-Mistral-7B}}}"
PORT="${VLLM_PORT:-8001}"
VLLM_DIR="${VLLM_DIR:-$PROJECT_DIR}"

# Install Python 3.11 if missing
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Installing Python 3.11..."
  apt-get update && apt-get install -y \
    build-essential libssl-dev zlib1g-dev libncurses5-dev \
    libncursesw5-dev libreadline-dev libsqlite3-dev libgdbm-dev \
    libdb5.3-dev libbz2-dev libexpat1-dev liblzma-dev tk-dev \
    libffi-dev uuid-dev wget curl git cmake ninja-build
  cd /usr/src
  wget -q "https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz" -O Python.tgz
  tar -xzf Python.tgz
  cd "Python-${PYTHON_VERSION}"
  ./configure --enable-optimizations --with-ensurepip=install
  make -j"$(nproc)"
  make altinstall
fi

# Install vLLM
"$PYTHON_BIN" -m pip install -U pip
"$PYTHON_BIN" -m pip install -U "vllm[serve]" "huggingface_hub>=0.23.0"

# Start vLLM
cd "$VLLM_DIR"
echo "Starting vLLM on port $PORT with model: $MODEL_ID"
nohup "$PYTHON_BIN" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_ID" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192 \
  ${VLLM_API_KEY:+--api-key "$VLLM_API_KEY"} \
  > "$VLLM_DIR/vllm.log" 2>&1 &

echo "vLLM started. Log: $VLLM_DIR/vllm.log"
