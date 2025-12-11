#!/bin/bash
set -euo pipefail

log() { echo -e "[\033[1;34mINFO\033[0m] $1"; }

log "🤖 Starting Ollama in Docker..."

# Remove old container if exists
if docker ps -a | grep -q ollama; then
    log "🗑️  Removing existing Ollama container..."
    docker rm -f ollama
fi

# Start Ollama
docker run -d \
    -p 11434:11434 \
    -v ollama:/root/.ollama \
    --name ollama \
    --restart always \
    ollama/ollama

log "⏳ Waiting for Ollama to start..."
sleep 5

log "📥 Pulling Qwen2.5 1.5B model..."
docker exec ollama ollama pull qwen2.5:1.5b

log "✅ Testing model..."
docker exec ollama ollama run qwen2.5:1.5b "Say hello"

log "🎉 Ollama Docker container running!"
log "📍 API: http://$(hostname -I | awk '{print $1}'):11434"
