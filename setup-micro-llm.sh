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

log "📥 Pulling Qwen3 4B 2507 model (July 2025 release)..."
docker exec ollama ollama pull qwen3:4b-2507

log "✅ Testing model..."
docker exec ollama ollama run qwen3:4b-2507 "Say hello in one sentence"

log "🎉 Ollama Docker container running!"
log "📍 API: http://$(hostname -I | awk '{print $1}'):11434"
log ""
log "💡 Model: Qwen3 4B 2507 (Alibaba, July 2025)"
log "   - 74% MMLU-Pro, 67% GPQA Diamond"
log "   - Thinking mode available"
log "   - Optimized for 4GB RAM systems"
log ""
log "🔧 Management commands:"
log "   docker stop ollama    # Stop container"
log "   docker start ollama   # Start container"
log "   docker logs -f ollama # View logs"
