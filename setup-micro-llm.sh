#!/bin/bash
set -euo pipefail

log() { echo -e "[\033[1;34mINFO\033[0m] $1"; }
warn() { echo -e "[\033[1;33mWARN\033[0m] $1"; }

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

log "📥 Pulling Qwen3 1.7B model (optimized for 4GB systems)..."
docker exec ollama ollama pull qwen3:1.7b

log "✅ Testing model..."
docker exec ollama ollama run qwen3:1.7b "Say hello in one sentence"

log "🎉 Ollama Docker container running!"
log "📍 API: http://$(hostname -I | awk '{print $1}'):11434"
log ""
log "💡 Model: Qwen3 1.7B (Alibaba, April 2025)"
log "   - Fits comfortably on 4GB RAM system (~1.5GB)"
log "   - Performs comparable to Qwen2.5 3B"
log "   - Good for chat, summaries, quick answers"
log ""
log "🔧 Management commands:"
log "   docker stop ollama    # Stop container"
log "   docker start ollama   # Start container"
log "   docker logs -f ollama # View logs"
log "   docker exec ollama ollama list  # List models"
