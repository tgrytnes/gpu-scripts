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

log "📥 Pulling Qwen3 4B model (April 2025 release)..."
docker exec ollama ollama pull qwen3:4b

log "✅ Testing model..."
docker exec ollama ollama run qwen3:4b "Say hello in one sentence"

log "🗑️  Removing old Qwen2.5 1.5B model to save space..."
docker exec ollama ollama rm qwen2.5:1.5b || warn "Old model not found"

log "🎉 Ollama Docker container running!"
log "📍 API: http://$(hostname -I | awk '{print $1}'):11434"
log ""
log "💡 Model: Qwen3 4B (Alibaba, April 2025)"
log "   - Better than Qwen2.5 in reasoning and multilingual"
log "   - Optimized for 4GB RAM systems"
log ""
log "🔧 Management commands:"
log "   docker stop ollama    # Stop container"
log "   docker start ollama   # Start container"
log "   docker logs -f ollama # View logs"
log "   docker exec ollama ollama list  # List all models"
