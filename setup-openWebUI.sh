#!/bin/bash
set -euo pipefail

log() { echo -e "[\033[1;34mINFO\033[0m] $1"; }
error() { echo -e "[\033[1;31mERROR\033[0m] $1"; exit 1; }

log "🌐 Setting up Open WebUI..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    log "📦 Installing Docker..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh || error "Failed to install Docker"
    sudo usermod -aG docker $USER
    rm get-docker.sh
    log "⚠️  Docker installed. You may need to log out/in or run: newgrp docker"
fi

# Remove old container if exists
if docker ps -a | grep -q open-webui; then
    log "🗑️  Removing existing container..."
    docker rm -f open-webui
fi

# Start Open WebUI
log "🚀 Starting Open WebUI..."
docker run -d \
    -p 3000:8080 \
    --add-host=host.docker.internal:host-gateway \
    -v open-webui:/app/backend/data \
    --name open-webui \
    --restart always \
    ghcr.io/open-webui/open-webui:main

log "⏳ Waiting for Open WebUI to start..."
sleep 5

log "✅ Open WebUI is running!"
log "📍 Access at: http://$(hostname -I | awk '{print $1}'):3000"
log "📍 Or: http://localhost:3000"
