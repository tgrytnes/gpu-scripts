#!/bin/bash
set -euo pipefail

log() { echo -e "[\033[1;34mINFO\033[0m] $1"; }

log "🤖 Installing Ollama..."
curl -fsSL https://ollama.com/install.sh | sh

log "📝 Configuring Ollama to accept connections from Docker..."
# Ollama muss auf 0.0.0.0 lauschen, nicht nur localhost
sudo mkdir -p /etc/systemd/system/ollama.service.d
cat <<EOF | sudo tee /etc/systemd/system/ollama.service.d/override.conf
[Service]
Environment="OLLAMA_HOST=0.0.0.0:11434"
EOF

log "🔄 Restarting Ollama service..."
sudo systemctl daemon-reload
sudo systemctl restart ollama

log "⏳ Waiting for Ollama to start..."
sleep 3

log "📥 Pulling Llama 3.2 3B model (this may take a few minutes)..."
ollama pull llama3.2:3b

log "✅ Testing model..."
ollama run llama3.2:3b "Say hello in one sentence"

log "🎉 Done! Ollama is running and accessible at:"
log "   - Local: http://localhost:11434"
log "   - Docker: http://host.docker.internal:11434"
log "   - Network: http://$(hostname -I | awk '{print $1}'):11434"
