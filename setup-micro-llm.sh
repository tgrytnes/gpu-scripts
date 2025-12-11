#!/bin/bash
set -euo pipefail

log() { echo -e "[\033[1;34mINFO\033[0m] $1"; }

log "🤖 Installing Ollama..."
curl -fsSL https://ollama.com/install.sh | sh

log "📝 Configuring Ollama to accept connections from Docker..."
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

log "📥 Pulling Qwen2.5 1.5B model (optimized for 4GB RAM)..."
ollama pull qwen2.5:1.5b

log "✅ Testing model..."
ollama run qwen2.5:1.5b "Say hello in one sentence"

log "🎉 Done! Ollama is running with Qwen2.5 1.5B"
log "📍 API: http://$(hostname -I | awk '{print $1}'):11434"
log ""
log "💡 Available smaller models you can also try:"
log "   - tinyllama:1.1b (smallest, ~650MB)"
log "   - gemma2:2b (may work, ~1.2GB)"
log "   To switch: ollama pull <model-name>"
