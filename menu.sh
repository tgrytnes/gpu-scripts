#!/bin/bash
set -e

clear
echo "╔════════════════════════════════════════╗"
echo "║   🚀 Multi-Project GPU Workspace       ║"
echo "╚════════════════════════════════════════╝"
echo ""
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'No GPU detected')"
echo "VRAM: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo ""

# Create workspace directories
mkdir -p /workspace/repos
mkdir -p /workspace/models
mkdir -p /workspace/data

# Clone repos if GITHUB_USERNAME is set
if [ ! -z "$GITHUB_USERNAME" ]; then
    echo "📦 Setting up repositories..."
    cd /workspace/repos
    
    # Only clone if doesn't exist
    [ ! -d "cv-satellite" ] && git clone https://github.com/${GITHUB_USERNAME}/cv-satellite.git 2>/dev/null && echo "  ✓ cv-satellite cloned" || echo "  ✓ cv-satellite exists"
    [ ! -d "llm-showcase" ] && git clone https://github.com/${GITHUB_USERNAME}/llm-showcase.git 2>/dev/null && echo "  ✓ llm-showcase cloned" || echo "  ✓ llm-showcase exists"
    [ ! -d "rag-project" ] && git clone https://github.com/${GITHUB_USERNAME}/rag-project.git 2>/dev/null && echo "  ✓ rag-project cloned" || echo "  ✓ rag-project exists"
    [ ! -d "flux-generation" ] && git clone https://github.com/${GITHUB_USERNAME}/flux-generation.git 2>/dev/null && echo "  ✓ flux-generation cloned" || echo "  ✓ flux-generation exists"
else
    echo "⚠️  GITHUB_USERNAME not set, skipping repo cloning"
fi

echo ""
echo "═══════════════════════════════════════"
echo "What do you want to work on today?"
echo "═══════════════════════════════════════"
echo ""
echo "1) 🛰️  CV Satellite Project (SAM/CLIP)"
echo "2) 🤖 LLM Showcase (Llama/Mistral)"
echo "3) 📚 RAG Project (OCR + Search)"
echo "4) 🎨 Z-Image Image Generation"
echo "5) 🖥️  JupyterLab (interactive)"
echo "6) 🐚 Bash Shell (manual setup)"
echo ""

# Read with timeout (defaults to option 5 if no input after 30 sec)
read -t 30 -p "Enter choice [1-6] (default: 5): " choice || choice=5

case $choice in
    1)
        echo ""
        echo "🛰️ Setting up CV Satellite environment..."
        curl -s https://raw.githubusercontent.com/${GITHUB_USERNAME}/gpu-scripts/main/setup-cv.sh | bash
        ;;
    2)
        echo ""
        echo "🤖 Setting up LLM environment..."
        curl -s https://raw.githubusercontent.com/${GITHUB_USERNAME}/gpu-scripts/main/setup-llm.sh | bash
        ;;
    3)
        echo ""
        echo "📚 Setting up RAG environment..."
        curl -s https://raw.githubusercontent.com/${GITHUB_USERNAME}/gpu-scripts/main/setup-rag.sh | bash
        ;;
    4)
        echo ""
        echo "🎨 Setting up Z-Image environment..."
        curl -s https://raw.githubusercontent.com/${GITHUB_USERNAME}/gpu-scripts/main/setup-z-image.sh | bash
        ;;
    5)
        echo ""
        echo "🖥️ Starting JupyterLab..."
        pip install -q jupyterlab 2>/dev/null || echo "JupyterLab already installed"
        echo ""
        echo "✅ Starting JupyterLab at http://localhost:8888"
        echo "   (Use the 'Connect' button in RunPod/Vast.ai UI)"
        echo ""
        jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root &
        /bin/bash
        ;;
    6)
        echo ""
        echo "🐚 Bash shell ready! Set up your environment manually."
        echo "   Repos are in: /workspace/repos"
        echo "   Models go in: /workspace/models"
        echo ""
        /bin/bash
        ;;
    *)
        echo ""
        echo "Invalid choice. Starting shell..."
        /bin/bash
        ;;
esac