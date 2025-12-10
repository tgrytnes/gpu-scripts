#!/bin/bash
set -e

echo "🛰️ CV Satellite Project Setup"
echo ""

cd /workspace/repos/cv-satellite 2>/dev/null || {
    echo "❌ cv-satellite repo not found!"
    echo "   Make sure GITHUB_USERNAME is set and repo exists"
    exit 1
}

# Install dependencies
echo "📦 Installing Python dependencies..."
pip install -q segment-anything torch torchvision pillow numpy opencv-python rasterio sentinelsat || {
    echo "⚠️ Installing common CV packages"
}

# Download models from Azure if available
if [ ! -z "$AZURE_STORAGE_ACCOUNT" ]; then
    echo "📥 Checking Azure for cached models..."
    
    mkdir -p /workspace/models
    
    # Try SAM
    az storage blob download \
        --account-name ${AZURE_STORAGE_ACCOUNT} \
        --account-key ${AZURE_STORAGE_KEY} \
        --container-name models \
        --name cv/sam_vit_h.pth \
        --file /workspace/models/sam_vit_h.pth 2>/dev/null && echo "  ✓ SAM downloaded" || echo "  ⚠️ SAM not in Azure"
    
    # Try CLIP
    az storage blob download \
        --account-name ${AZURE_STORAGE_ACCOUNT} \
        --account-key ${AZURE_STORAGE_KEY} \
        --container-name models \
        --name cv/clip_vit_l.pth \
        --file /workspace/models/clip_vit_l.pth 2>/dev/null && echo "  ✓ CLIP downloaded" || echo "  ⚠️ CLIP not in Azure"
fi

echo ""
echo "✅ CV environment ready!"
echo ""
echo "To start working:"
echo "  cd /workspace/repos/cv-satellite"
echo "  python process_satellite.py"
echo ""
/bin/bash