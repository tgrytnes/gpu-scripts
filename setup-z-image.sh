#!/bin/bash
set -e

echo "🎨 Z-Image Generation Setup"
echo ""

cd /workspace/repos/z-image 2>/dev/null || {
    echo "❌ z-image repo not found!"
    echo "   Creating placeholder directory..."
    mkdir -p /workspace/repos/z-image
    cd /workspace/repos/z-image
}

echo "📦 Installing Z-Image dependencies..."
pip install -q diffusers transformers accelerate torch safetensors pillow

# Download model from Azure if available
if [ ! -z "$AZURE_STORAGE_ACCOUNT" ]; then
    echo "📥 Checking Azure for cached Z-Image models..."
    mkdir -p /workspace/models
    
    az storage blob download \
        --account-name ${AZURE_STORAGE_ACCOUNT} \
        --account-key ${AZURE_STORAGE_KEY} \
        --container-name models \
        --name z-image/model.safetensors \
        --file /workspace/models/z-image-model.safetensors 2>/dev/null && echo "  ✓ Model found in Azure" || {
        echo "  ⚠️ Model not in Azure, will download from HuggingFace on first use"
    }
fi

# Download from HuggingFace if needed
if [ ! -z "$HF_TOKEN" ] && [ ! -f "/workspace/models/z-image-model.safetensors" ]; then
    echo "📥 Downloading Z-Image model from HuggingFace..."
    huggingface-cli login --token $HF_TOKEN
    # Add specific model download command here based on which Z-Image model you're using
    echo "  Configure the specific model in setup-z-image.sh"
fi

echo ""
echo "✅ Z-Image environment ready!"
echo ""
echo "Example usage:"
echo "  cd /workspace/repos/z-image"
echo "  python generate.py --prompt 'A beautiful landscape'"
echo ""
/bin/bash