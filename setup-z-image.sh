#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status

# Configuration
WORKSPACE_DIR="/workspace"
Z_IMAGE_REPO="https://github.com/YOUR_USERNAME/z-image.git" # REPLACE THIS
COMFY_REPO="https://github.com/comfyanonymous/ComfyUI.git"
MODEL_DIR="${WORKSPACE_DIR}/models"

echo "🎨 Setting up AI Environment (Z-Image + ComfyUI)"
echo "-----------------------------------------------"

# 1. System Updates & Basic Tools
echo "🛠️ Updating system tools..."
apt-get update && apt-get install -y git wget aria2 libgl1-mesa-glx > /dev/null 2>&1

# 2. Setup Z-Image
echo "📦 Setting up Z-Image..."
if [ -d "${WORKSPACE_DIR}/repos/z-image" ]; then
    echo "   z-image already exists. Pulling latest..."
    cd "${WORKSPACE_DIR}/repos/z-image"
    git pull
else
    echo "   Cloning z-image..."
    mkdir -p "${WORKSPACE_DIR}/repos"
    # If you don't have a real repo yet, we create the dir to prevent errors
    if [ "$Z_IMAGE_REPO" == "https://github.com/YOUR_USERNAME/z-image.git" ]; then
         mkdir -p "${WORKSPACE_DIR}/repos/z-image"
         echo "   ⚠️ Placeholder directory created (Update Z_IMAGE_REPO in script to clone actual code)"
    else
         git clone "$Z_IMAGE_REPO" "${WORKSPACE_DIR}/repos/z-image"
    fi
fi

# Install Z-Image common libs
echo "   Installing Z-Image pip dependencies..."
pip install -q diffusers transformers accelerate torch safetensors pillow

# 3. Setup ComfyUI
echo "🛋️ Setting up ComfyUI..."
if [ -d "${WORKSPACE_DIR}/ComfyUI" ]; then
    echo "   ComfyUI already exists."
else
    cd "${WORKSPACE_DIR}"
    git clone "$COMFY_REPO"
fi

echo "   Installing ComfyUI requirements..."
cd "${WORKSPACE_DIR}/ComfyUI"
pip install -r requirements.txt
# Install ComfyUI Manager (Highly recommended)
cd "${WORKSPACE_DIR}/ComfyUI/custom_nodes"
git clone https://github.com/ltdrdata/ComfyUI-Manager.git 2>/dev/null || echo "   ComfyUI Manager already installed"

# 4. Model Management (Shared Storage)
echo "📥 Managing Models..."
mkdir -p "$MODEL_DIR"

# Download Logic (Azure > HuggingFace > Skip)
TARGET_MODEL="${MODEL_DIR}/z-image-model.safetensors"

if [ -f "$TARGET_MODEL" ]; then
    echo "   ✓ Model already present."
else
    if [ ! -z "$AZURE_STORAGE_ACCOUNT" ]; then
        echo "   Checking Azure..."
        az storage blob download \
            --account-name ${AZURE_STORAGE_ACCOUNT} \
            --account-key ${AZURE_STORAGE_KEY} \
            --container-name models \
            --name z-image/model.safetensors \
            --file "$TARGET_MODEL" 2>/dev/null && echo "   ✓ Downloaded from Azure"
    fi

    # Fallback to HuggingFace if Azure failed or skipped
    if [ ! -f "$TARGET_MODEL" ] && [ ! -z "$HF_TOKEN" ]; then
        echo "   Downloading from HuggingFace..."
        huggingface-cli login --token $HF_TOKEN
        # Example download - REPLACE with your actual model ID
        huggingface-cli download stabilityai/stable-diffusion-xl-base-1.0 --include "*.safetensors" --local-dir "$MODEL_DIR" --local-dir-use-symlinks False
    fi
fi

# 5. Link Models to ComfyUI
# This creates a "shortcut" so ComfyUI sees the model without duplicating the file
echo "🔗 Linking models to ComfyUI..."
mkdir -p "${WORKSPACE_DIR}/ComfyUI/models/checkpoints"
ln -sf "$TARGET_MODEL" "${WORKSPACE_DIR}/ComfyUI/models/checkpoints/z-image-model.safetensors"

echo ""
echo "✅ Setup Complete!"
echo "-----------------------------------------------"
echo "To run Z-Image:"
echo "  cd ${WORKSPACE_DIR}/repos/z-image && python generate.py"
echo ""
echo "To run ComfyUI:"
echo "  cd ${WORKSPACE_DIR}/ComfyUI && python main.py --listen --port 8188"
echo "-----------------------------------------------"
