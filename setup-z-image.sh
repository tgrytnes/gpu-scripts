#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status

# Configuration
WORKSPACE_DIR="/workspace"
Z_IMAGE_REPO="https://github.com/YOUR_USERNAME/z-image.git" # REPLACE THIS
COMFY_REPO="https://github.com/comfyanonymous/ComfyUI.git"
Z_IMAGE_MODEL_REPO="Comfy-Org/z_image_turbo"
Z_IMAGE_LORA_REPO="tarn59/pixel_art_style_lora_z_image_turbo"
MODEL_DIR="${WORKSPACE_DIR}/models"
Z_IMAGE_DOWNLOAD_TEMP="${MODEL_DIR}/.z-image-downloads"

# Support both HF_TOKEN and HF_Token (Vast.ai default) plus PID1 env exports
if [ -z "$HF_TOKEN" ]; then
    if [ ! -z "$HF_Token" ]; then
        export HF_TOKEN="$HF_Token"
    elif [ -r /proc/1/environ ]; then
        PID_HF_TOKEN=$(tr '\0' '\n' </proc/1/environ | grep '^HF_Token=' | head -n 1 | cut -d '=' -f2-)
        [ ! -z "$PID_HF_TOKEN" ] && export HF_TOKEN="$PID_HF_TOKEN"
    fi
fi

HF_CACHE_DIR="${WORKSPACE_DIR}/.cache/huggingface"
mkdir -p "$HF_CACHE_DIR"
export HF_HOME="$HF_CACHE_DIR"

download_z_image_asset() {
    local repo="$1"
    local remote_path="$2"
    local dest_rel="$3"
    local dest_path="${Z_IMAGE_MODEL_ROOT}/${dest_rel}"

    if [ -f "$dest_path" ]; then
        echo "      ✓ ${dest_rel} already present."
        return
    fi

    mkdir -p "$(dirname "$dest_path")"
    mkdir -p "$Z_IMAGE_DOWNLOAD_TEMP"
    local staged_path="${Z_IMAGE_DOWNLOAD_TEMP}/${remote_path}"
    rm -f "$staged_path"

    echo "      ↳ Fetching ${dest_rel} from ${repo}"
    huggingface-cli download "$repo" "$remote_path" --local-dir "$Z_IMAGE_DOWNLOAD_TEMP" --local-dir-use-symlinks False --token "$HF_TOKEN" >/dev/null

    if [ ! -f "$staged_path" ]; then
        echo "      ✗ Failed to download ${remote_path} from ${repo}"
        exit 1
    fi

    mv "$staged_path" "$dest_path"
    find "$Z_IMAGE_DOWNLOAD_TEMP" -type d -empty -delete
}

echo "🎨 Setting up AI Environment (Z-Image + ComfyUI)"
echo "-----------------------------------------------"

# 1. System Updates & Basic Tools
echo "🛠️ Updating system tools..."
apt-get update && apt-get install -y git wget aria2 libgl1-mesa-glx > /dev/null 2>&1

# 2. Setup Z-Image
echo "📦 Setting up Z-Image..."
if [ -d "${WORKSPACE_DIR}/repos/z-image" ]; then
    if [ -d "${WORKSPACE_DIR}/repos/z-image/.git" ]; then
        echo "   z-image already exists. Pulling latest..."
        cd "${WORKSPACE_DIR}/repos/z-image"
        git pull
    else
        echo "   z-image directory exists but is not a git repo (skipping pull)."
    fi
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
pip install -q diffusers transformers accelerate safetensors pillow

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
# Align PyTorch stack with ComfyUI/Transformers requirements
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# Install ComfyUI Manager (Highly recommended)
cd "${WORKSPACE_DIR}/ComfyUI/custom_nodes"
git clone https://github.com/ltdrdata/ComfyUI-Manager.git 2>/dev/null || echo "   ComfyUI Manager already installed"

# 4. Model Management (Shared Storage)
echo "📥 Managing Models..."
mkdir -p "$MODEL_DIR"
Z_IMAGE_MODEL_ROOT="${MODEL_DIR}/z-image"
mkdir -p "$Z_IMAGE_MODEL_ROOT"

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
        huggingface-cli login --token "$HF_TOKEN"
        huggingface-cli download stabilityai/stable-diffusion-xl-base-1.0 --include "*.safetensors" --local-dir "$MODEL_DIR" --local-dir-use-symlinks False
    fi
fi

if [ -z "$HF_TOKEN" ]; then
    echo "   ⚠️ HF_TOKEN not set. Skipping Z-Image asset downloads from HuggingFace."
else
    echo "   Downloading required Z-Image assets from HuggingFace..."
    huggingface-cli login --token "$HF_TOKEN"

    MODEL_FILE_SPECS=(
        "split_files/text_encoders/qwen_3_4b.safetensors|text_encoders/qwen_3_4b.safetensors"
        "split_files/vae/ae.safetensors|vae/ae.safetensors"
        "split_files/diffusion_models/z_image_turbo_bf16.safetensors|diffusion_models/z_image_turbo_bf16.safetensors"
    )
    OLDIFS="$IFS"
    IFS="|"
    for spec in "${MODEL_FILE_SPECS[@]}"; do
        read -r remote dest <<<"$spec"
        download_z_image_asset "$Z_IMAGE_MODEL_REPO" "$remote" "$dest"
    done
    IFS="$OLDIFS"

    download_z_image_asset "$Z_IMAGE_LORA_REPO" "pixel_art_style_z_image_turbo.safetensors" "loras/pixel_art_style_z_image_turbo.safetensors"
fi

# 5. Link Models to ComfyUI
# This creates a "shortcut" so ComfyUI sees the model without duplicating the file
echo "🔗 Linking models to ComfyUI..."
mkdir -p "${WORKSPACE_DIR}/ComfyUI/models/checkpoints"
ln -sf "$TARGET_MODEL" "${WORKSPACE_DIR}/ComfyUI/models/checkpoints/z-image-model.safetensors"
ln -sf "${Z_IMAGE_MODEL_ROOT}/diffusion_models/z_image_turbo_bf16.safetensors" "${WORKSPACE_DIR}/ComfyUI/models/checkpoints/z_image_turbo_bf16.safetensors"
mkdir -p "${WORKSPACE_DIR}/ComfyUI/models/clip"
ln -sf "${Z_IMAGE_MODEL_ROOT}/text_encoders/qwen_3_4b.safetensors" "${WORKSPACE_DIR}/ComfyUI/models/clip/qwen_3_4b.safetensors"
mkdir -p "${WORKSPACE_DIR}/ComfyUI/models/vae"
ln -sf "${Z_IMAGE_MODEL_ROOT}/vae/ae.safetensors" "${WORKSPACE_DIR}/ComfyUI/models/vae/z_image_ae.safetensors"
mkdir -p "${WORKSPACE_DIR}/ComfyUI/models/loras"
ln -sf "${Z_IMAGE_MODEL_ROOT}/loras/pixel_art_style_z_image_turbo.safetensors" "${WORKSPACE_DIR}/ComfyUI/models/loras/pixel_art_style_z_image_turbo.safetensors"

echo ""
echo "✅ Setup Complete!"
echo "-----------------------------------------------"
echo "To run Z-Image:"
echo "  cd ${WORKSPACE_DIR}/repos/z-image && python generate.py"
echo ""
echo "To run ComfyUI:"
echo "  cd ${WORKSPACE_DIR}/ComfyUI && python main.py --listen --port 8188"
echo "-----------------------------------------------"
