#!/bin/bash
set -e # Exit immediately if errors occur

# --- Configuration ---
WORKSPACE_DIR="/workspace"

# ✅ FIX 1: Use the correct Official Z-Image Repo URL
Z_IMAGE_REPO="https://github.com/Tongyi-MAI/Z-Image.git" 

COMFY_REPO="https://github.com/comfyanonymous/ComfyUI.git"
MANAGER_REPO="https://github.com/ltdrdata/ComfyUI-Manager.git"

# Define Paths
MODEL_STORAGE="${WORKSPACE_DIR}/models/z-image"
COMFY_BASE="${WORKSPACE_DIR}/ComfyUI"

# --- Fix VS Code Git Errors ---
# ✅ FIX 2: Stop git from trying to use VS Code authentication socket
unset GIT_ASKPASS
unset SSH_ASKPASS
git config --global --unset credential.helper || true
git config --global credential.helper store

# Ensure HF_TOKEN is grabbed from environment if present (for private repos)
if [ -z "$HF_TOKEN" ] && [ -r /proc/1/environ ]; then
    PID_HF_TOKEN=$(tr '\0' '\n' < /proc/1/environ | grep '^HF_Token=' | head -n 1 | cut -d '=' -f2-)
    [ ! -z "$PID_HF_TOKEN" ] && export HF_TOKEN="$PID_HF_TOKEN"
fi

# Download Helper (Uses aria2c for speed and reliability)
download_file() {
    local url="$1"
    local dest_dir="$2"
    local filename="$3"
    local dest_path="${dest_dir}/${filename}"

    if [ -f "$dest_path" ]; then
        echo "  ✓ ${filename} already exists in storage."
        return
    fi

    echo "  ⬇️ Downloading ${filename}..."
    mkdir -p "$dest_dir"
    
    if [ ! -z "$HF_TOKEN" ]; then
        aria2c -x 16 -s 16 -k 1M --header="Authorization: Bearer $HF_TOKEN" -d "$dest_dir" -o "$filename" "$url"
    else
        aria2c -x 16 -s 16 -k 1M -d "$dest_dir" -o "$filename" "$url"
    fi
}

echo "🎨 Setting up Z-Image Turbo Environment"
echo "---------------------------------------"

# 1. Install System Tools
echo "🛠️ Updating system tools..."
apt-get update && apt-get install -y git wget aria2 libgl1-mesa-glx > /dev/null 2>&1

# 1.5 ✅ FIX 3: Upgrade PyTorch to fix the 'register_pytree_node' error
# We need to ensure Torch is new enough to handle the latest Transformers library
echo "🔄 Upgrading PyTorch to match new Transformers requirements..."
pip install --upgrade torch torchvision torchaudio xformers --index-url https://download.pytorch.org/whl/cu121

# 2. Setup ComfyUI (if missing)
echo "🛋️ Setting up ComfyUI..."
if [ ! -d "$COMFY_BASE" ]; then
    git clone "$COMFY_REPO" "$COMFY_BASE"
fi
# Install Manager
if [ ! -d "${COMFY_BASE}/custom_nodes/ComfyUI-Manager" ]; then
    cd "${COMFY_BASE}/custom_nodes" && git clone "$MANAGER_REPO"
fi

# 3. Setup Z-Image Repo
echo "📦 Cloning Z-Image Repository..."
mkdir -p "${WORKSPACE_DIR}/repos"

if [ ! -d "${WORKSPACE_DIR}/repos/z-image" ]; then
    git clone "$Z_IMAGE_REPO" "${WORKSPACE_DIR}/repos/z-image"
fi

# Install dependencies (This installs the new Transformers that caused the error, but we fixed Torch above)
echo "🐍 Installing Python Requirements..."
pip install -q diffusers transformers accelerate safetensors pillow sentencepiece protobuf
cd "$COMFY_BASE" && pip install -r requirements.txt

# 4. DOWNLOAD MODELS
echo "📥 Downloading Models..."

# URLs
URL_VAE="https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/vae/ae.safetensors"
URL_DIFF="https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/diffusion_models/z_image_turbo_bf16.safetensors"
URL_TEXT="https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors"
# Added Pixel Art LoRA URL
URL_LORA_PIXEL="https://huggingface.co/tarn59/pixel_art_style_lora_z_image_turbo/resolve/main/pixel_art_style_z_image_turbo.safetensors"

download_file "$URL_VAE"        "${MODEL_STORAGE}/vae"              "ae.safetensors"
download_file "$URL_DIFF"       "${MODEL_STORAGE}/diffusion_models" "z_image_turbo_bf16.safetensors"
download_file "$URL_TEXT"       "${MODEL_STORAGE}/text_encoders"    "qwen_3_4b.safetensors"
# Download Pixel Art LoRA
download_file "$URL_LORA_PIXEL" "${MODEL_STORAGE}/loras"            "pixel_art_style_z_image_turbo.safetensors"

# 5. LINK MODELS
echo "🔗 Linking models to ComfyUI paths..."

link_to_comfy() {
    local src_file="$1"
    local comfy_subfolder="$2"
    local target_name="$3"

    local target_dir="${COMFY_BASE}/models/${comfy_subfolder}"
    mkdir -p "$target_dir"
    echo "  Linking: models/${comfy_subfolder}/${target_name}"
    ln -sf "$src_file" "${target_dir}/${target_name}"
}

link_to_comfy "${MODEL_STORAGE}/vae/ae.safetensors" "vae" "ae.safetensors"
link_to_comfy "${MODEL_STORAGE}/diffusion_models/z_image_turbo_bf16.safetensors" "diffusion_models" "z_image_turbo_bf16.safetensors"
link_to_comfy "${MODEL_STORAGE}/text_encoders/qwen_3_4b.safetensors" "text_encoders" "qwen_3_4b.safetensors"
# Link Pixel Art LoRA
link_to_comfy "${MODEL_STORAGE}/loras/pixel_art_style_z_image_turbo.safetensors" "loras" "pixel_art_style_z_image_turbo.safetensors"

echo "---------------------------------------"
echo "✅ Setup Complete!"
echo ""
echo "🚀 TO START COMFYUI:"
echo "   cd /workspace/ComfyUI"
echo "   python main.py --listen 0.0.0.0"
echo ""
echo "Then open the URL provided by your cloud provider (port 8188)."
echo "---------------------------------------"
