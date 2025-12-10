#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status

# --- Configuration ---
WORKSPACE_DIR="/workspace"
# You can leave this placeholder or change it if you have a real custom repo
Z_IMAGE_REPO="https://github.com/Comfy-Org/z-image.git" 
COMFY_REPO="https://github.com/comfyanonymous/ComfyUI.git"
MANAGER_REPO="https://github.com/ltdrdata/ComfyUI-Manager.git"

# Define local storage paths
MODEL_DIR="${WORKSPACE_DIR}/models"
Z_IMAGE_STORE="${MODEL_DIR}/z-image"
COMFY_MODEL_DIR="${WORKSPACE_DIR}/ComfyUI/models"

# Ensure HF_TOKEN is grabbed from environment if present (for private repos only)
if [ -z "$HF_TOKEN" ] && [ -r /proc/1/environ ]; then
    PID_HF_TOKEN=$(tr '\0' '\n' < /proc/1/environ | grep '^HF_Token=' | head -n 1 | cut -d '=' -f2-)
    [ ! -z "$PID_HF_TOKEN" ] && export HF_TOKEN="$PID_HF_TOKEN"
fi

# Helper function to download files reliably using aria2c (fast & resumable)
download_file() {
    local url="$1"
    local dest_dir="$2"
    local filename="$3"
    local dest_path="${dest_dir}/${filename}"

    if [ -f "$dest_path" ]; then
        echo "  ✓ ${filename} already exists."
        return
    fi

    echo "  ⬇️ Downloading ${filename}..."
    mkdir -p "$dest_dir"
    
    # Use HF_TOKEN header only if the token exists
    if [ ! -z "$HF_TOKEN" ]; then
        aria2c -x 16 -s 16 -k 1M --header="Authorization: Bearer $HF_TOKEN" -d "$dest_dir" -o "$filename" "$url"
    else
        aria2c -x 16 -s 16 -k 1M -d "$dest_dir" -o "$filename" "$url"
    fi
}

echo "🎨 Setting up AI Environment (Z-Image Turbo + ComfyUI)"
echo "-----------------------------------------------------"

# 1. System Updates & Basic Tools
echo "🛠️ Updating system tools..."
apt-get update && apt-get install -y git wget aria2 libgl1-mesa-glx > /dev/null 2>&1

# 2. Setup Z-Image (The Code Repository)
echo "📦 Setting up Z-Image Code..."
mkdir -p "${WORKSPACE_DIR}/repos"
if [ -d "${WORKSPACE_DIR}/repos/z-image" ]; then
    echo "  ✓ z-image repo exists."
else
    # Cloning the official Comfy-Org repo as a safe default if yours isn't set
    if [[ "$Z_IMAGE_REPO" == *"YOUR_USERNAME"* ]]; then
        echo "  ⚠️ No custom Z-Image repo set. Cloning official Comfy-Org repo..."
        git clone "https://github.com/Comfy-Org/z-image.git" "${WORKSPACE_DIR}/repos/z-image"
    else
        git clone "$Z_IMAGE_REPO" "${WORKSPACE_DIR}/repos/z-image"
    fi
fi

# Install Z-Image pip dependencies
echo "  Installing Z-Image pip dependencies..."
pip install -q diffusers transformers accelerate safetensors pillow sentencepiece protobuf

# 3. Setup ComfyUI
echo "🛋️ Setting up ComfyUI..."
if [ ! -d "${WORKSPACE_DIR}/ComfyUI" ]; then
    cd "${WORKSPACE_DIR}"
    git clone "$COMFY_REPO"
fi

echo "  Installing ComfyUI requirements..."
cd "${WORKSPACE_DIR}/ComfyUI"
pip install -r requirements.txt
# Install ComfyUI Manager
if [ ! -d "${WORKSPACE_DIR}/ComfyUI/custom_nodes/ComfyUI-Manager" ]; then
    cd "${WORKSPACE_DIR}/ComfyUI/custom_nodes"
    git clone "$MANAGER_REPO"
fi

# 4. Download Models (The Heavy Lifting)
echo "📥 Downloading Z-Image Models..."
mkdir -p "$Z_IMAGE_STORE"

# --- Define URLs ---
# These are the direct download links for the files you requested
URL_VAE="https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/vae/ae.safetensors"
URL_DIFFUSION="https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/diffusion_models/z_image_turbo_bf16.safetensors"
URL_TEXT_ENC="https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors"
URL_LORA="https://huggingface.co/tarn59/pixel_art_style_lora_z_image_turbo/resolve/main/pixel_art_style_z_image_turbo.safetensors"

# --- Execute Downloads ---
# We store them in a central "z-image" folder to keep things clean
download_file "$URL_VAE"       "${Z_IMAGE_STORE}/vae"              "ae.safetensors"
download_file "$URL_DIFFUSION" "${Z_IMAGE_STORE}/diffusion_models" "z_image_turbo_bf16.safetensors"
download_file "$URL_TEXT_ENC"  "${Z_IMAGE_STORE}/text_encoders"    "qwen_3_4b.safetensors"
download_file "$URL_LORA"      "${Z_IMAGE_STORE}/loras"            "pixel_art_style_z_image_turbo.safetensors"

# 5. Link Models to ComfyUI
echo "🔗 Linking models to ComfyUI..."

# Helper to link files
link_model() {
    local src="$1"
    local dest_folder="$2"
    local dest_filename="$3"
    mkdir -p "$dest_folder"
    ln -sf "$src" "${dest_folder}/${dest_filename}"
}

# Link VAE
link_model "${Z_IMAGE_STORE}/vae/ae.safetensors" \
           "${COMFY_MODEL_DIR}/vae" "z_image_ae.safetensors"

# Link Diffusion Model (IMPORTANT: Goes to diffusion_models, NOT checkpoints)
link_model "${Z_IMAGE_STORE}/diffusion_models/z_image_turbo_bf16.safetensors" \
           "${COMFY_MODEL_DIR}/diffusion_models" "z_image_turbo_bf16.safetensors"

# Link Text Encoder
link_model "${Z_IMAGE_STORE}/text_encoders/qwen_3_4b.safetensors" \
           "${COMFY_MODEL_DIR}/text_encoders" "qwen_3_4b.safetensors"

# Link LoRA
link_model "${Z_IMAGE_STORE}/loras/pixel_art_style_z_image_turbo.safetensors" \
           "${COMFY_MODEL_DIR}/loras" "pixel_art_style_z_image_turbo.safetensors"

echo ""
echo "✅ Setup Complete!"
echo "-----------------------------------------------"
echo "To run ComfyUI:"
echo "  cd ${WORKSPACE_DIR}/ComfyUI && python main.py --listen --port 8188"
echo "-----------------------------------------------"
