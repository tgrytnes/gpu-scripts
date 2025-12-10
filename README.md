# GPU Workspace Scripts

Setup scripts for multi-project GPU workspace on RunPod/Vast.ai.

## Projects Supported

1. **CV Satellite** - SAM/CLIP for satellite imagery analysis
2. **LLM Showcase** - Llama/Mistral language model demos
3. **RAG Project** - OCR + semantic search RAG system
4. **Z-Image Generation** - AI image generation with Z-image

## Quick Start

### On GPU Instance:
```bash
curl -s https://raw.githubusercontent.com/thomasthaddeus/gpu-scripts/main/menu.sh | bash
```

### Environment Variables Required:
```bash
GITHUB_USERNAME=thomasthaddeus
AZURE_STORAGE_ACCOUNT=your_storage_name
AZURE_STORAGE_KEY=your_key
HF_TOKEN=hf_xxxxx  # Optional, for gated models
```

## Files

- `menu.sh` - Interactive project selector
- `setup-cv.sh` - Computer vision environment setup
- `setup-llm.sh` - LLM environment setup
- `setup-rag.sh` - RAG environment setup
- `setup-z-image.sh` - Z-Image image generation setup

## Usage in RunPod Template

**On-start Script:**
```bash
apt-get update && apt-get install -y git curl && \
curl -sL https://aka.ms/InstallAzureCLIDeb | bash && \
pip install transformers accelerate azure-storage-blob jupyterlab huggingface-hub && \
curl -s https://raw.githubusercontent.com/${GITHUB_USERNAME}/gpu-scripts/main/menu.sh | bash
```

## License

MIT
