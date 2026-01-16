# GPU Workspace Scripts

Setup scripts for multi-project GPU workspace on RunPod/Vast.ai.

## Projects Supported

1. **CV Satellite** - SAM/CLIP for satellite imagery analysis
2. **LLM Showcase** - Llama/Mistral language model demos
3. **RAG Project** - OCR + semantic search RAG system
4. **Z-Image Generation** - AI image generation with Z-image
5. **vLLM Production Server** - High-performance LLM serving with OpenAI-compatible API
6. **TabbyAPI Server** - EXL2 quantized model serving (perfect for 32B models on 24GB GPUs)

## Quick Start

### TabbyAPI Server (Best for Large EXL2 Models)

Perfect for running 32B+ models on 24GB GPUs using EXL2 quantization:

```bash
# One-line setup (includes Qwen2.5-Coder-32B-Instruct 4.0bpw)
bash ~/gpu-scripts/setup_tabbyapi.sh

# Or with custom configuration
cat ~/gpu-scripts/.env.tabbyapi.example >> ~/gpu-scripts/.env
# Edit .env with your model choice
bash ~/gpu-scripts/setup_tabbyapi.sh
```

**Manage the server:**
```bash
~/gpu-scripts/tabbyapi_control.sh status     # Check server status
~/gpu-scripts/tabbyapi_control.sh logs -f    # View logs
~/gpu-scripts/tabbyapi_control.sh test       # Run test
~/gpu-scripts/tabbyapi_control.sh gpu        # Check GPU usage
~/gpu-scripts/tabbyapi_control.sh restart    # Restart server
```

**Features:**
- ⭐ EXL2 quantization support (fits 32B models on 24GB VRAM)
- 🚀 Extremely fast inference (ExLlamaV2 backend)
- ✅ OpenAI-compatible API
- ✅ Built-in model downloader
- ✅ Perfect for Qwen2.5-Coder-32B-Instruct
- ✅ Port 5000 (doesn't conflict with vLLM)

📖 **[Read TABBYAPI_QUICKSTART.md](TABBYAPI_QUICKSTART.md) for detailed guide**

### vLLM Production Server (Best for Standard Models)

Production-ready vLLM server for standard quantization formats:

```bash
# One-line setup
bash ~/gpu-scripts/setup_vllm.sh

# Or with custom configuration
cp ~/gpu-scripts/.env.example ~/gpu-scripts/.env
# Edit .env with your model choice and settings
bash ~/gpu-scripts/setup_vllm.sh
```

**Manage the server:**
```bash
~/gpu-scripts/vllm_control.sh status    # Check server status
~/gpu-scripts/vllm_control.sh logs -f   # View logs
~/gpu-scripts/vllm_control.sh test      # Run test
~/gpu-scripts/vllm_control.sh restart   # Restart server
```

**Features:**
- ✅ OpenAI-compatible API (drop-in replacement)
- ✅ High-throughput batch inference
- ✅ Automatic GPU detection and optimization
- ✅ GPTQ/AWQ/FP16 support
- ✅ Port 8000 (doesn't conflict with TabbyAPI)

📖 **[Read VLLM_QUICKSTART.md](VLLM_QUICKSTART.md) for detailed guide**

### Which One to Choose?

| Use Case | Recommendation |
|----------|----------------|
| 32B models on 24GB GPU | **TabbyAPI** with EXL2 4.0bpw |
| Coding (Qwen2.5-Coder-32B) | **TabbyAPI** ⭐ |
| High-throughput serving | **vLLM** |
| GPTQ/AWQ models | **vLLM** |
| 8B or smaller models | Either (vLLM slightly faster) |
| Maximum VRAM efficiency | **TabbyAPI** with EXL2 |

### Other Projects - On GPU Instance:
```bash
curl -s https://raw.githubusercontent.com/thomasthaddeus/gpu-scripts/main/menu.sh | bash
```

### Environment Variables

Create a `.env` file in `~/gpu-scripts/` (see `.env.example` for all options):

```bash
# vLLM Configuration
VLLM_MODEL_ID=meta-llama/Llama-3.2-3B-Instruct
VLLM_PORT=8000
VLLM_GPU_MEMORY_UTILIZATION=0.90
HF_TOKEN=hf_xxxxx  # Optional, for gated models

# Other projects
GITHUB_USERNAME=thomasthaddeus
AZURE_STORAGE_ACCOUNT=your_storage_name
AZURE_STORAGE_KEY=your_key
```

## Files

### vLLM Production Server
- `setup_vllm.sh` - Complete vLLM setup with health checks
- `vllm_control.sh` - Server management (start/stop/status/logs)
- `test_vllm.py` - API test script (auto-generated)
- `.env.example` - Configuration template

### Other Projects
- `menu.sh` - Interactive project selector
- `setup-cv.sh` - Computer vision environment setup
- `setup-llm.sh` - Ollama-based LLM setup
- `setup-rag.sh` - RAG environment setup
- `setup-z-image.sh` - Z-Image image generation setup

## vLLM API Usage

Once running, use the OpenAI-compatible API:

**Python:**
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"  # or your VLLM_API_KEY if set
)

response = client.chat.completions.create(
    model="meta-llama/Llama-3.2-3B-Instruct",
    messages=[
        {"role": "user", "content": "Hello!"}
    ]
)
print(response.choices[0].message.content)
```

**cURL:**
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.2-3B-Instruct",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

**Endpoints:**
- Health: `http://localhost:8000/health`
- Models: `http://localhost:8000/v1/models`
- Chat: `http://localhost:8000/v1/chat/completions`
- Completions: `http://localhost:8000/v1/completions`
- Docs: `http://localhost:8000/docs`

## Model Recommendations for RTX 3090/4090 (24GB)

**Small (fast, fits easily):**
- `meta-llama/Llama-3.2-3B-Instruct` (~6GB)
- `Qwen/Qwen2.5-3B-Instruct` (~6GB)

**Medium (recommended balance):**
- `meta-llama/Llama-3.1-8B-Instruct` (~16GB) ⭐
- `mistralai/Mistral-7B-Instruct-v0.3` (~14GB)
- `Qwen/Qwen2.5-7B-Instruct` (~14GB)

**Large (near limit):**
- `google/gemma-2-9b-it` (~18GB)
- Use quantized models (GPTQ/AWQ) for larger models

## Usage in RunPod Template

**On-start Script for vLLM:**
```bash
#!/bin/bash
cd /root/gpu-scripts && \
git pull && \
bash setup_vllm.sh
```

**On-start Script for Other Projects:**
```bash
apt-get update && apt-get install -y git curl && \
curl -sL https://aka.ms/InstallAzureCLIDeb | bash && \
pip install transformers accelerate azure-storage-blob jupyterlab huggingface-hub && \
curl -s https://raw.githubusercontent.com/${GITHUB_USERNAME}/gpu-scripts/main/menu.sh | bash
```

## License

MIT
