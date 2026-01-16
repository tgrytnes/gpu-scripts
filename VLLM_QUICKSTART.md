# vLLM Quick Start Guide

Production-ready vLLM setup for RunPod GPU instances with OpenAI-compatible API.

## 🚀 One-Line Setup

```bash
bash ~/gpu-scripts/setup_vllm.sh
```

This will:
- ✅ Install Python 3.11 and vLLM
- ✅ Configure GPU optimizations
- ✅ Start OpenAI-compatible API server
- ✅ Run health checks
- ✅ Create management and test scripts

## ⚙️ Configuration

### Quick Start (uses defaults)
Just run the setup script - it will use sensible defaults:
- Model: `meta-llama/Llama-3.2-3B-Instruct`
- Port: `8000`
- GPU Memory: `90%`

### Custom Configuration
1. Copy the example config:
   ```bash
   cp ~/gpu-scripts/.env.example ~/gpu-scripts/.env
   ```

2. Edit `~/gpu-scripts/.env`:
   ```bash
   VLLM_MODEL_ID=meta-llama/Llama-3.1-8B-Instruct
   VLLM_PORT=8000
   VLLM_GPU_MEMORY_UTILIZATION=0.90
   HF_TOKEN=hf_your_token_here  # Optional
   ```

3. Run setup:
   ```bash
   bash ~/gpu-scripts/setup_vllm.sh
   ```

## 🎮 Server Management

```bash
# Check server status
~/gpu-scripts/vllm_control.sh status

# View logs (live)
~/gpu-scripts/vllm_control.sh logs -f

# Test the API
~/gpu-scripts/vllm_control.sh test

# Restart server
~/gpu-scripts/vllm_control.sh restart

# Stop server
~/gpu-scripts/vllm_control.sh stop

# Start server
~/gpu-scripts/vllm_control.sh start
```

## 📡 API Endpoints

Once running (default port 8000):

- **Health Check**: `http://localhost:8000/health`
- **List Models**: `http://localhost:8000/v1/models`
- **Chat Completions**: `http://localhost:8000/v1/chat/completions`
- **Text Completions**: `http://localhost:8000/v1/completions`
- **API Docs**: `http://localhost:8000/docs`

## 💻 Usage Examples

### Python (OpenAI SDK)

```python
from openai import OpenAI

# Connect to vLLM server
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"  # required but can be anything if auth disabled
)

# Chat completion
response = client.chat.completions.create(
    model="meta-llama/Llama-3.2-3B-Instruct",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain quantum computing in simple terms."}
    ],
    max_tokens=200,
    temperature=0.7
)

print(response.choices[0].message.content)
```

### cURL

```bash
# Chat completion
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.2-3B-Instruct",
    "messages": [
      {"role": "user", "content": "What is vLLM?"}
    ],
    "max_tokens": 100
  }'

# List available models
curl http://localhost:8000/v1/models

# Health check
curl http://localhost:8000/health
```

### JavaScript/TypeScript

```typescript
import OpenAI from 'openai';

const client = new OpenAI({
  baseURL: 'http://localhost:8000/v1',
  apiKey: 'dummy'
});

const response = await client.chat.completions.create({
  model: 'meta-llama/Llama-3.2-3B-Instruct',
  messages: [
    { role: 'user', content: 'Hello!' }
  ]
});

console.log(response.choices[0].message.content);
```

## 🎯 Model Selection Guide

For **RTX 3090/4090 (24GB VRAM)**:

### Small Models (Fast, ~6GB)
- `meta-llama/Llama-3.2-3B-Instruct` - Great for quick responses
- `Qwen/Qwen2.5-3B-Instruct` - Excellent multilingual

### Medium Models (Recommended, ~14-16GB)
- `meta-llama/Llama-3.1-8B-Instruct` ⭐ **Best balance**
- `mistralai/Mistral-7B-Instruct-v0.3` - Very capable
- `Qwen/Qwen2.5-7B-Instruct` - Strong reasoning

### Large Models (~18-20GB, tight fit)
- `google/gemma-2-9b-it` - Near VRAM limit
- Consider quantized versions (GPTQ/AWQ) for 13B+ models

### Change Model
Edit `~/gpu-scripts/.env` and restart:
```bash
VLLM_MODEL_ID=meta-llama/Llama-3.1-8B-Instruct
```
```bash
~/gpu-scripts/vllm_control.sh restart
```

## 🔒 Security (Optional)

### Enable API Key Authentication

1. Add to `.env`:
   ```bash
   VLLM_API_KEY=your-secret-key-here
   ```

2. Restart server:
   ```bash
   ~/gpu-scripts/vllm_control.sh restart
   ```

3. Use in requests:
   ```python
   client = OpenAI(
       base_url="http://localhost:8000/v1",
       api_key="your-secret-key-here"
   )
   ```

## 🐛 Troubleshooting

### Check if server is running
```bash
~/gpu-scripts/vllm_control.sh status
```

### View logs
```bash
~/gpu-scripts/vllm_control.sh logs -f
```

### Server won't start
1. Check GPU availability: `nvidia-smi`
2. Check logs: `tail -f ~/gpu-scripts/logs/vllm.log`
3. Verify Python: `python3.11 --version`
4. Check port: `lsof -i :8000`

### Out of memory errors
Reduce GPU memory utilization in `.env`:
```bash
VLLM_GPU_MEMORY_UTILIZATION=0.80
VLLM_MAX_MODEL_LEN=4096  # Reduce context window
```

### Model download issues
Set HuggingFace token for gated models:
```bash
HF_TOKEN=hf_your_token_here
```

### Port already in use
Change port in `.env`:
```bash
VLLM_PORT=8001
```

## 📊 Performance Tips

1. **GPU Memory**: Set to 0.90 for single model, 0.80 if running other GPU tasks
2. **Quantization**: Use GPTQ/AWQ models for larger models on 24GB GPU
3. **Context Length**: Reduce `MAX_MODEL_LEN` if memory constrained
4. **Batch Size**: vLLM automatically optimizes batching
5. **Data Type**: Use `bfloat16` for newer GPUs (Ampere+)

## 🔗 Resources

- **vLLM Documentation**: https://docs.vllm.ai
- **Model Hub**: https://huggingface.co/models
- **OpenAI API Reference**: https://platform.openai.com/docs/api-reference

## 📝 File Locations

- Setup script: `~/gpu-scripts/setup_vllm.sh`
- Control script: `~/gpu-scripts/vllm_control.sh`
- Config file: `~/gpu-scripts/.env`
- Logs: `~/gpu-scripts/logs/vllm.log`
- PID file: `~/gpu-scripts/vllm.pid`
- Test script: `~/gpu-scripts/test_vllm.py`

## 🆘 Getting Help

1. Check status: `~/gpu-scripts/vllm_control.sh status`
2. View logs: `~/gpu-scripts/vllm_control.sh logs -f`
3. Run test: `~/gpu-scripts/vllm_control.sh test`
4. Check GPU: `nvidia-smi`

---

**Quick Commands Summary:**
```bash
# Setup
bash ~/gpu-scripts/setup_vllm.sh

# Control
~/gpu-scripts/vllm_control.sh [start|stop|restart|status|logs|test]

# Test
~/gpu-scripts/vllm_control.sh test
curl http://localhost:8000/health
```
