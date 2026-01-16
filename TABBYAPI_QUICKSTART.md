# TabbyAPI Quick Start Guide

Production-ready TabbyAPI setup for RunPod with EXL2 quantized models. Perfect for running large models like Qwen2.5-Coder-32B on 24GB GPUs!

## 🚀 One-Line Setup

```bash
bash ~/gpu-scripts/setup_tabbyapi.sh
```

This will:
- ✅ Install Python 3.11 and TabbyAPI
- ✅ Clone TabbyAPI from official repo
- ✅ Configure for EXL2 models
- ✅ Download Qwen2.5-Coder-32B-Instruct (4.0bpw)
- ✅ Start OpenAI-compatible API server
- ✅ Run health checks

**Note:** First run downloads ~20-25GB model, allow 10-20 minutes.

## ⚙️ Configuration

### Quick Start (uses defaults)
The setup script uses these defaults:
- Model: `bartowski/Qwen2.5-Coder-32B-Instruct-exl2` (4.0bpw)
- Port: `5000`
- Model Dir: `/workspace/models` (or `~/models`)

### Custom Configuration

1. Copy the example config:
   ```bash
   cat ~/gpu-scripts/.env.tabbyapi.example >> ~/gpu-scripts/.env
   ```

2. Edit `~/gpu-scripts/.env`:
   ```bash
   # Choose your model
   TABBY_MODEL_REPO=bartowski/Qwen2.5-Coder-32B-Instruct-exl2
   TABBY_MODEL_REVISION=4.0bpw

   # Server settings
   TABBY_PORT=5000
   TABBY_MAX_SEQ_LEN=8192

   # Optional: HuggingFace token for gated models
   HF_TOKEN=hf_your_token_here
   ```

3. Run setup:
   ```bash
   bash ~/gpu-scripts/setup_tabbyapi.sh
   ```

## 🎮 Server Management

```bash
# Check server status (shows GPU usage, endpoints)
~/gpu-scripts/tabbyapi_control.sh status

# View logs (live)
~/gpu-scripts/tabbyapi_control.sh logs -f

# Test the API
~/gpu-scripts/tabbyapi_control.sh test

# Check GPU usage
~/gpu-scripts/tabbyapi_control.sh gpu

# Restart server
~/gpu-scripts/tabbyapi_control.sh restart

# Stop server
~/gpu-scripts/tabbyapi_control.sh stop

# Download additional models
~/gpu-scripts/tabbyapi_control.sh download bartowski/Qwen2.5-32B-Instruct-exl2 4.0bpw
```

## 📡 API Endpoints

Once running (default port 5000):

- **Health Check**: `http://localhost:5000/health`
- **List Models**: `http://localhost:5000/v1/models`
- **Chat Completions**: `http://localhost:5000/v1/chat/completions`
- **Text Completions**: `http://localhost:5000/v1/completions`
- **API Docs**: `http://localhost:5000/docs`

## 💻 Usage Examples

### Python (OpenAI SDK)

```python
from openai import OpenAI

# Connect to TabbyAPI server
client = OpenAI(
    base_url="http://localhost:5000/v1",
    api_key="dummy"  # required but can be anything if auth disabled
)

# Chat completion (great for coding!)
response = client.chat.completions.create(
    model="bartowski/Qwen2.5-Coder-32B-Instruct-exl2",
    messages=[
        {"role": "system", "content": "You are an expert Python programmer."},
        {"role": "user", "content": "Write a binary search function with proper error handling."}
    ],
    max_tokens=500,
    temperature=0.7
)

print(response.choices[0].message.content)
```

### cURL

```bash
# Chat completion
curl http://localhost:5000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bartowski/Qwen2.5-Coder-32B-Instruct-exl2",
    "messages": [
      {"role": "user", "content": "Explain async/await in Python"}
    ],
    "max_tokens": 200
  }'

# List available models
curl http://localhost:5000/v1/models

# Health check
curl http://localhost:5000/health
```

### JavaScript/TypeScript

```typescript
import OpenAI from 'openai';

const client = new OpenAI({
  baseURL: 'http://localhost:5000/v1',
  apiKey: 'dummy'
});

const response = await client.chat.completions.create({
  model: 'bartowski/Qwen2.5-Coder-32B-Instruct-exl2',
  messages: [
    { role: 'user', content: 'Write a React component for a todo list' }
  ]
});

console.log(response.choices[0].message.content);
```

## 🎯 Model Selection Guide

### For RTX 3090/4090 (24GB VRAM)

#### Best for Coding (Qwen2.5-Coder)
```bash
# 32B model - Perfect fit for 24GB
TABBY_MODEL_REPO=bartowski/Qwen2.5-Coder-32B-Instruct-exl2
TABBY_MODEL_REVISION=4.0bpw  # ⭐ Recommended (~20GB)
# TABBY_MODEL_REVISION=3.0bpw  # Lower quality, more context (~16GB)
# TABBY_MODEL_REVISION=5.0bpw  # Higher quality, tight fit (~24GB)
```

#### General Purpose Models
```bash
# Qwen2.5 32B Instruct
TABBY_MODEL_REPO=bartowski/Qwen2.5-32B-Instruct-exl2
TABBY_MODEL_REVISION=4.0bpw

# DeepSeek Coder 33B
TABBY_MODEL_REPO=bartowski/deepseek-coder-33b-instruct-exl2
TABBY_MODEL_REVISION=4.0bpw
```

#### For 48GB+ VRAM (A6000, etc.)
```bash
# Llama 3.1 70B
TABBY_MODEL_REPO=turboderp/Llama-3.1-70B-Instruct-exl2
TABBY_MODEL_REVISION=4.0bpw  # ~35GB
```

### EXL2 Quantization Guide

**bits-per-weight (bpw)** controls quality vs VRAM tradeoff:

| BPW  | 32B Model VRAM | Quality | Notes |
|------|----------------|---------|-------|
| 3.0  | ~16GB          | Good    | More context possible |
| 4.0  | ~20GB          | ⭐ Best | Recommended balance |
| 5.0  | ~24GB          | Great   | Tight fit on 24GB |
| 6.0  | ~28GB          | Best    | Needs 32GB+ VRAM |

**For 24GB GPUs: 4.0bpw is the sweet spot!**

## 🔄 Switching Models

1. Edit `.env`:
   ```bash
   TABBY_MODEL_REPO=bartowski/different-model-exl2
   TABBY_MODEL_REVISION=4.0bpw
   ```

2. Download new model (optional, auto-downloads on start):
   ```bash
   ~/gpu-scripts/tabbyapi_control.sh download bartowski/different-model-exl2 4.0bpw
   ```

3. Restart:
   ```bash
   ~/gpu-scripts/tabbyapi_control.sh restart
   ```

## 🚀 Performance Comparison

### TabbyAPI (ExLlamaV2) vs vLLM

| Feature | TabbyAPI | vLLM |
|---------|----------|------|
| EXL2 Support | ✅ Yes | ❌ No |
| Speed (single) | 🚀 Very Fast | Fast |
| Batch Inference | Good | 🚀 Excellent |
| VRAM Efficiency | 🌟 Excellent | Good |
| 32B on 24GB | ✅ Yes (EXL2) | ❌ Tight fit |
| Setup | Simple | Simple |

**Use TabbyAPI when:**
- You want to run 32B+ models on 24GB VRAM
- You have EXL2 quantized models
- You prioritize VRAM efficiency
- Single-user or low-concurrency use case

**Use vLLM when:**
- High-throughput batch inference
- Multi-user concurrent requests
- Using GPTQ/AWQ/FP16 models

## 🔒 Security (Optional)

### Enable API Key Authentication

1. Add to `.env`:
   ```bash
   TABBY_API_KEY=your-secret-key-here
   ```

2. Restart server:
   ```bash
   ~/gpu-scripts/tabbyapi_control.sh restart
   ```

3. Use in requests:
   ```python
   client = OpenAI(
       base_url="http://localhost:5000/v1",
       api_key="your-secret-key-here"
   )
   ```

## 🐛 Troubleshooting

### Check server status
```bash
~/gpu-scripts/tabbyapi_control.sh status
```

### View logs (most helpful!)
```bash
~/gpu-scripts/tabbyapi_control.sh logs -f
```

### Server won't start

1. **Check GPU**: `nvidia-smi`
2. **Check logs**: `tail -f ~/gpu-scripts/logs/tabbyapi.log`
3. **Verify model downloaded**: `ls -lh /workspace/models/`
4. **Check port**: `lsof -i :5000`

### Out of memory errors

Reduce model size or context in `.env`:
```bash
TABBY_MODEL_REVISION=3.0bpw  # Lower quantization
TABBY_MAX_SEQ_LEN=4096       # Reduce context window
```

### Model loading takes forever

- Normal for first load (1-3 minutes for 32B models)
- Check logs: `~/gpu-scripts/tabbyapi_control.sh logs -f`
- Ensure SSD/NVMe storage (HDD will be slow)

### Model not found

Download manually:
```bash
~/gpu-scripts/tabbyapi_control.sh download \
  bartowski/Qwen2.5-Coder-32B-Instruct-exl2 \
  4.0bpw
```

### API not responding

1. Check if model finished loading: `~/gpu-scripts/tabbyapi_control.sh logs -f`
2. Wait 1-3 minutes for initial model load
3. Verify health: `curl http://localhost:5000/health`

## 📊 Performance Tips

1. **Use EXL2 format** - Most VRAM efficient for large models
2. **4.0bpw quantization** - Best quality/size tradeoff for 24GB
3. **NVMe/SSD storage** - Faster model loading
4. **Reduce max_seq_len** - If you don't need long context
5. **Monitor GPU**: `watch -n 1 nvidia-smi`

## 🔗 Resources

- **TabbyAPI GitHub**: https://github.com/theroyallab/tabbyAPI
- **TabbyAPI Wiki**: https://github.com/theroyallab/tabbyapi/wiki
- **ExLlamaV2**: https://github.com/turboderp/exllamav2
- **EXL2 Models (bartowski)**: https://huggingface.co/bartowski
- **EXL2 Models (turboderp)**: https://huggingface.co/turboderp

## 📝 File Locations

- Setup script: `~/gpu-scripts/setup_tabbyapi.sh`
- Control script: `~/gpu-scripts/tabbyapi_control.sh`
- Config example: `~/gpu-scripts/.env.tabbyapi.example`
- Config file: `~/gpu-scripts/.env`
- TabbyAPI dir: `~/gpu-scripts/tabbyAPI/`
- Models: `/workspace/models/` (or `~/models/`)
- Logs: `~/gpu-scripts/logs/tabbyapi.log`
- PID file: `~/gpu-scripts/tabbyapi.pid`
- Test script: `~/gpu-scripts/test_tabbyapi.py`

## 🆘 Quick Commands

```bash
# Setup
bash ~/gpu-scripts/setup_tabbyapi.sh

# Control
~/gpu-scripts/tabbyapi_control.sh [start|stop|restart|status|logs|test|gpu|download]

# Test
~/gpu-scripts/tabbyapi_control.sh test
curl http://localhost:5000/health

# Monitor
~/gpu-scripts/tabbyapi_control.sh gpu
watch -n 1 nvidia-smi
```

## 🎯 Why TabbyAPI for Qwen2.5-Coder-32B?

1. **Fits on 24GB** - EXL2 4.0bpw uses ~20GB, perfect for RTX 3090/4090
2. **Fast inference** - ExLlamaV2 backend is extremely fast
3. **Great for coding** - Qwen2.5-Coder is state-of-the-art for code
4. **OpenAI compatible** - Works with existing tools
5. **Easy to use** - Simple setup and management

Perfect for local development, coding assistance, and experimentation!

---

**Sources:**
- [TabbyAPI GitHub](https://github.com/theroyallab/tabbyAPI)
- [TabbyAPI Wiki](https://github.com/theroyallab/tabbyAPI/wiki/01.-Getting-Started)
- [ExLlamaV2](https://github.com/turboderp/exllamav2)
