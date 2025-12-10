#!/bin/bash
set -e

echo "🤖 LLM Showcase Setup"
echo ""

cd /workspace/repos/llm-showcase 2>/dev/null || {
    echo "❌ llm-showcase repo not found!"
    echo "   Creating placeholder directory..."
    mkdir -p /workspace/repos/llm-showcase
    cd /workspace/repos/llm-showcase
}

# Install dependencies
echo "📦 Installing LLM dependencies..."
pip install -q transformers accelerate torch huggingface-hub bitsandbytes

# Download model from Azure or HuggingFace
if [ ! -z "$AZURE_STORAGE_ACCOUNT" ]; then
    echo "📥 Checking Azure for cached models..."
    mkdir -p /workspace/models
    
    az storage blob download \
        --account-name ${AZURE_STORAGE_ACCOUNT} \
        --account-key ${AZURE_STORAGE_KEY} \
        --container-name models \
        --name llm/mistral-7b-instruct.gguf \
        --file /workspace/models/mistral-7b-instruct.gguf 2>/dev/null && echo "  ✓ Model found in Azure" || {
        echo "  ⚠️ Model not in Azure, will download from HuggingFace on first use"
    }
fi

echo ""
echo "✅ LLM environment ready!"
echo ""
echo "Example usage:"
echo "  cd /workspace/repos/llm-showcase"
echo "  python -c 'from transformers import pipeline; pipe = pipeline(\"text-generation\", model=\"mistralai/Mistral-7B-Instruct-v0.2\"); print(pipe(\"Hello\"))'"
echo ""
/bin/bash