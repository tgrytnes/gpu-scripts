#!/bin/bash

echo "🚀 Setting up LangChain container..."

# Create working directory
mkdir -p ~/langchain-projects
cd ~/langchain-projects

# Create Dockerfile
cat > Dockerfile << 'EOF'
FROM python:3.11-slim

WORKDIR /app

# Install LangChain and dependencies
RUN pip install --no-cache-dir \
    langchain \
    langchain-community \
    langchain-ollama \
    requests \
    python-dotenv

# Keep container running
CMD ["tail", "-f", "/dev/null"]
EOF

# Build the image
echo "📦 Building Docker image..."
docker build -t langchain-dev .

# Stop and remove existing container if it exists
docker stop langchain 2>/dev/null || true
docker rm langchain 2>/dev/null || true

# Run the container (using host network for simplicity)
echo "🐳 Starting LangChain container..."
docker run -d \
  --name langchain \
  --network host \
  -v ~/langchain-projects:/app \
  -e OLLAMA_HOST=http://localhost:11434 \
  langchain-dev

# Create test script
cat > test_ollama.py << 'EOF'
from langchain_ollama import OllamaLLM

llm = OllamaLLM(
    model="qwen3:1.7b",
    base_url="http://localhost:11434"
)

response = llm.invoke("Hello! Respond in one sentence.")
print(response)
EOF

echo "✅ Setup complete!"
echo ""
echo "To test, run:"
echo "  docker exec -it langchain python /app/test_ollama.py"
echo ""
echo "To enter container:"
echo "  docker exec -it langchain bash"
