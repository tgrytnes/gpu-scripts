#!/bin/bash
set -e

echo "📚 RAG Project Setup"
echo ""

cd /workspace/repos/rag-project 2>/dev/null || {
    echo "❌ rag-project repo not found!"
    echo "   Creating placeholder directory..."
    mkdir -p /workspace/repos/rag-project
    cd /workspace/repos/rag-project
}

echo "📦 Installing RAG dependencies..."
pip install -q sentence-transformers chromadb fastapi uvicorn langchain

echo ""
echo "✅ RAG environment ready!"
echo ""
echo "Example usage:"
echo "  cd /workspace/repos/rag-project"
echo "  # Run your RAG application"
echo ""
/bin/bash