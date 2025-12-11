cat > ~/langchain-projects/setup_chainlit.sh << 'EOF'
#!/bin/bash

echo "🚀 Setting up ChainLit Query Interface..."

cd ~/langchain-projects

# Create Dockerfile for ChainLit
cat > Dockerfile.chainlit << 'DOCKERFILE'
FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    chainlit \
    langchain \
    langchain-community \
    langchain-ollama \
    langchain-chroma \
    chromadb

EXPOSE 8000

CMD ["chainlit", "run", "app.py", "--host", "0.0.0.0", "--port", "8000"]
DOCKERFILE

# Create basic ChainLit app
cat > chainlit_app.py << 'PYCODE'
"""
ChainLit Query Interface for Agentic RAG
"""
import chainlit as cl
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
import os

VECTOR_DB_DIR = "/app/chroma_db"

@cl.on_chat_start
async def start():
    """Initialize RAG system when chat starts"""
    
    # Check if vector DB exists
    if not os.path.exists(VECTOR_DB_DIR):
        await cl.Message(
            content="❌ Vector database not found! Please run ingestion first."
        ).send()
        return
    
    await cl.Message(content="🔄 Initializing RAG system...").send()
    
    # Load embeddings
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url="http://localhost:11434"
    )
    
    # Load vector database
    vectorstore = Chroma(
        persist_directory=VECTOR_DB_DIR,
        embedding_function=embeddings
    )
    
    # Create retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # Initialize LLM
    llm = OllamaLLM(
        model="qwen3:1.7b",
        base_url="http://localhost:11434"
    )
    
    # Create RAG chain
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    
    # Store in user session
    cl.user_session.set("rag_chain", rag_chain)
    
    await cl.Message(content="✅ RAG system ready! Ask me anything.").send()

@cl.on_message
async def main(message: cl.Message):
    """Handle incoming messages"""
    
    # Get RAG chain from session
    rag_chain = cl.user_session.get("rag_chain")
    
    if not rag_chain:
        await cl.Message(content="❌ System not initialized").send()
        return
    
    # Show thinking message
    msg = cl.Message(content="")
    await msg.send()
    
    # Query RAG
    result = rag_chain.invoke({"query": message.content})
    
    # Update message with answer
    msg.content = result["result"]
    await msg.update()
    
    # Show sources
    if result.get("source_documents"):
        sources_text = "\n\n**📚 Sources:**\n"
        for i, doc in enumerate(result["source_documents"], 1):
            sources_text += f"\n{i}. {doc.page_content[:200]}...\n"
        
        await cl.Message(content=sources_text).send()

PYCODE

# Build image
echo "📦 Building ChainLit image..."
docker build -f Dockerfile.chainlit -t chainlit-rag .

# Stop existing container
docker stop chainlit-rag 2>/dev/null || true
docker rm chainlit-rag 2>/dev/null || true

# Run ChainLit container
echo "🐳 Starting ChainLit container..."
docker run -d \
  --name chainlit-rag \
  --network host \
  -v ~/langchain-projects:/app \
  -v ~/langchain-projects/chroma_db:/app/chroma_db \
  -p 8000:8000 \
  chainlit-rag

sleep 3

if docker ps | grep -q chainlit-rag; then
    echo ""
    echo "=" * 60
    echo "✅ ChainLit Query Interface is running!"
    echo "=" * 60
    echo ""
    echo "🌐 Access at: http://YOUR_SERVER_IP:8000"
    echo ""
    echo "📝 Next: Run ingestion to build knowledge base"
    echo "   (We'll create that container next)"
    echo ""
else
    echo "❌ Failed to start"
    docker logs chainlit-rag
fi

EOF

chmod +x ~/langchain-projects/setup_chainlit.sh
