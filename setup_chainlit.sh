#!/bin/bash

echo "Setting up ChainLit Query Interface..."

# Create Dockerfile for ChainLit
cat > Dockerfile.chainlit << 'DOCKERFILE'
FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir chromadb

RUN pip install --no-cache-dir \
    chainlit \
    langchain \
    langchain-community \
    langchain-ollama \
    langchain-chroma

EXPOSE 8000

CMD ["chainlit", "run", "chainlit_app.py", "--host", "0.0.0.0", "--port", "8000"]
DOCKERFILE

echo "Dockerfile created"

# Create ChainLit app
cat > chainlit_app.py << 'PYCODE'
import chainlit as cl
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
import os

VECTOR_DB_DIR = "/app/chroma_db"

@cl.on_chat_start
async def start():
    if not os.path.exists(VECTOR_DB_DIR):
        await cl.Message(content="Vector database not found! Run ingestion first.").send()
        return
    
    await cl.Message(content="Initializing RAG system...").send()
    
    embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")
    vectorstore = Chroma(persist_directory=VECTOR_DB_DIR, embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = OllamaLLM(model="qwen3:1.7b", base_url="http://localhost:11434")
    
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    
    cl.user_session.set("rag_chain", rag_chain)
    await cl.Message(content="RAG system ready! Ask me anything.").send()

@cl.on_message
async def main(message: cl.Message):
    rag_chain = cl.user_session.get("rag_chain")
    if not rag_chain:
        await cl.Message(content="System not initialized").send()
        return
    
    msg = cl.Message(content="")
    await msg.send()
    
    result = rag_chain.invoke({"query": message.content})
    msg.content = result["result"]
    await msg.update()
    
    if result.get("source_documents"):
        sources = "\n\nSources:\n"
        for i, doc in enumerate(result["source_documents"], 1):
            sources += f"{i}. {doc.page_content[:200]}...\n"
        await cl.Message(content=sources).send()
PYCODE

echo "App created"

# Build
echo "Building image..."
docker build -f Dockerfile.chainlit -t chainlit-rag .

# Run
echo "Starting container..."
docker stop chainlit-rag 2>/dev/null || true
docker rm chainlit-rag 2>/dev/null || true

docker run -d \
  --name chainlit-rag \
  --network host \
  -v $(pwd):/app \
  -p 8000:8000 \
  chainlit-rag

sleep 2
docker ps | grep chainlit-rag

echo ""
echo "Done! Access at http://YOUR_IP:8000"
echo ""

