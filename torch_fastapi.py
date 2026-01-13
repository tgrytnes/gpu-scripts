import os
import gc
import json
import time
import uuid
import logging
import asyncio
import torch
import numpy as np
from contextlib import asynccontextmanager
from typing import List, Optional, Union, Tuple, Any
from threading import Thread

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import (
    AutoModel, 
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TextIteratorStreamer, 
    BitsAndBytesConfig
)

# --- CONFIG ---
HF_TOKEN = os.getenv("HF_TOKEN")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODE_CONFIG = {
    "speedster": "Qwen/Qwen2.5-3B-Instruct", 
    "daily": "Qwen/Qwen2.5-7B-Instruct",     
    "smart": "Qwen/Qwen2.5-14B-Instruct",
    "heavy": "Qwen/Qwen2.5-32B-Instruct",
    "deepseek": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
}
DEFAULT_MODE = "daily"
EMBEDDING_MODEL_ID = "jinaai/jina-embeddings-v3"

# --- GLOBALS ---
llm_model = None
llm_tokenizer = None
current_llm_mode = None
embedding_model = None
model_lock = asyncio.Lock()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("torch_server")

# --- HELPERS ---
def cleanup_memory():
    global llm_model, embedding_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def load_llm(mode_name: str):
    global llm_model, llm_tokenizer, current_llm_mode
    if mode_name not in MODE_CONFIG:
        # 400 Bad Request if model doesn't exist
        raise HTTPException(400, f"Unknown mode. Available: {list(MODE_CONFIG.keys())}")
    
    model_id = MODE_CONFIG[mode_name]
    
    # Do nothing if already loaded
    if llm_model is not None and current_llm_mode == mode_name:
        return

    # Unload previous model if exists
    if llm_model is not None:
        logger.info("Unloading previous model...")
        del llm_model
        del llm_tokenizer
        cleanup_memory()

    logger.info(f"Loading LLM: {model_id} (4-bit)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    try:
        llm_tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN, trust_remote_code=True)
        llm_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            token=HF_TOKEN
        )
        current_llm_mode = mode_name
        logger.info("✅ LLM Loaded.")
    except Exception as e:
        logger.error(f"Failed to load LLM: {e}")
        raise HTTPException(500, f"Model load failed: {str(e)}")

def load_embedding_model():
    global embedding_model
    if embedding_model is not None:
        return

    logger.info(f"Loading Embedding Model: {EMBEDDING_MODEL_ID}...")
    try:
        embedding_model = AutoModel.from_pretrained(
            EMBEDDING_MODEL_ID, 
            trust_remote_code=True, 
            device_map="auto"
        )
        embedding_model.tokenizer = AutoTokenizer.from_pretrained(
            EMBEDDING_MODEL_ID, trust_remote_code=True
        )
        logger.info("✅ Embedding Model Loaded.")
    except Exception as e:
        logger.error(f"Failed to load embedding model: {e}")
        raise HTTPException(500, f"Embedding load failed: {str(e)}")

def compute_late_chunking(text: str, spans: List[List[int]], task: str = "retrieval.passage"):
    if embedding_model is None:
        load_embedding_model()
    
    tokenizer = embedding_model.tokenizer
    inputs = tokenizer(text, return_tensors="pt", return_offsets_mapping=True, truncation=True, max_length=8192).to(DEVICE)
    input_ids = inputs["input_ids"]
    offset_mapping = inputs["offset_mapping"][0].cpu().numpy()
    
    with torch.no_grad():
        outputs = embedding_model(input_ids, task=task)
        token_embeddings = outputs.last_hidden_state[0]

    chunk_vectors = []
    for start_char, end_char in spans:
        token_mask = (offset_mapping[:, 0] >= start_char) & (offset_mapping[:, 1] <= end_char)
        indices = np.where(token_mask)[0]
        
        if len(indices) == 0:
            pooled = torch.zeros(token_embeddings.shape[1], device=DEVICE)
        else:
            slice_emb = token_embeddings[indices[0]:indices[-1]+1]
            pooled = torch.mean(slice_emb, dim=0)
            norm = torch.norm(pooled)
            if norm > 0: pooled = pooled / norm
        chunk_vectors.append(pooled.cpu().tolist())

    return chunk_vectors

# --- API MODELS ---
class LoadModelRequest(BaseModel):
    model: str  # e.g., "speedster", "daily", "smart", "deepseek"

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    model: Optional[str] = DEFAULT_MODE
    max_tokens: int = 2048
    temperature: float = 0.7
    stream: bool = True

class EmbeddingOptions(BaseModel):
    late_chunking: bool = False
    chunk_spans: Optional[List[List[int]]] = None
    adapter: str = "retrieval.passage"

class EmbeddingRequest(BaseModel):
    prompt: str
    options: Optional[EmbeddingOptions] = None

# --- APP ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Server Starting on {DEVICE}...")
    yield
    cleanup_memory()
    logger.info("Server Shutdown.")

app = FastAPI(lifespan=lifespan)

@app.post("/load")
async def manual_load_endpoint(req: LoadModelRequest):
    """
    Manually pre-load a model into GPU memory.
    """
    async with model_lock:
        try:
            load_llm(req.model)
            return {
                "status": "ok", 
                "message": f"Loaded model {req.model}", 
                "model_id": MODE_CONFIG[req.model]
            }
        except HTTPException as he:
            raise he
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/chat/completions")
async def chat_endpoint(req: ChatRequest):
    async with model_lock:
        target_mode = req.model if req.model in MODE_CONFIG else DEFAULT_MODE
        # Load logic handles skipping if already loaded
        load_llm(target_mode)
        
        prompt = llm_tokenizer.apply_chat_template([m.model_dump() for m in req.messages], tokenize=False, add_generation_prompt=True)
        inputs = llm_tokenizer(prompt, return_tensors="pt").to(DEVICE)
        gen_kwargs = dict(inputs, max_new_tokens=req.max_tokens, temperature=req.temperature, do_sample=True, top_p=0.9)

        if req.stream:
            streamer = TextIteratorStreamer(llm_tokenizer, skip_prompt=True, skip_special_tokens=True)
            gen_kwargs["streamer"] = streamer
            Thread(target=llm_model.generate, kwargs=gen_kwargs).start()

            async def stream_generator():
                req_id = f"chatcmpl-{uuid.uuid4().hex}"
                created = int(time.time())
                for text in streamer:
                    chunk = {"id": req_id, "object": "chat.completion.chunk", "created": created, "model": target_mode, "choices": [{"index": 0, "delta": {"content": text}}]}
                    yield f"data: {json.dumps(chunk)}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(stream_generator(), media_type="text/event-stream")
        else:
            outputs = llm_model.generate(**gen_kwargs)
            text = llm_tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            return {"id": f"chatcmpl-{uuid.uuid4().hex}", "object": "chat.completion", "model": target_mode, "choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}]}

@app.post("/api/embeddings")
async def embeddings_endpoint(req: EmbeddingRequest):
    async with model_lock:
        load_embedding_model()
        opt = req.options or EmbeddingOptions()
        if opt.late_chunking and opt.chunk_spans:
            vectors = compute_late_chunking(req.prompt, opt.chunk_spans, opt.adapter)
            return {"model": EMBEDDING_MODEL_ID, "embeddings": vectors, "late_chunking": True}
        else:
            with torch.no_grad():
                encoded = embedding_model.encode([req.prompt], task=opt.adapter)
                if isinstance(encoded, torch.Tensor): encoded = encoded.cpu().tolist()
                elif isinstance(encoded, np.ndarray): encoded = encoded.tolist()
            return {"model": EMBEDDING_MODEL_ID, "embedding": encoded[0]}

@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE, "current_llm": current_llm_mode}
