import os
import gc
import json
import time
import uuid
import logging
import asyncio
import torch
import numpy as np
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from typing import List, Optional
from threading import Thread

from transformers import AutoConfig

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
    BitsAndBytesConfig,
)

try:
    from auto_gptq import AutoGPTQForCausalLM
except Exception:
    AutoGPTQForCausalLM = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

from dotenv import load_dotenv
load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODE_CONFIG = {
    "speed": "Qwen/Qwen3-4B-Instruct-2507",
    "daily_8b": "Qwen/Qwen3-8B",
    "daily_14b": "Qwen/Qwen3-14B",
    "reasoning": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    "mistral_7b": "mistralai/Mistral-7B-Instruct-v0.3",
    "llama_31_8b": "meta-llama/Llama-3.1-8B-Instruct",
    "heavy_dense": "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
    "heavy_moe_general": "Qwen/Qwen3-30B-A3B-Instruct-2507",
    "heavy_moe_coder": "Qwen/Qwen3-Coder-30B-A3B-Instruct",
    "moe_20b": "openai/gpt-oss-20b",
}



DEFAULT_MODE = os.getenv("DEFAULT_MODE", "smart")

EMBEDDING_MODELS = {
    "jina": "jinaai/jina-embeddings-v3",
    "qwen4b": "Qwen/Qwen3-Embedding-4B",
    "bgem3": "BAAI/bge-m3",
}
EMBEDDING_MODEL_ID = os.getenv("EMBEDDING_MODEL_ID", "jinaai/jina-embeddings-v3")

llm_model = None
llm_tokenizer = None
current_llm_mode = None
embedding_model = None
embedding_backend = None
model_lock = asyncio.Lock()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("torch_server")

def cleanup_memory():
    global llm_model, embedding_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def _resolve_embedding_model(model_id: Optional[str]) -> str:
    if model_id in EMBEDDING_MODELS:
        return EMBEDDING_MODELS[model_id]
    return model_id or EMBEDDING_MODEL_ID

def load_llm(mode_name: str):
    global llm_model, llm_tokenizer, current_llm_mode
    if mode_name not in MODE_CONFIG:
        raise HTTPException(400, f"Unknown mode. Available: {list(MODE_CONFIG.keys())}")

    model_id = MODE_CONFIG[mode_name]

    if llm_model is not None and current_llm_mode == mode_name:
        return

    if llm_model is not None:
        del llm_model
        del llm_tokenizer
        cleanup_memory()

    # GPTQ branch (keep this)
    if "gptq" in model_id.lower():
        if AutoGPTQForCausalLM is None:
            raise HTTPException(500, "AutoGPTQ not available. Install auto-gptq.")
        llm_tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN, trust_remote_code=True)
        llm_model = AutoGPTQForCausalLM.from_quantized(
            model_id,
            device="cuda:0",
            trust_remote_code=True,
            use_safetensors=True,
            revision=os.getenv("GPTQ_REVISION") or None,
        )
        current_llm_mode = mode_name
        return

    # NEW: respect model’s built-in quantization (e.g., MXFP4)
    config = AutoConfig.from_pretrained(model_id, token=HF_TOKEN, trust_remote_code=True)
    if getattr(config, "quantization_config", None):
        llm_tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN, trust_remote_code=True)
        llm_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            trust_remote_code=True,
            token=HF_TOKEN,
            torch_dtype="auto",
        )
        current_llm_mode = mode_name
        return

    # Default: BitsAndBytes
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    llm_tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN, trust_remote_code=True)
    llm_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        token=HF_TOKEN,
    )
    current_llm_mode = mode_name


def load_embedding_model(model_id: Optional[str] = None):
    global embedding_model, EMBEDDING_MODEL_ID, embedding_backend
    resolved = _resolve_embedding_model(model_id)
    if resolved != EMBEDDING_MODEL_ID:
        EMBEDDING_MODEL_ID = resolved
        embedding_model = None
        embedding_backend = None

    if embedding_model is not None:
        return

    if "bge-m3" in EMBEDDING_MODEL_ID.lower() and SentenceTransformer:
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_ID, device=DEVICE)
        embedding_backend = "sentence_transformers"
        return

    embedding_model = AutoModel.from_pretrained(
        EMBEDDING_MODEL_ID, trust_remote_code=True, device_map="auto"
    )
    embedding_model.tokenizer = AutoTokenizer.from_pretrained(
        EMBEDDING_MODEL_ID, trust_remote_code=True
    )
    embedding_backend = "transformers"

def compute_late_chunking(text: str, spans: List[List[int]], task: str = "retrieval.passage"):
    if embedding_model is None:
        load_embedding_model()

    tokenizer = embedding_model.tokenizer
    inputs = tokenizer(
        text,
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation=True,
        max_length=8192,
    ).to(DEVICE)
    offset_mapping = inputs["offset_mapping"][0].cpu().numpy()

    with torch.no_grad():
        outputs = embedding_model(
            inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            task=task,
        )
        token_embeddings = outputs.last_hidden_state[0]

    chunk_vectors = []
    for start_char, end_char in spans:
        token_mask = (offset_mapping[:, 0] >= start_char) & (offset_mapping[:, 1] <= end_char)
        indices = np.where(token_mask)[0]
        if len(indices) == 0:
            pooled = torch.zeros(token_embeddings.shape[1], device=DEVICE)
        else:
            slice_emb = token_embeddings[indices[0] : indices[-1] + 1]
            pooled = torch.mean(slice_emb, dim=0)
            norm = torch.norm(pooled)
            if norm > 0:
                pooled = pooled / norm
        chunk_vectors.append(pooled.cpu().tolist())
    return chunk_vectors

class LoadModelRequest(BaseModel):
    model: str

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    model: Optional[str] = DEFAULT_MODE
    max_tokens: int = 2048
    temperature: float = 0.7
    stream: bool = True

class GenerateRequest(BaseModel):
    model: Optional[str] = None
    prompt: str
    format: Optional[str] = None
    options: Optional[dict] = None

class EmbeddingOptions(BaseModel):
    late_chunking: bool = False
    chunk_spans: Optional[List[List[int]]] = None
    adapter: str = "retrieval.passage"

class EmbeddingRequest(BaseModel):
    model: Optional[str] = None
    prompt: str
    options: Optional[EmbeddingOptions] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    cleanup_memory()

app = FastAPI(lifespan=lifespan)

@app.post("/load")
async def manual_load_endpoint(req: LoadModelRequest):
    load_llm(req.model)
    return {"status": "ok", "message": f"Loaded model {req.model}", "model_id": MODE_CONFIG[req.model]}

@app.post("/v1/chat/completions")
async def chat_endpoint(req: ChatRequest):
    target_mode = req.model if req.model in MODE_CONFIG else DEFAULT_MODE
    load_llm(target_mode)
    prompt = llm_tokenizer.apply_chat_template(
        [m.model_dump() for m in req.messages],
        tokenize=False,
        add_generation_prompt=True,
    )
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

    outputs = llm_model.generate(**gen_kwargs)
    text = llm_tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return {"id": f"chatcmpl-{uuid.uuid4().hex}", "object": "chat.completion", "model": target_mode, "choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}]}

@app.post("/api/generate")
async def generate_endpoint(req: GenerateRequest):
    target_mode = req.model if (req.model in MODE_CONFIG) else DEFAULT_MODE
    load_llm(target_mode)
    prompt = llm_tokenizer.apply_chat_template(
        [{"role": "user", "content": req.prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = llm_tokenizer(prompt, return_tensors="pt").to(DEVICE)
    options = req.options or {}
    outputs = llm_model.generate(
        **inputs,
        max_new_tokens=options.get("max_tokens", 2048),
        temperature=options.get("temperature", 0.7),
        do_sample=True,
        top_p=options.get("top_p", 0.9),
    )
    text = llm_tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return {"model": target_mode, "created_at": datetime.now(timezone.utc).isoformat(), "response": text, "done": True}

@app.post("/api/embeddings")
async def embeddings_endpoint(req: EmbeddingRequest):
    load_embedding_model(req.model)
    opt = req.options or EmbeddingOptions()
    if opt.late_chunking and opt.chunk_spans:
        vectors = compute_late_chunking(req.prompt, opt.chunk_spans, opt.adapter)
        return {"model": EMBEDDING_MODEL_ID, "embeddings": vectors, "late_chunking": True}

    with torch.no_grad():
        if embedding_backend == "sentence_transformers":
            encoded = embedding_model.encode([req.prompt])
        else:
            encoded = embedding_model.encode([req.prompt], task=opt.adapter)
        if isinstance(encoded, torch.Tensor):
            encoded = encoded.cpu().tolist()
        elif isinstance(encoded, np.ndarray):
            encoded = encoded.tolist()
    return {"model": EMBEDDING_MODEL_ID, "embedding": encoded[0]}

@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE, "current_llm": current_llm_mode}
