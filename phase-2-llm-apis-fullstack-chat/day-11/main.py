import uuid
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import logging
import time
from gemini_client import generate_reply
from fastapi import FastAPI, HTTPException

app = FastAPI()

import logging
import time

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("chat_api")
# In-memory session store: session_id -> list of message dicts
chat_sessions = {}

SYSTEM_PROMPT = "You are a helpful assistant."


class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str


class ChatResponse(BaseModel):
    session_id: str
    response: str
    message_count: int

class SessionSummary(BaseModel):
    session_id: str
    message_count: int


@app.get("/api/sessions", response_model=list[SessionSummary])
def list_sessions():
    return [
        SessionSummary(session_id=sid, message_count=len(history))
        for sid, history in chat_sessions.items()
    ]

@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="message cannot be empty")

    session_id = request.session_id

    if session_id is not None and session_id not in chat_sessions:
        raise HTTPException(status_code=404, detail=f"session_id '{session_id}' not found")

    if session_id is None:
        session_id = str(uuid.uuid4())
        chat_sessions[session_id] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

    chat_sessions[session_id].append({"role": "user", "content": request.message})

    try:
        start = time.time()
        result = generate_reply(chat_sessions[session_id])
        latency_ms = round((time.time() - start) * 1000, 2)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"model generation failed: {str(e)}")

    model_reply = result["text"]

    logger.info({
        "timestamp": time.time(),
        "session_id": session_id,
        "model": "gemini-2.0-flash-lite",
        "token_usage": {
            "input": result["input_tokens"],
            "output": result["output_tokens"],
        },
        "latency_ms": latency_ms,
    })

    chat_sessions[session_id].append({"role": "model", "content": model_reply})