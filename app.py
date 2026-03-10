from __future__ import annotations

import os
import uuid
from contextlib import asynccontextmanager
from typing import List, Optional

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agent.graph import PixellotAgentGraph

# ── Lifespan ──────────────────────────────────────────────────────────────────

agent_graph: PixellotAgentGraph | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent_graph
    agent_graph = PixellotAgentGraph()
    yield


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Pixellot AI Agent",
    version="1.0",
    lifespan=lifespan,
)

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────────────────────────────

class Message(BaseModel):
    role: str   # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    history: List[Message] = []


class ChatResponse(BaseModel):
    session_id: str
    response: str
    language: str
    intent: str
    should_handoff: bool


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "agent": "pixellot"}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())
    result = await agent_graph.invoke(
        user_message=req.message,
        history=[(m.role, m.content) for m in req.history],
    )
    return ChatResponse(
        session_id=session_id,
        response=result["response"],
        language=result.get("language", "en"),
        intent=result.get("intent", "GENERAL"),
        should_handoff=result.get("should_handoff", False),
    )
