"""FastAPI server that exposes a lightweight chat completion endpoint."""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .config import GenerationConfig, ModelConfig
from .model import PersonalGPT


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str


class ResetRequest(BaseModel):
    confirm: bool = True


app = FastAPI(title="Personal GPT", version="1.0.0")


@lru_cache(maxsize=1)
def get_assistant(
    model_name: str = "distilgpt2",
    device: Optional[str] = None,
) -> PersonalGPT:
    model_config = ModelConfig(model_name=model_name, device=device)
    generation_config = GenerationConfig()
    return PersonalGPT(model_config=model_config, generation_config=generation_config)


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message must not be empty")
    assistant = get_assistant()
    reply = assistant.chat(request.message)
    return ChatResponse(response=reply)


@app.post("/reset")
async def reset_conversation(request: ResetRequest) -> dict[str, str]:
    if not request.confirm:
        raise HTTPException(status_code=400, detail="Confirmation required to reset")
    assistant = get_assistant()
    assistant.reset()
    return {"status": "ok"}
