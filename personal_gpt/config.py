"""Configuration dataclasses for the Personal GPT project."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """Configuration controlling how the base language model is loaded."""

    model_name: str = "distilgpt2"
    """Name or path of the Hugging Face model to load."""

    device: Optional[str] = None
    """Target device ("cpu" or "cuda"). ``None`` lets PyTorch decide automatically."""

    trust_remote_code: bool = False
    """Whether to allow execution of custom model code provided by the repository."""


@dataclass
class GenerationConfig:
    """Parameters that shape the style of generated responses."""

    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.0
    stop_sequences: list[str] = field(default_factory=lambda: ["User:", "Assistant:"])
    system_prompt: str = (
        "You are a helpful, concise personal assistant."
        " Respond directly to the user's request using the information"
        " provided in the conversation."
    )
