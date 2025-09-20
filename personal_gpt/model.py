"""Model wrapper used to generate chat completions."""

from __future__ import annotations

from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import GenerationConfig, ModelConfig
from .memory import ConversationMemory


class PersonalGPT:
    """High level helper that mimics a lightweight ChatGPT-style interface."""

    def __init__(
        self,
        model_config: Optional[ModelConfig] = None,
        generation_config: Optional[GenerationConfig] = None,
        memory: Optional[ConversationMemory] = None,
    ) -> None:
        self.model_config = model_config or ModelConfig()
        self.generation_config = generation_config or GenerationConfig()
        self.memory = memory or ConversationMemory()

        self.device = self._resolve_device(self.model_config.device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config.model_name,
            trust_remote_code=self.model_config.trust_remote_code,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_config.model_name,
            trust_remote_code=self.model_config.trust_remote_code,
        ).to(self.device)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.eos_token_id = self.tokenizer.eos_token_id

    @staticmethod
    def _resolve_device(user_choice: Optional[str]) -> torch.device:
        if user_choice:
            return torch.device(user_choice)
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def build_prompt(self, user_message: str) -> str:
        parts: list[str] = []
        system_prompt = self.generation_config.system_prompt.strip()
        if system_prompt:
            parts.append(f"System: {system_prompt}")

        history = self.memory.formatted_history().strip()
        if history:
            parts.append(history)

        parts.append(f"User: {user_message.strip()}")
        parts.append("Assistant:")
        return "\n".join(parts)

    def _apply_stop_sequences(self, text: str) -> str:
        for stop in self.generation_config.stop_sequences:
            if not stop:
                continue
            idx = text.find(stop)
            if idx != -1:
                text = text[:idx]
        return text.strip()

    def generate(self, user_message: str) -> str:
        prompt = self.build_prompt(user_message)
        encoded = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output = self.model.generate(
                **encoded,
                max_new_tokens=self.generation_config.max_new_tokens,
                temperature=self.generation_config.temperature,
                top_p=self.generation_config.top_p,
                repetition_penalty=self.generation_config.repetition_penalty,
                do_sample=True,
                pad_token_id=self.eos_token_id,
            )

        generated_tokens = output[0][encoded["input_ids"].shape[-1] :]
        raw_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return self._apply_stop_sequences(raw_text)

    def chat(self, user_message: str) -> str:
        self.memory.add("User", user_message)
        assistant_reply = self.generate(user_message)
        self.memory.add("Assistant", assistant_reply)
        return assistant_reply

    def reset(self) -> None:
        self.memory.clear()
