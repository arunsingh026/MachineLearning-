"""Conversation memory utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence


@dataclass
class Message:
    """Represents a single chat message."""

    role: str
    content: str

    def format(self) -> str:
        return f"{self.role}: {self.content.strip()}"


class ConversationMemory:
    """Simple in-memory storage for conversation history."""

    def __init__(self, messages: Iterable[Message] | None = None) -> None:
        self._messages: List[Message] = list(messages) if messages else []

    def add(self, role: str, content: str) -> None:
        self._messages.append(Message(role=role, content=content))

    def clear(self) -> None:
        self._messages.clear()

    @property
    def messages(self) -> Sequence[Message]:
        return tuple(self._messages)

    def formatted_history(self) -> str:
        return "\n".join(message.format() for message in self._messages)
