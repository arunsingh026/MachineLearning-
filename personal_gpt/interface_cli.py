"""Command line chat interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from .config import GenerationConfig, ModelConfig
from .model import PersonalGPT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chat with a local Personal GPT model")
    parser.add_argument("--model", default="distilgpt2", help="Model name or path to load")
    parser.add_argument("--device", default=None, help="Device to run on (cpu or cuda)")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional path to a JSON file containing generation parameters",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=None, help="Maximum number of generated tokens"
    )
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=None, help="Nucleus sampling parameter")
    parser.add_argument(
        "--repetition-penalty", type=float, default=None, help="Penalty for repeated tokens"
    )
    return parser.parse_args()


def load_generation_config(args: argparse.Namespace) -> GenerationConfig:
    config = GenerationConfig()
    if args.config and args.config.exists():
        data: Dict[str, Any] = json.loads(args.config.read_text())
        config = GenerationConfig(**data)

    if args.max_new_tokens is not None:
        config.max_new_tokens = args.max_new_tokens
    if args.temperature is not None:
        config.temperature = args.temperature
    if args.top_p is not None:
        config.top_p = args.top_p
    if args.repetition_penalty is not None:
        config.repetition_penalty = args.repetition_penalty

    return config


def run_cli() -> None:
    args = parse_args()
    model_config = ModelConfig(model_name=args.model, device=args.device)
    generation_config = load_generation_config(args)

    assistant = PersonalGPT(model_config=model_config, generation_config=generation_config)

    print("Personal GPT - type 'exit' or Ctrl+C to quit.\n")
    while True:
        try:
            user_input = input("You: ")
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if user_input.strip().lower() in {"exit", "quit", "bye"}:
            print("Goodbye!")
            break

        response = assistant.chat(user_input)
        print(f"Assistant: {response}\n")


if __name__ == "__main__":
    run_cli()
