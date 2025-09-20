# Personal GPT

This repository contains a lightweight, local-first conversational AI project inspired by ChatGPT. It combines a Hugging Face causal language model with a minimal conversation manager, a command-line interface, and an optional FastAPI server so you can self-host a personal assistant without relying on external APIs.

## Features

- 🧠 **Composable core** – `personal_gpt.PersonalGPT` wraps any Hugging Face causal language model and keeps track of conversation history.
- 💬 **Interactive CLI** – start chatting from your terminal with configurable sampling parameters.
- 🌐 **REST API** – deploy the assistant with FastAPI + Uvicorn and integrate it with your own tools.
- ⚙️ **Configurable** – adjust model name, device (CPU/GPU), temperature, token limits, and more.

## Getting started

### 1. Create and activate a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use `.venv\\Scripts\\activate`
```

### 2. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> ℹ️ The default model is `distilgpt2`, which downloads automatically on first use. You may swap in any other causal language model available on [Hugging Face](https://huggingface.co/models) (for example `microsoft/DialoGPT-medium`).

## Usage

### Chat from the terminal

```bash
python -m personal_gpt.interface_cli --model distilgpt2 --max-new-tokens 200 --temperature 0.8
```

During a chat session you can type `exit`, `quit`, or press `Ctrl+C` to stop. Supply `--device cuda` if you have a GPU available.

You can also provide a JSON configuration file with custom generation parameters:

```json
{
  "max_new_tokens": 256,
  "temperature": 0.7,
  "top_p": 0.85,
  "repetition_penalty": 1.05,
  "system_prompt": "You are my friendly personal assistant."
}
```

Save it as `config.json` and start the CLI with `--config config.json`.

### Run the REST API

1. Start the server:

   ```bash
   uvicorn personal_gpt.server:app --reload --port 8000
   ```

2. Send a request:

   ```bash
   curl -X POST http://localhost:8000/chat \
        -H "Content-Type: application/json" \
        -d '{"message": "Hello, who are you?"}'
   ```

   Example response:

   ```json
   {"response": "Hello! I'm a lightweight assistant ready to help."}
   ```

3. Reset the conversation memory when needed:

   ```bash
   curl -X POST http://localhost:8000/reset -H "Content-Type: application/json" -d '{"confirm": true}'
   ```

Open the automatically generated API docs at [http://localhost:8000/docs](http://localhost:8000/docs).

## Project structure

```
personal_gpt/
├── __init__.py          # Package exports
├── config.py            # Model and generation configuration dataclasses
├── interface_cli.py     # Terminal chat entry point
├── memory.py            # Conversation history helpers
├── model.py             # PersonalGPT core wrapper
└── server.py            # FastAPI server exposing /chat and /reset endpoints
```

## Customisation tips

- Swap models by editing `ModelConfig.model_name` or passing `--model` on the CLI.
- Modify `GenerationConfig.system_prompt` to control the assistant's persona.
- Extend `ConversationMemory` (e.g. for persistence) or implement your own storage backend.
- Use the FastAPI server as a base for building automations, Slack bots, or GUI front-ends.

## Disclaimer

This project is intentionally lightweight and does not implement advanced safety features, moderation, or retrieval augmentation. For serious production deployments you should evaluate additional safeguards and monitor model outputs carefully.
