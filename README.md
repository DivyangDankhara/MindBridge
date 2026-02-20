# MindBridge

MindBridge is an intent-driven AI runtime that converts structured human intent into an execution plan using an LLM and prepares actions for execution.

## Requirements

- Python 3.11+
- OpenAI API key

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Set `OPENAI_API_KEY` in `.env`.

## Run

```bash
cd mindbridge
python main.py
```

## Intent Format

MindBridge expects `KEY: VALUE` structured intent text. Supported keys:

- `TASK`
- `GOAL`
- `CONSTRAINTS` (optional)
- `OUTPUT` (optional)
