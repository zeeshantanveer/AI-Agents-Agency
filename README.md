# AI Agents Agency

Production-ready built-in AI agents, and a prompt-to-agent generator that turns a
natural-language description into a working, integrated agent.

Built on **LangGraph** (orchestration) + **LangChain** (tools/models/memory), with
**LlamaIndex** as an optional data-ingestion layer for RAG-heavy agents. Multi-LLM
via a provider router (Anthropic, OpenAI, ...). Self-hosted via Docker Compose —
no cloud account required to try it.

See [`docs/architecture.md`](docs/architecture.md) for the full design.

## Quickstart

```bash
cp .env.example .env
# fill in ANTHROPIC_API_KEY and/or OPENAI_API_KEY in .env
docker compose up --build
```

- Backend API: http://localhost:8000 (docs at `/docs`)
- Frontend: http://localhost:3000

## Status

This project is in early scaffolding (Phase 0 of the build plan in
`docs/architecture.md`). The core LangGraph runtime, built-in agent library, and
prompt-to-agent generator land in subsequent phases.

## Local development (without Docker)

```bash
# backend
cd backend
python -m venv .venv && source .venv/Scripts/activate  # or .venv/bin/activate on macOS/Linux
pip install -e ".[dev]"
pytest

# frontend
cd frontend
npm install
npm run dev
```
