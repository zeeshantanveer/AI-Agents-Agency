# Architecture

## The core idea

Everything is an **AgentSpec**. A hand-built "built-in" agent and a prompt-generated
agent are the same object type, compiled by the same compiler, run by the same
executor. The only difference is how the spec was authored. This is enforced
everywhere — there is no separate code path for generated agents.

## Layers

- **`backend/agent_runtime/`** — the engine. AgentSpec pydantic models, a compiler
  that turns a spec into a LangGraph `StateGraph`, a provider router (multi-LLM via
  LiteLLM/LangChain), memory (short-term via LangGraph checkpoints, long-term via
  pgvector), and guardrails (budget limits, human-approval interrupts). This package
  has no FastAPI dependency — it's usable standalone from a CLI, a test, or a notebook.
- **`backend/tools/`** — the integration registry. Each tool exports an LLM-oriented
  description (used both for the prompt-to-agent tool matcher and the manual
  tool-picker UI), a sensitivity flag, required credentials, and a LangChain
  `BaseTool` factory.
- **`backend/generator/`** — the prompt-to-agent pipeline: intent extraction → tool
  matching → spec assembly → validation (including a real dry-run compile) →
  human preview/confirm. Nothing is persisted or runnable until the user confirms.
- **`backend/agents_library/`** — the built-in agents, as AgentSpec YAML. These
  double as the runtime's integration test suite: together they exercise every
  graph type, every guardrail, and the RAG/memory path.
- **`backend/app/`** — FastAPI. Accepts requests, enqueues work onto Redis (arq),
  and relays streamed run events over SSE/WS. The web process never executes an
  agent graph inline — that always happens in the worker.
- **`frontend/`** — Next.js. Agent library browser, per-agent run UI, run
  history/trace viewer, the prompt-to-agent builder flow, and integration
  credential settings.

## Execution flow

```
Next.js --REST/SSE/WS--> FastAPI --enqueue--> Redis --> worker
  worker: agent_runtime.executor.run_spec(spec)
    -> compiler.graph_builder.build(spec)   # AgentSpec -> LangGraph graph
    -> graph.astream_events(...)            # model calls, tool calls, checkpoints
    -> events published to Redis pub/sub `run:{run_id}`
  FastAPI subscribes and relays events back to the frontend
```

See the plan history / `docs/agentspec.md` (added alongside the first real
AgentSpec implementation in Phase 1) for the full field reference.
