"""Budget tracking and enforcement, checked inside the compiled graph itself
(not just monitored after the fact) so a runaway agent hard-stops mid-run.
"""

from __future__ import annotations

import time
from typing import Any

from langchain_core.messages import AIMessage

from agent_runtime.spec.models import BudgetGuardrails

# Rough $/1K-token pricing, used only for the soft cost_usd guardrail estimate.
# Unknown models cost 0 — the budget check degrades gracefully, it never blocks
# on a pricing-table miss.
_PRICE_PER_1K_TOKENS: dict[str, tuple[float, float]] = {
    "claude-sonnet-4-5": (0.003, 0.015),
    "claude-haiku-4-5": (0.0008, 0.004),
    "gpt-4.1": (0.002, 0.008),
    "gpt-4.1-mini": (0.0004, 0.0016),
}


def estimate_cost_usd(model_name: str, message: AIMessage) -> float:
    usage = getattr(message, "usage_metadata", None)
    if not usage:
        return 0.0
    in_price, out_price = _PRICE_PER_1K_TOKENS.get(model_name, (0.0, 0.0))
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    return (input_tokens / 1000) * in_price + (output_tokens / 1000) * out_price


def elapsed_seconds(started_at: float) -> float:
    return time.monotonic() - started_at


def budget_exceeded(used: dict[str, Any], budget: BudgetGuardrails) -> str | None:
    if used["tool_calls"] >= budget.max_tool_calls:
        return f"max_tool_calls exceeded ({used['tool_calls']}/{budget.max_tool_calls})"
    if used["seconds"] >= budget.max_run_seconds:
        return f"max_run_seconds exceeded ({used['seconds']:.0f}/{budget.max_run_seconds})"
    if used["cost_usd"] >= budget.max_cost_usd:
        return f"max_cost_usd exceeded (${used['cost_usd']:.4f}/${budget.max_cost_usd})"
    return None


def new_run_clock() -> float:
    return time.monotonic()
