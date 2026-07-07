from __future__ import annotations

import re

from agent_runtime.spec.models import InputGuardrails


class InputRejected(ValueError):
    pass


def check_input(text: str, guardrails: InputGuardrails) -> None:
    for pattern in guardrails.blocked_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            raise InputRejected(f"input matched blocked pattern: {pattern}")
    approx_tokens = len(text) // 4
    if approx_tokens > guardrails.max_input_tokens:
        raise InputRejected(
            f"input too long (~{approx_tokens} tokens > {guardrails.max_input_tokens} limit)"
        )
