from __future__ import annotations

import re

from agent_runtime.spec.models import OutputGuardrails

_PII_PATTERNS = [
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),  # SSN-shaped
    re.compile(r"\b\d{4}[ -]?\d{4}[ -]?\d{4}[ -]?\d{4}\b"),  # card-shaped
]


def apply_output_guardrails(text: str, guardrails: OutputGuardrails) -> str:
    if guardrails.pii_redaction:
        for pattern in _PII_PATTERNS:
            text = pattern.sub("[REDACTED]", text)
    approx_tokens = len(text) // 4
    if approx_tokens > guardrails.max_output_tokens:
        char_limit = guardrails.max_output_tokens * 4
        text = text[:char_limit] + "\n...[truncated by output guardrail]"
    return text
