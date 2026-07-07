"""Test doubles shared by unit tests and the eval harness. Lives in
agent_runtime (not tests/) so evals/ — a separate top-level package — can
import it without reaching into the test suite.
"""

from __future__ import annotations

import uuid

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field


class ScriptedFakeChatModel(BaseChatModel):
    """Replays `responses` in order, repeating the last one once exhausted.

    Each returned message gets a fresh `id` — reusing the same AIMessage
    object (and therefore the same auto-assigned id) across turns would
    collide with LangGraph's id-based replace semantics in `add_messages`
    and silently corrupt message ordering in the graph state.
    """

    responses: list[AIMessage]
    calls: list = Field(default_factory=list)

    def bind_tools(self, tools, **kwargs):  # noqa: ANN001, ANN003
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):  # noqa: ANN001
        idx = len(self.calls)
        self.calls.append(messages)
        template = self.responses[min(idx, len(self.responses) - 1)]
        response = template.model_copy(update={"id": str(uuid.uuid4())})
        return ChatResult(generations=[ChatGeneration(message=response)])

    @property
    def _llm_type(self) -> str:
        return "scripted-fake"
