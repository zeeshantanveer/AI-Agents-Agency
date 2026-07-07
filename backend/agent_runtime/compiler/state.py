from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class BudgetUsed(TypedDict):
    tool_calls: int
    cost_usd: float
    seconds: float


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    budget_used: BudgetUsed
    scratchpad: dict[str, Any]
    terminal_reason: str | None


def initial_state(input_messages: list[BaseMessage]) -> AgentState:
    return AgentState(
        messages=input_messages,
        budget_used=BudgetUsed(tool_calls=0, cost_usd=0.0, seconds=0.0),
        scratchpad={},
        terminal_reason=None,
    )
