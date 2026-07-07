from langchain_core.messages import AIMessage
from langchain_core.tools import tool as lc_tool

from agent_runtime import executor
from agent_runtime.compiler import graph_builder
from agent_runtime.spec.models import (
    AgentSpec,
    ApprovalGuardrails,
    BudgetGuardrails,
    GuardrailsConfig,
    ModelConfig,
    ToolRef,
)
from tools.base import ToolDefinition
from tools.registry import ToolRegistry

from .fakes import ScriptedFakeChatModel


@lc_tool
def echo_tool(x: int) -> str:
    """Echoes back a number, prefixed."""
    return f"result={x}"


def _test_registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(
        ToolDefinition(
            id="test.echo",
            name="Echo",
            description="Echoes a number.",
            category="test",
            factory=lambda config: echo_tool,
        )
    )
    return reg


def _base_spec(**overrides) -> AgentSpec:
    fields = dict(
        id="test-agent",
        name="Test Agent",
        description="A test agent.",
        model=ModelConfig(provider="anthropic", model_name="claude-sonnet-4-5"),
        system_prompt="You are a test agent.",
        tools=[ToolRef(id="test.echo")],
    )
    fields.update(overrides)
    return AgentSpec(**fields)


async def test_react_loop_calls_tool_then_ends(monkeypatch):
    responses = [
        AIMessage(content="", tool_calls=[{"name": "echo_tool", "args": {"x": 1}, "id": "call_1"}]),
        AIMessage(content="done"),
    ]
    fake_model = ScriptedFakeChatModel(responses=responses)
    monkeypatch.setattr(graph_builder, "get_chat_model", lambda config: fake_model)

    outcome = await executor.start_run(
        _base_spec(), "hello", thread_id="t1", tool_registry=_test_registry()
    )

    assert outcome.status == "succeeded"
    assert outcome.output == "done"
    assert outcome.budget_used["tool_calls"] == 1
    assert len(fake_model.calls) == 2


async def test_budget_guardrail_hard_stops_the_loop(monkeypatch):
    looping_call = AIMessage(
        content="", tool_calls=[{"name": "echo_tool", "args": {"x": 1}, "id": "call_x"}]
    )
    fake_model = ScriptedFakeChatModel(responses=[looping_call])  # never stops calling the tool
    monkeypatch.setattr(graph_builder, "get_chat_model", lambda config: fake_model)

    budget = BudgetGuardrails(max_tool_calls=2, max_run_seconds=300, max_cost_usd=10)
    spec = _base_spec(guardrails=GuardrailsConfig(budget=budget))
    outcome = await executor.start_run(spec, "hello", thread_id="t2", tool_registry=_test_registry())

    assert outcome.status == "succeeded"
    assert outcome.terminal_reason is not None
    assert "max_tool_calls" in outcome.terminal_reason
    assert outcome.budget_used["tool_calls"] <= 3  # stops shortly after crossing the limit


async def test_sensitive_tool_triggers_approval_interrupt_then_resumes(monkeypatch):
    from langgraph.checkpoint.memory import MemorySaver

    responses = [
        AIMessage(content="", tool_calls=[{"name": "echo_tool", "args": {"x": 9}, "id": "call_9"}]),
        AIMessage(content="approved and done"),
    ]
    fake_model = ScriptedFakeChatModel(responses=responses)
    monkeypatch.setattr(graph_builder, "get_chat_model", lambda config: fake_model)

    spec = _base_spec(
        guardrails=GuardrailsConfig(approvals=ApprovalGuardrails(require_human_approval_for=["test.echo"]))
    )
    checkpointer = MemorySaver()

    first = await executor.start_run(
        spec, "hello", thread_id="t3", checkpointer=checkpointer, tool_registry=_test_registry()
    )
    assert first.status == "waiting_approval"
    assert first.interrupt["tool_id"] == "test.echo"

    second = await executor.resume_run(
        spec,
        {"approved": True},
        thread_id="t3",
        checkpointer=checkpointer,
        tool_registry=_test_registry(),
    )
    assert second.status == "succeeded"
    assert second.output == "approved and done"


async def test_start_run_returns_failed_outcome_when_graph_build_raises(monkeypatch):
    """Regression test: a real deploy hit this — a missing-credentials error
    raised while resolving the model (i.e. before the graph even starts
    streaming) propagated out of start_run uncaught, leaving the caller's Run
    row stuck at status="running" forever with no error ever recorded."""

    def boom(spec, **kwargs):
        raise RuntimeError("boom: missing model credentials")

    monkeypatch.setattr(executor, "build", boom)

    outcome = await executor.start_run(_base_spec(), "hello", thread_id="t5", tool_registry=_test_registry())

    assert outcome.status == "failed"
    assert "boom" in outcome.error


async def test_resume_run_returns_failed_outcome_when_graph_build_raises(monkeypatch):
    from langgraph.checkpoint.memory import MemorySaver

    def boom(spec, **kwargs):
        raise RuntimeError("boom: missing model credentials")

    monkeypatch.setattr(executor, "build", boom)

    outcome = await executor.resume_run(
        _base_spec(),
        {"approved": True},
        thread_id="t6",
        checkpointer=MemorySaver(),
        tool_registry=_test_registry(),
    )

    assert outcome.status == "failed"
    assert "boom" in outcome.error
