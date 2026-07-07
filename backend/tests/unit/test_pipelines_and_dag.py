from langchain_core.messages import AIMessage
from langchain_core.tools import tool as lc_tool

from agent_runtime import executor
from agent_runtime.compiler import custom_dag, graph_builder, pipelines
from agent_runtime.spec.models import AgentSpec, GraphConfig, GraphEdge, GraphNode, ModelConfig, ToolRef
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


def _patch_all_model_factories(monkeypatch, fake_model) -> None:
    monkeypatch.setattr(graph_builder, "get_chat_model", lambda config: fake_model)
    monkeypatch.setattr(pipelines, "get_chat_model", lambda config: fake_model)
    monkeypatch.setattr(custom_dag, "get_chat_model", lambda config: fake_model)


async def test_linear_pipeline_runs_stages_in_order(monkeypatch):
    responses = [
        AIMessage(content="", tool_calls=[{"name": "echo_tool", "args": {"x": 1}, "id": "call_1"}]),
        AIMessage(content="search stage done"),
        AIMessage(content="final summary"),
    ]
    fake_model = ScriptedFakeChatModel(responses=responses)
    _patch_all_model_factories(monkeypatch, fake_model)

    spec = AgentSpec(
        id="pipeline-agent",
        name="Pipeline Agent",
        description="test",
        model=ModelConfig(provider="anthropic", model_name="claude-sonnet-4-5"),
        system_prompt="You are a research pipeline.",
        tools=[ToolRef(id="test.echo")],
        graph=GraphConfig(
            type="linear_pipeline",
            nodes=[
                GraphNode(id="search", kind="llm_call"),
                GraphNode(id="summarize", kind="llm_call"),
            ],
        ),
    )

    outcome = await executor.start_run(spec, "hello", thread_id="p1", tool_registry=_test_registry())

    assert outcome.status == "succeeded"
    assert outcome.output == "final summary"
    assert outcome.budget_used["tool_calls"] == 1
    assert len(fake_model.calls) == 3


async def test_custom_dag_router_picks_branch(monkeypatch):
    responses = [
        AIMessage(content="gathered context"),  # intro node
        AIMessage(content="path_a"),  # router picks path_a
        AIMessage(content="handled by path A"),  # path_a node
    ]
    fake_model = ScriptedFakeChatModel(responses=responses)
    _patch_all_model_factories(monkeypatch, fake_model)

    spec = AgentSpec(
        id="dag-agent",
        name="DAG Agent",
        description="test",
        model=ModelConfig(provider="anthropic", model_name="claude-sonnet-4-5"),
        system_prompt="You route work.",
        graph=GraphConfig(
            type="custom_dag",
            nodes=[
                GraphNode(id="intro", kind="llm_call"),
                GraphNode(
                    id="router",
                    kind="router",
                    config={"options": {"path_a": "Handle via A", "path_b": "Handle via B"}},
                ),
                GraphNode(id="path_a", kind="llm_call"),
                GraphNode(id="path_b", kind="llm_call"),
            ],
            edges=[
                GraphEdge(source="intro", target="router"),
                GraphEdge(source="router", target="path_a", condition="path_a"),
                GraphEdge(source="router", target="path_b", condition="path_b"),
                GraphEdge(source="path_a", target="END"),
                GraphEdge(source="path_b", target="END"),
            ],
        ),
    )

    outcome = await executor.start_run(spec, "hello", thread_id="d1", tool_registry=_test_registry())

    assert outcome.status == "succeeded"
    assert outcome.output == "handled by path A"


async def test_custom_dag_human_review_interrupts_then_resumes(monkeypatch):
    from langgraph.checkpoint.memory import MemorySaver

    responses = [
        AIMessage(content="draft summary"),  # draft node
        AIMessage(content="finalized after approval"),  # finalize node
    ]
    fake_model = ScriptedFakeChatModel(responses=responses)
    _patch_all_model_factories(monkeypatch, fake_model)

    spec = AgentSpec(
        id="review-agent",
        name="Review Agent",
        description="test",
        model=ModelConfig(provider="anthropic", model_name="claude-sonnet-4-5"),
        system_prompt="You draft then finalize.",
        graph=GraphConfig(
            type="custom_dag",
            nodes=[
                GraphNode(id="draft", kind="llm_call"),
                GraphNode(id="review", kind="human_review"),
                GraphNode(id="finalize", kind="llm_call"),
            ],
            edges=[
                GraphEdge(source="draft", target="review"),
                GraphEdge(source="review", target="finalize", condition="approved"),
                GraphEdge(source="review", target="END", condition="rejected"),
                GraphEdge(source="finalize", target="END"),
            ],
        ),
    )
    checkpointer = MemorySaver()

    first = await executor.start_run(
        spec, "hello", thread_id="r1", checkpointer=checkpointer, tool_registry=_test_registry()
    )
    assert first.status == "waiting_approval"
    assert first.interrupt["type"] == "human_review"

    second = await executor.resume_run(
        spec, {"approved": True}, thread_id="r1", checkpointer=checkpointer, tool_registry=_test_registry()
    )
    assert second.status == "succeeded"
    assert second.output == "finalized after approval"
