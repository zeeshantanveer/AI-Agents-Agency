"""Every built-in agent must load, validate against the real tool registry,
and actually compile — this is the runtime's own integration test suite, per
the project design (the built-in library doubles as the compiler's test bed).
"""

from unittest.mock import patch

from langchain_core.messages import AIMessage

from agent_runtime.compiler import custom_dag, graph_builder, pipelines
from agent_runtime.spec.loader import load_builtin_specs
from agent_runtime.spec.validate import validate_spec
from tools.registry import registry

from .fakes import ScriptedFakeChatModel


def test_all_builtin_specs_are_discovered():
    specs = load_builtin_specs()
    assert len(specs) == 7
    assert {s.id for s in specs} == {
        "code-reviewer",
        "test-generator",
        "document-qna",
        "web-researcher",
        "customer-support",
        "meeting-notes",
        "sales-outreach",
    }


def test_all_builtin_specs_validate_against_the_real_tool_registry():
    registry.discover()
    for spec in load_builtin_specs():
        result = validate_spec(spec, known_tool_ids=registry.ids())
        assert result.ok, f"{spec.id}: {result.errors}"


def test_all_builtin_specs_compile(monkeypatch):
    registry.discover()
    fake = ScriptedFakeChatModel(responses=[AIMessage(content="ok")])

    with (
        patch.object(graph_builder, "get_chat_model", lambda c: fake),
        patch.object(pipelines, "get_chat_model", lambda c: fake),
        patch.object(custom_dag, "get_chat_model", lambda c: fake),
    ):
        for spec in load_builtin_specs():
            graph_builder.build(spec, tool_registry=registry)  # raises on failure
