"""Loads AgentSpec YAML files from agents_library/ on disk."""

from __future__ import annotations

from pathlib import Path

import yaml

from agent_runtime.spec.models import AgentSpec

AGENTS_LIBRARY_ROOT = Path(__file__).resolve().parent.parent.parent / "agents_library"


def load_spec_from_yaml(path: Path) -> AgentSpec:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return AgentSpec.model_validate(data)


def load_builtin_specs(root: Path = AGENTS_LIBRARY_ROOT) -> list[AgentSpec]:
    specs = []
    for agent_dir in sorted(root.iterdir()):
        spec_file = agent_dir / "agent.yaml"
        if spec_file.exists():
            specs.append(load_spec_from_yaml(spec_file))
    return specs
