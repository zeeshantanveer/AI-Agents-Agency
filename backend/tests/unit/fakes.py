"""Re-exports the shared test double from agent_runtime.testing, which also
backs the eval harness — kept here too so existing `from .fakes import ...`
test imports keep working.
"""

from agent_runtime.testing import ScriptedFakeChatModel

__all__ = ["ScriptedFakeChatModel"]
