"""Manually (re-)seeds agents_library/*/agent.yaml into the `agents` table.

The backend also does this automatically on startup; this script is for
re-seeding on demand (e.g. after editing a built-in agent's YAML without
restarting the backend container).

Run from the backend container (or with backend/ on PYTHONPATH):
    python ../infra/scripts/seed_agents.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parent.parent.parent / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.db.seed import seed_builtin_agents  # noqa: E402

if __name__ == "__main__":
    asyncio.run(seed_builtin_agents())
    print("done")
