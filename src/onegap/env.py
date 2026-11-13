"""Minimal .env loader (no dependency). Existing environment variables always win; values are never printed."""

from __future__ import annotations

import os
import re
from pathlib import Path

_LINE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$")


def load_dotenv(path: str | Path | None = None) -> list[str]:
    """Load KEY=value lines into os.environ if not already set. Returns the key names loaded."""
    path = Path(path) if path else Path(__file__).resolve().parents[2] / ".env"
    if not path.exists():
        return []
    loaded = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        m = _LINE.match(line)
        if not m or line.lstrip().startswith("#"):
            continue
        key, val = m.group(1), m.group(2)
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        if val and key not in os.environ:
            os.environ[key] = val
            loaded.append(key)
    return loaded
