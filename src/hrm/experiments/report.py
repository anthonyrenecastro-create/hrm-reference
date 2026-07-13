"""Read a run summary for programmatic reporting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast


def load_summary(path: str | Path) -> dict[str, Any]:
    root = Path(path)
    target = root / "summary.json" if root.is_dir() else root
    return cast(dict[str, Any], json.loads(target.read_text(encoding="utf-8")))
