from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class MetricsSink:
    def __init__(self, *, runs_dir: str | Path, enabled: bool = True) -> None:
        self.runs_dir = Path(runs_dir)
        self.enabled = enabled
        self._lock = threading.Lock()
        self._ready_dirs: set[str] = set()

    def emit(self, session_id: str, event: str, **fields: Any) -> None:
        if not self.enabled:
            return
        payload = {
            "timestamp": datetime.now(UTC).isoformat(),
            "session_id": session_id,
            "event": event,
            **_json_safe(fields),
        }
        line = json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n"
        with self._lock:
            session_dir = self.runs_dir / session_id
            if session_id not in self._ready_dirs:
                session_dir.mkdir(parents=True, exist_ok=True)
                self._ready_dirs.add(session_id)
            with (session_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(line)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, int | float | str | bool) or value is None:
        return value
    return str(value)
