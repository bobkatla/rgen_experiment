from __future__ import annotations
import json, os, time
from typing import Any, Dict, Optional

class JsonlLogger:
    """
    Minimal JSONL logger: appends one JSON object per line.
    Good for metrics you’ll later aggregate into reports.
    """
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path

    def write(self, record: Dict[str, Any]) -> None:
        rec = dict(time=time.time(), **record)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

def save_text(text: str, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
