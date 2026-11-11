"""Append-only JSONL evidence files. One line per real HTTP attempt, written immediately.

Real and stand-in evidence live in different folders and are never mixed:
    data/raw/real/      responses from the real Groq service
    data/raw/standin/   responses from the stand-in (added later)
"""
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from app.config import Settings
from app.providers.base import LLMResult


class RawLog:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def for_real(cls, settings: Settings, name: str) -> "RawLog":
        return cls(settings.raw_dir / "real" / f"{name}.jsonl")

    @classmethod
    def for_standin(cls, settings: Settings, name: str) -> "RawLog":
        return cls(settings.raw_dir / "standin" / f"{name}.jsonl")

    def write(self, record: Dict[str, Any]) -> None:
        line = {"logged_at": datetime.now(timezone.utc).isoformat(timespec="milliseconds"), **record}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(line, ensure_ascii=False, sort_keys=True) + "\n")


def result_record(result: LLMResult, *, source: str, label: str, model: str, messages: Any) -> Dict[str, Any]:
    """Everything we know about one attempt: the prompt, the answer, timings, usage and headers."""
    return {"source": source, "label": label, "model": model, "messages": messages, "result": asdict(result)}
