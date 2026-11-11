"""One-off, FREE: re-state stage-A style job lines whose latency wrongly included our own pacing wait. No API calls.

The first version of eval_run counted the time it deliberately waited (to stay under 8,000 tokens per minute) as latency, so some
answers that the service returned in about a second were marked 'ok_late'. This script
  * copies data/raw/real/eval_jobs.jsonl to data/raw/real_archive/eval_jobs_before_pacing_fix.jsonl (the original is kept, never edited),
  * writes the corrected lines to data/raw/real/eval_jobs.jsonl: arrival_s = first send time, pacing_wait_s = the old queue wait,
    latency_s = end_s - first_send_s, ttft = the service's own first_token_ms, outcome recomputed against the 45 s deadline.
Lines that already have pacing_wait_s are left alone, so running it twice changes nothing.
    python -m app.fix_eval_pacing
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

JOBS = Path("data/raw/real/eval_jobs.jsonl")
ATTEMPTS = Path("data/raw/eval/eval_attempts.jsonl")
ARCHIVE = Path("data/raw/real_archive/eval_jobs_before_pacing_fix.jsonl")


def fix_line(job: Dict[str, Any], first_token_ms: Optional[float]) -> Dict[str, Any]:
    if job.get("scenario") != "eval" or "pacing_wait_s" in job or job.get("first_send_s") is None:
        return job
    out = dict(job)
    out["pacing_wait_s"] = round(job.get("queue_wait_s") or 0.0, 3)
    out["arrival_s"] = job["first_send_s"]
    out["queue_wait_s"] = 0.0
    if job.get("end_s") is not None:
        out["latency_s"] = round(job["end_s"] - job["first_send_s"], 3)
    if job["outcome"] in ("ok", "ok_late"):
        out["outcome"] = "ok" if out["latency_s"] <= job["deadline_s"] else "ok_late"
        if first_token_ms is not None:
            out["ttft_from_arrival_s"] = round(first_token_ms / 1000.0, 3)
    out["pacing_fix"] = "arrival moved to first send; original line kept in real_archive"
    return out


def first_tokens(attempts: Sequence[Dict[str, Any]]) -> Dict[tuple, Optional[float]]:
    out: Dict[tuple, Optional[float]] = {}
    for a in attempts:
        if a["result"].get("status") == "ok":
            out[(a["retrieval_setting"], a["question_id"])] = a["result"].get("first_token_ms")
    return out


def main(argv: Optional[Sequence[str]] = None, jobs_path: Path = JOBS, attempts_path: Path = ATTEMPTS, archive: Path = ARCHIVE) -> int:
    lines: List[Dict[str, Any]] = [json.loads(x) for x in jobs_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    attempts = [json.loads(x) for x in attempts_path.read_text(encoding="utf-8").splitlines() if x.strip()] if attempts_path.exists() else []
    ft = first_tokens(attempts)
    fixed = [fix_line(j, ft.get((j.get("retrieval_setting"), j.get("question_id")))) for j in lines]
    changed = sum(1 for a, b in zip(lines, fixed) if a != b)
    if changed:
        archive.parent.mkdir(parents=True, exist_ok=True)
        if not archive.exists():
            shutil.copy2(jobs_path, archive)
        jobs_path.write_text("".join(json.dumps(j, ensure_ascii=False, sort_keys=True) + "\n" for j in fixed), encoding="utf-8")
    print(f"{changed} of {len(lines)} job lines re-stated (original kept at {archive})")
    for a, b in zip(lines, fixed):
        if a != b and a["outcome"] != b["outcome"]:
            print(f"   {a['retrieval_setting']} {a['question_id']}: {a['outcome']} -> {b['outcome']} (pacing wait {b['pacing_wait_s']} s, service latency {b['latency_s']} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
