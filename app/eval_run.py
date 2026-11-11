"""The REAL question-answering evaluation: 16 questions x retrieval settings A, B, C, one stage at a time.

THIS MODULE SENDS REAL REQUESTS to Groq, so it is locked. By default it only PLANS (no key is read, nothing is sent):

    python -m app.eval_run --stage A                      # dry run: prints the plan and the token reservations
    python -m app.eval_run --stage A --approve-real-calls # real calls. Also needs the environment variable RELAY_APPROVE_REAL=yes

Safety rules (all enforced in code, see tests/test_eval_run.py):
  * every attempt first reserves its worst-case tokens in app.budget_guard.TokenBudget (stage caps A 30,000 / B 41,000 / C 37,000,
    global ceiling 110,000, one ledger file shared by all stages and all runs). A refused reservation stops the whole run;
  * a retry reserves again; a 429 or an unknown result counts as fully spent; actual usage above the reservation halts everything;
  * requests are sent ONE AT A TIME and paced so the last 60 seconds never hold more than 85% of the 8,000 tokens-per-minute limit;
  * a question already recorded for this stage is skipped, so re-running a stage never spends tokens twice;
  * the run refuses to start if .env is not listed in .gitignore. The key is read by GroqProvider only and is never written anywhere.
Evidence: data/raw/eval/eval_attempts.jsonl (every HTTP attempt, with prompt and headers) and data/raw/real/eval_jobs.jsonl (one line
per question, the file results.csv is built from).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import uuid
from collections import deque
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

from app.answer import NOT_FOUND, build_messages, estimate_tokens
from app.budget_guard import STAGE_CAPS, BudgetExceeded, TokenBudget, upper_bound_tokens
from app.eval_budget import COMPLETION_HIGH, MEASURED_B_PROMPT, PROMPT_HIGH_FACTOR, load_questions, prompt_estimate
from app.providers.base import LLMRequest
from app.rag import verify

TPM_LIMIT = 8000
TPM_MARGIN = 0.85          # stay at 85% of the limit
MAX_ATTEMPTS = 3           # first try + 2 retries
DEADLINE_S = 45.0
MAX_TOKENS = COMPLETION_HIGH   # the completion cap the approved budget assumes
JOBS_FILE = "eval_jobs.jsonl"
ATTEMPTS_FILE = "eval_attempts.jsonl"
LEDGER_FILE = "eval_budget_ledger.json"


class NotApproved(RuntimeError):
    pass


def env_file_is_ignored(gitignore: Path = Path(".gitignore")) -> bool:
    try:
        lines = [x.strip() for x in gitignore.read_text(encoding="utf-8").splitlines()]
    except OSError:
        return False
    return ".env" in lines or ".env*" in lines


def require_approval(approve_flag: bool, env: Optional[Dict[str, str]] = None) -> None:
    env = os.environ if env is None else env
    if not approve_flag or env.get("RELAY_APPROVE_REAL") != "yes":
        raise NotApproved("real calls are locked: pass --approve-real-calls AND set RELAY_APPROVE_REAL=yes (only after the owner has approved the final run)")


def load_done(jobs_path: Path) -> set:
    done = set()
    if jobs_path.exists():
        for line in jobs_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                if d.get("scenario") == "eval":
                    done.add((d["retrieval_setting"], d["question_id"]))
    return done


def append_line(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


class Pacer:
    """Keeps the tokens sent in any 60 s window under TPM_LIMIT * TPM_MARGIN (tokens count at their reserved or settled size)."""

    def __init__(self, tpm: int = TPM_LIMIT, margin: float = TPM_MARGIN, now=time.monotonic, sleep=asyncio.sleep) -> None:
        self.limit = tpm * margin
        self.events: Deque[Tuple[float, int]] = deque()
        self.now, self.sleep = now, sleep

    def _trim(self) -> int:
        t = self.now()
        while self.events and t - self.events[0][0] >= 60.0:
            self.events.popleft()
        return sum(n for _, n in self.events)

    async def wait_for(self, tokens: int) -> float:
        waited = 0.0
        while self._trim() + tokens > self.limit and self.events:
            gap = max(60.0 - (self.now() - self.events[0][0]), 0.05)
            await self.sleep(gap)
            waited += gap
        return waited

    def record(self, tokens: int) -> None:
        self.events.append((self.now(), tokens))


def prompt_scale(retrievers: Dict[str, Any], questions: Sequence[Dict[str, str]]) -> float:
    b = [prompt_estimate(retrievers["B"], q["question"]) for q in questions]
    return MEASURED_B_PROMPT / (sum(b) / len(b))


def plan_stage(stage: str, retriever: Any, questions: Sequence[Dict[str, str]], scale: float) -> List[Dict[str, Any]]:
    out = []
    for q in questions:
        hits = retriever.retrieve(q["question"])
        messages, passages = build_messages(q["question"], hits)
        est = sum(estimate_tokens(m["content"]) for m in messages)
        out.append({"q": q, "messages": messages, "passages": passages, "est_prompt": est,
                    "reserve": upper_bound_tokens(est * scale, MAX_TOKENS, PROMPT_HIGH_FACTOR)})
    return out


def _retry_wait(result: Any) -> float:
    ra = getattr(getattr(result, "ratelimit", None), "retry_after_s", None)
    return min(max(ra or 5.0, 1.0), 30.0)


async def run_stage(stage: str, items: Sequence[Dict[str, Any]], provider: Any, guard: TokenBudget, out_dir: Path,
                    *, run_id: str, pacer: Optional[Pacer] = None, sleep=asyncio.sleep, model: str = "") -> Dict[str, Any]:
    pacer = pacer or Pacer()
    jobs_path, attempts_path = out_dir / "real" / JOBS_FILE, out_dir / "eval" / ATTEMPTS_FILE
    done = load_done(jobs_path)
    started = datetime.now(timezone.utc)
    t0 = time.monotonic()
    summary = {"stage": stage, "sent": 0, "skipped": 0, "ok": 0, "stopped": ""}
    for idx, it in enumerate(items):
        q = it["q"]
        if (stage, q["id"]) in done:
            summary["skipped"] += 1
            continue
        arrival = time.monotonic() - t0
        rec: Dict[str, Any] = {"scenario": "eval", "run_id": run_id, "run_started_at": started.isoformat(timespec="milliseconds"),
                               "job_id": idx, "question_id": q["id"], "question": q["question"], "kind": q["kind"],
                               "retrieval_setting": stage, "arrival_s": round(arrival, 3), "deadline_s": DEADLINE_S, "attempts": 0,
                               "n_429": 0, "n_other_failures": 0, "degraded": False, "outcome": "", "last_status": "",
                               "passages": it["passages"], "answer": "", "cited": [], "citation_status": "n/a",
                               "unsupported_numbers": [], "prompt_tokens": None, "completion_tokens": None}
        result = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                resv = guard.reserve(stage, it["reserve"])
            except BudgetExceeded as e:
                summary["stopped"] = str(e)
                rec.update(outcome="budget_exhausted", end_s=time.monotonic() - t0, latency_s=time.monotonic() - t0 - arrival)
                if rec["attempts"] == 0:                       # nothing was sent for this question: record the refusal and stop
                    append_line(jobs_path, rec)
                else:                                          # sent before, then the guard stopped a retry
                    rec.update(outcome="failed")
                    append_line(jobs_path, rec)
                return summary
            waited = await pacer.wait_for(it["reserve"])        # our OWN throttle (TPM pacing): not service time, so it is kept out of latency
            sent_at = time.monotonic() - t0
            if attempt == 1:
                arrival = sent_at
                rec["arrival_s"], rec["pacing_wait_s"] = round(arrival, 3), round(waited, 3)
            pacer.record(it["reserve"])
            result = await provider.generate(LLMRequest(messages=it["messages"], max_tokens=MAX_TOKENS, timeout_s=30.0))
            rec["attempts"] += 1
            rec["last_status"] = result.status
            append_line(attempts_path, {"logged_at": datetime.now(timezone.utc).isoformat(timespec="milliseconds"), "run_id": run_id,
                                        "question_id": q["id"], "retrieval_setting": stage, "attempt": attempt, "model": model,
                                        "reserved": it["reserve"], "messages": it["messages"], "result": asdict(result)})
            if result.status == "rate_limited":
                guard.settle_rejected(resv)
                rec["n_429"] += 1
            elif result.ok and result.prompt_tokens is not None and result.completion_tokens is not None:
                guard.settle(resv, result.prompt_tokens + result.completion_tokens)
            else:
                guard.settle(resv, None)                       # unknown usage counts as fully spent
                if not result.ok:
                    rec["n_other_failures"] += 1
            if guard.halted:
                summary["stopped"] = guard.halt_reason
            if result.ok or guard.halted or attempt == MAX_ATTEMPTS:
                break
            await sleep(_retry_wait(result))
        end = time.monotonic() - t0
        rec.update(end_s=round(end, 3), latency_s=round(end - arrival, 3), first_send_s=round(sent_at, 3), queue_wait_s=round(sent_at - arrival, 3))
        if result is not None and result.ok:
            rec["latency_s"] = round(end - arrival, 3)
            rec["outcome"] = "ok" if rec["latency_s"] <= DEADLINE_S else "ok_late"
            rec["ttft_from_arrival_s"] = round((sent_at - arrival) + (result.first_token_ms or 0.0) / 1000.0, 3)
            rec.update(prompt_tokens=result.prompt_tokens, completion_tokens=result.completion_tokens, answer=result.text)
            v = verify(result.text, [p["text"] for p in it["passages"]])
            rec.update(cited=v.cited, citation_status=v.status, unsupported_numbers=v.unsupported_numbers)
            summary["ok"] += 1
        else:
            rec["outcome"] = "failed"
        append_line(jobs_path, rec)
        summary["sent"] += 1
        if guard.halted:
            return summary
    return summary


def print_plan(stage: str, plan: Sequence[Dict[str, Any]], guard: TokenBudget, done: set) -> None:
    todo = [p for p in plan if (stage, p["q"]["id"]) not in done]
    total = sum(p["reserve"] for p in todo)
    print(f"stage {stage}: {len(todo)} question(s) to send ({len(plan) - len(todo)} already done); worst-case reservation per request:")
    for p in todo:
        print(f"   {p['q']['id']} reserve {p['reserve']:>5}  (prompt estimate {p['est_prompt']}, {len(p['passages'])} passages)")
    print(f"   sum of first-try reservations {total:,}; stage cap {guard.caps[stage]:,}; already committed {guard.committed(stage):,}; "
          f"global committed {guard.committed():,} of {guard.ceiling:,}")
    print("   retries reserve again; 429 / unknown results count as fully spent; the guard refuses before any cap or the ceiling is passed")


async def amain(args: argparse.Namespace) -> int:
    from app.config import RETRIEVAL_CONFIGS, load_settings
    from app.retrieval import Retriever
    settings = load_settings()
    out_dir = settings.raw_dir
    questions = load_questions(Path(args.questions))
    retrievers = {k: Retriever.from_settings(settings, c) for k, c in RETRIEVAL_CONFIGS.items()}
    scale = prompt_scale(retrievers, questions)
    guard = TokenBudget(STAGE_CAPS, ledger_path=out_dir / "eval" / LEDGER_FILE)
    plan = plan_stage(args.stage, retrievers[args.stage], questions, scale)
    done = load_done(out_dir / "real" / JOBS_FILE)
    print_plan(args.stage, plan, guard, done)
    if not args.approve_real_calls:
        print("\nDRY RUN: nothing was sent and no key was read. (Real run is locked until the owner approves.)")
        return 0
    require_approval(args.approve_real_calls)
    if not env_file_is_ignored():
        print("STOP: .env is not listed in .gitignore. Fix that first.")
        return 2
    from app.providers.groq_adapter import GroqProvider
    provider = GroqProvider(settings)
    try:
        summary = await run_stage(args.stage, plan, provider, guard, out_dir, run_id=f"eval-{args.stage}-{uuid.uuid4().hex[:8]}", model=settings.groq_model or "")
    finally:
        await provider.aclose()
    print(json.dumps({**summary, "spent_stage": guard.spent(args.stage), "spent_total": guard.spent(), "ceiling": guard.ceiling}, indent=2))
    return 1 if summary["stopped"] else 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=sorted(STAGE_CAPS))
    ap.add_argument("--questions", default="data/eval/questions.csv")
    ap.add_argument("--approve-real-calls", action="store_true")
    args = ap.parse_args(argv)
    try:
        return asyncio.run(amain(args))
    except NotApproved as e:
        print("LOCKED:", e)
        return 3


if __name__ == "__main__":
    sys.exit(main())
