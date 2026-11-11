"""Step C: controlled per-minute TOKEN limit test against the real Groq service.

Plan (one request at a time, no concurrency):
  1. wait --warmup-s seconds so the token bucket refills;
  2. send real answer requests back to back (no pacing) until the service answers HTTP 429
     or --burst requests are done;
  3. if a 429 happened: retry once almost immediately (shows what an impatient retry does),
     then wait the service's Retry-After and retry (max --max-recovery times).

A 429 is only reported if it was actually received. If none arrives, the report says so.
Cost: a 429 normally costs no tokens; each successful request is about 1.9K tokens.

Run:
    python -m app.probe_limit --dry-run      # prints the plan and a token estimate, NO API calls
    python -m app.probe_limit                # the real test
Evidence: data/raw/real/limit_probe.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
from typing import Any, Dict, List, Optional

from app.answer import AnswerRecord, answer_question, build_messages, estimate_tokens
from app.config import RETRIEVAL_CONFIGS, load_settings
from app.probe_latency import PROBE_QUESTIONS
from app.providers.groq_adapter import GroqProvider
from app.raw_log import RawLog, result_record
from app.retrieval import Retriever

EARLY_RETRY_DELAY_S = 0.5
DEFAULT_WAIT_S = 10.0  # used only when a 429 carries no Retry-After


def summarize_events(events: List[Dict[str, Any]]) -> List[str]:
    """Plain-language report built ONLY from what was recorded."""
    burst = [e for e in events if e["phase"] == "burst"]
    lines = [f"burst requests sent: {len(burst)}   ok: {sum(e['status'] == 'ok' for e in burst)}"]
    first = next((e for e in events if e["status"] == "rate_limited"), None)
    if first is None:
        lines.append(f"no rate limit observed in {len(burst)} requests (nothing may be claimed about a 429)")
        return lines
    tokens_before = sum(e["tokens"] or 0 for e in burst if e["seq"] < first["seq"] and e["status"] == "ok")
    elapsed = first["t_start_s"]
    lines.append(f"429 observed at request #{first['seq']} ({elapsed:.1f} s after the first request); "
                 f"actual tokens used by the {first['seq'] - 1} requests before it: {tokens_before}")
    lines.append(f"  limit_kind guess={first['limit_kind']} (a wording guess), retry_after_s={first['retry_after_s']}, "
                 f"remaining_tokens header={first['remaining_tokens']}")
    lines.append(f"  message: {first['message']}")
    after = [e for e in events if e["seq"] > first["seq"]]
    for e in after:
        lines.append(f"  {e['phase']:<11} after {e['waited_s']:.1f} s wait -> {e['status']} "
                     f"(retry_after_s={e['retry_after_s']}, remaining_tokens={e['remaining_tokens']})")
    early = next((e for e in after if e["phase"] == "early_retry"), None)
    if early is not None:
        lines.append(f"  early retry: {early['status']}")
    ok_after = next((e for e in after if e["status"] == "ok"), None)
    if ok_after is None:
        lines.append("  not recovered within the allowed attempts")
    else:
        lines.append(f"  recovered after waiting {ok_after['waited_s']:.1f} s "
                     f"(the previous 429 said retry_after_s={ok_after['prev_retry_after_s']})")
    return lines


async def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--burst", type=int, default=8, help="max back-to-back requests")
    ap.add_argument("--warmup-s", type=float, default=60.0)
    ap.add_argument("--max-recovery", type=int, default=3)
    ap.add_argument("--budget", type=int, default=20000, help="stop the burst after this many ACTUAL tokens")
    ap.add_argument("--config", choices=sorted(RETRIEVAL_CONFIGS), default="B")
    ap.add_argument("--max-tokens", type=int, default=700)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    settings = load_settings()
    retriever = Retriever.from_settings(settings, RETRIEVAL_CONFIGS[args.config])
    questions = [PROBE_QUESTIONS[i % len(PROBE_QUESTIONS)] for i in range(args.burst)]

    if args.dry_run:
        est = 0
        for _, q in questions:
            messages, _ = build_messages(q, retriever.retrieve(q))
            est += sum(estimate_tokens(m["content"]) for m in messages)
        print(f"plan: wait {args.warmup_s:.0f}s, then up to {args.burst} back-to-back requests (stop at first 429 or "
              f"{args.budget} actual tokens), then 1 early retry and up to {args.max_recovery} waited retries")
        print(f"estimated prompt tokens if all {args.burst} succeed: {est} (a 429 request normally costs none)")
        return 0

    provider = GroqProvider(settings)
    log = RawLog.for_real(settings, "limit_probe")
    events: List[Dict[str, Any]] = []
    t0 = time.monotonic()
    seq = 0
    used = 0

    async def send(phase: str, category: str, q: str, waited: float, prev_ra: Optional[float]) -> AnswerRecord:
        nonlocal seq, used
        seq += 1
        start = time.monotonic() - t0
        rec = await answer_question(provider, retriever, q, max_tokens=args.max_tokens)
        r = rec.result
        assert r is not None  # the probe questions always retrieve passages
        tokens = (r.prompt_tokens or 0) + (r.completion_tokens or 0) if r.ok else None
        used += tokens or 0
        event = dict(phase=phase, seq=seq, t_start_s=start, status=r.status, limit_kind=r.limit_kind,
                     retry_after_s=r.ratelimit.retry_after_s, remaining_tokens=r.ratelimit.remaining_tokens,
                     tokens=tokens, message=r.error_message, waited_s=waited, prev_retry_after_s=prev_ra)
        events.append(event)
        entry = result_record(r, source="real", label=f"limit_probe_{seq}", model=settings.groq_model or "",
                              messages=rec.messages)
        entry.update(phase=phase, seq=seq, question=q, category=category, t_start_s=start, waited_s=waited,
                     est_prompt_tokens=rec.est_prompt_tokens)
        log.write(entry)
        print(f"{seq:>2}. {phase:<11} {r.status:<12} t={start:6.1f}s total_ms={r.total_ms:6.0f} "
              f"tokens={tokens} remaining_tokens={r.ratelimit.remaining_tokens} retry_after_s={r.ratelimit.retry_after_s}",
              flush=True)
        return rec

    try:
        print(f"waiting {args.warmup_s:.0f} s for the token bucket to refill ...", flush=True)
        await asyncio.sleep(args.warmup_s)
        t0 = time.monotonic()
        last: Optional[AnswerRecord] = None
        for category, q in questions:
            last = await send("burst", category, q, 0.0, None)
            if last.result.status == "rate_limited":
                break
            if used >= args.budget:
                print(f"Actual-token budget {args.budget} reached; stopping the burst.")
                break
        if last is not None and last.result.status == "rate_limited":
            category, q = questions[0]
            await asyncio.sleep(EARLY_RETRY_DELAY_S)
            last = await send("early_retry", category, q, EARLY_RETRY_DELAY_S, last.result.ratelimit.retry_after_s)
            for _ in range(args.max_recovery):
                if last.result.status == "ok":
                    break
                wait = last.result.ratelimit.retry_after_s or DEFAULT_WAIT_S
                prev = last.result.ratelimit.retry_after_s
                await asyncio.sleep(wait)
                last = await send("recovery", category, q, wait, prev)
    finally:
        await provider.aclose()

    print()
    for line in summarize_events(events):
        print(line)
    print(f"evidence appended to {log.path}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
