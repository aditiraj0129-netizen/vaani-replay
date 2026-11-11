"""Real-service probe: does sending several requests AT THE SAME TIME change anything?

Fills Table 1's "Requests in flight at once" row and checks the stand-in's biggest assumption (that the queue
time grows with the number of requests already in flight). Uses tiny requests (about 100 tokens each).

Waves of 1, 2, 4 and 8 simultaneous requests, with a pause between waves. Reports only what was observed:
a wave with no rejection does NOT prove there is no limit.

Run:
    python -m app.probe_concurrency --dry-run     # free: prints the plan
    python -m app.probe_concurrency               # real: about 15 tiny requests, about 1,500 tokens
Evidence: data/raw/real/concurrency_probe.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence

from app.config import load_settings
from app.providers.base import LLMRequest, LLMResult
from app.raw_log import RawLog, result_record

PROMPT = [{"role": "user", "content": "Reply with exactly the word: OK"}]


def _med(values: Sequence[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return statistics.median(vals) if vals else None


def summarize(waves: Dict[int, List[LLMResult]]) -> List[str]:
    """Plain-language report built ONLY from what was recorded."""
    lines = [f"{'in flight':>9} {'ok':>3} {'429':>4} {'other':>6} {'median queue ms':>16} {'median first-token ms':>22}"]
    rejected_at: Optional[int] = None
    for level in sorted(waves):
        rs = waves[level]
        ok = [r for r in rs if r.ok]
        n429 = sum(r.status == "rate_limited" for r in rs)
        other = len(rs) - len(ok) - n429
        q = _med([None if r.server_queue_s is None else r.server_queue_s * 1000 for r in ok])
        t = _med([r.first_token_ms for r in ok])
        fmt = lambda v: "-" if v is None else f"{v:.0f}"  # noqa: E731
        lines.append(f"{level:>9} {len(ok):>3} {n429:>4} {other:>6} {fmt(q):>16} {fmt(t):>22}")
        if n429 and rejected_at is None:
            rejected_at = level
    if rejected_at is None:
        top = max(waves) if waves else 0
        lines.append(f"no request was rejected with up to {top} in flight at once "
                     f"(that does not prove there is no limit; it was not tested higher)")
    else:
        kinds = {r.limit_kind for rs in waves.values() for r in rs if r.status == "rate_limited"}
        lines.append(f"first rejection appeared in the wave of {rejected_at} (limit kinds reported: {sorted(map(str, kinds))})")
    return lines


async def run_waves(provider: Any, levels: Sequence[int], gap_s: float, log: Optional[RawLog], model: str,
                    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep) -> Dict[int, List[LLMResult]]:
    waves: Dict[int, List[LLMResult]] = {}
    for i, level in enumerate(levels):
        if i:
            await sleep(gap_s)
        req = LLMRequest(messages=PROMPT, max_tokens=50, timeout_s=30.0)
        results = await asyncio.gather(*[provider.generate(req) for _ in range(level)])
        waves[level] = list(results)
        for j, r in enumerate(results):
            print(f"wave={level} #{j + 1}: {r.status} http={r.http_status} first_token_ms={r.first_token_ms} "
                  f"queue_s={r.server_queue_s} remaining_tokens={r.ratelimit.remaining_tokens}", flush=True)
            if log is not None:
                entry = result_record(r, source="real", label=f"concurrency_{level}_{j + 1}", model=model, messages=PROMPT)
                entry.update(wave=level, index=j + 1)
                log.write(entry)
    return waves


async def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--gap-s", type=float, default=6.0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    total = sum(args.levels)
    if args.dry_run:
        print(f"plan: waves of {args.levels} simultaneous tiny requests, {args.gap_s:.0f}s apart: {total} requests, "
              f"about {total * 100} tokens. NO API calls made.")
        return 0
    from app.providers.groq_adapter import GroqProvider
    settings = load_settings()
    provider = GroqProvider(settings)
    try:
        waves = await run_waves(provider, args.levels, args.gap_s, RawLog.for_real(settings, "concurrency_probe"),
                                settings.groq_model or "")
    finally:
        await provider.aclose()
    print()
    for line in summarize(waves):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
