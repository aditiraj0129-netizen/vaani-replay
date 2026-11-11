"""Step B: sequential latency probe with REAL answer prompts (retrieval + Groq).

Sequential on purpose: one request at a time, paced so the token bucket stays safe, so the
numbers describe an UNLOADED service. This is not a load test.

Run:
    python -m app.probe_latency --dry-run          # builds prompts, estimates tokens, NO API calls
    python -m app.probe_latency --n 12             # real calls (about 2K tokens each)
Safety: stops at the first rate-limit error, or when the actual-token budget (--budget) is used.
Evidence: data/raw/real/latency_probe.jsonl (one line per request, with headers).
"""
from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
from typing import Any, Dict, List, Optional, Tuple

from app.answer import AnswerRecord, answer_question, build_messages, estimate_tokens
from app.config import RETRIEVAL_CONFIGS, load_settings
from app.providers.groq_adapter import GroqProvider
from app.raw_log import RawLog, result_record
from app.retrieval import Retriever

# Probe questions are NOT the evaluation questions. Answers are not graded here.
# The category is what we EXPECT from the papers (confirm with `python -m app.check_probe_terms`).
ANSWERABLE, PARTIAL, UNANSWERABLE = "answerable", "partial", "unanswerable"
PROBE_QUESTIONS: List[Tuple[str, str]] = [
    (ANSWERABLE, "How many layers does the base Transformer encoder stack have?"),
    (ANSWERABLE, "Why does scaled dot-product attention divide by the square root of d_k?"),
    (ANSWERABLE, "What is Inner Monologue in Moshi?"),
    (ANSWERABLE, "What frame rate does the Mimi codec operate at?"),
    (ANSWERABLE, "What is the theoretical latency of Moshi?"),
    (ANSWERABLE, "What is the dynamic chunking mechanism in H-Net?"),
    (ANSWERABLE, "What does the routing module of H-Net compute?"),
    (PARTIAL, "How long did training the base Transformer take, and what did it cost in dollars?"),
    (PARTIAL, "What is Helium in the Moshi paper, and how much did it cost to pretrain?"),
    (UNANSWERABLE, "What BLEU score does the Transformer reach on WMT 2014 English-to-Russian?"),
    (UNANSWERABLE, "What word error rate does Moshi achieve on Mandarin speech recognition?"),
    (UNANSWERABLE, "What top-1 accuracy does H-Net reach on ImageNet?"),
]

DEFAULT_TPM = 8000


def pace_seconds(remaining_tokens: Optional[int], needed_tokens: int, tpm: int, margin: int = 500) -> float:
    """How long to wait so `needed_tokens + margin` fit in the bucket, assuming a steady refill of tpm/60 per second.

    The refill model is an inference from the headers (reset-tokens = missing tokens / (tpm/60)); Step C tests it under load.
    """
    if remaining_tokens is None:
        return 15.0  # unknown state: wait a safe fixed time
    deficit = needed_tokens + margin - remaining_tokens
    return max(0.0, deficit / (tpm / 60.0))


def _spread(values: List[float]) -> str:
    if not values:
        return "n=0"
    return f"n={len(values)} min={min(values):.0f} median={statistics.median(values):.0f} max={max(values):.0f}"


def summarize(records: List[AnswerRecord]) -> List[str]:
    ok = [r.result for r in records if r.result is not None and r.result.ok]
    lines = [f"requests made: {sum(r.result is not None for r in records)}   complete answers: {len(ok)}"]

    def col(name: str, getter) -> None:
        vals = [getter(x) for x in ok if getter(x) is not None]
        lines.append(f"  {name:<22} {_spread(vals)}")

    col("first_token_ms (TTFT)", lambda x: x.first_token_ms)
    col("total_ms", lambda x: x.total_ms)
    col("server_queue_ms", lambda x: None if x.server_queue_s is None else x.server_queue_s * 1000)
    col("prompt_tokens", lambda x: x.prompt_tokens)
    col("completion_tokens", lambda x: x.completion_tokens)
    col("reasoning_tokens", lambda x: x.reasoning_tokens)
    ratios = [r.est_prompt_tokens / r.result.prompt_tokens for r in records
              if r.result is not None and r.result.prompt_tokens]
    if ratios:
        lines.append(f"  estimate/actual prompt tokens: min={min(ratios):.2f} max={max(ratios):.2f}")
    bad = [r for r in records if r.result is not None and not r.result.ok]
    if bad:
        kinds = sorted({f"{r.result.status}/{r.result.error_kind}" for r in bad})
        lines.append(f"  NOT ok: {len(bad)} ({', '.join(kinds)})")
    return lines


async def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--config", choices=sorted(RETRIEVAL_CONFIGS), default="B")
    ap.add_argument("--max-tokens", type=int, default=700, help="includes hidden reasoning tokens")
    ap.add_argument("--budget", type=int, default=30000, help="stop after this many ACTUAL tokens")
    ap.add_argument("--dry-run", action="store_true", help="no API calls")
    args = ap.parse_args(argv)

    settings = load_settings()
    retriever = Retriever.from_settings(settings, RETRIEVAL_CONFIGS[args.config])
    questions = PROBE_QUESTIONS[: args.n]

    if args.dry_run:
        total = 0
        for category, q in questions:
            messages, passages = build_messages(q, retriever.retrieve(q))
            est = sum(estimate_tokens(m["content"]) for m in messages)
            total += est
            print(f"est_prompt_tokens={est:>5}  passages={len(passages)}  [{category}] {q}")
        print(f"estimated PROMPT tokens for {len(questions)} requests: {total} (plus answers, up to {args.max_tokens} each)")
        return 0

    provider = GroqProvider(settings)
    log = RawLog.for_real(settings, "latency_probe")
    records: List[AnswerRecord] = []
    used = 0
    typical_completion = 300  # starting guess for pacing only; replaced by the largest completion seen
    try:
        for i, (category, q) in enumerate(questions):
            rec = await answer_question(provider, retriever, q, max_tokens=args.max_tokens)
            records.append(rec)
            r = rec.result
            if r is None:
                print(f"{i + 1:>2}. no passages matched, no call made: {q}")
                continue
            entry: Dict[str, Any] = result_record(r, source="real", label=f"latency_probe_{i + 1}",
                                                  model=settings.groq_model or "", messages=rec.messages)
            entry.update(question=q, category=category, config=rec.config, passages=rec.passages, est_prompt_tokens=rec.est_prompt_tokens,
                         cited=rec.cited, invalid_cited=rec.invalid_cited, said_not_found=rec.said_not_found)
            log.write(entry)
            actual = (r.prompt_tokens or 0) + (r.completion_tokens or 0)
            used += actual or rec.est_prompt_tokens
            ttft = "-" if r.first_token_ms is None else f"{r.first_token_ms:.0f}"
            print(f"{i + 1:>2}. {category[:6]:<6} {r.status:<12} ttft_ms={ttft:>6} total_ms={r.total_ms:>6.0f} "
                  f"prompt={r.prompt_tokens} completion={r.completion_tokens} reasoning={r.reasoning_tokens} "
                  f"remaining_tokens={r.ratelimit.remaining_tokens} cited={rec.cited} used_so_far={used}", flush=True)
            if r.status == "rate_limited":
                print(f"RATE LIMIT hit: limit_kind={r.limit_kind} retry_after_s={r.ratelimit.retry_after_s} "
                      f"message={r.error_message}")
                print("Stopping. Paste this output; do not rerun until we have looked at it.")
                break
            if used >= args.budget:
                print(f"Actual-token budget {args.budget} reached; stopping.")
                break
            if r.completion_tokens:
                typical_completion = max(typical_completion, r.completion_tokens)
            if i + 1 < len(questions):
                nxt_msgs, _ = build_messages(questions[i + 1][1], retriever.retrieve(questions[i + 1][1]))
                needed = sum(estimate_tokens(m["content"]) for m in nxt_msgs) + typical_completion
                wait = pace_seconds(r.ratelimit.remaining_tokens, needed, r.ratelimit.limit_tokens or DEFAULT_TPM)
                wait = max(wait, 3.0)  # also never faster than one request per 3 s
                await asyncio.sleep(wait)
    finally:
        await provider.aclose()

    print()
    for line in summarize(records):
        print(line)
    print(f"evidence appended to {log.path}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
