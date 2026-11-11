"""Step A: does Groq count max_completion_tokens against the TPM bucket when the request starts?

Sends 4 tiny requests that differ ONLY in max_tokens, waiting between them so the token
bucket refills (8000 tokens/min is about 133 tokens/s). Compares the bucket drop shown in the
headers with the tokens actually used.

Run:  python -m app.probe_reservation
Cost: 4 requests, about 400 actual tokens. Evidence: data/raw/real/reservation_probe.jsonl
"""
from __future__ import annotations

import asyncio
import sys

from app.config import load_settings
from app.providers.base import LLMRequest
from app.providers.groq_adapter import GroqProvider
from app.raw_log import RawLog, result_record

MAX_TOKENS = [50, 200, 800, 1500]
GAP_S = 15  # refill time for the biggest possible reservation: ~1600 tokens / 133 per s = 12 s
TPM = 8000


def show(value) -> str:
    return "-" if value is None else str(value)


async def main() -> int:
    settings = load_settings()
    provider = GroqProvider(settings)
    log = RawLog.for_real(settings, "reservation_probe")
    messages = [{"role": "user", "content": "Reply with exactly the word: OK"}]
    rows = []
    try:
        for i, m in enumerate(MAX_TOKENS):
            if i:
                await asyncio.sleep(GAP_S)
            r = await provider.generate(LLMRequest(messages=messages, max_tokens=m, timeout_s=30.0))
            log.write(result_record(r, source="real", label=f"reservation_max{m}", model=settings.groq_model or "", messages=messages))
            rows.append((m, r))
            print(f"max_tokens={m}: status={r.status} http={r.http_status} error_kind={r.error_kind}", flush=True)
            if r.status == "rate_limited":
                print("Got a rate limit; stopping. Paste this output.")
                break
    finally:
        await provider.aclose()

    print()
    print("max_tokens | status | prompt | completion | actual_total | remaining | bucket_drop | prompt+max_tokens")
    for m, r in rows:
        actual = None
        if r.prompt_tokens is not None and r.completion_tokens is not None:
            actual = r.prompt_tokens + r.completion_tokens
        rem = r.ratelimit.remaining_tokens
        drop = None if rem is None else TPM - rem
        reserved = None if r.prompt_tokens is None else r.prompt_tokens + m
        print(f"{m:>10} | {r.status:<6} | {show(r.prompt_tokens):>6} | {show(r.completion_tokens):>10} | "
              f"{show(actual):>12} | {show(rem):>9} | {show(drop):>11} | {show(reserved):>17}")
    print(f"evidence appended to {log.path}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
