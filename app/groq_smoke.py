"""ONE tiny streaming request to the real Groq service, to check the adapter works live.

Run:  python -m app.groq_smoke
Cost: roughly 100-300 tokens and 1 request of your daily quota.
The result (including headers) is appended to data/raw/real/smoke.jsonl. The key is never logged.
"""
from __future__ import annotations

import asyncio
import sys

from app.config import load_settings
from app.providers.base import LLMRequest
from app.providers.groq_adapter import GroqProvider
from app.raw_log import RawLog, result_record


async def main() -> int:
    settings = load_settings()
    provider = GroqProvider(settings)  # raises a clear error if key/model are missing
    messages = [{"role": "user", "content": "Reply with exactly the word: OK"}]
    print("streamed text: ", end="", flush=True)
    try:
        result = await provider.generate(LLMRequest(messages=messages, max_tokens=200), on_token=lambda t: print(t, end="", flush=True))
    finally:
        await provider.aclose()
    print()
    r = result
    print(f"status={r.status} http={r.http_status} error_kind={r.error_kind} finish={r.finish_reason}")
    print(f"first_chunk_ms={r.first_chunk_ms} first_token_ms={r.first_token_ms} total_ms={r.total_ms:.1f}")
    print(f"prompt_tokens={r.prompt_tokens} completion_tokens={r.completion_tokens} reasoning_tokens={r.reasoning_tokens}")
    print(f"server_queue_s={r.server_queue_s} server_total_s={r.server_total_s}")
    print(f"ratelimit={r.ratelimit}")
    if r.error_message:
        print(f"error_message={r.error_message}")
    log = RawLog.for_real(settings, "smoke")
    log.write(result_record(r, source="real", label="smoke", model=settings.groq_model or "", messages=messages))
    print(f"evidence appended to {log.path}")
    return 0 if r.ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
