"""Check your Groq key, list models, and capture rate-limit headers from ONE tiny request.

Run:  python -m app.check_groq            list the model IDs your key can use (no completion)
      python -m app.check_groq --probe    also send ONE tiny chat request to GROQ_MODEL
The key is read from the environment/.env and is never printed or saved. The probe counts
toward your daily quota. Header meanings are NOT interpreted here; we verify them against
Groq's documentation before putting anything in Table 1.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from typing import Dict, List, Mapping, Optional

import httpx

from app.config import load_settings

_KEEP_PREFIXES = ("x-ratelimit", "retry-after")


def pick_headers(headers: Mapping[str, str]) -> Dict[str, str]:
    """Keep only rate-limit related response headers (lower-cased names)."""
    return {k.lower(): v for k, v in headers.items() if k.lower().startswith(_KEEP_PREFIXES)}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Check Groq access and capture rate-limit headers.")
    parser.add_argument("--probe", action="store_true", help="send one tiny chat completion")
    args = parser.parse_args(argv)
    settings = load_settings()
    if not settings.groq_api_key:
        print("GROQ_API_KEY is not set. Put it in .env (copy .env.example) or export it in your shell.")
        return 2
    auth = {"Authorization": f"Bearer {settings.groq_api_key}"}
    try:
        with httpx.Client(base_url=settings.groq_base_url, headers=auth, timeout=30.0) as client:
            resp = client.get("/models")
            print(f"GET /models -> HTTP {resp.status_code}")
            if resp.status_code != 200:
                print(resp.text[:300])
                return 1
            ids = sorted(m["id"] for m in resp.json().get("data", []))
            print(f"{len(ids)} models available to your key:")
            for model_id in ids:
                print(f"  {model_id}")
            if not args.probe:
                return 0
            if not settings.groq_model:
                print("Set GROQ_MODEL (one of the IDs above) before using --probe.")
                return 2
            body = {
                "model": settings.groq_model,
                "messages": [{"role": "user", "content": "Reply with the single word: OK"}],
                "max_tokens": 5,
                "temperature": 0,
            }
            t0 = time.perf_counter()
            resp = client.post("/chat/completions", json=body)
            latency_ms = (time.perf_counter() - t0) * 1000
    except httpx.HTTPError as exc:
        print(f"network problem: {type(exc).__name__}")
        return 1

    headers = pick_headers(resp.headers)
    usage = resp.json().get("usage") if resp.status_code == 200 else None
    print(f"POST /chat/completions -> HTTP {resp.status_code} in {latency_ms:.0f} ms")
    print(f"usage: {usage}")
    for name in sorted(headers):
        print(f"  {name}: {headers[name]}")
    if resp.status_code != 200:
        print(resp.text[:300])
    settings.raw_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = settings.raw_dir / f"groq_check_{stamp}.json"
    out.write_text(
        json.dumps(
            {"model": settings.groq_model, "status": resp.status_code, "latency_ms": round(latency_ms, 1),
             "usage": usage, "headers": headers, "captured_at": stamp},
            indent=1,
        ),
        encoding="utf-8",
    )
    print(f"saved {out}")
    return 0 if resp.status_code == 200 else 1


if __name__ == "__main__":
    raise SystemExit(main())
