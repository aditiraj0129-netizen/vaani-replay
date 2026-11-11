"""Stand-in LLM service: behaves like the real Groq service we measured, but runs on a fast simulated clock.

WHAT IS COPIED FROM REAL MEASUREMENTS (2026-11-10, openai/gpt-oss-20b, free tier):
  - token bucket: capacity 8,000, refill 8,000/60 per second, a request is admitted only if the bucket holds
    at least its token need; the 429 message and a whole-second Retry-After mimic the real one;
  - latency pieces: server queue time, time from queue end to first answer token, time after the first token,
    and completion lengths are SAMPLED from the 12 real Step B requests (EMPIRICAL_* below);
  - rejected (429) requests did not reduce the daily request count in the real headers.
WHAT IS ASSUMED (never measured, so least like the real thing):
  - behavior under CONCURRENCY (queue time grows by slowdown_per_inflight_ms per request already in flight);
  - the RPM limit (30/min) and the RPD refill (1/86.4 s) wording and behavior at the limit;
  - error and timeout rates (default 0; scenarios turn them on);
  - the answer text (a fixed sentence), so answer quality is NOT simulated.
"""
from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional

from app.answer import estimate_tokens
from app.clock import Clock
from app.providers.base import (
    STATUS_ERROR, STATUS_OK, STATUS_RATE_LIMITED, STATUS_TIMEOUT,
    LLMRequest, LLMResult, RateLimitInfo, TokenCallback,
)
from app.providers.groq_adapter import parse_ratelimit

# Real Step B samples (latency_probe.jsonl). Seconds / tokens.
EMPIRICAL_QUEUE_S = [0.164, 0.704, 0.157, 1.875, 0.540, 0.374, 0.374, 0.283, 0.674, 1.272, 0.373, 0.411]
EMPIRICAL_TTFT_AFTER_QUEUE_S = [0.930, 0.703, 1.020, 0.489, 0.531, 0.395, 0.634, 0.942, 0.476, 0.386, 0.534, 0.722]
EMPIRICAL_AFTER_FIRST_TOKEN_S = [0.007, 0.119, 0.162, 0.024, 0.024, 0.161, 0.061, 0.008, 0.066, 0.013, 0.015, 0.004]
EMPIRICAL_COMPLETION_TOKENS = [53, 133, 119, 75, 52, 155, 136, 51, 124, 22, 26, 23]
REJECT_LATENCY_S = 0.2  # real 429 replies took 0.18 to 0.27 s
STANDIN_TEXT = "Stand-in answer used for load testing only. [1]"


@dataclass
class StandInConfig:
    tpm: int = 8000
    rpm: int = 30
    rpd: int = 1000
    admit_extra_tokens: int = 133  # real 'Requested' exceeded the measured prompt by about this much (unexplained)
    slowdown_per_inflight_ms: float = 150.0  # ASSUMPTION: queue time grows with requests already in flight
    error_rate: float = 0.0  # chance of an HTTP 503
    timeout_rate: float = 0.0  # chance the request hangs until the caller's timeout
    retry_after_mode: str = "ceil"  # real hints were whole seconds and once slightly too short ("floor")
    seed: int = 1234


class StandInProvider:
    def __init__(self, clock: Clock, config: Optional[StandInConfig] = None) -> None:
        self.clock = clock
        self.cfg = config or StandInConfig()
        self._rng = random.Random(self.cfg.seed)
        self._bucket = float(self.cfg.tpm)
        self._rpd_bucket = float(self.cfg.rpd)
        self._last = clock.now()
        self._minute: Deque[float] = deque()
        self.in_flight = 0
        self.log: List[Dict[str, float]] = []  # one record per call, for calibration checks

    # ----- limiter state -----
    def _refill(self) -> None:
        now = self.clock.now()
        dt = max(0.0, now - self._last)
        self._last = now
        self._bucket = min(float(self.cfg.tpm), self._bucket + dt * self.cfg.tpm / 60.0)
        self._rpd_bucket = min(float(self.cfg.rpd), self._rpd_bucket + dt * self.cfg.rpd / 86400.0)
        while self._minute and now - self._minute[0] >= 60.0:
            self._minute.popleft()

    def _hint(self, seconds: float) -> float:
        if self.cfg.retry_after_mode == "floor":
            return float(max(1, math.floor(seconds)))
        return float(max(1, math.ceil(seconds)))

    def _headers(self) -> Dict[str, str]:
        missing = self.cfg.tpm - self._bucket
        return {
            "x-ratelimit-limit-requests": str(self.cfg.rpd),
            "x-ratelimit-limit-tokens": str(self.cfg.tpm),
            "x-ratelimit-remaining-requests": str(int(self._rpd_bucket)),
            "x-ratelimit-remaining-tokens": str(int(self._bucket)),
            "x-ratelimit-reset-requests": f"{86400.0 / self.cfg.rpd:.1f}s",
            "x-ratelimit-reset-tokens": f"{missing / (self.cfg.tpm / 60.0):.3f}s",
        }

    def _reject(self, kind: str, message: str, wait_s: float, t0: float) -> LLMResult:
        headers = self._headers()
        headers["retry-after"] = str(int(self._hint(wait_s)))
        return LLMResult(
            status=STATUS_RATE_LIMITED, http_status=429, error_kind="rate_limit", limit_kind=kind,
            error_message=message, total_ms=0.0, ratelimit=parse_ratelimit(headers), raw_headers=headers,
        )

    # ----- the call -----
    async def generate(self, req: LLMRequest, on_token: Optional[TokenCallback] = None) -> LLMResult:
        t0 = self.clock.now()
        self._refill()
        prompt_tokens = sum(estimate_tokens(m["content"]) for m in req.messages)
        completion = self._rng.choice(EMPIRICAL_COMPLETION_TOKENS)
        need = prompt_tokens + self.cfg.admit_extra_tokens

        # admission, in the order the real service reports: per-minute requests, daily requests, tokens
        rejection: Optional[LLMResult] = None
        if len(self._minute) >= self.cfg.rpm:
            wait = 60.0 - (t0 - self._minute[0])
            rejection = self._reject("rpm", f"Rate limit reached for requests per minute (RPM): Limit {self.cfg.rpm}. "
                                            f"Please try again in {wait:.2f}s.", wait, t0)
        elif self._rpd_bucket < 1.0:
            wait = (1.0 - self._rpd_bucket) * 86400.0 / self.cfg.rpd
            rejection = self._reject("rpd", f"Rate limit reached for requests per day (RPD): Limit {self.cfg.rpd}. "
                                            f"Please try again in {wait:.2f}s.", wait, t0)
        elif self._bucket < need:
            wait = (need - self._bucket) / (self.cfg.tpm / 60.0)
            rejection = self._reject("tpm", f"Rate limit reached for model `stand-in` on tokens per minute (TPM): "
                                            f"Limit {self.cfg.tpm}, Used {int(self.cfg.tpm - self._bucket)}, "
                                            f"Requested {need}. Please try again in {wait:.2f}s.", wait, t0)
        if rejection is not None:
            await self.clock.sleep(REJECT_LATENCY_S)
            rejection.total_ms = (self.clock.now() - t0) * 1000
            self.log.append({"t": t0, "outcome": 429.0})
            return rejection  # a rejected request costs nothing (real headers: remaining requests unchanged)

        # admitted: charge now
        self._bucket -= prompt_tokens + completion
        self._rpd_bucket -= 1.0
        self._minute.append(t0)
        headers = self._headers()
        ahead = self.in_flight
        self.in_flight += 1
        try:
            roll = self._rng.random()
            if roll < self.cfg.error_rate:
                await self.clock.sleep(0.3)
                return LLMResult(status=STATUS_ERROR, http_status=503, error_kind="server",
                                 error_message="stand-in injected 503", total_ms=(self.clock.now() - t0) * 1000,
                                 ratelimit=parse_ratelimit(headers), raw_headers=headers)
            queue = self._rng.choice(EMPIRICAL_QUEUE_S) + ahead * self.cfg.slowdown_per_inflight_ms / 1000.0
            to_first = queue + self._rng.choice(EMPIRICAL_TTFT_AFTER_QUEUE_S)
            tail = self._rng.choice(EMPIRICAL_AFTER_FIRST_TOKEN_S)
            if roll < self.cfg.error_rate + self.cfg.timeout_rate or to_first + tail > req.timeout_s:
                await self.clock.sleep(req.timeout_s)
                return LLMResult(status=STATUS_TIMEOUT, error_kind="timeout",
                                 error_message=f"no complete answer within {req.timeout_s}s",
                                 total_ms=(self.clock.now() - t0) * 1000, ratelimit=parse_ratelimit(headers),
                                 raw_headers=headers)
            await self.clock.sleep(to_first)
            first_ms = (self.clock.now() - t0) * 1000
            pieces = STANDIN_TEXT.split(" ")
            for i, word in enumerate(pieces):
                if on_token is not None:
                    on_token(word + ("" if i == len(pieces) - 1 else " "))
                await self.clock.sleep(tail / len(pieces))
            return LLMResult(
                status=STATUS_OK, text=STANDIN_TEXT, complete=True, http_status=200, finish_reason="stop",
                first_chunk_ms=first_ms, first_token_ms=first_ms, total_ms=(self.clock.now() - t0) * 1000,
                prompt_tokens=prompt_tokens, completion_tokens=completion, reasoning_tokens=completion // 3,
                server_queue_s=queue, ratelimit=parse_ratelimit(headers), raw_headers=headers,
            )
        finally:
            self.in_flight -= 1
