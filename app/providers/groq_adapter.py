"""Real Groq streaming adapter (OpenAI-compatible /chat/completions with stream=true).

What it measures per attempt: time to first chunk, time to first ANSWER token (TTFT),
total time, token usage (as reported), the provider's own queue/model time, and the
rate-limit headers. It does NOT retry; retry policy belongs to the scheduler.
"""
from __future__ import annotations

import asyncio
import json
import re
import time
from typing import Any, Dict, List, Mapping, Optional

import httpx

from app.config import Settings
from app.providers.base import (
    STATUS_ERROR,
    STATUS_OK,
    STATUS_RATE_LIMITED,
    STATUS_TIMEOUT,
    LLMRequest,
    LLMResult,
    RateLimitInfo,
    TokenCallback,
)

_DURATION_PART = re.compile(r"(\d+(?:\.\d+)?)(ms|d|h|m|s)")
_UNIT_SECONDS = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0}


def parse_duration(text: str) -> Optional[float]:
    """'1m26.4s' -> 86.4, '622ms' -> 0.622. Returns None if the text is not a duration."""
    text = text.strip()
    parts = _DURATION_PART.findall(text)
    if not parts or "".join(n + u for n, u in parts) != text:
        return None
    return sum(float(n) * _UNIT_SECONDS[u] for n, u in parts)


def _to_int(value: Optional[str]) -> Optional[int]:
    try:
        return int(value) if value is not None else None
    except ValueError:
        return None


def _to_float(value: Optional[str]) -> Optional[float]:
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None


def parse_ratelimit(headers: Mapping[str, str]) -> RateLimitInfo:
    h = {k.lower(): v for k, v in headers.items()}
    reset_r, reset_t = h.get("x-ratelimit-reset-requests"), h.get("x-ratelimit-reset-tokens")
    return RateLimitInfo(
        limit_requests=_to_int(h.get("x-ratelimit-limit-requests")),
        remaining_requests=_to_int(h.get("x-ratelimit-remaining-requests")),
        reset_requests_s=parse_duration(reset_r) if reset_r else None,
        limit_tokens=_to_int(h.get("x-ratelimit-limit-tokens")),
        remaining_tokens=_to_int(h.get("x-ratelimit-remaining-tokens")),
        reset_tokens_s=parse_duration(reset_t) if reset_t else None,
        retry_after_s=_to_float(h.get("retry-after")),
    )


def classify_limit(message: str) -> str:
    """Guess WHICH limit a 429 refers to from its message text.

    ASSUMPTION: the wording below is not yet confirmed against a real Groq 429.
    The first real 429 we capture will confirm or correct it.
    """
    m = message.lower()
    for needle, kind in (
        ("tokens per minute", "tpm"), ("(tpm)", "tpm"),
        ("requests per minute", "rpm"), ("(rpm)", "rpm"),
        ("tokens per day", "tpd"), ("(tpd)", "tpd"),
        ("requests per day", "rpd"), ("(rpd)", "rpd"),
    ):
        if needle in m:
            return kind
    return "unknown"


def _error_message(raw: bytes) -> str:
    text = raw.decode("utf-8", errors="replace")
    try:
        data = json.loads(text)
        message = data["error"]["message"]
        if isinstance(message, str):
            return message[:500]
    except (ValueError, KeyError, TypeError):
        pass
    return text[:300]


class _Trace:
    """Mutable record of one attempt. Survives a timeout so partial timings are kept."""

    def __init__(self) -> None:
        self.status: Optional[str] = None
        self.http_status: Optional[int] = None
        self.error_kind: Optional[str] = None
        self.limit_kind: Optional[str] = None
        self.error_message: Optional[str] = None
        self.text_parts: List[str] = []
        self.finish_reason: Optional[str] = None
        self.saw_done = False
        self.first_chunk_ms: Optional[float] = None
        self.first_token_ms: Optional[float] = None
        self.total_ms = 0.0
        self.usage: Optional[Dict[str, Any]] = None
        self.headers: Dict[str, str] = {}

    def fail(self, status: str, kind: str, message: str) -> None:
        if self.status is None:  # first failure wins
            self.status, self.error_kind, self.error_message = status, kind, message

    def fail_http(self, code: int, message: str) -> None:
        if code == 429:
            self.fail(STATUS_RATE_LIMITED, "rate_limit", message)
            self.limit_kind = classify_limit(message)
        elif code == 408:
            self.fail(STATUS_TIMEOUT, "timeout", message)
        elif code in (401, 403):
            self.fail(STATUS_ERROR, "auth", message)
        elif code >= 500:
            self.fail(STATUS_ERROR, "server", message)
        elif 400 <= code < 500:
            self.fail(STATUS_ERROR, "client", message)
        else:
            self.fail(STATUS_ERROR, "http", message)

    def conclude(self) -> None:
        """Decide whether a stream that ended normally is a complete, usable answer."""
        if self.status is not None:
            return
        if not self.saw_done or self.finish_reason is None:
            self.fail(STATUS_ERROR, "stream_interrupted", "stream ended without [DONE] or finish_reason")
        elif self.finish_reason == "length":
            self.fail(STATUS_ERROR, "truncated", "hit max tokens (reasoning tokens count too)")
        elif not "".join(self.text_parts).strip():
            self.fail(STATUS_ERROR, "empty_answer", "finished without any answer text")
        else:
            self.status = STATUS_OK

    def build(self) -> LLMResult:
        text = "".join(self.text_parts)
        complete = self.status == STATUS_OK
        usage = self.usage or {}
        details = usage.get("completion_tokens_details") or {}
        return LLMResult(
            status=self.status or STATUS_ERROR,
            text=text if complete else "",
            partial_text="" if complete else text,
            complete=complete,
            http_status=self.http_status,
            error_kind=self.error_kind,
            limit_kind=self.limit_kind,
            error_message=self.error_message,
            finish_reason=self.finish_reason,
            first_chunk_ms=self.first_chunk_ms,
            first_token_ms=self.first_token_ms,
            total_ms=self.total_ms,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            reasoning_tokens=details.get("reasoning_tokens"),
            server_queue_s=usage.get("queue_time"),
            server_total_s=usage.get("total_time"),
            ratelimit=parse_ratelimit(self.headers),
            raw_headers=dict(self.headers),
            raw_usage=self.usage,
        )


class GroqProvider:
    def __init__(
        self,
        settings: Settings,
        *,
        client: Optional[httpx.AsyncClient] = None,
        reasoning_effort: str = "low",
        include_reasoning: bool = False,
    ) -> None:
        if not settings.groq_api_key:
            raise ValueError("GROQ_API_KEY is not set")
        if not settings.groq_model:
            raise ValueError("GROQ_MODEL is not set")
        self._model = settings.groq_model
        self._reasoning_effort = reasoning_effort
        self._include_reasoning = include_reasoning
        self._auth = {"Authorization": f"Bearer {settings.groq_api_key}"}  # sent, never stored in results
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url=settings.groq_base_url, timeout=httpx.Timeout(10.0, read=60.0)
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def generate(self, req: LLMRequest, on_token: Optional[TokenCallback] = None) -> LLMResult:
        t0 = time.perf_counter()
        trace = _Trace()
        try:
            await asyncio.wait_for(self._stream(req, trace, t0, on_token), timeout=req.timeout_s)
        except asyncio.TimeoutError:
            trace.fail(STATUS_TIMEOUT, "timeout", f"no complete answer within {req.timeout_s}s")
        except httpx.TimeoutException:
            trace.fail(STATUS_TIMEOUT, "timeout", "network timeout")
        except httpx.HTTPError as exc:
            kind = "stream_interrupted" if trace.text_parts else "network"
            trace.fail(STATUS_ERROR, kind, type(exc).__name__)
        trace.total_ms = (time.perf_counter() - t0) * 1000
        return trace.build()

    async def _stream(self, req: LLMRequest, trace: _Trace, t0: float, on_token: Optional[TokenCallback]) -> None:
        body = {
            "model": self._model,
            "messages": req.messages,
            "stream": True,
            "stream_options": {"include_usage": True},
            "max_completion_tokens": req.max_tokens,
            "temperature": req.temperature,
            "reasoning_effort": self._reasoning_effort,
            "include_reasoning": self._include_reasoning,
        }
        async with self._client.stream("POST", "/chat/completions", json=body, headers=self._auth) as resp:
            trace.http_status = resp.status_code
            trace.headers = {
                k.lower(): v for k, v in resp.headers.items()
                if k.lower().startswith(("x-ratelimit", "retry-after"))
            }
            if resp.status_code != 200:
                trace.fail_http(resp.status_code, _error_message(await resp.aread()))
                return
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                now_ms = (time.perf_counter() - t0) * 1000
                if trace.first_chunk_ms is None:
                    trace.first_chunk_ms = now_ms
                payload = line[5:].strip()
                if payload == "[DONE]":
                    trace.saw_done = True
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    trace.fail(STATUS_ERROR, "bad_stream", "unparseable SSE chunk")
                    return
                self._absorb(chunk, trace, now_ms, on_token)
        trace.conclude()

    @staticmethod
    def _absorb(chunk: Dict[str, Any], trace: _Trace, now_ms: float, on_token: Optional[TokenCallback]) -> None:
        for choice in chunk.get("choices") or []:
            piece = (choice.get("delta") or {}).get("content")
            if piece:
                if trace.first_token_ms is None:
                    trace.first_token_ms = now_ms
                trace.text_parts.append(piece)
                if on_token is not None:
                    on_token(piece)
            if choice.get("finish_reason"):
                trace.finish_reason = choice["finish_reason"]
        usage = (chunk.get("x_groq") or {}).get("usage") or chunk.get("usage")
        if usage:
            trace.usage = usage
