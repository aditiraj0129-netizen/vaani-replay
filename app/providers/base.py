"""Provider-neutral request/result types, shared by the real Groq adapter and the stand-in.

Rule used everywhere: `text` is filled ONLY for a complete answer. Anything that arrived
before a failure goes in `partial_text` and must never be shown as a final answer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol

STATUS_OK = "ok"
STATUS_RATE_LIMITED = "rate_limited"
STATUS_TIMEOUT = "timeout"
STATUS_ERROR = "error"


@dataclass(frozen=True)
class LLMRequest:
    messages: List[Dict[str, str]]
    max_tokens: int = 800  # includes hidden reasoning tokens on reasoning models
    temperature: float = 0.0
    timeout_s: float = 30.0  # wall-clock budget for this single attempt


@dataclass(frozen=True)
class RateLimitInfo:
    """Parsed x-ratelimit-* / retry-after headers. Groq documents: *-requests = per DAY, *-tokens = per MINUTE."""

    limit_requests: Optional[int] = None
    remaining_requests: Optional[int] = None
    reset_requests_s: Optional[float] = None
    limit_tokens: Optional[int] = None
    remaining_tokens: Optional[int] = None
    reset_tokens_s: Optional[float] = None
    retry_after_s: Optional[float] = None


@dataclass
class LLMResult:
    status: str  # ok | rate_limited | timeout | error
    text: str = ""
    partial_text: str = ""
    complete: bool = False
    http_status: Optional[int] = None
    error_kind: Optional[str] = None
    limit_kind: Optional[str] = None  # rpm | rpd | tpm | tpd | unknown (rate-limit errors only)
    error_message: Optional[str] = None
    finish_reason: Optional[str] = None
    first_chunk_ms: Optional[float] = None  # first SSE data line of any kind
    first_token_ms: Optional[float] = None  # first non-empty ANSWER text (what a person would see)
    total_ms: float = 0.0
    prompt_tokens: Optional[int] = None  # None = the service did not report it (never invented)
    completion_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None
    server_queue_s: Optional[float] = None  # the provider's own queue_time
    server_total_s: Optional[float] = None  # the provider's own model time
    ratelimit: RateLimitInfo = field(default_factory=RateLimitInfo)
    raw_headers: Dict[str, str] = field(default_factory=dict)
    raw_usage: Optional[Dict[str, Any]] = None

    @property
    def ok(self) -> bool:
        return self.status == STATUS_OK and self.complete

    @property
    def is_daily_limit(self) -> bool:
        return self.limit_kind in ("rpd", "tpd")


TokenCallback = Callable[[str], None]


class LLMProvider(Protocol):
    async def generate(self, req: LLMRequest, on_token: Optional[TokenCallback] = None) -> LLMResult:
        """`on_token` is for live display only. The only valid final answer is result.text when result.ok."""
        ...
