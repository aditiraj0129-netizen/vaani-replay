"""The two ways of serving a stream of questions, run on identical traces so they can be compared.

BASELINE ("Architecture A", uncontrolled): every question is sent the moment it arrives, with no queue, no limit
on concurrent calls and no deadline. A failed call is retried after a fixed short pause, at most `max_attempts`
times in total. Optionally it honors Retry-After ("polite" variant), which separates good retry manners from
the other controls. It is still memory-safe: it keeps only one small record per question.

CONTROLLED ("Architecture B", bounded async pipeline with admission control):
  - bounded queue: a question arriving at a full queue is refused at once (shed_queue_full);
  - fixed pool of workers, so concurrency is bounded;
  - local token limiter (a mirror of the provider's token bucket, kept `margin` below it) and an independent
    requests-per-minute limiter; both are corrected from the provider's own x-ratelimit-* headers;
  - deadlines: a question whose estimated wait already exceeds its deadline is refused on arrival (not queued,
    not sent late); one that runs out of time while waiting is dropped before sending;
  - on a 429: every worker pauses for Retry-After + cushion + random jitter (no retry storms);
  - 5xx / timeouts: exponential backoff with full jitter, and a circuit breaker that pauses all sending
    after `breaker_failures` failures in a row;
  - a daily budget guard: when the provider reports a daily limit (or our own budget is used up), questions are
    refused at once instead of being retried;
  - one retrieval degradation: when a question would wait long for tokens, fewer passages go in the prompt
    (shorter prompts = fewer tokens = more answers per minute);
  - telemetry: one JobRecord per question, with every number needed for the report.

All waiting uses the Clock, so the same code runs against the real service (speed 1) and the stand-in (fast).
"""
from __future__ import annotations

import asyncio
import random
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any, Deque, Dict, List, Optional, Sequence

from app.answer import build_messages, estimate_tokens
from app.rag import analyze, verify
from app.clock import Clock
from app.providers.base import (
    STATUS_OK, STATUS_RATE_LIMITED, LLMRequest, LLMResult,
)

OK = "ok"                      # complete answer inside the deadline
OK_LATE = "ok_late"            # complete answer, but after the deadline (counts as a miss)
SHED_QUEUE_FULL = "shed_queue_full"
SHED_DEADLINE = "shed_deadline"
BUDGET = "budget_exhausted"
FAILED = "failed"              # attempts used up


@dataclass(frozen=True)
class Job:
    job_id: int
    question: str
    arrival_s: float   # seconds from the start of the run
    deadline_s: float  # useful only if answered within this many seconds of arrival


@dataclass
class JobRecord:
    job_id: int
    question: str
    arrival_s: float
    deadline_s: float
    outcome: str = ""
    attempts: int = 0            # calls actually sent to the provider
    n_429: int = 0
    n_other_failures: int = 0    # timeouts and server errors
    degraded: bool = False       # fewer passages were used because the queue was long
    first_send_s: Optional[float] = None
    end_s: Optional[float] = None
    queue_wait_s: Optional[float] = None   # arrival -> first send
    latency_s: Optional[float] = None      # arrival -> end (answer or refusal to answer)
    ttft_from_arrival_s: Optional[float] = None  # arrival -> first answer token of the successful call
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    citation_status: str = "n/a"
    unsupported_numbers: List[str] = field(default_factory=list)  # numbers in the answer not found in its cited passages
    paper_hint: List[str] = field(default_factory=list)  # papers the question names (step 1)
    last_status: str = ""          # status of the last call sent: ok | rate_limited | timeout | error
    answer: str = ""
    cited: List[int] = field(default_factory=list)
    passages: List[Dict[str, Any]] = field(default_factory=list)


class CountingProvider:
    """Wraps any provider and counts what it was asked, so both modes are measured the same way."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.calls = 0
        self.by_status: Dict[str, int] = {}
        self.tokens_ok = 0
        self.in_flight = 0
        self.peak_in_flight = 0

    async def generate(self, req: LLMRequest, on_token: Any = None) -> LLMResult:
        self.calls += 1
        self.in_flight += 1
        self.peak_in_flight = max(self.peak_in_flight, self.in_flight)
        try:
            r = await self.inner.generate(req, on_token)
        finally:
            self.in_flight -= 1
        self.by_status[r.status] = self.by_status.get(r.status, 0) + 1
        if r.ok:
            self.tokens_ok += (r.prompt_tokens or 0) + (r.completion_tokens or 0)
        return r


def _finish_ok(rec: JobRecord, r: LLMResult, send_s: float, end_s: float, passages: List[Dict[str, Any]]) -> None:
    rec.end_s = end_s
    rec.latency_s = end_s - rec.arrival_s
    rec.outcome = OK if rec.latency_s <= rec.deadline_s else OK_LATE
    rec.ttft_from_arrival_s = (send_s - rec.arrival_s) + (r.first_token_ms or 0.0) / 1000.0
    rec.prompt_tokens, rec.completion_tokens = r.prompt_tokens, r.completion_tokens
    rec.answer = r.text
    rec.passages = passages
    v = verify(r.text, [p["text"] for p in passages])  # step 4: citations real? numbers supported?
    rec.cited, rec.citation_status, rec.unsupported_numbers = v.cited, v.status, v.unsupported_numbers


def _finish_other(rec: JobRecord, outcome: str, now: float) -> JobRecord:
    rec.outcome = outcome
    rec.end_s = now
    rec.latency_s = now - rec.arrival_s
    return rec


# --------------------------------------------------------------------------------------------- baseline
@dataclass
class BaselineConfig:
    max_attempts: int = 5
    retry_delay_s: float = 0.5
    honor_retry_after: bool = False  # True = the "polite" variant
    max_retry_after_s: float = 60.0  # the polite variant never waits longer than this (a daily-limit hint is hours)
    timeout_s: float = 30.0
    max_tokens: int = 700
    use_paper_hint: bool = False  # steps 1-2: restrict retrieval to the paper the question names


def _retrieve(retriever: Any, question: str, use_hint: bool):
    """Steps 1 and 2: analyze the question, then call the retrieval tool."""
    plan = analyze(question)
    papers = plan.papers if use_hint and plan.single_paper else None
    return plan, (retriever.retrieve(question, papers) if papers else retriever.retrieve(question))


async def _baseline_job(job: Job, provider: Any, retriever: Any, clock: Clock, cfg: BaselineConfig) -> JobRecord:
    rec = JobRecord(job.job_id, job.question, job.arrival_s, job.deadline_s)
    plan, hits = _retrieve(retriever, job.question, cfg.use_paper_hint)
    rec.paper_hint = plan.papers
    messages, passages = build_messages(job.question, hits)
    for attempt in range(cfg.max_attempts):
        send_s = clock.now()
        if rec.first_send_s is None:
            rec.first_send_s = send_s
            rec.queue_wait_s = send_s - job.arrival_s
        rec.attempts += 1
        r = await provider.generate(LLMRequest(messages=messages, max_tokens=cfg.max_tokens, timeout_s=cfg.timeout_s))
        if r.ok:
            _finish_ok(rec, r, send_s, clock.now(), passages)
            return rec
        rec.last_status = r.status
        if r.status == STATUS_RATE_LIMITED:
            rec.n_429 += 1
        else:
            rec.n_other_failures += 1
        if attempt + 1 < cfg.max_attempts:
            pause = cfg.retry_delay_s
            if cfg.honor_retry_after and r.status == STATUS_RATE_LIMITED and r.ratelimit.retry_after_s:
                pause = max(pause, min(r.ratelimit.retry_after_s, cfg.max_retry_after_s))
            await clock.sleep(pause)
    return _finish_other(rec, FAILED, clock.now())


async def run_baseline(trace: Sequence[Job], provider: Any, retriever: Any, clock: Clock,
                       cfg: Optional[BaselineConfig] = None) -> List[JobRecord]:
    cfg = cfg or BaselineConfig()
    tasks: List["asyncio.Task[JobRecord]"] = []
    for job in sorted(trace, key=lambda j: j.arrival_s):
        await clock.sleep(job.arrival_s - clock.now())
        tasks.append(asyncio.create_task(_baseline_job(job, provider, retriever, clock, cfg)))
    return list(await asyncio.gather(*tasks))


# ------------------------------------------------------------------------------------------- controlled
@dataclass
class ControlConfig:
    workers: int = 4
    max_queue: int = 30
    tpm: int = 8000                 # what we believe the limit is; corrected from response headers
    rpm: int = 30
    margin: float = 0.15            # we stay this fraction below the provider's limits
    completion_guess: int = 150     # tokens we expect an answer to use (real max seen: 155, median 64)
    extra_tokens: int = 133         # real 'Requested' exceeded the measured prompt by about this much
    daily_token_budget: Optional[int] = None
    max_retries: int = 3            # extra attempts after the first
    cushion_s: float = 0.5          # added to Retry-After (real hints were sometimes 0.1 to 1 s short)
    jitter_s: float = 0.5           # plus a random 0..jitter_s so workers do not wake together
    backoff_base_s: float = 1.0
    backoff_cap_s: float = 8.0
    breaker_failures: int = 5
    breaker_open_s: float = 10.0
    use_paper_hint: bool = False    # steps 1-2: restrict retrieval to the paper the question names
    degrade_wait_s: float = 15.0    # if a question would wait longer than this, use fewer passages
    degraded_k: int = 3
    min_service_s: float = 2.5      # a call needs at least this long (real worst case so far: 2.4 s)
    timeout_s: float = 30.0
    max_tokens: int = 700
    seed: int = 1


class Gate:
    """Decides when the next call may be sent. Holds the local token and request limiters and every pause."""

    def __init__(self, clock: Clock, cfg: ControlConfig, rng: random.Random) -> None:
        self.clock, self.cfg, self.rng = clock, cfg, rng
        self.tpm = cfg.tpm
        self.level = self.capacity
        self._last = clock.now()
        self.sent: Deque[float] = deque()
        self.blocked_until = 0.0
        self.daily_until = 0.0
        self.tokens_today = 0
        self.fail_streak = 0
        self.cooldowns = 0
        self.breaker_opens = 0
        self.lock = asyncio.Lock()

    @property
    def capacity(self) -> float:
        return self.tpm * (1.0 - self.cfg.margin)

    @property
    def margin_tokens(self) -> float:
        return self.tpm * self.cfg.margin

    def refill(self) -> None:
        now = self.clock.now()
        self.level = min(self.capacity, self.level + (now - self._last) * self.tpm / 60.0)
        self._last = now
        while self.sent and now - self.sent[0] >= 60.0:
            self.sent.popleft()

    def wait_needed(self, need: float) -> float:
        now = self.clock.now()
        waits = [self.blocked_until - now, (need - self.level) * 60.0 / self.tpm]
        rpm_cap = max(1, int(self.cfg.rpm * (1.0 - self.cfg.margin)))
        if len(self.sent) >= rpm_cap:
            waits.append(self.sent[0] + 60.0 - now)
        return max(0.0, *waits)

    def daily_blocked(self, need: float) -> bool:
        if self.clock.now() < self.daily_until:
            return True
        budget = self.cfg.daily_token_budget
        return budget is not None and self.tokens_today + need > budget

    async def acquire(self, need: float, deadline_abs: float) -> Optional[str]:
        """None = go ahead (tokens and a request slot are reserved). Otherwise the reason not to send."""
        need = min(need, self.capacity)
        async with self.lock:  # one waiter at a time keeps the order fair
            while True:
                self.refill()
                if self.daily_blocked(need):
                    return BUDGET
                wait = self.wait_needed(need)
                if wait <= 0:
                    if self.clock.now() + self.cfg.min_service_s > deadline_abs:
                        return SHED_DEADLINE  # waiting used up the time; do not send a call that cannot finish
                    self.level -= need
                    self.sent.append(self.clock.now())
                    return None
                if self.clock.now() + wait + self.cfg.min_service_s > deadline_abs:
                    return SHED_DEADLINE
                await self.clock.sleep(min(wait, 1.0) + 0.01)

    def on_result(self, r: LLMResult, need: float) -> None:
        need = min(need, self.capacity)
        now = self.clock.now()
        limit = r.ratelimit.limit_tokens
        if limit and limit != self.tpm:  # the provider told us the real limit (e.g. it was halved)
            self.tpm = limit
            self.level = min(self.level, self.capacity)
        if r.ok:
            actual = (r.prompt_tokens or 0) + (r.completion_tokens or 0)
            self.level = min(self.capacity, self.level + max(0.0, need - actual))
            self.tokens_today += actual
            self.fail_streak = 0
        elif r.status == STATUS_RATE_LIMITED:
            self.level = min(self.capacity, self.level + need)  # a rejected call is not charged
            hint = r.ratelimit.retry_after_s if r.ratelimit.retry_after_s is not None else 2.0
            pause = hint + self.cfg.cushion_s + self.rng.uniform(0.0, self.cfg.jitter_s)
            self.blocked_until = max(self.blocked_until, now + pause)
            self.cooldowns += 1
            if r.is_daily_limit:
                self.daily_until = max(self.daily_until, now + hint)
        else:
            self.fail_streak += 1
            if self.fail_streak >= self.cfg.breaker_failures:
                self.blocked_until = max(self.blocked_until, now + self.cfg.breaker_open_s)
                self.fail_streak = self.cfg.breaker_failures - 1  # half-open: one more failure reopens
                self.breaker_opens += 1
        remaining = r.ratelimit.remaining_tokens
        if remaining is not None:  # the provider knows best; never trust our own count over a lower header
            self.level = min(self.level, remaining - self.margin_tokens)


class Controller:
    def __init__(self, provider: Any, retriever: Any, clock: Clock, cfg: Optional[ControlConfig] = None) -> None:
        self.provider, self.retriever, self.clock = provider, retriever, clock
        self.cfg = cfg or ControlConfig()
        self.rng = random.Random(self.cfg.seed)
        self.gate = Gate(clock, self.cfg, self.rng)
        self.queue: "asyncio.Queue[tuple]" = asyncio.Queue()
        self.records: List[JobRecord] = []
        self.max_queue_depth = 0
        self.busy = 0  # workers currently holding a question (waiting for tokens or waiting for the provider)

    def _need(self, messages: List[Dict[str, str]]) -> int:
        return sum(estimate_tokens(m["content"]) for m in messages) + self.cfg.completion_guess + self.cfg.extra_tokens

    def _estimated_wait_s(self, need: int) -> float:
        """How long a new question probably waits for tokens: everything already ahead of it, at our token rate."""
        ahead = self.queue.qsize() + self.busy
        missing = (ahead + 1) * need - self.gate.level
        return max(self.gate.blocked_until - self.clock.now(), missing * 60.0 / self.gate.tpm, 0.0)

    def submit(self, job: Job) -> None:
        cfg = self.cfg
        rec = JobRecord(job.job_id, job.question, job.arrival_s, job.deadline_s)
        now = self.clock.now()
        if self.queue.qsize() >= cfg.max_queue:
            self.records.append(_finish_other(rec, SHED_QUEUE_FULL, now))
            return
        if self.gate.daily_blocked(0):
            self.records.append(_finish_other(rec, BUDGET, now))
            return
        plan, hits = _retrieve(self.retriever, job.question, cfg.use_paper_hint)
        rec.paper_hint = plan.papers
        messages, passages = build_messages(job.question, hits)
        need = self._need(messages)
        wait = self._estimated_wait_s(need)
        if wait > cfg.degrade_wait_s:  # the one retrieval degradation: fewer passages, shorter prompt
            messages, passages = build_messages(job.question, hits[: cfg.degraded_k])
            need = self._need(messages)
            rec.degraded = True
            wait = self._estimated_wait_s(need)
        if now + wait + cfg.min_service_s > job.arrival_s + job.deadline_s:  # it cannot make its deadline: say no now
            self.records.append(_finish_other(rec, SHED_DEADLINE, now))
            return
        self.queue.put_nowait((job, rec, messages, passages, need))
        self.max_queue_depth = max(self.max_queue_depth, self.queue.qsize())

    async def _worker(self) -> None:
        while True:
            item = await self.queue.get()
            self.busy += 1
            try:
                self.records.append(await self._handle(*item))
            finally:
                self.busy -= 1
                self.queue.task_done()

    async def _handle(self, job: Job, rec: JobRecord, messages: List[Dict[str, str]],
                      passages: List[Dict[str, Any]], need: int) -> JobRecord:
        cfg, clock = self.cfg, self.clock
        deadline_abs = job.arrival_s + job.deadline_s  # Gate.acquire refuses when too little time is left
        for attempt in range(cfg.max_retries + 1):
            reason = await self.gate.acquire(need, deadline_abs)
            if reason is not None:
                return _finish_other(rec, reason, clock.now())
            send_s = clock.now()
            if rec.first_send_s is None:
                rec.first_send_s = send_s
                rec.queue_wait_s = send_s - job.arrival_s
            timeout = min(cfg.timeout_s, deadline_abs - send_s)  # acquire() guarantees at least min_service_s
            rec.attempts += 1
            r = await self.provider.generate(LLMRequest(messages=messages, max_tokens=cfg.max_tokens, timeout_s=timeout))
            self.gate.on_result(r, need)
            rec.last_status = r.status
            if r.ok:
                _finish_ok(rec, r, send_s, clock.now(), passages)
                return rec
            if r.status == STATUS_RATE_LIMITED:
                rec.n_429 += 1
            else:
                rec.n_other_failures += 1
                await clock.sleep(self.rng.uniform(0.0, min(cfg.backoff_cap_s, cfg.backoff_base_s * 2 ** attempt)))
        return _finish_other(rec, FAILED, clock.now())


async def run_controlled(trace: Sequence[Job], provider: Any, retriever: Any, clock: Clock,
                         cfg: Optional[ControlConfig] = None) -> "tuple[List[JobRecord], Controller]":
    ctl = Controller(provider, retriever, clock, cfg)
    workers = [asyncio.create_task(ctl._worker()) for _ in range(ctl.cfg.workers)]
    for job in sorted(trace, key=lambda j: j.arrival_s):
        await clock.sleep(job.arrival_s - clock.now())
        ctl.submit(job)
    await ctl.queue.join()
    for w in workers:
        w.cancel()
    await asyncio.gather(*workers, return_exceptions=True)
    return ctl.records, ctl


# ---------------------------------------------------------------------------------------------- summary
def _pct(values: List[float], q: float) -> Optional[float]:
    if not values:
        return None
    s = sorted(values)
    return s[min(len(s) - 1, int(round(q * (len(s) - 1))))]


def summarize_run(records: Sequence[JobRecord], counting: CountingProvider, controller: Optional[Controller]) -> Dict[str, Any]:
    n = len(records)
    count = lambda o: sum(r.outcome == o for r in records)  # noqa: E731
    answered = [r for r in records if r.outcome in (OK, OK_LATE)]
    shed = [r for r in records if r.outcome in (SHED_QUEUE_FULL, SHED_DEADLINE, BUDGET)]
    calls = counting.calls
    wasted = calls - counting.by_status.get(STATUS_OK, 0)
    return {
        "jobs": n,
        "ok_on_time": count(OK), "ok_late": count(OK_LATE), "failed": count(FAILED),
        "shed_queue_full": count(SHED_QUEUE_FULL), "shed_deadline": count(SHED_DEADLINE), "budget_exhausted": count(BUDGET),
        "goodput": count(OK) / n if n else 0.0,
        "latency_p50_s": _pct([r.latency_s for r in answered], 0.5), "latency_p95_s": _pct([r.latency_s for r in answered], 0.95),
        "ttft_p50_s": _pct([r.ttft_from_arrival_s for r in answered if r.ttft_from_arrival_s is not None], 0.5),
        "ttft_p95_s": _pct([r.ttft_from_arrival_s for r in answered if r.ttft_from_arrival_s is not None], 0.95),
        "time_to_refusal_p50_s": _pct([r.latency_s for r in shed], 0.5),
        "calls": calls, "calls_429": counting.by_status.get(STATUS_RATE_LIMITED, 0), "calls_wasted": wasted,
        "wasted_fraction": wasted / calls if calls else 0.0,
        "tokens_ok": counting.tokens_ok, "peak_in_flight": counting.peak_in_flight,
        "degraded_jobs": sum(r.degraded for r in records),
        "max_queue_depth": controller.max_queue_depth if controller else None,
        "cooldowns": controller.gate.cooldowns if controller else None,
        "breaker_opens": controller.gate.breaker_opens if controller else None,
        "uncited_answers": sum(r.citation_status == "uncited" for r in records),
        "answers_with_unsupported_numbers": sum(bool(r.unsupported_numbers) for r in records),
    }


def record_dict(rec: JobRecord) -> Dict[str, Any]:
    return asdict(rec)
