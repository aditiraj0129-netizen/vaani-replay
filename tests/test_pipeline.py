import asyncio

import pytest

from app.clock import Clock
from app.pipeline import (
    BUDGET, FAILED, OK, OK_LATE, SHED_DEADLINE, SHED_QUEUE_FULL, BaselineConfig, ControlConfig, Controller, CountingProvider, Gate,
    Job, run_baseline, run_controlled, summarize_run,
)
from app.providers.base import LLMResult, RateLimitInfo
from app.scenarios import SCENARIOS, FakeRetriever, make_trace, run_one

SPEED = 100.0


def ok_result(text="An answer. [1]", prompt=1500, completion=60, remaining=None, limit=None):
    return LLMResult(status="ok", text=text, complete=True, http_status=200, first_token_ms=800.0, total_ms=900.0,
                     prompt_tokens=prompt, completion_tokens=completion,
                     ratelimit=RateLimitInfo(limit_tokens=limit, remaining_tokens=remaining))


def rl_result(retry_after=5.0, kind="tpm", remaining=None):
    return LLMResult(status="rate_limited", http_status=429, error_kind="rate_limit", limit_kind=kind,
                     ratelimit=RateLimitInfo(retry_after_s=retry_after, remaining_tokens=remaining))


def err_result():
    return LLMResult(status="error", http_status=503, error_kind="server")


class Scripted:
    """Returns the scripted results in order (the last one repeats). Remembers when and with what it was called."""

    def __init__(self, clock, results, latency_s=1.0):
        self.clock, self.results, self.latency_s = clock, list(results), latency_s
        self.times, self.requests = [], []

    async def generate(self, req, on_token=None):
        self.times.append(self.clock.now())
        self.requests.append(req)
        await self.clock.sleep(self.latency_s)
        i = len(self.times) - 1
        return self.results[min(i, len(self.results) - 1)]


def job(i=0, t=0.0, deadline=600.0):
    return Job(i, "What is attention?", t, deadline)


def run(coro):
    return asyncio.run(coro)


def controlled(trace, provider, clock, **cfg):
    return run_controlled(trace, provider, FakeRetriever(), clock, ControlConfig(**cfg))


# ---------- baseline ----------
def test_baseline_retries_after_a_short_fixed_pause_and_ignores_retry_after():
    async def go():
        c = Clock(SPEED)
        p = Scripted(c, [rl_result(retry_after=30.0), ok_result()])
        recs = await run_baseline([job()], p, FakeRetriever(), c)
        return p, recs
    p, recs = run(go())
    assert recs[0].outcome == OK and recs[0].attempts == 2 and recs[0].n_429 == 1
    assert p.times[1] - p.times[0] < 5.0  # waited about 0.5 s + the call itself, not 30 s


def test_polite_baseline_waits_for_retry_after_but_never_more_than_its_cap():
    async def go(hint):
        c = Clock(SPEED)
        p = Scripted(c, [rl_result(retry_after=hint), ok_result()])
        await run_baseline([job()], p, FakeRetriever(), c, BaselineConfig(honor_retry_after=True))
        return p.times[1] - p.times[0]
    assert run(go(20.0)) >= 20.0
    assert 60.0 <= run(go(9000.0)) < 70.0  # capped at 60 s


def test_baseline_gives_up_after_max_attempts():
    async def go():
        c = Clock(SPEED)
        p = Scripted(c, [rl_result(retry_after=1.0)])
        return p, await run_baseline([job()], p, FakeRetriever(), c, BaselineConfig(max_attempts=3))
    p, recs = run(go())
    assert recs[0].outcome == FAILED and len(p.times) == 3 and recs[0].n_429 == 3


# ---------- controlled ----------
def test_controller_waits_retry_after_plus_cushion_before_retrying():
    async def go():
        c = Clock(SPEED)
        p = Scripted(c, [rl_result(retry_after=5.0), ok_result()])
        recs, _ = await controlled([job()], p, c)
        return p, recs
    p, recs = run(go())
    assert recs[0].outcome == OK and recs[0].attempts == 2 and recs[0].n_429 == 1
    assert p.times[1] - p.times[0] >= 5.0 + 0.5 + 1.0 - 0.3  # hint + cushion (+ the first call's own 1 s)


def test_daily_limit_stops_all_sending():
    async def go():
        c = Clock(SPEED)
        p = Scripted(c, [rl_result(retry_after=10000.0, kind="rpd")])
        recs, ctl = await controlled([job(0, 0.0), job(1, 10.0), job(2, 20.0)], p, c)
        return p, recs
    p, recs = run(go())
    assert len(p.times) == 1  # one call told us; nothing else was sent
    assert {r.outcome for r in recs} == {BUDGET}


def test_own_daily_token_budget_refuses_when_it_would_be_exceeded():
    async def go():
        c = Clock(SPEED)
        p = Scripted(c, [ok_result(prompt=1500, completion=100)])
        recs, _ = await controlled([job(i, i * 30.0) for i in range(5)], p, c, daily_token_budget=4000)
        return p, recs
    p, recs = run(go())
    outcomes = [r.outcome for r in sorted(recs, key=lambda r: r.job_id)]
    assert outcomes[:2] == [OK, OK] and BUDGET in outcomes[2:] and len(p.times) == 2


def test_circuit_breaker_pauses_all_sending_after_repeated_failures():
    async def go():
        c = Clock(SPEED)
        p = Scripted(c, [err_result()], latency_s=0.1)
        recs, ctl = await controlled([job(deadline=600.0)], p, c, breaker_failures=3, breaker_open_s=10.0,
                                     backoff_base_s=0.01, backoff_cap_s=0.01, max_retries=5)
        return p, recs, ctl
    p, recs, ctl = run(go())
    assert ctl.gate.breaker_opens >= 1
    assert p.times[3] - p.times[2] >= 10.0  # nothing was sent for the open period after the 3rd failure
    assert recs[0].outcome == FAILED


def test_headers_correct_the_local_limiter():
    c = Clock(SPEED)
    g = Gate(c, ControlConfig(), __import__("random").Random(1))
    g.on_result(ok_result(prompt=1000, completion=0, limit=4000, remaining=1000), need=1200)
    assert g.tpm == 4000  # the provider told us its real limit
    assert g.level <= 1000 - 4000 * 0.15  # remaining minus the safety margin


def test_a_rejected_call_gives_its_reserved_tokens_back():
    c = Clock(SPEED)
    g = Gate(c, ControlConfig(), __import__("random").Random(1))

    async def go():
        assert await g.acquire(2000, 1e9) is None
        after_reserve = g.level
        g.on_result(rl_result(retry_after=1.0), need=2000)
        return after_reserve, g.level
    reserved, back = run(go())
    assert back > reserved + 1500


def test_burst_is_split_into_answered_and_refused_and_refusals_are_immediate():
    async def go():
        c = Clock(SPEED)
        p = Scripted(c, [ok_result()], latency_s=1.0)
        recs, ctl = await controlled([job(i, 0.0, deadline=30.0) for i in range(30)], p, c, tpm=8000)
        return p, recs
    p, recs = run(go())
    answered = [r for r in recs if r.outcome == OK]
    refused = [r for r in recs if r.outcome == SHED_DEADLINE and r.first_send_s is None]
    assert answered and refused and len(answered) + len(refused) + sum(r.outcome == SHED_QUEUE_FULL for r in recs) \
        + sum(r.outcome in (SHED_DEADLINE, FAILED, BUDGET) and r.first_send_s is not None for r in recs) == 30
    assert sum(r.latency_s < 1.0 for r in refused) >= 0.5 * len(refused)  # most refusals come at once, on arrival
    assert len(p.times) <= len(answered) + 2  # (almost) every call sent was one that got answered


def test_full_queue_refuses_at_once():
    async def go():
        c = Clock(SPEED)
        p = Scripted(c, [ok_result()], latency_s=5.0)
        recs, ctl = await controlled([job(i, 0.0, deadline=100000.0) for i in range(6)], p, c,
                                     workers=1, max_queue=2, tpm=10**9, rpm=10**6)
        return recs
    recs = run(go())
    assert sum(r.outcome == SHED_QUEUE_FULL for r in recs) == 4
    assert all(r.latency_s == pytest.approx(0.0, abs=0.5) for r in recs if r.outcome == SHED_QUEUE_FULL)


def test_a_question_that_cannot_make_its_deadline_is_refused_on_arrival():
    async def go():
        c = Clock(SPEED)
        p = Scripted(c, [ok_result()], latency_s=1.0)
        recs, ctl = await controlled([job(i, 0.0, deadline=12.0) for i in range(20)], p, c, tpm=2000)
        return recs
    recs = run(go())
    refused = [r for r in recs if r.outcome == SHED_DEADLINE and r.first_send_s is None]
    assert refused and all(r.latency_s < 1.0 for r in refused)


def test_long_expected_wait_switches_to_fewer_passages():
    async def go():
        c = Clock(SPEED)
        p = Scripted(c, [ok_result()], latency_s=1.0)
        recs, ctl = await controlled([job(i, i * 0.1, deadline=600.0) for i in range(10)], p, c)
        return p, recs
    p, recs = run(go())
    degraded = [r for r in recs if r.degraded]
    assert degraded and len(degraded) < len(recs)  # early questions use the full context
    sizes = sorted({sum(len(m["content"]) for m in rq.messages) for rq in p.requests})
    assert len(sizes) == 2 and sizes[0] < 0.7 * sizes[1]  # the shorter prompt really is shorter


def test_uncited_answer_is_flagged_and_counted():
    async def go():
        c = Clock(SPEED)
        cp = CountingProvider(Scripted(c, [ok_result(text="Plain answer without any citation.")]))
        recs, ctl = await controlled([job()], cp, c)
        return summarize_run(recs, cp, ctl), recs
    summary, recs = run(go())
    assert recs[0].citation_status == "uncited" and summary["uncited_answers"] == 1


def test_a_refusal_text_is_a_refusal_not_uncited():
    async def go():
        c = Clock(SPEED)
        p = Scripted(c, [ok_result(text="Not found in the provided passages.")])
        recs, _ = await controlled([job()], p, c)
        return recs
    assert run(go())[0].citation_status == "refusal"


# ---------- whole scenarios on the stand-in ----------
def test_same_trace_for_every_mode():
    a = make_trace(SCENARIOS["burst"].times, ["q1", "q2"])
    b = make_trace(SCENARIOS["burst"].times, ["q1", "q2"])
    assert a == b


@pytest.mark.parametrize("mode", ["baseline", "baseline_polite", "controlled_no_degrade", "controlled"])
def test_every_question_gets_exactly_one_outcome(mode):
    res = run(run_one(SCENARIOS["burst"], mode, 1, FakeRetriever(), ["q"], SPEED))
    s = res["summary"]
    ids = [r["job_id"] for r in res["records"]]
    assert sorted(ids) == list(range(s["jobs"]))
    total = (s["ok_on_time"] + s["ok_late"] + s["failed"] + s["shed_queue_full"] + s["shed_deadline"] + s["budget_exhausted"])
    assert total == s["jobs"] and s["source"] == "standin"


def test_on_the_burst_the_controlled_pipeline_wastes_no_calls_and_answers_at_least_as_many():
    base = run(run_one(SCENARIOS["burst"], "baseline", 1, FakeRetriever(), ["q"], SPEED))["summary"]
    ctl = run(run_one(SCENARIOS["burst"], "controlled", 1, FakeRetriever(), ["q"], SPEED))["summary"]
    assert ctl["calls_429"] == 0 and ctl["wasted_fraction"] == 0
    assert base["wasted_fraction"] > 0.5  # most baseline calls were rejected
    assert ctl["ok_on_time"] >= base["ok_on_time"]
    assert base["peak_in_flight"] > ctl["peak_in_flight"]


def test_cushion_is_added_on_top_of_the_retry_after_hint():
    async def go():
        c = Clock(SPEED)
        p = Scripted(c, [rl_result(retry_after=5.0), ok_result()], latency_s=1.0)
        await controlled([job()], p, c, cushion_s=2.0, jitter_s=0.0)
        return p.times[1] - p.times[0]
    assert run(go()) >= 5.0 + 2.0 + 1.0 - 0.3  # hint + cushion + the first call's own 1 s


def test_request_limiter_is_independent_of_the_token_limiter():
    c = Clock(SPEED)
    g = Gate(c, ControlConfig(tpm=10**9, rpm=4), __import__("random").Random(1))

    async def go():
        first = [await g.acquire(10, c.now() + 1000) for _ in range(3)]  # int(4 * 0.85) = 3 requests per minute
        fourth = await g.acquire(10, c.now() + 10)  # would have to wait about 60 s
        return first, fourth
    first, fourth = run(go())
    assert first == [None, None, None] and fourth == SHED_DEADLINE


def test_waiting_for_tokens_gives_up_when_the_deadline_would_pass():
    c = Clock(SPEED)
    g = Gate(c, ControlConfig(), __import__("random").Random(1))
    g.level = 0.0
    start = c.now()
    assert run(g.acquire(2000, c.now() + 5.0)) == SHED_DEADLINE  # needs about 15 s of refill
    assert c.now() - start < 3.0  # it said no at once; it did not sit and wait the 15 s first


def test_no_call_is_sent_when_less_than_the_minimum_service_time_is_left():
    c = Clock(SPEED)
    g = Gate(c, ControlConfig(), __import__("random").Random(1))  # full bucket: no wait needed
    assert run(g.acquire(2000, c.now() + 1.0)) == SHED_DEADLINE  # min_service_s is 2.5
    assert run(g.acquire(2000, c.now() + 10.0)) is None


def test_answer_that_arrives_after_the_deadline_is_late_not_on_time():
    async def go():
        c = Clock(SPEED)
        p = Scripted(c, [ok_result()], latency_s=5.0)
        recs, _ = await controlled([job(deadline=4.0)], p, c)
        return recs
    assert run(go())[0].outcome == OK_LATE


def test_no_degrade_variant_never_shortens_prompts_but_full_controller_does():
    plain = run(run_one(SCENARIOS["overload"], "controlled_no_degrade", 1, FakeRetriever(), ["q"], SPEED))["summary"]
    full = run(run_one(SCENARIOS["overload"], "controlled", 1, FakeRetriever(), ["q"], SPEED))["summary"]
    assert plain["degraded_jobs"] == 0 and full["degraded_jobs"] > 0
    assert full["ok_on_time"] > plain["ok_on_time"]  # shorter prompts let more answers through the same token limit


# ---------- steps 1, 2 and 4 inside the pipeline ----------
class HintRetriever(FakeRetriever):
    def __init__(self):
        self.calls = []

    def retrieve(self, query, papers=None):
        self.calls.append(papers)
        return super().retrieve(query, papers)


def test_paper_hint_is_used_only_when_switched_on_and_the_question_names_one_paper():
    async def go(use_hint, question):
        c = Clock(SPEED)
        r = HintRetriever()
        p = Scripted(c, [ok_result()])
        j = Job(0, question, 0.0, 600.0)
        recs, _ = await run_controlled([j], p, r, c, ControlConfig(use_paper_hint=use_hint))
        return r.calls, recs[0]
    calls, rec = run(go(True, "What is Inner Monologue in Moshi?"))
    assert calls == [["moshi"]] and rec.paper_hint == ["moshi"]
    assert run(go(False, "What is Inner Monologue in Moshi?"))[0] == [None]
    assert run(go(True, "How does attention work?"))[0] == [None]  # no single named paper -> no filter


def test_baseline_uses_the_same_steps():
    async def go():
        c = Clock(SPEED)
        r = HintRetriever()
        recs = await run_baseline([Job(0, "What is Helium in Moshi?", 0.0, 600.0)], Scripted(c, [ok_result()]), r, c,
                                  BaselineConfig(use_paper_hint=True))
        return r.calls, recs[0]
    calls, rec = run(go())
    assert calls == [["moshi"]] and rec.paper_hint == ["moshi"]


def test_answer_with_a_number_not_in_its_cited_passage_is_flagged():
    async def go():
        c = Clock(SPEED)
        cp = CountingProvider(Scripted(c, [ok_result(text="The model has 99 layers [1].")]))
        recs, ctl = await controlled([job()], cp, c)
        return summarize_run(recs, cp, ctl), recs[0]
    summary, rec = run(go())
    assert rec.unsupported_numbers == ["99"] and summary["answers_with_unsupported_numbers"] == 1


def test_last_status_records_how_a_failed_question_ended():
    async def go(results):
        c = Clock(SPEED)
        recs, _ = await controlled([job()], Scripted(c, results), c, max_retries=1, breaker_failures=99)
        return recs[0]
    assert run(go([rl_result(retry_after=1.0)])).last_status == "rate_limited"
    assert run(go([err_result()])).last_status == "error"
    assert run(go([ok_result()])).last_status == "ok"


def test_a_question_naming_two_papers_is_not_filtered_even_when_the_hint_is_on():
    async def go():
        c = Clock(SPEED)
        r = HintRetriever()
        j = Job(0, "How does H-Net's dynamic chunking differ from multi-head attention?", 0.0, 600.0)
        recs, _ = await run_controlled([j], Scripted(c, [ok_result()]), r, c, ControlConfig(use_paper_hint=True))
        return r.calls, recs[0]
    calls, rec = run(go())
    assert calls == [None] and sorted(rec.paper_hint) == ["attention", "dynamic_chunking"]


# ---------- steps 1, 2 and 4 inside the pipeline ----------
class HintRetriever(FakeRetriever):
    def __init__(self):
        self.calls = []

    def retrieve(self, query, papers=None):
        self.calls.append(papers)
        return super().retrieve(query, papers)


def test_paper_hint_is_used_only_when_switched_on_and_the_question_names_one_paper():
    async def go(use_hint, question):
        c = Clock(SPEED)
        r = HintRetriever()
        p = Scripted(c, [ok_result()])
        j = Job(0, question, 0.0, 600.0)
        recs, _ = await run_controlled([j], p, r, c, ControlConfig(use_paper_hint=use_hint))
        return r.calls, recs[0]
    calls, rec = run(go(True, "What is Inner Monologue in Moshi?"))
    assert calls == [["moshi"]] and rec.paper_hint == ["moshi"]
    assert run(go(False, "What is Inner Monologue in Moshi?"))[0] == [None]
    assert run(go(True, "How does attention work?"))[0] == [None]  # no single named paper -> no filter


def test_baseline_uses_the_same_steps():
    async def go():
        c = Clock(SPEED)
        r = HintRetriever()
        recs = await run_baseline([Job(0, "What is Helium in Moshi?", 0.0, 600.0)], Scripted(c, [ok_result()]), r, c,
                                  BaselineConfig(use_paper_hint=True))
        return r.calls, recs[0]
    calls, rec = run(go())
    assert calls == [["moshi"]] and rec.paper_hint == ["moshi"]


def test_answer_with_a_number_not_in_its_cited_passage_is_flagged():
    async def go():
        c = Clock(SPEED)
        cp = CountingProvider(Scripted(c, [ok_result(text="The model has 99 layers [1].")]))
        recs, ctl = await controlled([job()], cp, c)
        return summarize_run(recs, cp, ctl), recs[0]
    summary, rec = run(go())
    assert rec.unsupported_numbers == ["99"] and summary["answers_with_unsupported_numbers"] == 1


def test_last_status_records_how_a_failed_question_ended():
    async def go(results):
        c = Clock(SPEED)
        recs, _ = await controlled([job()], Scripted(c, results), c, max_retries=1, breaker_failures=99)
        return recs[0]
    assert run(go([rl_result(retry_after=1.0)])).last_status == "rate_limited"
    assert run(go([err_result()])).last_status == "error"
    assert run(go([ok_result()])).last_status == "ok"


def test_a_question_naming_two_papers_is_not_filtered_even_when_the_hint_is_on():
    async def go():
        c = Clock(SPEED)
        r = HintRetriever()
        j = Job(0, "How does H-Net's dynamic chunking differ from multi-head attention?", 0.0, 600.0)
        recs, _ = await run_controlled([j], Scripted(c, [ok_result()]), r, c, ControlConfig(use_paper_hint=True))
        return r.calls, recs[0]
    calls, rec = run(go())
    assert calls == [None] and sorted(rec.paper_hint) == ["attention", "dynamic_chunking"]
