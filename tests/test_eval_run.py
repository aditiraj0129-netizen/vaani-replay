import asyncio
import json
from types import SimpleNamespace

import pytest

from app import eval_run as er
from app.budget_guard import BudgetExceeded, TokenBudget
from app.providers.base import LLMResult, RateLimitInfo


def run(coro):
    return asyncio.run(coro)


def item(qid="Q01", reserve=1000, kind="answerable"):
    passages = [{"n": 1, "chunk_id": "c", "paper_key": "attention", "version": 7, "pages": "p.1", "section": "s", "text": "scaled attention uses 64 dimensions"}]
    return {"q": {"id": qid, "question": "q?", "kind": kind, "paper": "attention"}, "messages": [{"role": "user", "content": "x"}],
            "passages": passages, "est_prompt": 800, "reserve": reserve}


def ok(prompt=700, completion=60, text="It uses 64 [1]."):
    return LLMResult(status="ok", text=text, complete=True, first_token_ms=300.0, total_ms=900.0, prompt_tokens=prompt, completion_tokens=completion)


class Fake:
    def __init__(self, results):
        self.results, self.calls = list(results), 0

    async def generate(self, req, on_token=None):
        self.calls += 1
        return self.results.pop(0) if self.results else ok()


class FastPacer(er.Pacer):
    async def wait_for(self, tokens):
        return 0.0


async def nosleep(_):
    return None


def go(tmp_path, items, provider, guard, **kw):
    return run(er.run_stage("A", items, provider, guard, tmp_path, run_id="t", pacer=FastPacer(), sleep=nosleep, **kw))


def jobs(tmp_path):
    p = tmp_path / "real" / er.JOBS_FILE
    return [json.loads(x) for x in p.read_text().splitlines()]


def test_real_calls_are_locked_without_both_the_flag_and_the_environment_variable():
    with pytest.raises(er.NotApproved):
        er.require_approval(False, {"RELAY_APPROVE_REAL": "yes"})
    with pytest.raises(er.NotApproved):
        er.require_approval(True, {})
    er.require_approval(True, {"RELAY_APPROVE_REAL": "yes"})


def test_env_must_be_git_ignored(tmp_path):
    assert not er.env_file_is_ignored(tmp_path / "missing")
    g = tmp_path / ".gitignore"
    g.write_text("__pycache__/\n")
    assert not er.env_file_is_ignored(g)
    g.write_text(".env\n.venv/\n")
    assert er.env_file_is_ignored(g)


def test_main_without_approval_is_a_dry_run_that_never_touches_a_provider(monkeypatch, tmp_path, capsys):
    called = []
    monkeypatch.setattr("app.providers.groq_adapter.GroqProvider", lambda *a, **k: called.append(1))
    # plan needs the real corpus; replace the heavy parts with tiny stand-ins
    monkeypatch.setattr("app.config.load_settings", lambda: SimpleNamespace(raw_dir=tmp_path))
    monkeypatch.setattr("app.eval_run.load_questions", lambda p: [{"id": "Q01", "question": "q?", "kind": "answerable", "paper": "attention"}])
    monkeypatch.setattr("app.config.RETRIEVAL_CONFIGS", {"A": 1, "B": 2, "C": 3})
    fake_r = SimpleNamespace(retrieve=lambda q: [SimpleNamespace(chunk=SimpleNamespace(chunk_id="c", paper_key="attention", version=7, page_start=1, page_end=1, section="s", text="t " * 50))])
    monkeypatch.setattr("app.retrieval.Retriever.from_settings", classmethod(lambda cls, s, c: fake_r))
    monkeypatch.setattr("app.eval_run.prompt_estimate", lambda r, q: 500)
    code = er.main(["--stage", "A"])
    out = capsys.readouterr().out
    assert code == 0 and "DRY RUN" in out and not called
    assert er.main(["--stage", "A", "--approve-real-calls"]) == 3          # flag without the environment variable: locked


def test_a_good_run_records_one_job_line_and_one_attempt_line_and_books_actual_usage(tmp_path):
    guard = TokenBudget({"A": 5000}, ceiling=5000)
    s = go(tmp_path, [item("Q01"), item("Q02")], Fake([ok(), ok(prompt=650, completion=40)]), guard)
    assert s["sent"] == 2 and s["ok"] == 2 and not s["stopped"] and guard.spent("A") == 760 + 690
    j = jobs(tmp_path)
    assert [x["question_id"] for x in j] == ["Q01", "Q02"] and j[0]["outcome"] == "ok" and j[0]["retrieval_setting"] == "A"
    assert j[0]["citation_status"] == "cited" and j[0]["prompt_tokens"] == 700 and j[0]["run_started_at"]
    attempts = (tmp_path / "eval" / er.ATTEMPTS_FILE).read_text().splitlines()
    assert len(attempts) == 2 and "gsk_" not in "".join(attempts)


def test_rerunning_a_stage_skips_finished_questions_so_no_tokens_are_spent_twice(tmp_path):
    guard = TokenBudget({"A": 5000}, ceiling=5000)
    p = Fake([ok(), ok()])
    go(tmp_path, [item("Q01"), item("Q02")], p, guard)
    s = go(tmp_path, [item("Q01"), item("Q02")], p, guard)
    assert s["skipped"] == 2 and s["sent"] == 0 and p.calls == 2 and guard.spent("A") == 1520


def test_a_429_is_retried_with_a_new_reservation_and_counted_as_spent(tmp_path):
    rl = LLMResult(status="rate_limited", http_status=429, limit_kind="tpm", ratelimit=RateLimitInfo(retry_after_s=2.0))
    guard = TokenBudget({"A": 5000}, ceiling=5000)
    s = go(tmp_path, [item("Q01", reserve=1000)], Fake([rl, ok()]), guard)
    j = jobs(tmp_path)[0]
    assert j["attempts"] == 2 and j["n_429"] == 1 and j["outcome"] == "ok" and guard.spent("A") == 1000 + 760


def test_timeouts_count_as_fully_spent_and_attempts_are_capped(tmp_path):
    bad = LLMResult(status="timeout", error_kind="timeout")
    guard = TokenBudget({"A": 10_000}, ceiling=10_000)
    go(tmp_path, [item("Q01", reserve=1000)], Fake([bad, bad, bad, bad]), guard)
    j = jobs(tmp_path)[0]
    assert j["attempts"] == er.MAX_ATTEMPTS == 3 and j["outcome"] == "failed" and j["n_other_failures"] == 3 and guard.spent("A") == 3000


def test_the_stage_cap_stops_the_run_before_a_request_that_would_pass_it(tmp_path):
    guard = TokenBudget({"A": 2500}, ceiling=10_000)
    p = Fake([ok(prompt=900, completion=60)] * 5)
    s = go(tmp_path, [item(f"Q{i:02d}", reserve=1000) for i in range(1, 6)], p, guard)
    assert s["stopped"] and "cap" in s["stopped"] and p.calls == 2 and guard.committed("A") <= 2500
    assert jobs(tmp_path)[-1]["outcome"] == "budget_exhausted" and jobs(tmp_path)[-1]["attempts"] == 0


def test_the_global_ceiling_stops_the_run_even_when_the_stage_cap_has_room(tmp_path):
    guard = TokenBudget({"A": 10_000}, ceiling=2500, prior_spend=1000)
    p = Fake([ok(prompt=400, completion=50)] * 5)
    s = go(tmp_path, [item(f"Q{i:02d}", reserve=1000) for i in range(1, 6)], p, guard)
    assert "global" in s["stopped"] and p.calls == 2 and guard.committed() <= 2500


def test_actual_usage_above_the_reservation_halts_the_run(tmp_path):
    guard = TokenBudget({"A": 10_000}, ceiling=10_000)
    p = Fake([ok(prompt=900, completion=300)] + [ok()] * 3)
    s = go(tmp_path, [item(f"Q{i:02d}", reserve=1000) for i in range(1, 4)], p, guard)
    assert guard.halted and "above reserved" in s["stopped"] and p.calls == 1


def test_pacer_waits_until_the_minute_window_has_room():
    t = {"now": 0.0}
    slept = []

    async def fake_sleep(s):
        slept.append(s)
        t["now"] += s

    pc = er.Pacer(tpm=1000, margin=1.0, now=lambda: t["now"], sleep=fake_sleep)
    pc.record(900)
    waited = run(pc.wait_for(200))
    assert waited >= 59.9 and t["now"] >= 60.0 and slept
    assert run(pc.wait_for(500)) == 0.0                      # window is empty again


def test_our_own_pacing_wait_is_not_counted_as_latency(tmp_path):
    class SlowPacer(er.Pacer):
        async def wait_for(self, tokens):
            await asyncio.sleep(0.3)
            return 0.3

    guard = TokenBudget({"A": 5000}, ceiling=5000)
    run(er.run_stage("A", [item("Q01")], Fake([ok()]), guard, tmp_path, run_id="t", pacer=SlowPacer(), sleep=nosleep))
    j = jobs(tmp_path)[0]
    assert j["pacing_wait_s"] >= 0.3 and j["latency_s"] < 0.25 and j["queue_wait_s"] < 0.05 and j["arrival_s"] >= 0.3
