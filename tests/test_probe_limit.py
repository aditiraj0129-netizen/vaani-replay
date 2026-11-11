import asyncio
import json
from types import SimpleNamespace

import app.probe_limit as pk
from app.providers.base import LLMResult, RateLimitInfo


def ok_res(prompt=1800, completion=100):
    return LLMResult(status="ok", text="answer [1]", complete=True, http_status=200, finish_reason="stop",
                     prompt_tokens=prompt, completion_tokens=completion, total_ms=1000.0,
                     ratelimit=RateLimitInfo(remaining_tokens=3000))


def rl_res(retry_after=7.0):
    return LLMResult(status="rate_limited", http_status=429, error_kind="rate_limit", limit_kind="tpm",
                     error_message="Rate limit reached for tokens per minute (TPM)", total_ms=100.0,
                     ratelimit=RateLimitInfo(remaining_tokens=0, retry_after_s=retry_after))


class FakeRetriever:
    config = SimpleNamespace(name="B")

    def retrieve(self, q):
        c = SimpleNamespace(chunk_id="c1", paper_key="attention", version=7, page_start=1, page_end=1,
                            section="s", text="passage")
        return [SimpleNamespace(chunk=c, score=1.0, rank=1)]


def run_main(monkeypatch, tmp_path, results, argv):
    calls, sleeps = [], []

    class FakeProvider:
        def __init__(self, settings):
            pass

        async def generate(self, req, on_token=None):
            calls.append(req)
            return results[len(calls) - 1]

        async def aclose(self):
            pass

    async def fake_sleep(s):
        sleeps.append(s)

    settings = SimpleNamespace(groq_model="m", raw_dir=tmp_path / "raw")
    monkeypatch.setattr(pk, "load_settings", lambda: settings)
    monkeypatch.setattr(pk.Retriever, "from_settings", classmethod(lambda cls, s, c: FakeRetriever()))
    monkeypatch.setattr(pk, "GroqProvider", FakeProvider)
    monkeypatch.setattr(pk.asyncio, "sleep", fake_sleep)
    code = asyncio.run(pk.main(argv))
    log = tmp_path / "raw" / "real" / "limit_probe.jsonl"
    lines = [json.loads(x) for x in log.read_text(encoding="utf-8").splitlines()] if log.exists() else []
    return code, calls, sleeps, lines


def test_429_then_early_retry_fails_then_waiting_retry_after_recovers(monkeypatch, tmp_path, capsys):
    results = [ok_res()] * 4 + [rl_res(7.0), rl_res(6.0), ok_res()]
    code, calls, sleeps, lines = run_main(monkeypatch, tmp_path, results, ["--warmup-s", "5"])
    assert len(calls) == 7 and sleeps == [5.0, 0.5, 6.0]  # warm-up, early retry delay, newest Retry-After
    assert [x["phase"] for x in lines] == ["burst"] * 5 + ["early_retry", "recovery"]
    assert lines[4]["result"]["status"] == "rate_limited" and lines[4]["source"] == "real"
    out = capsys.readouterr().out
    assert "429 observed at request #5" in out and "actual tokens used by the 4 requests before it: 7600" in out
    assert "early retry: rate_limited" in out and "recovered after waiting 6.0 s" in out
    assert "limit_kind guess=tpm" in out


def test_no_429_means_no_claim_and_no_recovery_calls(monkeypatch, tmp_path, capsys):
    code, calls, sleeps, lines = run_main(monkeypatch, tmp_path, [ok_res()] * 3, ["--burst", "3", "--warmup-s", "1"])
    assert len(calls) == 3 and sleeps == [1.0]
    out = capsys.readouterr().out
    assert "no rate limit observed in 3 requests" in out and "429 observed" not in out


def test_recovery_attempts_are_capped_and_failure_is_reported(monkeypatch, tmp_path, capsys):
    results = [rl_res(5.0)] * 10
    code, calls, sleeps, lines = run_main(monkeypatch, tmp_path, results, ["--warmup-s", "0", "--max-recovery", "3"])
    assert len(calls) == 1 + 1 + 3 and sleeps == [0.0, 0.5, 5.0, 5.0, 5.0]
    assert "not recovered within the allowed attempts" in capsys.readouterr().out


def test_missing_retry_after_uses_the_default_wait(monkeypatch, tmp_path):
    results = [rl_res(None), rl_res(None), ok_res()]
    code, calls, sleeps, lines = run_main(monkeypatch, tmp_path, results, ["--warmup-s", "0"])
    assert sleeps == [0.0, 0.5, pk.DEFAULT_WAIT_S]


def test_burst_stops_at_the_token_budget_without_inventing_a_429(monkeypatch, tmp_path, capsys):
    code, calls, sleeps, lines = run_main(monkeypatch, tmp_path, [ok_res(1500, 500)] * 10,
                                          ["--budget", "4000", "--warmup-s", "0"])
    assert len(calls) == 2  # 2000, 4000 >= 4000
    out = capsys.readouterr().out
    assert "budget 4000 reached" in out and "429 observed" not in out and sleeps == [0.0]


def test_dry_run_makes_no_provider_and_no_log(monkeypatch, tmp_path, capsys):
    def boom(settings):
        raise AssertionError("provider must not be created in a dry run")

    settings = SimpleNamespace(groq_model="m", raw_dir=tmp_path / "raw")
    monkeypatch.setattr(pk, "load_settings", lambda: settings)
    monkeypatch.setattr(pk.Retriever, "from_settings", classmethod(lambda cls, s, c: FakeRetriever()))
    monkeypatch.setattr(pk, "GroqProvider", boom)
    assert asyncio.run(pk.main(["--dry-run"])) == 0
    out = capsys.readouterr().out
    assert "plan: wait 60s" in out and "estimated prompt tokens" in out
    assert not (tmp_path / "raw").exists()


def test_summary_early_retry_that_succeeds_is_reported_as_recovery():
    events = [
        dict(phase="burst", seq=1, t_start_s=0.0, status="rate_limited", limit_kind="tpm", retry_after_s=2.0,
             remaining_tokens=0, tokens=None, message="m", waited_s=0.0, prev_retry_after_s=None),
        dict(phase="early_retry", seq=2, t_start_s=0.6, status="ok", limit_kind=None, retry_after_s=None,
             remaining_tokens=100, tokens=1900, message=None, waited_s=0.5, prev_retry_after_s=2.0),
    ]
    text = "\n".join(pk.summarize_events(events))
    assert "early retry: ok" in text and "recovered after waiting 0.5 s" in text
