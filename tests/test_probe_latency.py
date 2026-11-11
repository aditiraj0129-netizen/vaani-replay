import asyncio
import json
from types import SimpleNamespace

import pytest

import app.probe_latency as pl
from app.answer import AnswerRecord
from app.providers.base import LLMResult, RateLimitInfo


def test_pace_unknown_state_waits_a_safe_fixed_time():
    assert pl.pace_seconds(None, 2000, 8000) == 15.0


def test_pace_no_wait_when_bucket_has_room():
    assert pl.pace_seconds(7000, 2000, 8000) == 0.0


def test_pace_waits_for_the_deficit_at_tpm_over_60_per_second():
    # needs 2000 + 500 margin, has 1000 -> deficit 1500 tokens / (8000/60 per s) = 11.25 s
    assert pl.pace_seconds(1000, 2000, 8000) == pytest.approx(11.25)


def test_probe_questions_are_twelve_distinct_cover_all_papers_and_all_three_categories():
    qs = pl.PROBE_QUESTIONS
    texts = [q for _, q in qs]
    assert len(qs) == 12 and len(set(texts)) == 12
    joined = " ".join(texts).lower()
    assert "transformer" in joined and "moshi" in joined and "h-net" in joined
    cats = [c for c, _ in qs]
    assert {c: cats.count(c) for c in set(cats)} == {"answerable": 7, "partial": 2, "unanswerable": 3}


def rec(status="ok", ttft=500.0, total=900.0, prompt=1500, completion=300, est=1400, queue=0.3, remaining=6000,
        limit_kind=None):
    result = LLMResult(status=status, text="ans [1]" if status == "ok" else "", complete=status == "ok",
                       http_status=200 if status == "ok" else 429, first_token_ms=ttft, total_ms=total,
                       prompt_tokens=prompt, completion_tokens=completion, reasoning_tokens=100,
                       server_queue_s=queue, limit_kind=limit_kind, error_kind=None if status == "ok" else "rate_limit",
                       ratelimit=RateLimitInfo(remaining_tokens=remaining, limit_tokens=8000))
    return AnswerRecord("q", "B", [{"n": 1}], [{"role": "user", "content": "x"}], est, result, [1], [], False)


def test_summarize_reports_spreads_and_estimate_accuracy():
    lines = pl.summarize([rec(ttft=400), rec(ttft=600), rec(ttft=800)])
    text = "\n".join(lines)
    assert "complete answers: 3" in text
    assert "n=3 min=400 median=600 max=800" in text
    assert "estimate/actual prompt tokens: min=0.93 max=0.93" in text


def test_summarize_counts_failures_separately():
    text = "\n".join(pl.summarize([rec(), rec(status="rate_limited")]))
    assert "complete answers: 1" in text and "NOT ok: 1 (rate_limited/rate_limit)" in text


# ---------- the run loop, with the network and retrieval replaced ----------

class FakeRetriever:
    config = SimpleNamespace(name="B")

    def retrieve(self, q):
        c = SimpleNamespace(chunk_id="c1", paper_key="attention", version=7, page_start=1, page_end=1,
                            section="s", text="passage")
        return [SimpleNamespace(chunk=c, score=1.0, rank=1)]


def run_main(monkeypatch, tmp_path, results, argv):
    calls = []

    class FakeProvider:
        def __init__(self, settings):
            pass

        async def generate(self, req, on_token=None):
            calls.append(req)
            return results[len(calls) - 1]

        async def aclose(self):
            pass

    async def no_sleep(_):
        pass

    settings = SimpleNamespace(groq_model="m", raw_dir=tmp_path / "raw")
    monkeypatch.setattr(pl, "load_settings", lambda: settings)
    monkeypatch.setattr(pl.Retriever, "from_settings", classmethod(lambda cls, s, c: FakeRetriever()))
    monkeypatch.setattr(pl, "GroqProvider", FakeProvider)
    monkeypatch.setattr(pl.asyncio, "sleep", no_sleep)
    code = asyncio.run(pl.main(argv))
    log = tmp_path / "raw" / "real" / "latency_probe.jsonl"
    lines = log.read_text(encoding="utf-8").splitlines() if log.exists() else []
    return code, calls, [json.loads(x) for x in lines]


def test_run_stops_at_the_first_rate_limit_and_logs_it(monkeypatch, tmp_path, capsys):
    results = [rec().result, rec().result, rec(status="rate_limited", limit_kind="tpm").result, rec().result]
    code, calls, lines = run_main(monkeypatch, tmp_path, results, ["--n", "6"])
    assert code == 0 and len(calls) == 3 and len(lines) == 3
    assert lines[2]["result"]["status"] == "rate_limited" and lines[2]["source"] == "real"
    assert "RATE LIMIT hit" in capsys.readouterr().out


def test_run_stops_when_the_actual_token_budget_is_used(monkeypatch, tmp_path, capsys):
    results = [rec(prompt=1500, completion=500).result] * 10  # 2000 actual tokens per request
    code, calls, lines = run_main(monkeypatch, tmp_path, results, ["--n", "10", "--budget", "5000"])
    assert len(calls) == 3 and len(lines) == 3  # 2000, 4000, 6000 >= 5000
    assert "budget 5000 reached" in capsys.readouterr().out


def test_log_lines_carry_question_passages_and_headers_fields(monkeypatch, tmp_path):
    code, calls, lines = run_main(monkeypatch, tmp_path, [rec().result], ["--n", "1"])
    first = lines[0]
    assert first["question"] == pl.PROBE_QUESTIONS[0][1] and first["category"] == "answerable" and first["config"] == "B"
    assert first["passages"][0]["chunk_id"] == "c1" and first["cited"] == [1] and first["est_prompt_tokens"] > 0
    assert calls[0].max_tokens == 700


def test_dry_run_makes_no_provider_and_no_calls(monkeypatch, tmp_path, capsys):
    def boom(settings):
        raise AssertionError("provider must not be created in a dry run")

    settings = SimpleNamespace(groq_model="m", raw_dir=tmp_path / "raw")
    monkeypatch.setattr(pl, "load_settings", lambda: settings)
    monkeypatch.setattr(pl.Retriever, "from_settings", classmethod(lambda cls, s, c: FakeRetriever()))
    monkeypatch.setattr(pl, "GroqProvider", boom)
    assert asyncio.run(pl.main(["--dry-run", "--n", "3"])) == 0
    out = capsys.readouterr().out
    assert out.count("est_prompt_tokens=") == 3 and "NO API" not in out
    assert not (tmp_path / "raw").exists()


def test_budget_boundary_stops_when_used_equals_the_budget_exactly(monkeypatch, tmp_path):
    results = [rec(prompt=1500, completion=500).result] * 10  # 2000 per request
    code, calls, lines = run_main(monkeypatch, tmp_path, results, ["--n", "10", "--budget", "4000"])
    assert len(calls) == 2  # 2000, then 4000 >= 4000 -> stop


def test_a_failed_request_is_logged_and_the_run_continues_but_it_is_not_counted_as_an_answer(monkeypatch, tmp_path, capsys):
    timeout = LLMResult(status="timeout", partial_text="half", complete=False, error_kind="timeout", total_ms=30000.0)
    results = [rec().result, timeout, rec().result]
    code, calls, lines = run_main(monkeypatch, tmp_path, results, ["--n", "3"])
    assert len(calls) == 3 and len(lines) == 3
    assert lines[1]["result"]["status"] == "timeout" and lines[1]["result"]["text"] == ""
    out = capsys.readouterr().out
    assert "complete answers: 2" in out and "NOT ok: 1 (timeout/timeout)" in out


def test_categories_reach_the_log_for_every_line(monkeypatch, tmp_path):
    code, calls, lines = run_main(monkeypatch, tmp_path, [rec().result] * 12, ["--n", "12", "--budget", "10000000"])
    assert [x["category"] for x in lines] == [c for c, _ in pl.PROBE_QUESTIONS]


def test_count_terms_is_case_insensitive_and_whole_word():
    from app.check_probe_terms import count_terms

    got = count_terms(["Costs and COST, cost.", "the cost"], ["cost", "Russian"])
    assert got == {"cost": 3, "Russian": 0}
