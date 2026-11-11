import asyncio
import json

from app.probe_concurrency import run_waves, summarize
from app.providers.base import LLMResult, RateLimitInfo
from app.raw_log import RawLog


def ok(queue_s=0.2, ttft=800.0):
    return LLMResult(status="ok", text="OK", complete=True, http_status=200, first_token_ms=ttft, server_queue_s=queue_s,
                     prompt_tokens=78, completion_tokens=5, ratelimit=RateLimitInfo(remaining_tokens=7000))


def rl(kind="unknown"):
    return LLMResult(status="rate_limited", http_status=429, limit_kind=kind, ratelimit=RateLimitInfo(retry_after_s=2.0))


class Fake:
    def __init__(self, fail_from=None):
        self.in_flight = 0
        self.peak = 0
        self.fail_from = fail_from

    async def generate(self, req, on_token=None):
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        await asyncio.sleep(0.01)
        n = self.in_flight
        self.in_flight -= 1
        return rl("concurrency") if self.fail_from and n >= self.fail_from else ok(queue_s=0.1 * n)


async def nosleep(_):
    return None


def test_waves_are_really_simultaneous_and_logged(tmp_path):
    fake = Fake()
    log = RawLog(tmp_path / "c.jsonl")
    waves = asyncio.run(run_waves(fake, [1, 3], 0, log, "m", sleep=nosleep))
    assert fake.peak == 3 and [len(waves[k]) for k in (1, 3)] == [1, 3]
    rows = [json.loads(x) for x in (tmp_path / "c.jsonl").read_text().splitlines()]
    assert len(rows) == 4 and {r["wave"] for r in rows} == {1, 3} and all(r["source"] == "real" for r in rows)


def test_summary_without_rejection_does_not_claim_there_is_no_limit():
    waves = asyncio.run(run_waves(Fake(), [1, 2], 0, None, "m", sleep=nosleep))
    text = "\n".join(summarize(waves))
    assert "no request was rejected with up to 2 in flight" in text and "does not prove" in text


def test_summary_reports_the_first_wave_with_a_rejection():
    waves = asyncio.run(run_waves(Fake(fail_from=3), [1, 2, 4], 0, None, "m", sleep=nosleep))
    text = "\n".join(summarize(waves))
    assert "first rejection appeared in the wave of 4" in text and "concurrency" in text


def test_summary_table_shows_medians():
    rows = summarize({1: [ok(0.2, 700)], 2: [ok(0.4, 900), ok(0.6, 1100)]})
    assert rows[1].split()[-2:] == ["200", "700"] and rows[2].split()[-2:] == ["500", "1000"]
