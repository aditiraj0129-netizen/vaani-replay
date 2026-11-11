import asyncio

from app.clock import Clock
from app.providers.base import LLMRequest
from app.providers.standin import StandInConfig, StandInProvider

BIG = "x" * 7200  # about 1,800 estimated tokens, like a real config-B prompt


def req(timeout_s=30.0):
    return LLMRequest(messages=[{"role": "system", "content": "s"}, {"role": "user", "content": BIG}],
                      max_tokens=700, timeout_s=timeout_s)


def run(coro):
    return asyncio.run(coro)


async def until_rejected(p, limit=12):
    """Send back-to-back requests (the bucket also refills while they run) until the first 429."""
    out = []
    for _ in range(limit):
        r = await p.generate(req())
        out.append(r)
        if r.status == "rate_limited":
            break
    return out


def test_fresh_bucket_admits_a_few_requests_then_rejects_with_whole_second_retry_after():
    out = run(until_rejected(StandInProvider(Clock(speed=100))))
    assert 3 <= len(out) - 1 <= 6  # about 8000 / ~1,900 per request, plus a little refill while running
    assert all(r.status == "ok" for r in out[:-1])
    r = out[-1]
    assert r.status == "rate_limited" and r.http_status == 429 and r.limit_kind == "tpm"
    assert float(r.ratelimit.retry_after_s).is_integer()
    assert "Requested" in r.error_message and "Limit 8000" in r.error_message


def test_rejected_requests_do_not_use_daily_quota_but_successes_do():
    async def go():
        p = StandInProvider(Clock(speed=100))
        out = await until_rejected(p)
        again = await p.generate(req())
        return out, again
    out, again = run(go())
    assert again.status == "rate_limited"
    rem = [r.ratelimit.remaining_requests for r in out]
    assert rem[0] > rem[-2]  # successes decrement
    assert rem[-1] == rem[-2] == again.ratelimit.remaining_requests  # rejections do not


def test_waiting_the_hint_recovers():
    async def go():
        c = Clock(speed=100)
        p = StandInProvider(c)
        r = (await until_rejected(p))[-1]
        await c.sleep(r.ratelimit.retry_after_s + 1)
        return await p.generate(req())
    assert run(go()).status == "ok"


def test_retrying_at_once_is_still_rejected():
    async def go():
        c = Clock(speed=100)
        p = StandInProvider(c)
        first = (await until_rejected(p))[-1]
        wait_exact = float(first.error_message.split("try again in ")[1].rstrip("s."))
        second = await p.generate(req())  # immediately: only the 0.2 s a rejection takes has passed
        return wait_exact, second
    wait_exact, second = run(go())
    if wait_exact > 0.5:
        assert second.status == "rate_limited"
    else:
        assert second.status in ("ok", "rate_limited")


def test_floor_mode_gives_a_hint_that_can_be_too_short():
    async def go():
        c = Clock(speed=100)
        p = StandInProvider(c, StandInConfig(retry_after_mode="floor", seed=3))
        shorts = 0
        for _ in range(6):
            r = (await until_rejected(p))[-1]
            exact = float(r.error_message.split("try again in ")[1].rstrip("s."))
            if r.ratelimit.retry_after_s < exact:
                shorts += 1
            await c.sleep(r.ratelimit.retry_after_s)
        return shorts
    assert run(go()) >= 1


def test_rpm_limit():
    async def go():
        p = StandInProvider(Clock(speed=100), StandInConfig(tpm=10**9, rpm=3))
        small = LLMRequest(messages=[{"role": "user", "content": "hi"}], max_tokens=10, timeout_s=30)
        return [(await p.generate(small)).status for _ in range(5)]
    s = run(go())
    assert s[:3] == ["ok"] * 3 and s[3] == "rate_limited"


def test_timeout_and_error_injection():
    async def go():
        p = StandInProvider(Clock(speed=100), StandInConfig(timeout_rate=1.0))
        t = await p.generate(req(timeout_s=5))
        q = StandInProvider(Clock(speed=100), StandInConfig(error_rate=1.0))
        e = await q.generate(req())
        return t, e
    t, e = run(go())
    assert t.status == "timeout" and e.status == "error" and e.http_status == 503


def test_streaming_calls_back_and_text_is_complete():
    async def go():
        p = StandInProvider(Clock(speed=100))
        got = []
        r = await p.generate(req(), on_token=got.append)
        return r, got
    r, got = run(go())
    assert r.ok and "".join(got) == r.text and len(got) > 1
    assert r.first_token_ms is not None and r.first_token_ms <= r.total_ms


def test_load_slows_down_first_token():
    async def go():
        quiet = StandInProvider(Clock(speed=100), StandInConfig(tpm=10**9, slowdown_per_inflight_ms=500, seed=1))
        small = LLMRequest(messages=[{"role": "user", "content": "hi"}], max_tokens=10, timeout_s=60)
        one = await quiet.generate(small)
        busy = StandInProvider(Clock(speed=100), StandInConfig(tpm=10**9, slowdown_per_inflight_ms=500, seed=1))
        many = await asyncio.gather(*[busy.generate(small) for _ in range(10)])
        return one, many
    one, many = run(go())
    assert max(r.first_token_ms for r in many) > one.first_token_ms + 1000


def test_clock_runs_faster_than_real_time():
    import time
    async def go():
        c = Clock(speed=500)
        t = time.monotonic()
        await c.sleep(50)
        return time.monotonic() - t, c.now()
    real, sim = run(go())
    assert real < 1.0 and sim >= 49
