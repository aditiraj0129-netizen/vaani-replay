import asyncio
import json
import random
import threading

import pytest

from app import eval_budget as eb
from app.budget_guard import GLOBAL_CEILING, STAGE_CAPS, BudgetExceeded, TokenBudget, upper_bound_tokens


def test_the_approved_numbers():
    assert STAGE_CAPS == {"A": 30_000, "B": 41_000, "C": 37_000} and GLOBAL_CEILING == 110_000
    assert sum(STAGE_CAPS.values()) == 108_000 < GLOBAL_CEILING


def test_stage_cap_is_enforced_and_nothing_is_booked_for_a_refused_reserve():
    g = TokenBudget({"A": 100, "B": 100}, ceiling=200)
    r = g.reserve("A", 60)
    with pytest.raises(BudgetExceeded):
        g.reserve("A", 41)                  # 60 reserved + 41 > 100
    assert g.committed("A") == 60
    g.settle(r, 55)
    g.reserve("A", 45)                       # exactly the cap is allowed
    with pytest.raises(BudgetExceeded):
        g.reserve("A", 1)


def test_global_ceiling_stops_before_it_is_passed_even_when_each_stage_has_room():
    g = TokenBudget({"A": 100, "B": 100, "C": 100}, ceiling=250)
    g.settle(g.reserve("A", 100), 100)
    g.settle(g.reserve("B", 100), 100)
    with pytest.raises(BudgetExceeded) as e:
        g.reserve("C", 51)                   # C is under its cap but 200 + 51 > 250
    assert "global" in str(e.value)
    g.reserve("C", 50)
    assert g.committed() == 250 and g.remaining() == 0


def test_prior_spend_counts_against_the_ceiling():
    g = TokenBudget({"A": 100}, ceiling=150, prior_spend=100)
    with pytest.raises(BudgetExceeded):
        g.reserve("A", 51)
    g.reserve("A", 50)
    with pytest.raises(BudgetExceeded):
        TokenBudget({"A": 100}, ceiling=150, prior_spend=151)


def test_a_retry_is_a_new_reservation_and_cannot_pass_the_cap():
    g = TokenBudget({"A": 100}, ceiling=100)
    for _ in range(2):                       # first try and one retry, each fails with unknown usage -> counted as fully spent
        g.settle(g.reserve("A", 40), None)
    assert g.spent("A") == 80
    with pytest.raises(BudgetExceeded):
        g.reserve("A", 40)                   # a third try would pass the cap, so it is not sent


def test_rejected_429_counts_as_spent_by_default_and_can_be_switched_off():
    g = TokenBudget({"A": 100}, ceiling=100)
    g.settle_rejected(g.reserve("A", 30))
    assert g.spent("A") == 30
    h = TokenBudget({"A": 100}, ceiling=100, rejected_counts=False)
    h.settle_rejected(h.reserve("A", 30))
    assert h.spent("A") == 0


def test_actual_usage_above_the_reservation_halts_the_guard():
    g = TokenBudget({"A": 1000}, ceiling=1000)
    g.settle(g.reserve("A", 100), 130)
    assert g.halted and g.overruns == [("A", 100, 130)] and g.spent("A") == 130
    with pytest.raises(BudgetExceeded):
        g.reserve("A", 1)


def test_unknown_stage_and_double_settle_and_negative_tokens():
    g = TokenBudget({"A": 100}, ceiling=100)
    with pytest.raises(KeyError):
        g.reserve("Z", 1)
    with pytest.raises(ValueError):
        g.reserve("A", -1)
    r = g.reserve("A", 10)
    g.settle(r, 10)
    with pytest.raises(KeyError):
        g.settle(r, 10)


def test_threads_never_pass_a_cap_or_the_ceiling():
    caps = {"A": 3000, "B": 4000, "C": 3500}
    g = TokenBudget(caps, ceiling=10_000)
    rng = random.Random(1)
    sizes = [rng.randint(20, 300) for _ in range(600)]
    stages = [rng.choice("ABC") for _ in sizes]
    max_seen = []

    def worker(chunk):
        for st, n in chunk:
            try:
                r = g.reserve(st, n)
            except BudgetExceeded:
                continue
            max_seen.append(g.committed())
            g.settle(r, rng_actual(n))

    def rng_actual(n):
        return max(1, n - 5)

    jobs = list(zip(stages, sizes))
    ts = [threading.Thread(target=worker, args=(jobs[i::12],)) for i in range(12)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert g.committed() == g.spent() <= 10_000 and max(max_seen) <= 10_000
    assert all(g.spent(s) <= c for s, c in caps.items()) and not g.halted
    assert g.spent() > 9000                                  # the budget really was used up to the limit, then requests were refused


def test_asyncio_concurrent_requests_with_retries_stop_safely_at_the_ceiling():
    g = TokenBudget({"A": 30_000, "B": 41_000, "C": 37_000}, ceiling=110_000)
    sent = []

    async def call(stage, tokens, fail_first):
        for attempt in range(2):
            try:
                r = g.reserve(stage, tokens)
            except BudgetExceeded:
                return "stopped"
            await asyncio.sleep(0)
            sent.append(tokens)
            if attempt == 0 and fail_first:
                g.settle(r, None)                    # timeout: unknown usage, retry
                continue
            g.settle(r, tokens - 10)
            return "ok"
        return "gave up"

    async def main():
        jobs = [call(s, 2500, i % 3 == 0) for i in range(120) for s in "ABC"]
        return await asyncio.gather(*jobs)

    res = asyncio.run(main())
    assert "stopped" in res                          # 360 calls x 2500 cannot all fit, so some were refused before sending
    assert g.spent() <= 110_000 and sum(sent) >= g.spent()
    assert all(g.spent(s) <= c for s, c in {"A": 30_000, "B": 41_000, "C": 37_000}.items())
    assert g.spent() > 100_000


def test_ledger_survives_a_restart_and_an_open_reservation_counts_as_spent(tmp_path):
    path = tmp_path / "ledger.json"
    g = TokenBudget({"A": 1000, "B": 1000}, ceiling=2000, ledger_path=path)
    g.settle(g.reserve("A", 100), 90)
    g.reserve("B", 200)                              # "crash": never settled
    d = json.loads(path.read_text())
    assert d["spent"]["A"] == 90 and d["open"][0]["tokens"] == 200 and not (tmp_path / "ledger.json.tmp").exists()
    h = TokenBudget({"A": 1000, "B": 1000}, ceiling=2000, ledger_path=path)
    assert h.spent("A") == 90 and h.spent("B") == 200 and h.spent() == 290
    h.reserve("B", 800)                              # 200 + 800 = cap
    with pytest.raises(BudgetExceeded):
        h.reserve("B", 1)


def test_a_halt_is_remembered_across_restarts(tmp_path):
    path = tmp_path / "l.json"
    g = TokenBudget({"A": 100}, ceiling=100, ledger_path=path)
    g.halt("operator stop")
    h = TokenBudget({"A": 100}, ceiling=100, ledger_path=path)
    assert h.halted and "operator stop" in h.halt_reason
    with pytest.raises(BudgetExceeded):
        h.reserve("A", 1)


def test_upper_bound_tokens():
    assert upper_bound_tokens(1000) == 1150 + 250 and upper_bound_tokens(0, completion_cap=5) == 5


def _budget(prompt_b):
    return eb.estimate_budget({"A": [prompt_b * 0.7] * 16, "B": [prompt_b] * 16, "C": [prompt_b * 0.88] * 16}, 1.0)


def test_check_caps_passes_when_each_config_and_the_total_fit():
    assert eb.check_caps(_budget(1783)) == []


def test_check_caps_reports_an_over_cap_config_and_does_not_raise_anything():
    problems = eb.check_caps(_budget(2200))
    assert any(p.startswith("config B") and "above its cap 41000" in p for p in problems)


def test_check_caps_reports_the_global_ceiling():
    problems = eb.check_caps(_budget(1783), caps={"A": 10 ** 6, "B": 10 ** 6, "C": 10 ** 6}, ceiling=50_000)
    assert problems == [f"total upper bound {_budget(1783)['total_hi']:.0f} is above the global ceiling 50000"]
