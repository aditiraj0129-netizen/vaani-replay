"""Load scenarios on the STAND-IN service: baseline vs controlled pipeline, identical request traces.

Everything here is simulated (data/raw/standin/), never mixed with the real Groq evidence (data/raw/real/).
The stand-in copies the real token bucket and latency spread; how it behaves under concurrency, its RPM limit
and its error rates are ASSUMPTIONS (see app/providers/standin.py).

Run:
    python -m app.scenarios --repeats 3                 # all scenarios, all modes, real retriever (config B)
    python -m app.scenarios --retriever fake --repeats 1 --scenarios burst    # no PDFs needed
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Sequence

from app.clock import Clock
from app.pipeline import (
    BaselineConfig, ControlConfig, CountingProvider, Job, record_dict, run_baseline, run_controlled, summarize_run,
)
from app.providers.standin import StandInConfig, StandInProvider

DEADLINE_S = 45.0  # an answer is useful only if it arrives within 45 s of the question
FALLBACK_QUESTIONS = ["What is scaled dot-product attention?", "What is Inner Monologue in Moshi?",
                      "What does the routing module of H-Net compute?", "How many attention heads does the base Transformer use?"]


class FakeRetriever:
    """About 1,800-token prompts (5 passages of 1,450 characters), like a real config-B prompt. For tests only."""

    config = SimpleNamespace(name="fake")

    def retrieve(self, query: str, papers: Optional[Sequence[str]] = None) -> List[Any]:
        hits = []
        for i in range(5):
            chunk = SimpleNamespace(chunk_id=f"fake-{i}", paper_key="attention", version=7, page_start=1, page_end=1,
                                    section="3 Model", text=("passage " * 181).strip())
            hits.append(SimpleNamespace(chunk=chunk, score=1.0))
        return hits


def make_trace(times: Sequence[float], questions: Sequence[str], deadline_s: float = DEADLINE_S) -> List[Job]:
    return [Job(i, questions[i % len(questions)], t, deadline_s) for i, t in enumerate(times)]


def even(n: int, every_s: float) -> List[float]:
    return [i * every_s for i in range(n)]


@dataclass
class Scenario:
    name: str
    description: str
    times: List[float]
    standin: Dict[str, Any] = field(default_factory=dict)  # overrides of StandInConfig


SCENARIOS: Dict[str, Scenario] = {s.name: s for s in [
    Scenario("steady", "3 questions/min for 5 min (below the ~4.3/min the token limit allows)", even(15, 20.0)),
    Scenario("burst", "20 questions in 5 s, then nothing", [i * 0.25 for i in range(20)]),
    Scenario("overload", "10 questions/min for 5 min (about 2.3x what the token limit allows)", even(50, 6.0)),
    Scenario("halved_limit", "3 questions/min for 5 min, but the provider's token limit is 4,000 (we are not told)",
             even(15, 20.0), {"tpm": 4000}),
    Scenario("daily_exhausted", "3 questions/min for 5 min, but only 8 requests are left for the day",
             even(15, 20.0), {"rpd": 8}),
    Scenario("flaky", "3 questions/min for 5 min; 20% of calls fail with 503 and 5% hang", even(15, 20.0),
             {"error_rate": 0.20, "timeout_rate": 0.05}),
]}

MODES = ("baseline", "baseline_polite", "controlled_no_degrade", "controlled")


async def run_one(scn: Scenario, mode: str, seed: int, retriever: Any, questions: Sequence[str], speed: float) -> Dict[str, Any]:
    clock = Clock(speed)
    standin = StandInProvider(clock, StandInConfig(seed=seed, **scn.standin))
    counting = CountingProvider(standin)
    trace = make_trace(scn.times, questions)
    controller = None
    if mode.startswith("controlled"):
        # controlled_no_degrade = everything except the shorter-prompt degradation (shows what that one part adds)
        cfg = ControlConfig(seed=seed, degrade_wait_s=1e12) if mode == "controlled_no_degrade" else ControlConfig(seed=seed)
        records, controller = await run_controlled(trace, counting, retriever, clock, cfg)
    else:
        records = await run_baseline(trace, counting, retriever, clock,
                                     BaselineConfig(honor_retry_after=(mode == "baseline_polite")))
    summary = summarize_run(records, counting, controller)
    summary.update(scenario=scn.name, mode=mode, seed=seed, clock_speed=speed, source="standin")
    return {"summary": summary, "records": [record_dict(r) for r in sorted(records, key=lambda r: r.job_id)]}


def _fmt(values: List[float], digits: int = 0, spread: bool = False) -> str:
    vals = [v for v in values if v is not None]
    if not vals:
        return "-"
    text = f"{statistics.mean(vals):.{digits}f}"
    if spread and len(vals) > 1:
        text += f" ({min(vals):.{digits}f}-{max(vals):.{digits}f})"
    return text


def table(runs: List[Dict[str, Any]]) -> List[str]:
    cols = [("on-time (min-max)", 18), ("refused", 9), ("failed", 8), ("calls", 7), ("wasted %", 10), ("429s", 6),
            ("peak ||", 9), ("p50 s", 7), ("p95 s", 7), ("refusal s", 11)]
    lines = [f"{'scenario':<16}{'mode':<22}" + "".join(f"{name:>{w}}" for name, w in cols)]
    for scn in SCENARIOS:
        for mode in MODES:
            g = [r["summary"] for r in runs if r["summary"]["scenario"] == scn and r["summary"]["mode"] == mode]
            if not g:
                continue
            refused = [x["shed_queue_full"] + x["shed_deadline"] + x["budget_exhausted"] for x in g]
            cells = [_fmt([x["ok_on_time"] for x in g], spread=True), _fmt(refused), _fmt([x["failed"] for x in g]),
                     _fmt([x["calls"] for x in g]), _fmt([100 * x["wasted_fraction"] for x in g]),
                     _fmt([x["calls_429"] for x in g]), _fmt([x["peak_in_flight"] for x in g]),
                     _fmt([x["latency_p50_s"] for x in g]), _fmt([x["latency_p95_s"] for x in g]),
                     _fmt([x["time_to_refusal_p50_s"] for x in g])]
            lines.append(f"{scn:<16}{mode:<22}" + "".join(f"{c:>{w}}" for c, (_, w) in zip(cells, cols)))
    return lines


async def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--speed", type=float, default=100.0, help="simulated seconds per real second (keep <= 100)")
    ap.add_argument("--scenarios", nargs="*", choices=sorted(SCENARIOS), default=list(SCENARIOS))
    ap.add_argument("--modes", nargs="*", choices=MODES, default=list(MODES))
    ap.add_argument("--retriever", choices=["real", "fake"], default="real")
    ap.add_argument("--config", default="B")
    ap.add_argument("--out", default=None, help="folder for the evidence files (default data/raw/standin)")
    args = ap.parse_args(argv)
    if args.speed > 100:
        print("speed above 100 distorts measured latency; use <= 100", file=sys.stderr)
        return 2

    from app.config import RETRIEVAL_CONFIGS, load_settings
    from app.probe_latency import PROBE_QUESTIONS

    settings = load_settings()
    if args.retriever == "real":
        from app.retrieval import Retriever
        retriever: Any = Retriever.from_settings(settings, RETRIEVAL_CONFIGS[args.config])
        questions = [q for _, q in PROBE_QUESTIONS]
    else:
        retriever, questions = FakeRetriever(), FALLBACK_QUESTIONS
    out = Path(args.out) if args.out else settings.raw_dir / "standin"
    out.mkdir(parents=True, exist_ok=True)

    runs: List[Dict[str, Any]] = []
    with (out / "scenario_runs.jsonl").open("a", encoding="utf-8") as sf, \
            (out / "scenario_jobs.jsonl").open("a", encoding="utf-8") as jf:
        for name in args.scenarios:
            for mode in args.modes:
                for seed in range(1, args.repeats + 1):
                    result = await run_one(SCENARIOS[name], mode, seed, retriever, questions, args.speed)
                    cfg_name = args.config if args.retriever == "real" else "fake"
                    run_id = uuid.uuid4().hex[:8]  # one id per independently executed run, so repeats are never confused
                    result["summary"]["retrieval_config"] = cfg_name  # recorded so results.csv can fill retrieval_setting
                    result["summary"]["run_id"] = run_id
                    runs.append(result)
                    key = {"scenario": name, "mode": mode, "seed": seed, "retrieval_config": cfg_name, "run_id": run_id}
                    sf.write(json.dumps(result["summary"], sort_keys=True) + "\n")
                    for rec in result["records"]:
                        jf.write(json.dumps({**key, **rec}, sort_keys=True, default=str) + "\n")
                    s = result["summary"]
                    print(f"{name:<16}{mode:<22}seed={seed} on_time={s['ok_on_time']}/{s['jobs']} calls={s['calls']} "
                          f"429s={s['calls_429']}", flush=True)
    print()
    print("STAND-IN results (simulated). Cells are mean (min-max) over repeats.")
    for line in table(runs):
        print(line)
    print(f"\nevidence appended to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
