"""FREE: summarise the real evaluation so far (jobs, tokens, retries, latency, ledger). Reads files only; no API calls, no key.

    python -m app.eval_status
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path
from typing import Optional, Sequence

JOBS = Path("data/raw/real/eval_jobs.jsonl")
ATTEMPTS = Path("data/raw/eval/eval_attempts.jsonl")
LEDGER = Path("data/raw/eval/eval_budget_ledger.json")
CAPS = {"A": 30_000, "B": 41_000, "C": 37_000}
CEILING = 110_000


def lines(p: Path):
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()] if p.exists() else []


def main(argv: Optional[Sequence[str]] = None) -> int:
    jobs = [j for j in lines(JOBS) if j.get("scenario") == "eval"]
    attempts = lines(ATTEMPTS)
    problems = []
    for st in "ABC":
        js = [j for j in jobs if j["retrieval_setting"] == st]
        if not js:
            print(f"stage {st}: no jobs yet")
            continue
        at = [a for a in attempts if a["retrieval_setting"] == st]
        used = sum((a["result"].get("prompt_tokens") or 0) + (a["result"].get("completion_tokens") or 0) for a in at if a["result"].get("status") == "ok")
        lat = [j["latency_s"] for j in js if j["outcome"] in ("ok", "ok_late") and j.get("latency_s") is not None]
        ttft = [j["ttft_from_arrival_s"] for j in js if j.get("ttft_from_arrival_s") is not None]
        outcomes = {}
        for j in js:
            outcomes[j["outcome"]] = outcomes.get(j["outcome"], 0) + 1
        statuses = {}
        for a in at:
            statuses[a["result"]["status"]] = statuses.get(a["result"]["status"], 0) + 1
        print(f"stage {st}: jobs {len(js)}/16 outcomes {outcomes}; attempts {len(at)} {statuses}; retries {sum(max(j['attempts'] - 1, 0) for j in js)}; "
              f"429s {sum(j['n_429'] for j in js)}")
        print(f"   tokens reported on ok attempts {used:,}" + (f"; latency p50 {statistics.median(lat):.1f}s max {max(lat):.1f}s; first token p50 {statistics.median(ttft):.2f}s" if lat and ttft else ""))
        print("   citation provenance: " + str({k: sum(1 for j in js if j["citation_status"] == k) for k in sorted({j['citation_status'] for j in js})}))
        for j in js:
            if j["outcome"] not in ("ok",):
                problems.append(f"{st} {j['question_id']} outcome {j['outcome']} last_status {j['last_status']}")
            if j.get("passages") and "gsk_" in json.dumps(j):
                problems.append(f"{st} {j['question_id']}: key-like text in the job line")
    if LEDGER.exists():
        d = json.loads(LEDGER.read_text())
        spent = d["spent"]
        total = sum(spent.values()) + d.get("prior", 0)
        print(f"\nledger: spent {spent} total {total:,} of {CEILING:,}; open reservations {d['open']}; halted={d['halted']} {d['halt_reason']}")
        for st, cap in CAPS.items():
            print(f"   {st}: spent {spent.get(st, 0):,} of cap {cap:,} (left {cap - spent.get(st, 0):,})")
        print(f"   global tokens left: {CEILING - total:,}")
        if d["open"] or d["halted"] or total > CEILING or any(spent.get(s, 0) > c for s, c in CAPS.items()):
            problems.append("LEDGER NOT CLEAN (open reservation, halt, or limit passed)")
        for st in "ABC":
            at = [a for a in attempts if a["retrieval_setting"] == st]
            used = sum((a["result"].get("prompt_tokens") or 0) + (a["result"].get("completion_tokens") or 0) for a in at if a["result"].get("status") == "ok")
            if used > spent.get(st, 0):
                problems.append(f"stage {st}: reported usage {used} is ABOVE the ledger {spent.get(st, 0)}")
    else:
        print("\nno ledger yet")
    print("\nPROBLEMS: " + ("none" if not problems else ""))
    for p in problems:
        print("   -", p)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
