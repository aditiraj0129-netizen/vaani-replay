"""FREE (zero API calls): estimate what the real evaluation would cost, and show what retrieval returns for each question.

    python -m app.eval_budget                      # token budget for questions x configs A, B, C
    python -m app.eval_budget --verify --config B  # also print the top passages per question (to check each question
                                                   # against the pinned paper versions and fill must_include / evidence_page)

Token estimate: the chars/4 estimate of the real prompt (built by the real retriever and prompt builder), then scaled so
that config B matches the one measured value (about 1,783 prompt tokens per request, from the real probes). The scale
is applied to A and C too, which is an assumption. Completion: median 64 and max 155 tokens were measured on 12 real
requests; the upper bound uses 250. Calls rejected with 429 were not charged in the one episode we observed (n=1).
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from app.answer import build_messages, estimate_tokens
from app.budget_guard import GLOBAL_CEILING, STAGE_CAPS

MEASURED_B_PROMPT = 1783   # measured mean prompt tokens per request at config B (real probes)
COMPLETION_TYPICAL = 64    # measured median completion tokens
COMPLETION_HIGH = 250      # conservative (measured max was 155)
PROMPT_HIGH_FACTOR = 1.15  # conservative: prompts 15% larger than the calibrated estimate
RETRY_ALLOWANCE = 0.10     # conservative: 10% extra for calls repeated after 5xx or timeouts
TPM, MARGIN = 8000, 0.15


def load_questions(path: Path) -> List[Dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def prompt_estimate(retriever: Any, question: str) -> int:
    messages, _ = build_messages(question, retriever.retrieve(question))
    return sum(estimate_tokens(m["content"]) for m in messages)


def estimate_budget(prompts: Dict[str, List[int]], scale: float) -> Dict[str, Any]:
    """prompts[config] = raw prompt-token estimates, one per question. Returns per-config and overall numbers."""
    out: Dict[str, Any] = {"configs": {}}
    total_typ = total_hi = calls = 0.0
    for cfg, vals in prompts.items():
        n = len(vals)
        mean_prompt = scale * sum(vals) / n if n else 0.0
        typ = mean_prompt + COMPLETION_TYPICAL
        hi = mean_prompt * PROMPT_HIGH_FACTOR + COMPLETION_HIGH
        out["configs"][cfg] = {"questions": n, "prompt_typ": mean_prompt, "per_request_typ": typ, "per_request_hi": hi,
                               "total_typ": typ * n, "total_hi": hi * n}
        total_typ += typ * n
        total_hi += hi * n
        calls += n
    out["calls_min"] = int(calls)
    out["total_typ"] = total_typ
    out["total_hi"] = total_hi * (1 + RETRY_ALLOWANCE)
    out["minutes_at_limit"] = total_typ / (TPM * (1 - MARGIN)) 
    return out


def check_caps(budget: Dict[str, Any], caps: Optional[Dict[str, int]] = None, ceiling: int = GLOBAL_CEILING) -> List[str]:
    """Problems found when the upper bound (per config, with the retry allowance) is compared with the approved caps and the ceiling.
    An empty list means it fits. Nothing here raises a budget: if it does not fit, STOP and report."""
    caps = STAGE_CAPS if caps is None else caps
    problems = []
    for cfg, v in budget["configs"].items():
        bound = v["total_hi"] * (1 + RETRY_ALLOWANCE)
        if cfg not in caps:
            problems.append(f"config {cfg} has no cap")
        elif bound > caps[cfg]:
            problems.append(f"config {cfg}: upper bound {bound:.0f} is above its cap {caps[cfg]}")
    if budget["total_hi"] > ceiling:
        problems.append(f"total upper bound {budget['total_hi']:.0f} is above the global ceiling {ceiling}")
    return problems


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default="data/eval/questions.csv")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--config", default="B")
    ap.add_argument("--snippet", type=int, default=150)
    args = ap.parse_args(argv)
    from app.config import RETRIEVAL_CONFIGS, load_settings
    from app.retrieval import Retriever

    settings = load_settings()
    qs = load_questions(Path(args.questions))
    retrievers = {k: Retriever.from_settings(settings, c) for k, c in RETRIEVAL_CONFIGS.items()}
    prompts = {k: [prompt_estimate(r, q["question"]) for q in qs] for k, r in retrievers.items()}
    scale = MEASURED_B_PROMPT / (sum(prompts["B"]) / len(qs))
    b = estimate_budget(prompts, scale)
    print(f"{len(qs)} questions; chars/4 estimate for B averages {sum(prompts['B']) / len(qs):.0f}; scale to the measured "
          f"{MEASURED_B_PROMPT} = x{scale:.2f}")
    print(f"{'config':<8}{'prompt tok':>11}{'per req typ':>13}{'per req high':>14}{'total typ':>11}{'total high':>12}")
    for k, v in b["configs"].items():
        print(f"{k:<8}{v['prompt_typ']:>11.0f}{v['per_request_typ']:>13.0f}{v['per_request_hi']:>14.0f}"
              f"{v['total_typ']:>11.0f}{v['total_hi']:>12.0f}")
    print(f"\nreal calls (minimum): {b['calls_min']}   expected tokens: {b['total_typ']:.0f}   "
          f"upper bound (+{RETRY_ALLOWANCE:.0%} retries): {b['total_hi']:.0f}   "
          f"pure sending time at the token limit: {b['minutes_at_limit']:.1f} min")
    problems = check_caps(b)
    print("\napproved caps: " + ", ".join(f"{k} {v:,}" for k, v in STAGE_CAPS.items()) + f" (sum {sum(STAGE_CAPS.values()):,}); global ceiling {GLOBAL_CEILING:,}")
    for k, v in b["configs"].items():
        bound = v["total_hi"] * (1 + RETRY_ALLOWANCE)
        print(f"   {k}: upper bound {bound:>9,.0f} of cap {STAGE_CAPS.get(k, 0):>7,}  ({'fits' if bound <= STAGE_CAPS.get(k, 0) else 'OVER'}; headroom {STAGE_CAPS.get(k, 0) - bound:,.0f})")
    print(f"   total upper bound {b['total_hi']:,.0f} of ceiling {GLOBAL_CEILING:,}  (headroom {GLOBAL_CEILING - b['total_hi']:,.0f})")
    if problems:
        print("\nSTOP: the budget does NOT fit the approved limits. Nothing has been raised. Report this before any real run:")
        for p in problems:
            print("   -", p)
    else:
        print("   => fits every approved cap and the ceiling")
    if args.verify:
        r = retrievers[args.config]
        for q in qs:
            print(f"\n{q['id']} [{q['kind']}] {q['question']}")
            for h in r.retrieve(q["question"]):
                c = h.chunk
                pages = f"p.{c.page_start}" if c.page_start == c.page_end else f"p.{c.page_start}-{c.page_end}"
                print(f"   {h.rank}. {c.paper_key} {pages} [{c.section}] {' '.join(c.text.split())[:args.snippet]}")
            terms = [t for t in q.get("must_include", "").split("|") if t]
            if terms:
                joined = " ".join(h.chunk.text.lower() for h in r.retrieve(q["question"]))
                print("   must_include found in retrieved text:", {t: t.lower() in joined for t in terms})
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
