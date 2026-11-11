"""Tables 1, 2a, 2b and 3, computed from results.csv (Tables 2a, 2b, 3) and from the raw real-service evidence (Table 1).

Nothing here is typed in by hand: every number is calculated from the files, and a value that cannot be calculated
is printed as "Not measured". Units: milliseconds for time, one decimal for percentages.

Definitions (also printed under each table):
  * Percentiles use the nearest-rank method over rows with status ok (n is printed in the notes).
  * Duration = last (start + total) minus first start. Offered rate = (requests - 1) / (last start - first start) * 60.
  * Peak queue length = the most requests that were at the same moment ACCEPTED BUT NOT YET SENT to the provider (queue telemetry:
    each job's queue_wait_ms in results_detail.csv, or, for a job that was never sent, the time until it was refused). It does NOT
    count requests already being processed. Baseline modes have no queue, so they show 0. Real probe rows have no queue
    telemetry: "Not measured". It is NOT the controller's internal queue object size (that one leaves out workers waiting for tokens).
  * peak_pending_to_first_response (printed in the notes, computable from results.csv alone) = the most requests at once that were
    submitted and had not yet produced a first token (or been refused). It INCLUDES requests already sent and being processed, so
    it is a "waiting for any response" number, not a queue length.
  * Tokens per answer = mean (tokens_in + tokens_out) over ok rows.
  * Three kinds of numbers are kept apart. OBSERVED: what the service did in your own probes (headers, 429s, timings).
    PUBLISHED LIMITS: figures you copied from the provider's Limits page into data/eval/published_limits.json (shown only under
    Table 1, labelled "not observed"). PAID-EQUIVALENT PRICES: list prices from data/eval/pricing.json, used for the
    paid-equivalent cost; the free tier itself charged $0.00. A missing file or a null value is printed as "Not measured".
  * Table 2c says where each refusal or failure came from (provider 429, our own queue-full refusal, our own budget gate, our own
    deadline shed, a late answer, provider timeout or error), from results_detail.csv. The status columns of Table 2b follow the
    brief's four values and cannot say who refused; Table 2c can.
  * Table 3 percentages are of "Questions asked" (an unanswered or ungraded question is in no column). "Correct and
    cited" = the answer's content is correct AND the pipeline's provenance check says cited AND a person confirmed the
    citation supports the claim (citation_support = supported). "Wrong" = wrong + refused_wrongly.

Run:  python -m app.tables --results results.csv --out tables.md
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.export_csv import EVAL_SCENARIO, parse_time, read_detail, read_results, validate_detail, validate_rows

NOT_MEASURED = "Not measured"
SCENARIO_ORDER = ["steady", "burst", "overload", "halved_limit", "daily_exhausted", "flaky"]
MODE_ORDER = ["baseline", "baseline_polite", "controlled_no_degrade", "controlled"]


# ------------------------------------------------------------------------------------------------ basic maths
def percentile(values: Sequence[float], p: float) -> Optional[float]:
    """Nearest-rank percentile: the value at position ceil(p/100 * n) of the sorted list."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(math.ceil(p / 100.0 * len(ordered)), 1) - 1]


def pct(count: int, total: int) -> str:
    return NOT_MEASURED if total == 0 else f"{100.0 * count / total:.1f}"


def ms(value: Optional[float]) -> str:
    return NOT_MEASURED if value is None else f"{value:,.0f}"


def trio(values: Sequence[float], ps: Sequence[float]) -> str:
    return " / ".join(ms(percentile(values, p)) for p in ps)


def _f(text: str) -> Optional[float]:
    return float(text) if text not in ("", None) else None


def _start_s(row: Dict[str, str]) -> float:
    return parse_time(row["start_time"]).timestamp()


_NAME = re.compile(r"^(?P<base>.+?) \((?P<mode>[^)]*)\)(?P<rest>.*)$")


def split_name(scenario: str) -> Tuple[str, str, str]:
    """'burst (controlled) [seed 1, run r2]' -> ('burst', 'controlled', '[seed 1, run r2]')."""
    m = _NAME.match(scenario)
    return (m["base"], m["mode"], m["rest"].strip()) if m else (scenario, "", "")


def _name_key(scenario: str) -> Tuple[int, int, str]:
    base, mode, _ = split_name(scenario)
    return (SCENARIO_ORDER.index(base) if base in SCENARIO_ORDER else 99,
            MODE_ORDER.index(mode) if mode in MODE_ORDER else 99, scenario)


def group_rows(rows: Sequence[Dict[str, str]], modes: Optional[Sequence[str]] = None) -> List[Tuple[Tuple[str, str], List[Dict[str, str]]]]:
    groups: Dict[Tuple[str, str], List[Dict[str, str]]] = {}
    for r in rows:
        if r["scenario"] == EVAL_SCENARIO:
            continue  # the real question-answering evaluation is Table 3, not a load scenario
        _, mode, _ = split_name(r["scenario"])
        if modes and mode and mode not in modes:
            continue
        groups.setdefault((r["source"], r["scenario"]), []).append(r)
    order = {"real": 0, "stand-in": 1}
    return sorted(groups.items(), key=lambda kv: (order[kv[0][0]], _name_key(kv[0][1])))


def span_stats(rs: Sequence[Dict[str, str]]) -> Dict[str, Optional[float]]:
    starts = [_start_s(r) for r in rs]
    ends = [_start_s(r) + (_f(r["total_ms"]) or 0.0) / 1000.0 for r in rs]
    duration = max(ends) - min(starts)
    arrival_span = max(starts) - min(starts)
    rate = (len(rs) - 1) / arrival_span * 60.0 if len(rs) > 1 and arrival_span > 0 else None
    return {"duration_s": duration, "offered_per_min": rate}


def _peak_overlap(intervals: Sequence[Tuple[float, float]]) -> int:
    """Largest number of intervals open at the same moment. An interval that ends when it starts (nobody waited) counts for nothing;
    when one ends exactly as another starts, the ending one leaves first."""
    events: List[Tuple[float, int]] = []
    for begin, finish in intervals:
        if finish > begin:
            events.append((begin, +1))
            events.append((finish, -1))
    events.sort(key=lambda e: (e[0], e[1]))
    current = peak = 0
    for _, delta in events:
        current += delta
        peak = max(peak, current)
    return peak


def peak_pending_to_first_response(rs: Sequence[Dict[str, str]]) -> int:
    """Most requests at once that were submitted and had not yet produced a first token (or been refused / finished).
    Includes requests already being processed by the provider, so this is NOT a queue length. Needs results.csv only."""
    spans = []
    for r in rs:
        start = _start_s(r)
        wait_ms = _f(r["first_token_ms"])
        if wait_ms is None:
            wait_ms = _f(r["total_ms"]) or 0.0
        spans.append((start, start + wait_ms / 1000.0))
    return _peak_overlap(spans)


def peak_waiting_to_send(rs: Sequence[Dict[str, str]], detail_by_id: Dict[str, Dict[str, str]]) -> Optional[int]:
    """Peak queue length from queue telemetry: requests accepted but not yet sent to the provider.
    A request that was sent waited queue_wait_ms; one that was never sent (attempts 0) waited until it was refused (total_ms).
    None when any row has no queue telemetry (real probe rows, or no results_detail.csv)."""
    spans = []
    for r in rs:
        d = detail_by_id.get(r["request_id"])
        if d is None or d.get("attempts", "") == "":
            return None
        start = _start_s(r)
        if int(d["attempts"]) == 0:
            wait_ms = _f(r["total_ms"]) or 0.0
        else:
            if d.get("queue_wait_ms", "") == "":
                return None
            wait_ms = float(d["queue_wait_ms"])
        spans.append((start, start + wait_ms / 1000.0))
    return _peak_overlap(spans)


def load_pricing(path: Optional[Path]) -> Optional[Dict[str, float]]:
    if path is None or not Path(path).exists():
        return None
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    pin, pout = data.get("input_per_million_usd"), data.get("output_per_million_usd")
    if pin is None or pout is None:  # a null is "not filled in yet", never a price
        return None
    return {"in": float(pin), "out": float(pout), "source_url": data.get("source_url"), "checked_on": data.get("checked_on")}


def pricing_note(pricing: Optional[Dict[str, Any]]) -> str:
    if pricing is None:
        return "Paid-equivalent prices: Not measured (no prices in data/eval/pricing.json). The free tier charged $0.00."
    return (f"Paid-equivalent uses LIST PRICES from pricing.json: input ${pricing['in']} and output ${pricing['out']} per million tokens "
            f"(source {pricing.get('source_url') or 'Not measured'}; checked {pricing.get('checked_on') or 'Not measured'}). "
            "They are published prices, not observed charges; the free tier charged $0.00.")


def cost_text(ok_rows: Sequence[Dict[str, str]], pricing: Optional[Dict[str, float]], simulated: bool) -> str:
    tin = [_f(r["tokens_in"]) for r in ok_rows if _f(r["tokens_in"]) is not None]
    tout = [_f(r["tokens_out"]) for r in ok_rows if _f(r["tokens_out"]) is not None]
    if not tin or not tout or pricing is None:
        paid = f"paid-equivalent {NOT_MEASURED}" + ("" if pricing is not None else " (no prices in pricing.json)")
    else:
        per_answer = statistics.mean(tin) * pricing["in"] / 1e6 + statistics.mean(tout) * pricing["out"] / 1e6
        paid = f"paid-equivalent ${per_answer * 1000:.4f}"
    free = "free tier $0.00" if not simulated else "free tier n/a (simulated)"
    return f"{free}; {paid}"


# ------------------------------------------------------------------------------------------------ tables 2a / 2b / 3
def table_2a(rows: Sequence[Dict[str, str]], modes: Optional[Sequence[str]] = None) -> Tuple[List[str], List[List[str]], List[str]]:
    headers = ["Scenario", "Real or stand-in", "Requests sent", "Duration (s)", "Offered rate (per min)",
               "First token p50 / p95 / p99 (ms)", "Total time p50 / p95 / p99 (ms)"]
    out: List[List[str]] = []
    notes: List[str] = []
    for (source, scenario), rs in group_rows(rows, modes):
        ok = [r for r in rs if r["status"] == "ok"]
        first = [_f(r["first_token_ms"]) for r in ok if _f(r["first_token_ms"]) is not None]
        total = [_f(r["total_ms"]) for r in ok if _f(r["total_ms"]) is not None]
        sp = span_stats(rs)
        out.append([scenario, source, str(len(rs)), f"{sp['duration_s']:.1f}",
                    NOT_MEASURED if sp["offered_per_min"] is None else f"{sp['offered_per_min']:.1f}",
                    trio(first, (50, 95, 99)), trio(total, (50, 95, 99))])
        notes.append(f"{scenario} [{source}]: percentiles over {len(ok)} ok of {len(rs)} requests")
    return headers, out, notes


def table_2b(rows: Sequence[Dict[str, str]], pricing: Optional[Dict[str, float]], modes: Optional[Sequence[str]] = None,
             detail: Optional[Sequence[Dict[str, str]]] = None) -> Tuple[List[str], List[List[str]], List[str]]:
    headers = ["Scenario", "Succeeded %", "Rate-limited %", "Timed out or given up %", "Retries per request",
               "Peak queue length (accepted, not yet sent)", "Tokens per answer", "Cost per 1,000 answers"]
    detail_by_id = {d["request_id"]: d for d in detail} if detail is not None else None
    out: List[List[str]] = []
    notes: List[str] = []
    for (source, scenario), rs in group_rows(rows, modes):
        n = len(rs)
        cnt = Counter(r["status"] for r in rs)
        ok = [r for r in rs if r["status"] == "ok"]
        toks = [(_f(r["tokens_in"]) or 0) + (_f(r["tokens_out"]) or 0) for r in ok
                if _f(r["tokens_in"]) is not None and _f(r["tokens_out"]) is not None]
        retries = statistics.mean(int(r["retries"]) for r in rs)
        if detail_by_id is None:
            queue = f"{NOT_MEASURED} (no results_detail.csv)"
        else:
            depth = peak_waiting_to_send(rs, detail_by_id)
            queue = NOT_MEASURED + " (no queue telemetry)" if depth is None else str(depth)
        out.append([scenario, pct(cnt["ok"], n), pct(cnt["rate_limited"], n), pct(cnt["timeout"] + cnt["error"], n),
                    f"{retries:.2f}", queue,
                    NOT_MEASURED if not toks else f"{statistics.mean(toks):,.0f}",
                    cost_text(ok, pricing, simulated=(source == "stand-in"))])
        notes.append(f"{scenario} [{source}]: n={n}; ok={cnt['ok']} rate_limited={cnt['rate_limited']} "
                     f"timeout={cnt['timeout']} error={cnt['error']}; tokens averaged over {len(toks)} ok rows; "
                     f"peak_pending_to_first_response={peak_pending_to_first_response(rs)} (includes requests already being processed)")
    return headers, out, notes


ORIGIN_COLUMNS = [("provider_429", "Provider 429"), ("local_admission", "Local: queue full"), ("local_budget_gate", "Local: budget gate"),
                  ("local_deadline_shed", "Local: deadline shed"), ("late_answer", "Late answer"),
                  ("provider_timeout", "Provider timeout"), ("provider_error", "Provider error"), ("unknown_inferred", "Status inferred")]


def table_2c(rows: Sequence[Dict[str, str]], detail: Optional[Sequence[Dict[str, str]]],
             modes: Optional[Sequence[str]] = None) -> Tuple[List[str], List[List[str]], List[str]]:
    """Who refused or failed. Counts of requests per refusal_origin (results_detail.csv)."""
    headers = ["Scenario", "Requests sent"] + [label for _, label in ORIGIN_COLUMNS]
    if detail is None:
        return headers, [[f"{NOT_MEASURED} (no results_detail.csv)"] + [""] * (len(headers) - 1)], []
    by_id = {d["request_id"]: d for d in detail}
    out: List[List[str]] = []
    for (source, scenario), rs in group_rows(rows, modes):
        c = Counter(by_id[r["request_id"]]["refusal_origin"] for r in rs)
        out.append([scenario, str(len(rs))] + [str(c[key]) for key, _ in ORIGIN_COLUMNS])
    notes = ["Local = OUR controller refused (queue full, budget gate, deadline shed). Provider = the service answered 429, timed out or failed. "
             "Late answer = a complete answer after the deadline. Status inferred = a failed job whose last call status was not recorded.",
             "In results.csv, queue-full and budget-gate refusals have status rate_limited; deadline sheds and late answers have status timeout."]
    return headers, out, notes


def setting_label(name: str, cfg: Any) -> str:
    return (f"{name}: {cfg.chunk_words}-word pieces ({cfg.overlap_words} overlap), {cfg.top_k} fetched, "
            f"re-rank {'yes' if cfg.rerank else 'no'}")


def table_3(rows: Sequence[Dict[str, str]], configs: Dict[str, Any],
            detail: Optional[Sequence[Dict[str, str]]] = None) -> Tuple[List[str], List[List[str]], List[str]]:
    headers = ["Setting (piece size, pieces fetched, re-rank)", "Questions asked", "Correct and cited %", "Wrong %",
               "Correctly refused %", "Tokens per answer", "First token p50 / p95 (ms)", "Total time p50 / p95 (ms)"]
    out: List[List[str]] = []
    notes: List[str] = []
    evalrows = [r for r in rows if r["source"] == "real" and r["scenario"] == EVAL_SCENARIO]
    by_id = {d["request_id"]: d for d in detail} if detail is not None else None
    for name in sorted({r["retrieval_setting"] for r in evalrows}):
        rs = [r for r in evalrows if r["retrieval_setting"] == name]
        n = len(rs)
        grades = Counter(r["answer_grade"] for r in rs)
        ok = [r for r in rs if r["status"] == "ok"]
        toks = [(_f(r["tokens_in"]) or 0) + (_f(r["tokens_out"]) or 0) for r in ok]
        first = [_f(r["first_token_ms"]) for r in ok if _f(r["first_token_ms"]) is not None]
        total = [_f(r["total_ms"]) for r in ok]
        graded = sum(1 for r in rs if r["answer_grade"])
        label = setting_label(name, configs[name]) if name in configs else name
        out.append([label, str(n), pct(grades["correct"], n) if graded else NOT_MEASURED + " (no grades)",
                    pct(grades["wrong"] + grades["refused_wrongly"], n) if graded else NOT_MEASURED + " (no grades)",
                    pct(grades["refused_correctly"], n) if graded else NOT_MEASURED + " (no grades)",
                    NOT_MEASURED if not toks else f"{statistics.mean(toks):,.0f}",
                    trio(first, (50, 95)), trio(total, (50, 95))])
        notes.append(f"{name}: asked={n}, graded={graded}, answered (status ok)={len(ok)}, not answered={n - len(ok)}, "
                     f"correct={grades['correct']} wrong={grades['wrong']} refused_correctly={grades['refused_correctly']} "
                     f"refused_wrongly={grades['refused_wrongly']}")
        if by_id is not None:
            prov = Counter(by_id[r["request_id"]]["citation_provenance"] or "none" for r in ok)
            sup = Counter(by_id[r["request_id"]]["citation_support"] or "not reviewed" for r in ok)
            notes.append(f"{name}: citation provenance (pipeline check) over answered rows: " + ", ".join(f"{k}={v}" for k, v in sorted(prov.items()))
                         + "; citation support (person check): " + ", ".join(f"{k}={v}" for k, v in sorted(sup.items())))
    if not out:
        out.append([NOT_MEASURED + " (no real evaluation rows yet)"] + [""] * 7)
    return headers, out, notes


# ------------------------------------------------------------------------------------------------ table 1 (raw real evidence)
def _real_lines(real_dir: Path) -> List[Tuple[str, int, Dict[str, Any]]]:
    out: List[Tuple[str, int, Dict[str, Any]]] = []
    if not Path(real_dir).is_dir():
        return out
    for path in sorted(Path(real_dir).glob("*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for no, text in enumerate(handle, 1):
                if text.strip():
                    line = json.loads(text)
                    if "result" in line:
                        out.append((path.stem, no, line))
    return out


def _redact(text: str) -> str:
    return re.sub(r"org_[A-Za-z0-9]+", "org_<redacted>", text or "")


def _mode(values: Sequence[Any]) -> Optional[Any]:
    return Counter(values).most_common(1)[0][0] if values else None


def table_1(real_dir: Path, published: Optional[Dict[str, Any]] = None) -> Tuple[List[str], List[List[str]], List[str]]:
    headers = ["Limit", "Value you found", "How you found it", "Confidence (high, medium, low)"]
    lines = _real_lines(real_dir)
    published = {k: v for k, v in (published or {}).items() if v is not None}  # a null is "not filled in", never a value
    n = len(lines)
    results = [(stem, no, ln["result"], ln) for stem, no, ln in lines]
    hdr_req = [r["ratelimit"]["limit_requests"] for _, _, r, _ in results if (r.get("ratelimit") or {}).get("limit_requests")]
    hdr_tok = [r["ratelimit"]["limit_tokens"] for _, _, r, _ in results if (r.get("ratelimit") or {}).get("limit_tokens")]
    r429 = [(stem, no, r, ln) for stem, no, r, ln in results if r.get("status") == "rate_limited"]
    kinds = Counter(r.get("limit_kind") for _, _, r, _ in r429)

    rows: List[List[str]] = []
    rows.append(["Requests per minute",
                 f"{NOT_MEASURED} (no real 429 of kind rpm; {n} real attempts logged)" if kinds["rpm"] == 0 else f"429 of kind rpm seen {kinds['rpm']} times",
                 "Observed: no real rate-limit response named a per-minute request limit", "n/a"])
    tpm_msgs = [re.search(r"Limit (\d+)", r.get("error_message") or "") for _, _, r, _ in r429 if r.get("limit_kind") == "tpm"]
    tpm_msgs = [int(m.group(1)) for m in tpm_msgs if m]
    if hdr_tok:
        value = f"{_mode(hdr_tok):,} per minute"
        how = f"Observed: x-ratelimit-limit-tokens header on {len(hdr_tok)} responses" + (
            f"; {len(tpm_msgs)} real 429s quote 'Limit {_mode(tpm_msgs)}'" if tpm_msgs else "")
        conf = "high" if tpm_msgs and _mode(tpm_msgs) == _mode(hdr_tok) else "medium"
    else:
        value, how, conf = NOT_MEASURED, "Observed: no header seen", "n/a"
    rows.append(["Tokens per minute", value, how, conf])
    rows.append(["Requests per day",
                 f"{_mode(hdr_req):,} per day" if hdr_req else NOT_MEASURED,
                 (f"Observed: x-ratelimit-limit-requests header on {len(hdr_req)} responses; never exhausted, so the limit itself was not "
                  f"tested" if hdr_req else "Observed: no header seen"), "medium" if hdr_req else "n/a"])
    rows.append(["Tokens per day", NOT_MEASURED + (" (no real 429 of kind tpd)" if kinds["tpd"] == 0 else ""),
                 "Observed: the service sends no per-day token header; the daily token limit was never reached", "n/a"])
    waves: Dict[int, List[bool]] = {}
    for _, _, r, ln in results:
        if "wave" in ln:
            waves.setdefault(int(ln["wave"]), []).append(r.get("status") == "ok")
    if waves:
        top = max(waves)
        rejected = [w for w, oks in sorted(waves.items()) if not all(oks)]
        value = (f"First rejection in the wave of {rejected[0]} simultaneous requests" if rejected else
                 f"No rejection up to {top} simultaneous requests (not tested higher)")
        how = f"Observed: concurrency probe, waves {sorted(waves)} of tiny requests, {sum(len(v) for v in waves.values())} requests"
        conf = "low"
    else:
        value, how, conf = NOT_MEASURED, "no concurrency probe in the raw evidence", "n/a"
    rows.append(["Requests in flight at once", value, how, conf])
    if r429:
        statuses = sorted({str(r.get("http_status")) for _, _, r, _ in r429})
        ras = sorted({r["ratelimit"]["retry_after_s"] for _, _, r, _ in r429 if (r.get("ratelimit") or {}).get("retry_after_s") is not None})
        example = _redact(next((r.get("error_message") for _, _, r, _ in r429 if r.get("error_message")), ""))[:160]
        rows.append(["What it returns when a limit is hit",
                     f"HTTP {', '.join(statuses)}; limit kinds {dict(kinds)}; Retry-After seconds seen: {ras if ras else NOT_MEASURED}",
                     f"Observed: {len(r429)} real rate-limited responses. Example message: {example!r}", "high" if len(r429) >= 3 else "medium"])
    else:
        rows.append(["What it returns when a limit is hit", NOT_MEASURED + " (no real 429 in the raw evidence)", "", "n/a"])
    rec = _recovery(results)
    rows.append(["Time until it recovers",
                 NOT_MEASURED if rec is None else
                 f"Per-minute token limit: the first request sent {rec['wait_s']:.1f} s after the 429 succeeded (Retry-After said "
                 f"{rec['retry_after_s']} s). Daily limit: {NOT_MEASURED}",
                 NOT_MEASURED if rec is None else "Observed: limit probe, first 429 to first later success (an upper bound: the probe chose "
                 "when to send; one episode)", "low" if rec is not None else "n/a"])
    notes = [f"{n} real attempts read from {real_dir}. Values and 'How' text above are OBSERVED in your own probes."]
    observed = {"tpm": _mode(hdr_tok), "rpd": _mode(hdr_req)}
    if not any(k in published for k in ("rpm", "tpm", "rpd", "tpd")):
        notes.append(f"Published limits (provider Limits page, NOT observed): {NOT_MEASURED} (no values in data/eval/published_limits.json)")
    else:
        parts = []
        for key in ("rpm", "tpm", "rpd", "tpd"):
            if key in published:
                seen = observed.get(key)
                cmp = "" if seen is None else (" (matches the observed header)" if seen == published[key] else f" (DIFFERS from the observed header {seen:,})")
                parts.append(f"{key}={published[key]:,}{cmp}")
            else:
                parts.append(f"{key}={NOT_MEASURED}")
        notes.append(f"Published limits (provider Limits page, NOT observed; checked {published.get('checked_on', NOT_MEASURED)}): " + "; ".join(parts))
    return headers, rows, notes


def _recovery(results: Sequence[Tuple[str, int, Dict[str, Any], Dict[str, Any]]]) -> Optional[Dict[str, float]]:
    probe = [(stem, no, r, ln) for stem, no, r, ln in results if ln.get("t_start_s") is not None]
    probe.sort(key=lambda t: (t[0], t[3]["t_start_s"]))
    for i, (stem, _, r, ln) in enumerate(probe):
        if r.get("status") == "rate_limited":
            received = ln["t_start_s"] + (r.get("total_ms") or 0.0) / 1000.0
            for stem2, _, r2, ln2 in probe[i + 1:]:
                if stem2 == stem and r2.get("status") == "ok":
                    return {"wait_s": ln2["t_start_s"] - received, "retry_after_s": (r.get("ratelimit") or {}).get("retry_after_s")}
            return None
    return None


# ------------------------------------------------------------------------------------------------ rendering
def md_table(title: str, headers: Sequence[str], rows: Sequence[Sequence[str]], notes: Sequence[str]) -> str:
    lines = [f"### {title}", "", "| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(str(c).replace("|", "/") for c in row) + " |" for row in rows]
    if notes:
        lines += [""] + [f"- {n}" for n in notes]
    return "\n".join(lines) + "\n"


def build_all(results_path: Path, real_dir: Path, pricing_path: Optional[Path], published_path: Optional[Path],
              modes: Optional[Sequence[str]] = None, configs: Optional[Dict[str, Any]] = None,
              detail_path: Optional[Path] = None) -> Tuple[str, Dict[str, Any]]:
    header, rows = read_results(results_path)
    problems = validate_rows(rows, header)
    if problems:
        raise ValueError("results.csv is invalid: " + "; ".join(str(p) for p in problems[:5]))
    detail: Optional[List[Dict[str, str]]] = None
    if detail_path is not None and Path(detail_path).exists():
        dheader, detail = read_detail(detail_path)
        dproblems = validate_detail(dheader, detail, rows)
        if dproblems:
            raise ValueError("results_detail.csv is invalid: " + "; ".join(str(p) for p in dproblems[:5]))
    if configs is None:
        from app.config import RETRIEVAL_CONFIGS
        configs = RETRIEVAL_CONFIGS
    published = json.loads(Path(published_path).read_text(encoding="utf-8")) if published_path and Path(published_path).exists() else {}
    pricing = load_pricing(pricing_path)
    parts: Dict[str, Any] = {}
    md: List[str] = []
    b2 = table_2b(rows, pricing, modes, detail)
    b2[2].append(pricing_note(pricing))
    b3 = table_3(rows, configs, detail)
    b3[2].append("'Correct and cited' = content correct AND pipeline provenance 'cited' AND a person confirmed citation_support=supported. "
                 "'Wrong' = wrong + refused_wrongly.")
    for key, title, built in (
        ("table1", "Table 1: Limits found in the real service", table_1(real_dir, published)),
        ("table2a", "Table 2a: Latency, one row per load scenario", table_2a(rows, modes)),
        ("table2b", "Table 2b: Outcomes, same rows", b2),
        ("table2c", "Table 2c (supplement): Where refusals and failures came from", table_2c(rows, detail, modes)),
        ("table3", "Table 3: Retrieval settings (real evaluation)", b3),
    ):
        headers, body, notes = built
        parts[key] = {"headers": headers, "rows": body, "notes": notes}
        md.append(md_table(title, headers, body, notes))
    md.append("Rows ending in (mode) are stand-in runs of one scenario under one pipeline mode; [seed S, run R] marks one of several "
              "independent runs of the same scenario and mode. Stand-in times and tokens are SIMULATED; stand-in start_time counts "
              "simulated seconds from 1970-01-01 (results_detail.csv time_basis = simulated).\n")
    return "\n".join(md), parts


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results.csv")
    ap.add_argument("--real", default="data/raw/real")
    ap.add_argument("--pricing", default="data/eval/pricing.json")
    ap.add_argument("--published", default="data/eval/published_limits.json")
    ap.add_argument("--detail", default="results_detail.csv")
    ap.add_argument("--modes", nargs="*", default=None, help="stand-in modes to show (default: all)")
    ap.add_argument("--out", default="tables.md")
    args = ap.parse_args(argv)
    try:
        text, parts = build_all(Path(args.results), Path(args.real), Path(args.pricing), Path(args.published), args.modes,
                              detail_path=Path(args.detail))
    except (ValueError, FileNotFoundError) as exc:
        print(f"TABLES FAILED: {exc}")
        return 1
    Path(args.out).write_text(text, encoding="utf-8")
    Path(args.out).with_suffix(".json").write_text(json.dumps(parts, indent=2), encoding="utf-8")
    print(text)
    print(f"saved {args.out} and {Path(args.out).with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
