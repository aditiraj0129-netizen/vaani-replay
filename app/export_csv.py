"""Build results.csv (exactly the 12 columns the brief asks for) from the raw evidence, and validate it.

Raw evidence is never edited. Two kinds of JSONL lines are understood, found by their keys:
  * ATTEMPT lines (one HTTP attempt to the real service; key "result"): data/raw/real/*.jsonl
  * JOB lines   (one question run through the pipeline; key "outcome"): data/raw/standin/scenario_jobs.jsonl for the
    stand-in, and (later) data/raw/real/eval_jobs.jsonl for the real evaluation.
Real and stand-in evidence stay in separate folders; a line whose own "source" contradicts its folder is an error.

Repeated experiments are NEVER discarded. Every independently executed stand-in run is exported. A run is identified by the
"run_id" on its job lines; older lines without one are split into runs by file order (a run restarts when the job id goes back)
and numbered r1, r2 ... When one scenario/mode has more than one exported run (or seed), the scenario column carries
"[seed S, run R]" so each run stays a separate group. Only a REPEATED EXPORT of the same request (the same run and job id, with
identical content) is dropped, and the drop is reported; two different lines with the same id are an error.

Column rules (all derived from the raw lines, nothing invented; an unknown value is left EMPTY):
  request_id  unique, built from the file, line number (real attempts) or run and job id (jobs)
  scenario    real attempts: "real_<file name>"; stand-in jobs: "<scenario> (<mode>)"; real evaluation: "eval"
  source      "real" | "stand-in"
  start_time  ISO 8601 UTC. Real attempt: time logged minus total_ms. Job: run start + arrival offset. STAND-IN has no
              wall clock: its start_time counts SIMULATED seconds from 1970-01-01T00:00:00Z (so it can never be mistaken for a real time).
              results_detail.csv says so per row in its time_basis column: "wall_clock" or "simulated".
  first_token_ms, total_ms  measured from the moment the request arrived (queueing included) for jobs; per attempt for real attempts
  status      ok | rate_limited | timeout | error  (see JOB_STATUS below for how pipeline outcomes map)
  retries     attempts - 1 (0 when no call was sent)
  tokens_in, tokens_out  as reported by the service (empty when it did not report them)
  retrieval_setting  A | B | C when the raw line says so, else empty
  answer_grade  only from data/eval/grades.csv (a person's grade), only for real rows with status ok; else empty
A sibling results_detail.csv keeps what the 12 columns cannot (raw outcome, where a refusal came from, queue wait, citation checks ...).

Status is one of four values, so it cannot say WHO refused. results_detail.csv can, in refusal_origin:
  provider_429          the provider answered 429 (real attempts; a job whose last call was rate limited)
  local_admission       OUR controller refused at once because its queue was full      (status rate_limited)
  local_budget_gate     OUR controller refused because the provider had signalled the daily limit is gone (status rate_limited)
  local_deadline_shed   OUR controller refused because the answer could not arrive in time (status timeout)
  late_answer           a complete answer arrived after the deadline                    (status timeout)
  provider_timeout / provider_error   the last call to the provider timed out / failed
  unknown_inferred      a failed job whose last call status was NOT recorded: status error, status_basis inferred
A local refusal is not a provider 429 and a fast refusal is not a timeout of the provider; the status column follows the brief's
four allowed values, refusal_origin keeps the difference.

Citations are two separate checks. citation_provenance is MECHANICAL (from the pipeline: do the cited passage numbers exist, are the
numbers in the answer found in the cited passages). citation_support is HUMAN (a person read the cited passage and says it backs the
claim: supported | unsupported), read from data/eval/grades.csv. An answer may be graded "correct" only when its content is correct
AND provenance is "cited" AND support is "supported".

Run:  python -m app.export_csv                      # writes results.csv and results_detail.csv, after validating
      python -m app.export_csv --validate-only results.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

COLUMNS = ["request_id", "scenario", "source", "start_time", "first_token_ms", "total_ms", "status", "retries",
           "tokens_in", "tokens_out", "retrieval_setting", "answer_grade"]
DETAIL_COLUMNS = ["request_id", "outcome", "status_basis", "refusal_origin", "time_basis", "run", "attempts", "n_429",
                  "n_other_failures", "degraded", "queue_wait_ms", "citation_provenance", "citation_support",
                  "unsupported_numbers", "question_id", "source_file", "source_line"]
SOURCES = ("real", "stand-in")
STATUSES = ("ok", "rate_limited", "timeout", "error")
GRADES = ("correct", "wrong", "refused_correctly", "refused_wrongly")
SETTINGS = ("A", "B", "C")
CITATION_SUPPORT = ("supported", "unsupported")
STATUS_BASES = ("recorded", "inferred")
TIME_BASES = ("wall_clock", "simulated")
# refusal_origin -> the status it must go with (checked by the validator)
ORIGIN_STATUS = {"": "ok", "late_answer": "timeout", "local_admission": "rate_limited", "local_budget_gate": "rate_limited",
                 "local_deadline_shed": "timeout", "provider_429": "rate_limited", "provider_timeout": "timeout",
                 "provider_error": "error", "unknown_inferred": "error"}
EVAL_SCENARIO = "eval"
SIM_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)

# Pipeline outcome -> status. Documented here and in the README; the raw outcome is kept in results_detail.csv.
JOB_STATUS = {
    "ok": "ok",                          # complete answer inside the 45 s deadline
    "ok_late": "timeout",                # complete answer, but after the deadline: the person gave up, tokens were still spent
    "shed_queue_full": "rate_limited",   # we answered "busy" at once (our own admission control)
    "budget_exhausted": "rate_limited",  # daily budget gone: told "not today"
    "shed_deadline": "timeout",          # could not finish inside the deadline: refused at once / dropped
    # "failed" takes the status of its last call (see job_status)
}
JOB_ORIGIN = {"ok": "", "ok_late": "late_answer", "shed_queue_full": "local_admission",
              "budget_exhausted": "local_budget_gate", "shed_deadline": "local_deadline_shed"}
FAILED_ORIGIN = {"rate_limited": "provider_429", "timeout": "provider_timeout", "error": "provider_error"}


class ExportError(Exception):
    pass


@dataclass
class Problem:
    row: int            # 1 = first data row; 0 = file-level
    request_id: str
    field: str
    message: str

    def __str__(self) -> str:
        where = "file" if self.row == 0 else f"row {self.row} ({self.request_id})"
        return f"{where}: {self.field}: {self.message}"


# ------------------------------------------------------------------------------------------------ time helpers
def parse_time(text: str) -> datetime:
    dt = datetime.fromisoformat(text.strip().replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise ValueError("no timezone")
    return dt.astimezone(timezone.utc)


def fmt_time(dt: datetime) -> str:
    dt = dt.astimezone(timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


def _ms(value: Optional[float]) -> str:
    return "" if value is None else f"{value:.1f}"


def _int(value: Optional[Any]) -> str:
    return "" if value is None else str(int(value))


# ------------------------------------------------------------------------------------------------ line -> row
def job_status(line: Dict[str, Any]) -> Tuple[str, str]:
    """(status, basis). basis 'recorded' = taken from the raw line; 'inferred' = the raw line has no terminal status.

    A failed job takes the status of its LAST call (last_status). If that was not recorded it is NOT guessed from the
    429 counts: it becomes 'error' and is marked 'inferred'."""
    outcome = line["outcome"]
    if outcome in JOB_STATUS:
        return JOB_STATUS[outcome], "recorded"
    if outcome != "failed":
        raise ExportError(f"unknown job outcome {outcome!r}")
    last = line.get("last_status")
    if last in ("rate_limited", "timeout", "error"):
        return last, "recorded"
    return "error", "inferred"


def job_origin(line: Dict[str, Any], status: str, basis: str) -> str:
    outcome = line["outcome"]
    if outcome in JOB_ORIGIN:
        return JOB_ORIGIN[outcome]
    return "unknown_inferred" if basis == "inferred" else FAILED_ORIGIN[status]


def row_from_attempt(line: Dict[str, Any], stem: str, lineno: int) -> Tuple[Dict[str, str], Dict[str, str]]:
    for key in ("result", "logged_at"):
        if key not in line:
            raise ExportError(f"{stem}.jsonl line {lineno}: attempt line has no {key!r}")
    res = line["result"]
    status = res.get("status")
    if status == "ok" and not res.get("complete"):
        status = "error"  # an ok response that did not complete is not an answer
    if status not in STATUSES:
        raise ExportError(f"{stem}.jsonl line {lineno}: unknown status {status!r}")
    total = float(res.get("total_ms") or 0.0)
    start = parse_time(line["logged_at"]) - timedelta(milliseconds=total)
    setting = line.get("config") if line.get("config") in SETTINGS else ""
    rid = f"real-{stem}-{lineno:04d}-{line.get('label', '')}".rstrip("-")
    row = {"request_id": rid, "scenario": f"real_{stem}", "source": "real", "start_time": fmt_time(start),
           "first_token_ms": _ms(res.get("first_token_ms")), "total_ms": _ms(total), "status": status, "retries": "0",
           "tokens_in": _int(res.get("prompt_tokens")), "tokens_out": _int(res.get("completion_tokens")),
           "retrieval_setting": setting, "answer_grade": ""}
    origin = {"ok": "", **FAILED_ORIGIN}[status]
    detail = {"request_id": rid, "outcome": status, "status_basis": "recorded", "refusal_origin": origin,
              "time_basis": "wall_clock", "run": "", "attempts": "1",
              "n_429": "1" if status == "rate_limited" else "0", "n_other_failures": "1" if status in ("timeout", "error") else "0",
              "degraded": "", "queue_wait_ms": "", "citation_provenance": "", "citation_support": "", "unsupported_numbers": "",
              "question_id": "", "source_file": f"{stem}.jsonl", "source_line": str(lineno)}
    return row, detail


def row_from_job(line: Dict[str, Any], source: str, stem: str, lineno: int,
                 standin_setting: str = "", run_token: str = "") -> Tuple[Dict[str, str], Dict[str, str]]:
    missing = [k for k in ("job_id", "arrival_s", "outcome") if line.get(k) is None]
    if missing:
        raise ExportError(f"{stem}.jsonl line {lineno}: job line is missing {missing}")
    status, basis = job_status(line)
    origin = job_origin(line, status, basis)
    base = parse_time(line["run_started_at"]) if line.get("run_started_at") else SIM_EPOCH
    if source == "real" and not line.get("run_started_at"):
        raise ExportError(f"{stem}.jsonl line {lineno}: a REAL job line needs run_started_at (no simulated clock for real data)")
    start = base + timedelta(seconds=float(line["arrival_s"]))
    mode = line.get("mode")
    scenario = line.get("scenario") or EVAL_SCENARIO
    scenario_name = f"{scenario} ({mode})" if mode else scenario
    seed_part = f"s{line['seed']}" if line.get("seed") is not None else None
    if source == "real":
        run = line.get("run_id") or "-".join(str(x) for x in (scenario, mode, seed_part) if x)
        rid = f"real-{run}-{int(line['job_id']):03d}"
        run_col = str(line.get("run_id") or "")
    else:
        run_col = run_token or str(line.get("run_id") or "r1")
        rid = "standin-" + "-".join(str(x) for x in (scenario, mode, seed_part, run_col) if x) + f"-{int(line['job_id']):03d}"
    setting = line.get("retrieval_setting") or line.get("retrieval_config") or (standin_setting if source == "stand-in" else "")
    setting = setting if setting in SETTINGS else ""
    ttft = line.get("ttft_from_arrival_s")
    latency = line.get("latency_s")
    row = {"request_id": rid, "scenario": scenario_name, "source": source, "start_time": fmt_time(start),
           "first_token_ms": _ms(None if ttft is None else ttft * 1000.0),
           "total_ms": _ms(None if latency is None else latency * 1000.0), "status": status,
           "retries": str(max(int(line.get("attempts") or 0) - 1, 0)),
           "tokens_in": _int(line.get("prompt_tokens")), "tokens_out": _int(line.get("completion_tokens")),
           "retrieval_setting": setting, "answer_grade": ""}
    qw = line.get("queue_wait_s")
    detail = {"request_id": rid, "outcome": line["outcome"], "status_basis": basis, "refusal_origin": origin,
              "time_basis": "wall_clock" if line.get("run_started_at") else "simulated", "run": run_col,
              "attempts": _int(line.get("attempts")),
              "n_429": _int(line.get("n_429")), "n_other_failures": _int(line.get("n_other_failures")),
              "degraded": "" if line.get("degraded") is None else str(int(bool(line["degraded"]))),
              "queue_wait_ms": _ms(None if qw is None else qw * 1000.0),
              "citation_provenance": line.get("citation_status", ""), "citation_support": "",
              "unsupported_numbers": str(len(line.get("unsupported_numbers") or [])),
              "question_id": str(line.get("question_id", "")), "source_file": f"{stem}.jsonl", "source_line": str(lineno)}
    return row, detail


# ------------------------------------------------------------------------------------------------ reading raw files
def _lines(path: Path) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with path.open(encoding="utf-8") as handle:
        for no, text in enumerate(handle, 1):
            if text.strip():
                try:
                    yield no, json.loads(text)
                except json.JSONDecodeError as exc:
                    raise ExportError(f"{path.name} line {no}: not valid JSON ({exc})") from exc


def build_rows(real_dir: Optional[Path], standin_dir: Optional[Path], *, standin_seeds: Optional[Sequence[int]] = (1,),
               modes: Optional[Sequence[str]] = None, standin_setting: str = "",
               grades: Optional[Dict[str, str]] = None, support: Optional[Dict[str, str]] = None,
               notes: Optional[List[str]] = None) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    rows: List[Dict[str, str]] = []
    details: List[Dict[str, str]] = []
    note = notes.append if notes is not None else (lambda _m: None)

    def add(pair: Tuple[Dict[str, str], Dict[str, str]]) -> int:
        rows.append(pair[0])
        details.append(pair[1])
        return len(rows) - 1

    if real_dir is not None and Path(real_dir).is_dir():
        seen_real: Dict[str, Dict[str, Any]] = {}
        for path in sorted(Path(real_dir).glob("*.jsonl")):
            for no, line in _lines(path):
                if line.get("source") not in (None, "real"):
                    raise ExportError(f"{path.name} line {no}: source {line.get('source')!r} found in the REAL folder")
                if "result" in line:
                    add(row_from_attempt(line, path.stem, no))
                elif "outcome" in line:
                    pair = row_from_job(line, "real", path.stem, no)
                    rid = pair[0]["request_id"]
                    if rid in seen_real:  # the same request exported twice: keep one, but only if the two lines are identical
                        if seen_real[rid] != line:
                            raise ExportError(f"{path.name} line {no}: request_id {rid!r} appears twice with DIFFERENT content")
                        note(f"{path.name} line {no}: repeated export of {rid} dropped (identical content)")
                        continue
                    seen_real[rid] = line
                    add(pair)
    meta: List[Tuple[int, Any, Any, Any, str]] = []  # stand-in rows: (row index, scenario, mode, seed, run token)
    if standin_dir is not None and Path(standin_dir).is_dir():
        counters: Dict[tuple, int] = {}
        for path in sorted(Path(standin_dir).glob("scenario_jobs*.jsonl")):
            # Every independently executed run is kept. A run is identified by its run_id; lines without one are split into
            # runs by file order (a run restarts when the job id goes back). Nothing is chosen "latest".
            runs: Dict[tuple, List[Tuple[int, Dict[str, Any]]]] = {}
            prev_key3, prev_job, current = None, None, None
            for no, line in _lines(path):
                if line.get("source") not in (None, "stand-in", "standin"):
                    raise ExportError(f"{path.name} line {no}: source {line.get('source')!r} found in the STAND-IN folder")
                if "outcome" not in line:
                    continue
                key3 = (line.get("scenario"), line.get("mode"), line.get("seed"))
                if line.get("run_id"):
                    runs.setdefault(key3 + (str(line["run_id"]),), []).append((no, line))
                    prev_key3 = None
                    continue
                job_id = line.get("job_id")
                if key3 != prev_key3 or (prev_job is not None and job_id is not None and job_id <= prev_job):
                    counters[key3] = counters.get(key3, 0) + 1
                    current = key3 + (f"r{counters[key3]}",)
                    runs[current] = []
                runs[current].append((no, line))
                prev_key3, prev_job = key3, job_id
            for key, lines in runs.items():
                scn, mode, seed, token = key
                if standin_seeds and seed is not None and seed not in standin_seeds:
                    continue  # a line with no seed at all is kept, never dropped silently
                if modes and mode not in modes:
                    continue
                seen_ids: Dict[Any, Dict[str, Any]] = {}
                for no, line in lines:
                    if line.get("job_id") in seen_ids:
                        if seen_ids[line["job_id"]] != line:
                            raise ExportError(f"{path.name} line {no}: job {line['job_id']} of run {token} appears twice with DIFFERENT content")
                        note(f"{path.name} line {no}: repeated export of job {line['job_id']} in run {token} dropped (identical content)")
                        continue
                    seen_ids[line.get("job_id")] = line
                    meta.append((add(row_from_job(line, "stand-in", path.stem, no, standin_setting, token)), scn, mode, seed, token))
    if not rows:
        raise ExportError("no raw evidence found: nothing to export (check --real / --standin paths and filters)")

    # Label runs: when one scenario/mode has several exported runs (or seeds), each one is its own group in results.csv.
    runs_of: Dict[Tuple[Any, Any], set] = {}
    for _, scn, mode, seed, token in meta:
        runs_of.setdefault((scn, mode), set()).add((seed, token))
    for (scn, mode), found in sorted(runs_of.items(), key=lambda kv: str(kv[0])):
        by_seed: Dict[Any, int] = {}
        for seed, _ in found:
            by_seed[seed] = by_seed.get(seed, 0) + 1
        for seed, n in sorted(by_seed.items(), key=lambda kv: str(kv[0])):
            if n > 1:
                note(f"{scn} / {mode} / seed {seed}: {n} independent runs found; ALL are exported as separate groups")
    for idx, scn, mode, seed, token in meta:
        if len(runs_of[(scn, mode)]) > 1:
            who = f"seed {seed}, run {token}" if seed is not None else f"run {token}"
            rows[idx]["scenario"] = f"{scn} ({mode}) [{who}]" if mode else f"{scn} [{who}]"

    index = {r["request_id"]: i for i, r in enumerate(rows)}
    if grades:
        for rid, grade in grades.items():
            if rid not in index:
                raise ExportError(f"grades file names unknown request_id {rid!r}")
            row = rows[index[rid]]
            if row["source"] != "real" or row["status"] != "ok":
                raise ExportError(f"grade for {rid!r}: only real rows with status ok can be graded")
            row["answer_grade"] = grade
    if support:
        for rid, value in support.items():
            if rid not in index:
                raise ExportError(f"grades file names unknown request_id {rid!r}")
            details[index[rid]]["citation_support"] = value
    order = {"real": 0, "stand-in": 1}
    paired = sorted(zip(rows, details), key=lambda p: (order[p[0]["source"]], p[0]["scenario"], p[0]["start_time"], p[0]["request_id"]))
    return [p[0] for p in paired], [p[1] for p in paired]


def load_grades(path: Optional[Path]) -> Dict[str, str]:
    if path is None or not Path(path).exists():
        return {}
    out: Dict[str, str] = {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rid, grade = (row.get("request_id") or "").strip(), (row.get("answer_grade") or "").strip()
            if not rid or not grade:
                continue
            if rid in out:
                raise ExportError(f"grades file lists {rid!r} twice")
            out[rid] = grade
    return out


def load_support(path: Optional[Path]) -> Dict[str, str]:
    """Human citation-support checks from the same grades file: column citation_support = supported | unsupported."""
    if path is None or not Path(path).exists():
        return {}
    out: Dict[str, str] = {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rid, value = (row.get("request_id") or "").strip(), (row.get("citation_support") or "").strip()
            if not rid or not value:
                continue
            if value not in CITATION_SUPPORT:
                raise ExportError(f"grades file: citation_support {value!r} for {rid!r} must be one of {CITATION_SUPPORT}")
            out[rid] = value
    return out


# ------------------------------------------------------------------------------------------------ validation
def _num(text: str) -> Optional[float]:
    try:
        value = float(text)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def validate_rows(rows: Sequence[Dict[str, str]], header: Optional[Sequence[str]] = None) -> List[Problem]:
    problems: List[Problem] = []
    if header is not None and list(header) != COLUMNS:
        problems.append(Problem(0, "", "header", f"columns must be exactly {COLUMNS}, found {list(header)}"))
    if not rows:
        problems.append(Problem(0, "", "rows", "the file has no data rows"))
        return problems
    seen: Dict[str, int] = {}
    for i, row in enumerate(rows, 1):
        rid = (row.get("request_id") or "").strip()

        def bad(field: str, message: str) -> None:
            problems.append(Problem(i, rid, field, message))

        for col in COLUMNS:
            if col not in row or row[col] is None:
                bad(col, "missing")
        if not rid:
            bad("request_id", "empty")
        elif rid in seen:
            bad("request_id", f"duplicate (first seen at row {seen[rid]})")
        else:
            seen[rid] = i
        if not (row.get("scenario") or "").strip():
            bad("scenario", "empty")
        source, status = row.get("source", ""), row.get("status", "")
        if source not in SOURCES:
            bad("source", f"{source!r} is not one of {SOURCES}")
        if status not in STATUSES:
            bad("status", f"{status!r} is not one of {STATUSES}")
        try:
            parse_time(row.get("start_time", ""))
        except ValueError:
            bad("start_time", f"{row.get('start_time')!r} is not an ISO 8601 time with a timezone")
        total = _num(row.get("total_ms", ""))
        if total is None or total < 0:
            bad("total_ms", f"{row.get('total_ms')!r} must be a number of milliseconds >= 0")
        first_text = row.get("first_token_ms", "")
        first = _num(first_text) if first_text != "" else None
        if first_text != "" and (first is None or first < 0):
            bad("first_token_ms", f"{first_text!r} must be empty or a number >= 0")
        elif first is not None and total is not None and first > total + 1e-6:
            bad("first_token_ms", "is larger than total_ms")
        if status == "ok" and first_text == "":
            bad("first_token_ms", "an ok row needs a first token time")
        retries = row.get("retries", "")
        if not retries.isdigit():
            bad("retries", f"{retries!r} must be a whole number >= 0")
        for col in ("tokens_in", "tokens_out"):
            text = row.get(col, "")
            if text != "" and not text.isdigit():
                bad(col, f"{text!r} must be empty or a whole number >= 0")
            if status == "ok" and text == "":
                bad(col, "an ok row must report tokens (never invented, so it cannot be left out)")
        setting = row.get("retrieval_setting", "")
        if setting != "" and setting not in SETTINGS:
            bad("retrieval_setting", f"{setting!r} is not one of {SETTINGS} (or empty)")
        if row.get("scenario") == EVAL_SCENARIO and source == "real" and setting == "":
            bad("retrieval_setting", "a real evaluation row needs its retrieval setting")
        grade = row.get("answer_grade", "")
        if grade != "":
            if grade not in GRADES:
                bad("answer_grade", f"{grade!r} is not one of {GRADES} (or empty)")
            elif source != "real" or status != "ok":
                bad("answer_grade", "only a real row with status ok can have a grade")
    return problems


def read_results(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), [dict(r) for r in reader]


def validate_file(path: Path) -> List[Problem]:
    header, rows = read_results(path)
    return validate_rows(rows, header)


def check_grades_against_citations(rows: Sequence[Dict[str, str]], details: Sequence[Dict[str, str]]) -> List[Problem]:
    """'correct' = content right AND provenance 'cited' (mechanical) AND support 'supported' (a person checked). Two separate checks."""
    by_id = {d["request_id"]: d for d in details}
    out: List[Problem] = []
    for i, row in enumerate(rows, 1):
        if row["answer_grade"] != "correct":
            continue
        d = by_id.get(row["request_id"], {})
        prov, sup = d.get("citation_provenance", ""), d.get("citation_support", "")
        if prov != "cited":
            out.append(Problem(i, row["request_id"], "answer_grade",
                               f"graded correct but citation provenance is {prov!r}, not 'cited' (the pipeline's own check failed or is missing)"))
        if sup != "supported":
            out.append(Problem(i, row["request_id"], "answer_grade",
                               f"graded correct but citation_support is {sup!r}; a person must check the cited passage and set "
                               "citation_support=supported in grades.csv. Right answer with a citation that does not back it: grade it 'wrong' and set unsupported"))
    return out


def read_detail(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    return read_results(path)


def validate_detail(header: Sequence[str], details: Sequence[Dict[str, str]], rows: Optional[Sequence[Dict[str, str]]] = None) -> List[Problem]:
    problems: List[Problem] = []
    if list(header) != DETAIL_COLUMNS:
        problems.append(Problem(0, "", "header", f"detail columns must be exactly {DETAIL_COLUMNS}, found {list(header)}"))
        return problems
    status_of = {r["request_id"]: r["status"] for r in rows} if rows is not None else {}
    if rows is not None:
        ids_d = [d["request_id"] for d in details]
        if len(details) != len(rows) or set(ids_d) != set(status_of) or len(set(ids_d)) != len(ids_d):
            problems.append(Problem(0, "", "request_id", "results_detail.csv does not describe exactly the rows of results.csv"))
    for i, d in enumerate(details, 1):
        rid = d.get("request_id", "")
        origin = d.get("refusal_origin", "")
        if origin not in ORIGIN_STATUS:
            problems.append(Problem(i, rid, "refusal_origin", f"{origin!r} is not one of {sorted(ORIGIN_STATUS)}"))
        elif rid in status_of and ORIGIN_STATUS[origin] != status_of[rid]:
            problems.append(Problem(i, rid, "refusal_origin", f"{origin!r} goes with status {ORIGIN_STATUS[origin]!r}, but the row says {status_of[rid]!r}"))
        if d.get("status_basis") not in STATUS_BASES:
            problems.append(Problem(i, rid, "status_basis", f"{d.get('status_basis')!r} is not one of {STATUS_BASES}"))
        if d.get("time_basis") not in TIME_BASES:
            problems.append(Problem(i, rid, "time_basis", f"{d.get('time_basis')!r} is not one of {TIME_BASES}"))
        if d.get("citation_support", "") not in ("",) + CITATION_SUPPORT:
            problems.append(Problem(i, rid, "citation_support", f"{d.get('citation_support')!r} is not one of {CITATION_SUPPORT} (or empty)"))
    return problems


def write_csv(path: Path, columns: Sequence[str], rows: Sequence[Dict[str, str]]) -> None:
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", default="data/raw/real")
    ap.add_argument("--standin", default="data/raw/standin")
    ap.add_argument("--grades", default="data/eval/grades.csv")
    ap.add_argument("--out", default="results.csv")
    ap.add_argument("--detail", default="results_detail.csv")
    ap.add_argument("--standin-seeds", type=int, nargs="*", default=[1], help="stand-in seeds to include (default: 1)")
    ap.add_argument("--modes", nargs="*", default=None, help="stand-in modes to include (default: all)")
    ap.add_argument("--standin-setting", default="", help="retrieval setting of OLD stand-in runs whose raw lines did not record it")
    ap.add_argument("--validate-only", default=None)
    args = ap.parse_args(argv)
    if args.validate_only:
        problems = validate_file(Path(args.validate_only))
        for p in problems[:50]:
            print("PROBLEM", p)
        print(f"{args.validate_only}: {'INVALID, ' + str(len(problems)) + ' problems' if problems else 'valid'}")
        return 1 if problems else 0
    notes: List[str] = []
    try:
        rows, details = build_rows(Path(args.real), Path(args.standin), standin_seeds=args.standin_seeds, modes=args.modes,
                                   standin_setting=args.standin_setting, grades=load_grades(Path(args.grades)),
                                   support=load_support(Path(args.grades)), notes=notes)
    except ExportError as exc:
        print(f"EXPORT FAILED: {exc}")
        return 1
    problems = (validate_rows(rows, COLUMNS) + validate_detail(DETAIL_COLUMNS, details, rows)
                + check_grades_against_citations(rows, details))
    if problems:
        for p in problems[:50]:
            print("PROBLEM", p)
        print(f"NOT WRITTEN: {len(problems)} problems")
        return 1
    for note in notes:
        print("NOTE", note)
    write_csv(Path(args.out), COLUMNS, rows)
    write_csv(Path(args.detail), DETAIL_COLUMNS, details)
    counts: Dict[Tuple[str, str, str], int] = {}
    for r in rows:
        counts[(r["source"], r["scenario"], r["status"])] = counts.get((r["source"], r["scenario"], r["status"]), 0) + 1
    print(f"wrote {args.out} ({len(rows)} rows) and {args.detail}; validated: no problems")
    for (src, scn, st), n in sorted(counts.items()):
        print(f"  {src:<9} {scn:<40} {st:<13} {n}")
    origins: Dict[str, int] = {}
    for d in details:
        if d["refusal_origin"]:
            origins[d["refusal_origin"]] = origins.get(d["refusal_origin"], 0) + 1
    print("refusal / failure origins (from results_detail.csv): " + (", ".join(f"{k}={v}" for k, v in sorted(origins.items())) or "none"))
    inferred = sum(1 for d in details if d["status_basis"] == "inferred")
    print(f"rows whose status is inferred (no recorded terminal status): {inferred}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
