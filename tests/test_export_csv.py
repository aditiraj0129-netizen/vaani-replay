import asyncio
import csv
import json
from datetime import datetime, timezone

import pytest

from app import export_csv as ex

REAL_START = "2026-11-11T10:00:00.000Z"


def attempt(label="latency_probe_1", status="ok", complete=True, total=1200.0, first=900.0, p=1800, c=60, config="B",
            logged="2026-11-11T10:00:02.000+00:00", **extra):
    line = {"logged_at": logged, "source": "real", "label": label, "model": "m",
            "result": {"status": status, "complete": complete, "total_ms": total, "first_token_ms": first,
                       "prompt_tokens": p, "completion_tokens": c, "http_status": 200 if status == "ok" else 429}}
    if config:
        line["config"] = config
    line.update(extra)
    return line


def job(job_id=0, outcome="ok", arrival=0.0, latency=2.0, ttft=1.2, attempts=1, p=1800, c=70, **extra):
    line = {"job_id": job_id, "arrival_s": arrival, "deadline_s": 45.0, "outcome": outcome, "attempts": attempts, "n_429": 0,
            "n_other_failures": 0, "degraded": False, "queue_wait_s": 0.1, "latency_s": latency, "ttft_from_arrival_s": ttft,
            "prompt_tokens": p, "completion_tokens": c, "citation_status": "cited", "unsupported_numbers": [],
            "scenario": "burst", "mode": "controlled", "seed": 1}
    line.update(extra)
    return line


def write_jsonl(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for ln in lines:
            f.write(json.dumps(ln) + "\n")


@pytest.fixture
def raw(tmp_path):
    real, standin = tmp_path / "real", tmp_path / "standin"
    write_jsonl(real / "latency_probe.jsonl", [attempt("latency_probe_1"), attempt("latency_probe_2")])
    write_jsonl(standin / "scenario_jobs.jsonl", [job(0), job(1, arrival=0.25)])
    return real, standin


def test_columns_and_allowed_values_are_the_ones_in_the_brief():
    assert ex.COLUMNS == ["request_id", "scenario", "source", "start_time", "first_token_ms", "total_ms", "status", "retries",
                          "tokens_in", "tokens_out", "retrieval_setting", "answer_grade"]
    assert ex.SOURCES == ("real", "stand-in") and ex.STATUSES == ("ok", "rate_limited", "timeout", "error")
    assert ex.GRADES == ("correct", "wrong", "refused_correctly", "refused_wrongly")


def test_real_attempt_becomes_a_row_with_start_time_before_the_log_time(raw):
    rows, details = ex.build_rows(*raw)
    real = [r for r in rows if r["source"] == "real"]
    assert len(real) == 2 and real[0]["scenario"] == "real_latency_probe" and real[0]["status"] == "ok"
    assert real[0]["start_time"] == "2026-11-11T10:00:00.800Z"  # logged 10:00:02.000 minus total 1200 ms
    assert (real[0]["first_token_ms"], real[0]["total_ms"], real[0]["tokens_in"], real[0]["tokens_out"]) == ("900.0", "1200.0", "1800", "60")
    assert real[0]["retrieval_setting"] == "B" and real[0]["answer_grade"] == "" and real[0]["retries"] == "0"
    assert len(details) == len(rows)


def test_standin_start_time_is_simulated_and_scenario_names_the_mode(raw):
    rows, _ = ex.build_rows(*raw)
    s = [r for r in rows if r["source"] == "stand-in"]
    assert s[0]["scenario"] == "burst (controlled)" and s[0]["start_time"] == "1970-01-01T00:00:00.000Z"
    assert s[1]["start_time"] == "1970-01-01T00:00:00.250Z" and s[0]["first_token_ms"] == "1200.0" and s[0]["total_ms"] == "2000.0"


def test_real_rows_come_first_and_everything_validates(raw):
    rows, _ = ex.build_rows(*raw)
    assert [r["source"] for r in rows] == ["real", "real", "stand-in", "stand-in"]
    assert ex.validate_rows(rows, ex.COLUMNS) == []


@pytest.mark.parametrize("outcome, status", [("ok", "ok"), ("ok_late", "timeout"), ("shed_queue_full", "rate_limited"),
                                             ("budget_exhausted", "rate_limited"), ("shed_deadline", "timeout")])
def test_pipeline_outcomes_map_to_the_four_statuses(outcome, status):
    assert ex.job_status({"outcome": outcome}) == (status, "recorded")


def test_retries_are_attempts_minus_one_and_never_negative(tmp_path):
    d = tmp_path / "s"
    write_jsonl(d / "scenario_jobs.jsonl", [job(0, attempts=4), job(1, outcome="shed_queue_full", attempts=0, ttft=None, p=None, c=None, latency=0.0)])
    rows, _ = ex.build_rows(None, d)
    assert [r["retries"] for r in rows] == ["3", "0"]
    assert rows[1]["first_token_ms"] == "" and rows[1]["tokens_in"] == "" and rows[1]["status"] == "rate_limited"


def test_an_ok_response_that_did_not_complete_is_an_error_not_an_answer(tmp_path):
    d = tmp_path / "r"
    write_jsonl(d / "x.jsonl", [attempt(complete=False)])
    assert ex.build_rows(d, None)[0][0]["status"] == "error"


def test_only_selected_seeds_and_modes_are_exported(tmp_path):
    d = tmp_path / "s"
    write_jsonl(d / "scenario_jobs.jsonl", [job(0, seed=1), job(0, seed=2), job(0, seed=1, mode="baseline")])
    assert len(ex.build_rows(None, d)[0]) == 1 + 1  # seed 1 only: two modes
    assert len(ex.build_rows(None, d, modes=["controlled"])[0]) == 1
    assert len(ex.build_rows(None, d, standin_seeds=[1, 2])[0]) == 3


def test_ids_are_unique_even_when_a_probe_was_rerun(tmp_path):
    d = tmp_path / "r"
    write_jsonl(d / "latency_probe.jsonl", [attempt("latency_probe_1")])
    write_jsonl(d / "latency_probe.jsonl", [attempt("latency_probe_1")])  # same label appended again
    rows, _ = ex.build_rows(d, None)
    assert len({r["request_id"] for r in rows}) == 2


def test_empty_evidence_is_an_error_not_an_empty_file(tmp_path):
    with pytest.raises(ex.ExportError, match="nothing to export"):
        ex.build_rows(tmp_path / "nope", tmp_path / "nada")
    (tmp_path / "e").mkdir()
    with pytest.raises(ex.ExportError):
        ex.build_rows(tmp_path / "e", None)


def test_real_and_standin_evidence_must_not_be_mixed(tmp_path):
    d = tmp_path / "standin"
    write_jsonl(d / "scenario_jobs.jsonl", [job(0, source="real")])
    with pytest.raises(ex.ExportError, match="STAND-IN folder"):
        ex.build_rows(None, d)
    r = tmp_path / "real"
    write_jsonl(r / "x.jsonl", [attempt(source="stand-in")])
    with pytest.raises(ex.ExportError, match="REAL folder"):
        ex.build_rows(r, None)


def test_a_real_job_line_needs_a_wall_clock_start(tmp_path):
    r = tmp_path / "real"
    write_jsonl(r / "eval_jobs.jsonl", [job(0, scenario="eval", mode=None, source="real")])
    with pytest.raises(ex.ExportError, match="run_started_at"):
        ex.build_rows(r, None)
    r2 = tmp_path / "real2"
    write_jsonl(r2 / "eval_jobs.jsonl", [job(0, scenario="eval", mode=None, source="real", run_id="eval-B", retrieval_setting="B",
                                            run_started_at=REAL_START, question_id="Q01")])
    row = ex.build_rows(r2, None)[0][0]
    assert row["scenario"] == "eval" and row["start_time"] == REAL_START and row["retrieval_setting"] == "B"
    assert row["request_id"] == "real-eval-B-000"


def test_malformed_lines_and_missing_fields_are_reported_with_their_line(tmp_path):
    d = tmp_path / "s"
    d.mkdir()
    (d / "scenario_jobs.jsonl").write_text("{not json}\n")
    with pytest.raises(ex.ExportError, match="line 1"):
        ex.build_rows(None, d)
    d2 = tmp_path / "s2"
    write_jsonl(d2 / "scenario_jobs.jsonl", [{"outcome": "ok", "job_id": 0}])
    with pytest.raises(ex.ExportError, match="missing"):
        ex.build_rows(None, d2)


# ---------------------------------------------------------------- grades
def eval_raw(tmp_path):
    r = tmp_path / "real"
    write_jsonl(r / "eval_jobs.jsonl", [job(i, scenario="eval", mode=None, source="real", run_id="eval-B", retrieval_setting="B",
                                           run_started_at=REAL_START, question_id=f"Q{i}") for i in range(3)])
    return r


def test_grades_are_joined_by_request_id_and_never_invented(tmp_path):
    r = eval_raw(tmp_path)
    rows, _ = ex.build_rows(r, None, grades={"real-eval-B-001": "correct", "real-eval-B-002": "refused_wrongly"})
    assert [x["answer_grade"] for x in rows] == ["", "correct", "refused_wrongly"]


# ---------------------------------------------------------------- validator
def good(**kw):
    row = {"request_id": "r1", "scenario": "s", "source": "real", "start_time": REAL_START, "first_token_ms": "100.0",
           "total_ms": "200.0", "status": "ok", "retries": "0", "tokens_in": "10", "tokens_out": "5", "retrieval_setting": "B",
           "answer_grade": ""}
    row.update(kw)
    return row


def fields(rows, header=None):
    return {(p.row, p.field) for p in ex.validate_rows(rows, header)}


def test_a_good_row_is_valid():
    assert ex.validate_rows([good()], ex.COLUMNS) == []


def test_empty_results_and_wrong_header_are_invalid():
    assert (0, "rows") in fields([])
    assert (0, "header") in fields([good()], ["request_id", "scenario"])
    assert (0, "header") in fields([good()], list(reversed(ex.COLUMNS)))  # order matters too


@pytest.mark.parametrize("field, value", [
    ("request_id", ""), ("scenario", ""), ("source", "standin"), ("source", "Real"), ("status", "success"), ("status", ""),
    ("start_time", "yesterday"), ("start_time", "2026-11-11T10:00:00"), ("start_time", ""),
    ("total_ms", ""), ("total_ms", "-1"), ("total_ms", "abc"), ("total_ms", "nan"),
    ("first_token_ms", "x"), ("first_token_ms", "-5"), ("first_token_ms", "999.0"),  # larger than total_ms
    ("retries", "-1"), ("retries", "1.5"), ("retries", ""),
    ("tokens_in", "1.5"), ("tokens_out", "-3"),
    ("retrieval_setting", "D"), ("retrieval_setting", "b"),
    ("answer_grade", "right"), ("answer_grade", "CORRECT"),
])
def test_invalid_values_are_flagged(field, value):
    assert any(f == field for _, f in fields([good(**{field: value})]))


def test_missing_columns_are_flagged():
    row = good()
    del row["tokens_in"]
    assert (1, "tokens_in") in fields([row])


def test_ok_rows_need_a_first_token_and_tokens_but_failed_rows_may_lack_them():
    assert (1, "first_token_ms") in fields([good(first_token_ms="")])
    assert (1, "tokens_in") in fields([good(tokens_in="")])
    assert fields([good(status="rate_limited", first_token_ms="", tokens_in="", tokens_out="")]) == set()


def test_duplicate_request_ids_are_flagged():
    assert (2, "request_id") in fields([good(), good()])


def test_grades_only_on_real_ok_rows_and_real_eval_rows_need_a_setting():
    assert (1, "answer_grade") in fields([good(source="stand-in", answer_grade="correct")])
    assert (1, "answer_grade") in fields([good(status="timeout", answer_grade="wrong", first_token_ms="", tokens_in="", tokens_out="")])
    assert fields([good(answer_grade="refused_correctly")]) == set()
    assert (1, "retrieval_setting") in fields([good(scenario="eval", retrieval_setting="")])
    assert fields([good(scenario="eval", source="stand-in", retrieval_setting="")]) == set()


def test_file_round_trip_and_cli(tmp_path, raw, capsys):
    real, standin = raw
    out, detail = tmp_path / "results.csv", tmp_path / "results_detail.csv"
    code = ex.main(["--real", str(real), "--standin", str(standin), "--grades", str(tmp_path / "none.csv"),
                    "--out", str(out), "--detail", str(detail)])
    assert code == 0 and out.exists() and detail.exists()
    header, rows = ex.read_results(out)
    assert header == ex.COLUMNS and len(rows) == 4
    assert ex.validate_file(out) == []
    assert ex.main(["--validate-only", str(out)]) == 0
    bad = tmp_path / "bad.csv"
    bad.write_text(out.read_text().replace(",ok,", ",fine,", 1))
    assert ex.main(["--validate-only", str(bad)]) == 1
    assert list(csv.DictReader(detail.open()))[0].keys() >= {"request_id", "outcome", "status_basis"}


def test_cli_does_not_write_anything_when_validation_fails(tmp_path):
    d = tmp_path / "s"
    write_jsonl(d / "scenario_jobs.jsonl", [job(0, ttft=3.0, latency=2.0)])  # first token after the end: impossible
    out = tmp_path / "results.csv"
    assert ex.main(["--real", str(tmp_path / "x"), "--standin", str(d), "--out", str(out), "--detail", str(tmp_path / "d.csv")]) == 1
    assert not out.exists()
    assert ex.main(["--real", str(tmp_path / "x"), "--standin", str(tmp_path / "y"), "--out", str(out)]) == 1  # nothing to export


# ================================================================ corrections: status, origin, runs, citations, detail file
def test_failed_job_takes_its_last_call_status_and_a_missing_one_is_error_and_inferred():
    assert ex.job_status({"outcome": "failed", "last_status": "timeout"}) == ("timeout", "recorded")
    assert ex.job_status({"outcome": "failed", "last_status": "rate_limited"}) == ("rate_limited", "recorded")
    # no terminal status recorded: never guessed from the 429 counts
    assert ex.job_status({"outcome": "failed", "n_429": 3, "n_other_failures": 0}) == ("error", "inferred")
    assert ex.job_status({"outcome": "failed"}) == ("error", "inferred")
    with pytest.raises(ex.ExportError):
        ex.job_status({"outcome": "mystery"})


@pytest.mark.parametrize("line, origin", [
    ({"outcome": "ok"}, ""),
    ({"outcome": "ok_late"}, "late_answer"),
    ({"outcome": "shed_queue_full"}, "local_admission"),
    ({"outcome": "budget_exhausted"}, "local_budget_gate"),
    ({"outcome": "shed_deadline"}, "local_deadline_shed"),
    ({"outcome": "failed", "last_status": "rate_limited"}, "provider_429"),
    ({"outcome": "failed", "last_status": "timeout"}, "provider_timeout"),
    ({"outcome": "failed", "last_status": "error"}, "provider_error"),
    ({"outcome": "failed"}, "unknown_inferred"),
])
def test_refusal_origin_tells_local_refusals_from_provider_responses(line, origin):
    status, basis = ex.job_status(line)
    assert ex.job_origin(line, status, basis) == origin
    assert ex.ORIGIN_STATUS[origin] == status  # the validator uses this table, so the two can never disagree


def test_the_detail_file_says_who_refused_and_how_the_time_was_measured(tmp_path):
    d, r = tmp_path / "s", tmp_path / "r"
    write_jsonl(d / "scenario_jobs.jsonl", [job(0, outcome="shed_queue_full", attempts=0, ttft=None, p=None, c=None, latency=0.0),
                                            job(1, outcome="shed_deadline", attempts=0, ttft=None, p=None, c=None, latency=0.4),
                                            job(2, outcome="failed", last_status="rate_limited", attempts=3, ttft=None, p=None, c=None),
                                            job(3, outcome="failed", attempts=2, ttft=None, p=None, c=None)])
    write_jsonl(r / "x.jsonl", [attempt(label="a", status="rate_limited", complete=False, first=None, p=None, c=None)])
    rows, details = ex.build_rows(r, d)
    by = {x["request_id"]: (row, x) for row, x in zip(rows, details)}
    got = {rid: (row["status"], x["refusal_origin"], x["status_basis"], x["time_basis"]) for rid, (row, x) in by.items()}
    assert got["standin-burst-controlled-s1-r1-000"] == ("rate_limited", "local_admission", "recorded", "simulated")
    assert got["standin-burst-controlled-s1-r1-001"] == ("timeout", "local_deadline_shed", "recorded", "simulated")
    assert got["standin-burst-controlled-s1-r1-002"] == ("rate_limited", "provider_429", "recorded", "simulated")
    assert got["standin-burst-controlled-s1-r1-003"] == ("error", "unknown_inferred", "inferred", "simulated")
    real = [v for k, v in got.items() if k.startswith("real-")][0]
    assert real == ("rate_limited", "provider_429", "recorded", "wall_clock")
    assert ex.validate_rows(rows, ex.COLUMNS) == [] and ex.validate_detail(ex.DETAIL_COLUMNS, details, rows) == []


# ---------------------------------------------------------------- repeated runs are never discarded
def test_every_independent_run_is_kept_and_labelled_nothing_is_chosen_latest(tmp_path):
    d = tmp_path / "s"
    old = [job(0, latency=9.0, ttft=8.0), job(1, latency=9.0, ttft=8.0)]
    new = [job(0, latency=2.0, ttft=1.0), job(1, latency=2.0, ttft=1.0)]
    write_jsonl(d / "scenario_jobs.jsonl", old + new)
    notes = []
    rows, details = ex.build_rows(None, d, notes=notes)
    assert len(rows) == 4 and len({r["request_id"] for r in rows}) == 4
    assert sorted(r["total_ms"] for r in rows) == ["2000.0", "2000.0", "9000.0", "9000.0"]  # the older run is NOT dropped
    assert sorted({r["scenario"] for r in rows}) == ["burst (controlled) [seed 1, run r1]", "burst (controlled) [seed 1, run r2]"]
    assert sorted({x["run"] for x in details}) == ["r1", "r2"]
    assert any("2 independent runs" in n and "ALL are exported" in n for n in notes)
    assert ex.validate_rows(rows, ex.COLUMNS) == []


def test_a_single_run_keeps_the_plain_scenario_name(raw):
    rows, _ = ex.build_rows(*raw)
    assert {r["scenario"] for r in rows if r["source"] == "stand-in"} == {"burst (controlled)"}


def test_run_id_separates_runs_even_when_job_ids_interleave_or_repeat(tmp_path):
    d = tmp_path / "s"
    write_jsonl(d / "scenario_jobs.jsonl", [job(0, run_id="aaaa1111"), job(0, run_id="bbbb2222"),
                                           job(1, run_id="aaaa1111"), job(1, run_id="bbbb2222")])
    rows, _ = ex.build_rows(None, d)
    assert len(rows) == 4 and len({r["request_id"] for r in rows}) == 4
    assert sorted({r["scenario"] for r in rows}) == ["burst (controlled) [seed 1, run aaaa1111]", "burst (controlled) [seed 1, run bbbb2222]"]
    assert "standin-burst-controlled-s1-aaaa1111-000" in {r["request_id"] for r in rows}


def test_two_seeds_are_two_groups_not_one_pooled_timeline(tmp_path):
    d = tmp_path / "s"
    write_jsonl(d / "scenario_jobs.jsonl", [job(0, seed=1), job(0, seed=2)])
    rows, _ = ex.build_rows(None, d, standin_seeds=[1, 2])
    assert sorted(r["scenario"] for r in rows) == ["burst (controlled) [seed 1, run r1]", "burst (controlled) [seed 2, run r1]"]


def test_only_a_repeated_export_of_the_same_request_is_dropped_and_it_is_reported(tmp_path):
    d = tmp_path / "s"
    write_jsonl(d / "scenario_jobs.jsonl", [job(0, run_id="a1"), job(0, run_id="a1"), job(1, run_id="a1")])
    notes = []
    rows, _ = ex.build_rows(None, d, notes=notes)
    assert len(rows) == 2 and any("repeated export" in n and "dropped" in n for n in notes)
    write_jsonl(tmp_path / "s2" / "scenario_jobs.jsonl", [job(0, run_id="a1"), job(0, run_id="a1", latency=5.0)])
    with pytest.raises(ex.ExportError, match="DIFFERENT content"):
        ex.build_rows(None, tmp_path / "s2")


def test_a_real_request_exported_twice_is_dropped_with_a_note_but_a_conflict_is_an_error(tmp_path):
    r = tmp_path / "real"
    line = job(0, scenario="eval", mode=None, source="real", run_id="e", retrieval_setting="A", run_started_at=REAL_START)
    write_jsonl(r / "eval_jobs.jsonl", [line, line])
    notes = []
    rows, _ = ex.build_rows(r, None, notes=notes)
    assert len(rows) == 1 and any("repeated export" in n for n in notes)
    r2 = tmp_path / "real2"
    write_jsonl(r2 / "eval_jobs.jsonl", [line, dict(line, latency_s=9.0)])
    with pytest.raises(ex.ExportError, match="DIFFERENT content"):
        ex.build_rows(r2, None)


def test_scenarios_cli_gives_every_run_its_own_run_id_and_the_exporter_keeps_all_runs(tmp_path):
    from app import scenarios as sc
    out = tmp_path / "standin"
    args = ["--retriever", "fake", "--repeats", "1", "--scenarios", "burst", "--modes", "controlled", "--out", str(out)]
    assert asyncio.run(sc.main(args)) == 0
    assert asyncio.run(sc.main(args)) == 0  # the same experiment, run again
    lines = [json.loads(x) for x in (out / "scenario_jobs.jsonl").read_text().splitlines()]
    ids = {x["run_id"] for x in lines}
    assert len(ids) == 2 and all(len(i) == 8 for i in ids)
    rows, details = ex.build_rows(None, out)
    assert len(rows) == len(lines) and len({x["run"] for x in details}) == 2
    assert len({r["scenario"] for r in rows}) == 2


# ---------------------------------------------------------------- citations: provenance (mechanical) and support (human)
def citation_raw(tmp_path):
    r = tmp_path / "real"
    write_jsonl(r / "eval_jobs.jsonl", [
        job(0, scenario="eval", mode=None, source="real", run_id="e", retrieval_setting="B", run_started_at=REAL_START, citation_status="cited"),
        job(1, scenario="eval", mode=None, source="real", run_id="e", retrieval_setting="B", run_started_at=REAL_START, citation_status="uncited"),
        job(2, scenario="eval", mode=None, source="real", run_id="e", retrieval_setting="B", run_started_at=REAL_START, citation_status="cited")])
    return r


def test_correct_needs_provenance_and_human_support_as_two_separate_checks(tmp_path):
    r = citation_raw(tmp_path)
    grades = {"real-e-000": "correct", "real-e-001": "correct", "real-e-002": "correct"}
    support = {"real-e-000": "supported", "real-e-001": "supported", "real-e-002": "unsupported"}
    rows, details = ex.build_rows(r, None, grades=grades, support=support)
    problems = ex.check_grades_against_citations(rows, details)
    by = {p.request_id: str(p) for p in problems}
    assert set(by) == {"real-e-001", "real-e-002"}
    assert "provenance" in by["real-e-001"] and "citation_support" not in by["real-e-001"]   # mechanical check failed, human said yes
    assert "citation_support" in by["real-e-002"] and "provenance" not in by["real-e-002"]   # mechanical ok, human said no
    assert [d["citation_provenance"] for d in details] == ["cited", "uncited", "cited"]
    assert [d["citation_support"] for d in details] == ["supported", "supported", "unsupported"]


def test_correct_without_any_human_support_check_is_refused_and_wrong_never_needs_one(tmp_path):
    r = citation_raw(tmp_path)
    rows, details = ex.build_rows(r, None, grades={"real-e-000": "correct"})
    assert "citation_support" in str(ex.check_grades_against_citations(rows, details)[0])  # existence of a citation is not enough
    rows, details = ex.build_rows(r, None, grades={"real-e-000": "wrong", "real-e-001": "refused_wrongly"}, support={"real-e-000": "unsupported"})
    assert ex.check_grades_against_citations(rows, details) == []


def test_grade_file_problems_are_errors(tmp_path):
    r = eval_raw(tmp_path)
    with pytest.raises(ex.ExportError, match="unknown request_id"):
        ex.build_rows(r, None, grades={"real-eval-B-999": "correct"})
    with pytest.raises(ex.ExportError, match="unknown request_id"):
        ex.build_rows(r, None, support={"real-eval-B-999": "supported"})
    d = tmp_path / "s"
    write_jsonl(d / "scenario_jobs.jsonl", [job(0)])
    with pytest.raises(ex.ExportError, match="only real rows"):
        ex.build_rows(None, d, grades={"standin-burst-controlled-s1-r1-000": "correct"})
    g = tmp_path / "g.csv"
    g.write_text("request_id,answer_grade\nA,correct\nA,wrong\n")
    with pytest.raises(ex.ExportError, match="twice"):
        ex.load_grades(g)
    assert ex.load_grades(tmp_path / "missing.csv") == {} and ex.load_support(tmp_path / "missing.csv") == {}


def test_citation_support_values_are_checked_when_the_grades_file_is_read(tmp_path):
    g = tmp_path / "g.csv"
    g.write_text("request_id,answer_grade,citation_support,note\nA,correct,supported,\nB,wrong,,\nC,wrong,unsupported,x\n")
    assert ex.load_support(g) == {"A": "supported", "C": "unsupported"}
    g.write_text("request_id,answer_grade,citation_support\nA,correct,yes\n")
    with pytest.raises(ex.ExportError, match="citation_support"):
        ex.load_support(g)


def test_cli_with_a_human_support_check_writes_the_grade_and_without_it_writes_nothing(tmp_path):
    r = tmp_path / "real"
    write_jsonl(r / "eval_jobs.jsonl", [job(0, scenario="eval", mode=None, source="real", run_id="e", retrieval_setting="A",
                                           run_started_at=REAL_START, citation_status="cited")])
    g, out = tmp_path / "grades.csv", tmp_path / "results.csv"
    common = ["--real", str(r), "--standin", str(tmp_path / "none"), "--grades", str(g), "--out", str(out), "--detail", str(tmp_path / "d.csv")]
    g.write_text("request_id,answer_grade\nreal-e-000,correct\n")
    assert ex.main(common) == 1 and not out.exists()
    g.write_text("request_id,answer_grade,citation_support\nreal-e-000,correct,supported\n")
    assert ex.main(common) == 0
    assert ex.read_results(out)[1][0]["answer_grade"] == "correct"
    assert list(csv.DictReader((tmp_path / "d.csv").open()))[0]["citation_support"] == "supported"


# ---------------------------------------------------------------- the detail file is validated too
def test_detail_validator_catches_unknown_values_mismatches_and_a_wrong_file(raw):
    rows, details = ex.build_rows(*raw)
    assert ex.validate_detail(ex.DETAIL_COLUMNS, details, rows) == []
    bad = [dict(d) for d in details]
    bad[0]["refusal_origin"] = "local_admission"  # but the row status is ok
    assert any(p.field == "refusal_origin" and "goes with status" in p.message for p in ex.validate_detail(ex.DETAIL_COLUMNS, bad, rows))
    bad = [dict(d) for d in details]
    bad[0]["refusal_origin"], bad[1]["time_basis"], bad[2]["status_basis"], bad[3]["citation_support"] = "mystery", "now", "guess", "yes"
    assert {p.field for p in ex.validate_detail(ex.DETAIL_COLUMNS, bad, rows)} == {"refusal_origin", "time_basis", "status_basis", "citation_support"}
    assert any(p.field == "request_id" for p in ex.validate_detail(ex.DETAIL_COLUMNS, details[:-1], rows))  # a row is missing
    assert any(p.field == "header" for p in ex.validate_detail(["request_id"], details, rows))


def test_the_cli_reports_origins_and_inferred_rows(tmp_path, capsys):
    d = tmp_path / "s"
    write_jsonl(d / "scenario_jobs.jsonl", [job(0), job(1, outcome="shed_queue_full", attempts=0, ttft=None, p=None, c=None, latency=0.0),
                                            job(2, outcome="failed", attempts=2, ttft=None, p=None, c=None)])
    assert ex.main(["--real", str(tmp_path / "x"), "--standin", str(d), "--out", str(tmp_path / "r.csv"), "--detail", str(tmp_path / "d.csv")]) == 0
    text = capsys.readouterr().out
    assert "local_admission=1" in text and "unknown_inferred=1" in text and "status is inferred (no recorded terminal status): 1" in text
