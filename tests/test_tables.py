import json
from datetime import timedelta
from types import SimpleNamespace

import pytest

from app import export_csv as ex
from app import tables as tb


def t(seconds):
    return ex.fmt_time(ex.SIM_EPOCH + timedelta(seconds=seconds))


def row(rid, start_s, first, total, status="ok", retries=0, tin="100", tout="10", setting="", grade="",
        scenario="burst (controlled)", source="stand-in"):
    return {"request_id": rid, "scenario": scenario, "source": source, "start_time": t(start_s), "first_token_ms": first,
            "total_ms": total, "status": status, "retries": str(retries), "tokens_in": tin, "tokens_out": tout,
            "retrieval_setting": setting, "answer_grade": grade}


BURST = [
    row("r1", 0.0, "500.0", "1000.0", tin="100", tout="10", retries=0),
    row("r2", 1.0, "1500.0", "3000.0", tin="200", tout="20", retries=1),
    row("r3", 2.0, "", "100.0", status="rate_limited", tin="", tout="", retries=0),
    row("r4", 3.0, "", "50.0", status="timeout", tin="", tout="", retries=2),
]


def test_percentile_is_nearest_rank():
    assert tb.percentile([], 50) is None
    assert tb.percentile([5], 99) == 5
    v = list(range(1, 101))
    assert (tb.percentile(v, 50), tb.percentile(v, 95), tb.percentile(v, 99)) == (50, 95, 99)
    assert tb.percentile([3, 1, 2], 50) == 2 and tb.percentile([1, 2], 95) == 2


def test_table_2a_matches_hand_computed_numbers():
    headers, body, notes = tb.table_2a(BURST)
    assert headers[0] == "Scenario" and len(headers) == 7
    assert body == [["burst (controlled)", "stand-in", "4", "4.0", "60.0", "500 / 1,500 / 1,500", "1,000 / 3,000 / 3,000"]]
    assert "2 ok of 4" in notes[0]


def test_nothing_is_hardcoded_changing_the_data_changes_the_tables():
    faster = [dict(r, total_ms=str(float(r["total_ms"]) / 2), first_token_ms=("" if r["first_token_ms"] == "" else str(float(r["first_token_ms"]) / 2)))
              for r in BURST]
    assert tb.table_2a(faster)[1][0][6] == "500 / 1,500 / 1,500"  # totals halved: 500 / 1,500 / 1,500
    assert tb.table_2a(faster)[1][0][6] != tb.table_2a(BURST)[1][0][6]
    assert tb.table_2a(BURST[:3])[1][0][2] == "3"


def test_a_single_request_has_no_offered_rate():
    assert tb.table_2a([row("a", 0, "5.0", "9.0")])[1][0][4] == "Not measured"


def test_scenarios_are_ordered_real_first_then_by_scenario_and_mode():
    rows = [row("1", 0, "1.0", "2.0", scenario="overload (controlled)"), row("2", 0, "1.0", "2.0", scenario="steady (baseline)"),
            row("3", 0, "1.0", "2.0", scenario="real_probe", source="real"), row("4", 0, "1.0", "2.0", scenario="steady (controlled)")]
    assert [r[0] for r in tb.table_2a(rows)[1]] == ["real_probe", "steady (baseline)", "steady (controlled)", "overload (controlled)"]
    assert [r[0] for r in tb.table_2a(rows, modes=["controlled"])[1]] == ["real_probe", "steady (controlled)", "overload (controlled)"]


CFG = {"A": SimpleNamespace(chunk_words=80, overlap_words=20, top_k=8, rerank=False),
       "B": SimpleNamespace(chunk_words=200, overlap_words=40, top_k=5, rerank=False)}


def ev(rid, setting, status, first, total, tin, tout, grade):
    return row(rid, 0, first, total, status=status, tin=tin, tout=tout, setting=setting, grade=grade, scenario="eval", source="real")


def test_table_3_shows_counts_and_percent_of_questions_asked_for_every_answer_grade():
    rows = [ev("a1", "A", "ok", "800.0", "1500.0", "1000", "50", "correct"), ev("a2", "A", "ok", "1000.0", "2500.0", "1200", "70", "wrong"),
            ev("a3", "A", "ok", "900.0", "2000.0", "1100", "60", "refused_correctly"),
            ev("a4", "A", "timeout", "", "30000.0", "", "", ""),
            ev("b1", "B", "ok", "700.0", "1000.0", "1800", "60", ""), ev("b2", "B", "ok", "800.0", "1100.0", "1800", "60", "")]
    headers, body, notes = tb.table_3(rows, CFG)
    assert len(headers) == 11 and headers[2:8] == ["Correct n (%)", "Partial n (%)", "Wrong n (%)", "Refused correctly n (%)",
                                                    "Refused wrongly n (%)", "Not graded n (%)"]
    assert body[0] == ["A: 80-word pieces (20 overlap), 8 fetched, re-rank no", "4", "1 (25.0%)", "0 (0.0%)", "1 (25.0%)", "1 (25.0%)",
                       "0 (0.0%)", "1 (25.0%)", "1,160", "900 / 1,000", "2,000 / 2,500"]
    assert body[1][2:8] == ["0 (0.0%)"] * 5 + ["2 (100.0%)"] and body[1][1] == "2"
    assert "not answered=1" in notes[0]
    rows2 = [ev("c1", "A", "ok", "1.0", "2.0", "10", "1", "refused_wrongly"), ev("c2", "A", "ok", "1.0", "2.0", "10", "1", "partial")]
    b2 = tb.table_3(rows2, CFG)[1][0]
    assert b2[3] == "1 (50.0%)" and b2[6] == "1 (50.0%)"   # partial and refused_wrongly are their own columns, never merged into "wrong"


def test_table_3b_splits_by_review_status_and_labels_the_mixture():
    rows = [ev("a1", "A", "ok", "1", "2", "1", "1", "correct"), ev("a2", "A", "ok", "1", "2", "1", "1", "correct"),
            ev("a3", "A", "ok", "1", "2", "1", "1", "correct"), ev("a4", "A", "ok", "1", "2", "1", "1", "correct"),
            ev("a5", "A", "ok", "1", "2", "1", "1", "partial"), ev("a6", "A", "ok", "1", "2", "1", "1", "refused_correctly"),
            ev("a7", "A", "ok", "1", "2", "1", "1", "correct")]
    detail = [det("a1", 1, prov="cited", sup="supported"), det("a2", 1, prov="uncited", sup="uncited"),
              det("a3", 1, prov="cited", sup="unsupported"), det("a4", 1, prov="cited", sup="partial"),
              det("a5", 1, prov="cited", sup="supported"), det("a6", 1, prov="refusal", sup=""), det("a7", 1, prov="cited", sup="")]
    reviews = {"a1": "manual_review", "a2": "manual_review", "a3": "assistant_first_pass", "a4": "assistant_first_pass",
               "a5": "assistant_first_pass", "a6": "assistant_first_pass", "a7": "assistant_first_pass"}
    headers, body, notes = tb.table_3b(rows, CFG, detail, reviews)
    by = {r[1]: dict(zip(headers, r)) for r in body}
    assert list(by) == ["Manual review", "Assistant first pass (unverified)", "ALL = mixture of the lines above"]
    m, f, al = by["Manual review"], by["Assistant first pass (unverified)"], by["ALL = mixture of the lines above"]
    assert m["Rows n (%)"] == "2 (28.6%)" and f["Rows n (%)"] == "5 (71.4%)" and al["Rows n (%)"] == "7 (100.0%)"
    assert (m["Supported n (%)"], m["Uncited n (%)"], m["Correct, uncited n (%)"]) == ("1 (14.3%)", "1 (14.3%)", "1 (14.3%)")
    assert f["No citation (refused) n (%)"] == "1 (14.3%)" and f["Support missing n (%)"] == "1 (14.3%)"   # a7: answered, no support value
    assert (al["Correct and supported n (%)"], al["Correct, partial citation n (%)"], al["Correct, unsupported n (%)"],
            al["Correct, uncited n (%)"]) == ("1 (14.3%)",) * 4
    assert any("MIXES" in n and "not an independently human-verified" in n for n in notes)


def test_table_3c_counts_each_review_status_and_rounding_is_half_up():
    rows = [ev(f"a{i}", "A", "ok", "1", "2", "1", "1", "correct") for i in range(16)]
    reviews = {f"a{i}": ("manual_review" if i < 9 else "assistant_first_pass") for i in range(16)}
    body = tb.table_3c(rows, reviews)[1]
    assert body[0] == ["A", "16", "9 (56.3%)", "7 (43.8%)", "0 (0.0%)"] and body[-1][0] == "all settings"
    assert tb.pct(9, 16) == "56.3" and tb.pct(1, 16) == "6.3" and tb.pct(3, 16) == "18.8" and tb.pct(1, 3) == "33.3"
    assert tb.table_3c(rows, None)[1][0][0].startswith("Not measured")


def test_table_3_without_real_evaluation_rows_says_not_measured():
    assert tb.table_3(BURST, CFG)[1][0][0].startswith("Not measured")


def raw_line(status="ok", t_start=None, total=200.0, kind=None, retry_after=None, message=None, wave=None, limits=True):
    rl = {"limit_requests": 1000, "limit_tokens": 8000} if limits else {}
    if retry_after is not None:
        rl["retry_after_s"] = retry_after
    ln = {"logged_at": "2026-11-11T10:00:00+00:00", "result": {"status": status, "complete": status == "ok", "total_ms": total,
          "http_status": 200 if status == "ok" else 429, "limit_kind": kind, "error_message": message, "ratelimit": rl}}
    if t_start is not None:
        ln["t_start_s"] = t_start
    if wave is not None:
        ln["wave"] = wave
    return ln


def write(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(x) for x in lines) + "\n")


MSG = "Rate limit reached for model in organization org_abc123XYZ service tier on_demand on tokens per minute (TPM): Limit 8000, Used 6800, Requested 1900. Please try again in 5s"


def test_table_1_with_no_evidence_is_all_not_measured(tmp_path):
    rows = tb.table_1(tmp_path / "nothing")[1]
    assert all(r[1].startswith("Not measured") for r in rows)


def test_table_1_reports_a_rejection_wave_when_there_was_one(tmp_path):
    real = tmp_path / "real"
    write(real / "concurrency_probe.jsonl", [raw_line(wave=2), raw_line("rate_limited", wave=4, kind="concurrency")])
    assert "First rejection in the wave of 4" in {r[0]: r for r in tb.table_1(real)[1]}["Requests in flight at once"][1]


def test_build_all_refuses_an_invalid_results_file(tmp_path):
    bad = tmp_path / "results.csv"
    bad.write_text("request_id,scenario\nx,y\n")
    with pytest.raises(ValueError, match="invalid"):
        tb.build_all(bad, tmp_path, None, None, configs=CFG)


def test_errors_count_as_given_up_so_the_three_percentages_cover_every_row():
    rows = [row("a", 0, "1.0", "2.0"), row("b", 1, "", "5.0", status="error", tin="", tout=""),
            row("c", 2, "", "5.0", status="timeout", tin="", tout=""), row("d", 3, "", "1.0", status="rate_limited", tin="", tout="")]
    _, body, notes = tb.table_2b(rows, None)
    assert body[0][1:4] == ["25.0", "25.0", "50.0"]  # timeout + error = given up
    assert "error=1" in notes[0]


# ================================================================ corrections: queue, origins, three kinds of numbers
def det(rid, attempts, queue_wait_ms="", origin="", prov="", sup="", run="r1", time_basis="simulated"):
    d = {c: "" for c in ex.DETAIL_COLUMNS}
    d.update(request_id=rid, outcome="x", status_basis="recorded", refusal_origin=origin, time_basis=time_basis, run=run,
             attempts=str(attempts), queue_wait_ms=queue_wait_ms, citation_provenance=prov, citation_support=sup)
    return d


BURST_DETAIL = [det("r1", 1, "200.0"), det("r2", 1, "1500.0", origin=""), det("r3", 0, origin="local_admission"),
                det("r4", 0, origin="local_deadline_shed")]


def test_table_2b_matches_hand_computed_numbers():
    pricing = {"in": 0.1, "out": 0.5}
    headers, body, notes = tb.table_2b(BURST, pricing, detail=BURST_DETAIL)
    assert len(headers) == 8 and headers[5].startswith("Peak queue length")
    scenario, ok, rl, to, retries, peak, tokens, cost = body[0]
    assert (scenario, ok, rl, to, retries, tokens) == ("burst (controlled)", "50.0", "25.0", "25.0", "0.75", "165")
    # waiting to be sent: r1 0-0.2 s, r2 1.0-2.5 s, r3 2.0-2.1 s (refused after 100 ms), r4 3.0-3.05 s -> at most 2 at once (r2 and r3)
    assert peak == "2"
    assert cost == "free tier n/a (simulated); paid-equivalent $0.0225"  # (150 in * 0.1 + 15 out * 0.5) / 1e6 * 1000
    assert "ok=2 rate_limited=1 timeout=1 error=0" in notes[0] and "peak_pending_to_first_response=2" in notes[0]


def test_without_the_detail_file_the_queue_length_is_not_measured_never_guessed():
    assert tb.table_2b(BURST, None)[1][0][5] == "Not measured (no results_detail.csv)"
    real = [row("a", 0, "500.0", "900.0", scenario="real_x", source="real")]
    assert tb.table_2b(real, None, detail=[det("a", 1)])[1][0][5] == "Not measured (no queue telemetry)"  # a real probe has no queue


def test_the_two_peak_metrics_differ_when_requests_are_in_service_not_in_a_queue():
    a = row("a", 0.0, "5000.0", "6000.0")
    b = row("b", 1.0, "5000.0", "6000.0")
    detail = {"a": det("a", 1, "100.0"), "b": det("b", 1, "100.0")}
    assert tb.peak_pending_to_first_response([a, b]) == 2       # both are waiting for a first token...
    assert tb.peak_waiting_to_send([a, b], detail) == 1          # ...but each was sent after 100 ms: they were never queued together
    assert tb.peak_waiting_to_send([a, b], {"a": detail["a"]}) is None  # a row without telemetry: not measured


def test_peak_pending_counts_overlap_and_handles_ties():
    a = row("a", 0.0, "1000.0", "1000.0")
    b = row("b", 1.0, "1000.0", "1000.0")  # starts exactly when a gets its first token
    assert tb.peak_pending_to_first_response([a, b]) == 1
    c = row("c", 0.5, "1000.0", "1000.0")
    assert tb.peak_pending_to_first_response([a, b, c]) == 2
    assert tb.peak_pending_to_first_response([row("d", 0, "", "300.0", status="rate_limited")]) == 1  # a refusal waits until it is refused
    assert tb.peak_pending_to_first_response([row("e", 0, "", "0.0", status="rate_limited")]) == 0  # refused at once: nobody waited


def test_a_baseline_run_has_no_queue_and_a_request_refused_at_the_door_never_queued():
    rows = [row("a", 0.0, "300.0", "500.0"), row("b", 0.1, "300.0", "500.0"), row("c", 0.2, "", "0.0", status="rate_limited", tin="", tout="")]
    detail = [det("a", 1, "0.0"), det("b", 1, "0.0"), det("c", 0, origin="local_admission")]
    assert tb.table_2b(rows, None, detail=detail)[1][0][5] == "0"


def test_table_2c_counts_who_refused():
    rows = [row("a", 0, "1.0", "2.0"), row("b", 1, "", "1.0", status="rate_limited", tin="", tout=""),
            row("c", 2, "", "1.0", status="rate_limited", tin="", tout=""), row("d", 3, "", "5.0", status="timeout", tin="", tout=""),
            row("e", 4, "", "5.0", status="error", tin="", tout="")]
    detail = [det("a", 1, "0.0"), det("b", 0, origin="provider_429"), det("c", 0, origin="local_budget_gate"),
              det("d", 0, origin="local_deadline_shed"), det("e", 1, origin="unknown_inferred")]
    headers, body, notes = tb.table_2c(rows, detail)
    got = dict(zip(headers, body[0]))
    assert got["Requests sent"] == "5" and got["Provider 429"] == "1" and got["Local: budget gate"] == "1"
    assert got["Local: deadline shed"] == "1" and got["Status inferred"] == "1" and got["Local: queue full"] == "0"
    assert tb.table_2c(rows, None)[1][0][0].startswith("Not measured")
    assert "OUR controller" in notes[0]


def test_real_rows_report_zero_free_tier_spend_and_missing_prices_are_not_invented():
    real = [row("a", 0, "500.0", "900.0", scenario="real_x", source="real")]
    cost = tb.table_2b(real, None)[1][0][-1]
    assert cost == "free tier $0.00; paid-equivalent Not measured (no prices in pricing.json)"
    assert tb.table_2b(real, {"in": 1.0, "out": 1.0})[1][0][-1] == "free tier $0.00; paid-equivalent $0.1100"


def test_a_null_price_is_not_a_price_and_prices_are_labelled_as_published_not_observed(tmp_path):
    f = tmp_path / "pricing.json"
    f.write_text(json.dumps({"input_per_million_usd": None, "output_per_million_usd": None, "source_url": "u", "checked_on": "d"}))
    assert tb.load_pricing(f) is None and tb.load_pricing(tmp_path / "missing.json") is None
    assert "Not measured" in tb.pricing_note(None) and "$0.00" in tb.pricing_note(None)
    f.write_text(json.dumps({"input_per_million_usd": 0.5, "output_per_million_usd": 2, "source_url": "https://example.test/p", "checked_on": "2026-11-11"}))
    price = tb.load_pricing(f)
    assert price == {"in": 0.5, "out": 2.0, "source_url": "https://example.test/p", "checked_on": "2026-11-11"}
    note = tb.pricing_note(price)
    assert "LIST PRICES" in note and "not observed charges" in note and "https://example.test/p" in note and "2026-11-11" in note


def test_scenario_names_with_a_run_label_are_parsed_ordered_and_mode_filtered():
    assert tb.split_name("burst (controlled) [seed 1, run r2]") == ("burst", "controlled", "[seed 1, run r2]")
    assert tb.split_name("real_limit_probe") == ("real_limit_probe", "", "")
    rows = [row("1", 0, "1.0", "2.0", scenario="burst (baseline) [seed 1, run r2]"), row("2", 0, "1.0", "2.0", scenario="burst (controlled) [seed 1, run r1]"),
            row("3", 0, "1.0", "2.0", scenario="steady (controlled)")]
    assert [r[0] for r in tb.table_2a(rows)[1]] == ["steady (controlled)", "burst (baseline) [seed 1, run r2]", "burst (controlled) [seed 1, run r1]"]
    assert [r[0] for r in tb.table_2a(rows, modes=["controlled"])[1]] == ["steady (controlled)", "burst (controlled) [seed 1, run r1]"]


def test_table_3b_notes_keep_provenance_apart_from_support():
    rows = [ev("a1", "A", "ok", "800.0", "1500.0", "1000", "50", "correct"), ev("a2", "A", "ok", "1000.0", "2500.0", "1200", "70", "wrong")]
    detail = [det("a1", 1, prov="cited", sup="supported"), det("a2", 1, prov="uncited", sup="uncited")]
    notes = tb.table_3b(rows, CFG, detail)[2]
    assert any("provenance" in n and "cited=1" in n and "uncited=1" in n for n in notes)


def raw_published(tmp_path):
    real = tmp_path / "real"
    write(real / "limit_probe.jsonl", [raw_line(t_start=0.0), raw_line(t_start=2.0),
                                       raw_line("rate_limited", t_start=10.0, total=200.0, kind="tpm", retry_after=5, message=MSG),
                                       raw_line(t_start=16.0)])
    write(real / "concurrency_probe.jsonl", [raw_line(wave=1), raw_line(wave=2), raw_line(wave=2), raw_line(wave=4)])
    return real


def test_table_1_is_built_from_the_raw_real_evidence_and_published_limits_stay_apart(tmp_path):
    real = raw_published(tmp_path)
    headers, rows, notes = tb.table_1(real, {"rpm": 30, "tpm": 8000, "rpd": 14400, "checked_on": "2026-11-11"})
    assert headers == ["Limit", "Value you found", "How you found it", "Confidence (high, medium, low)"]
    by = {r[0]: r for r in rows}
    assert [r[0] for r in rows] == ["Requests per minute", "Tokens per minute", "Requests per day", "Tokens per day",
                                    "Requests in flight at once", "What it returns when a limit is hit", "Time until it recovers"]
    assert by["Tokens per minute"][1] == "8,000 per minute" and by["Tokens per minute"][3] == "high"
    assert by["Requests per day"][1] == "1,000 per day" and by["Requests per day"][3] == "medium"
    assert by["Requests per minute"][1].startswith("Not measured")          # a published 30/min is NOT an observation
    assert all("ublished" not in cell for r in rows for cell in r)          # published figures never leak into the table body
    assert all(r[2].startswith("Observed") for r in rows if not r[1].startswith("Not measured"))
    published = [n for n in notes if n.startswith("Published limits")][0]
    assert "NOT observed" in published and "rpm=30" in published and "tpm=8,000 (matches the observed header)" in published
    assert "rpd=14,400 (DIFFERS from the observed header 1,000)" in published and "tpd=Not measured" in published
    assert by["Tokens per day"][1].startswith("Not measured")
    assert by["Requests in flight at once"][1] == "No rejection up to 4 simultaneous requests (not tested higher)"
    ret = by["What it returns when a limit is hit"]
    assert "HTTP 429" in ret[1] and "'tpm': 1" in ret[1] and "5" in ret[1] and "org_<redacted>" in ret[2] and "org_abc123XYZ" not in ret[2]
    assert "5.8 s" in by["Time until it recovers"][1] and "Daily limit: Not measured" in by["Time until it recovers"][1]


def test_null_published_limits_are_missing_not_a_crash_and_not_a_number(tmp_path):
    real = raw_published(tmp_path)
    _, rows, notes = tb.table_1(real, {"rpm": None, "tpm": None, "rpd": None, "tpd": None, "checked_on": "2026-11-11"})
    assert any(n.startswith("Published limits") and "Not measured (no values" in n for n in notes)
    assert {r[0]: r for r in rows}["Requests per minute"][1].startswith("Not measured")


def test_end_to_end_tables_from_a_csv_file(tmp_path):
    out = tmp_path / "results.csv"
    ex.write_csv(out, ex.COLUMNS, BURST)
    ex.write_csv(tmp_path / "results_detail.csv", ex.DETAIL_COLUMNS, BURST_DETAIL)
    text, parts = tb.build_all(out, tmp_path / "noreal", None, None, configs=CFG, detail_path=tmp_path / "results_detail.csv")
    for title in ("### Table 1", "### Table 2a", "### Table 2b", "### Table 2c", "### Table 3", "### Table 3b", "### Table 3c"):
        assert title in text
    assert "burst (controlled)" in text and "Not measured" in text and set(parts) == {"table1", "table2a", "table2b", "table2c", "table3", "table3b", "table3c"}
    assert "Paid-equivalent prices: Not measured" in text and "simulated" in text
    assert tb.main(["--results", str(out), "--real", str(tmp_path / "noreal"), "--detail", str(tmp_path / "results_detail.csv"),
                    "--out", str(tmp_path / "t.md")]) == 0
    assert (tmp_path / "t.md").exists() and (tmp_path / "t.json").exists()
    assert tb.main(["--results", str(tmp_path / "missing.csv"), "--out", str(tmp_path / "x.md")]) == 1


def test_tables_refuse_a_detail_file_that_does_not_belong_to_the_results(tmp_path):
    out = tmp_path / "results.csv"
    ex.write_csv(out, ex.COLUMNS, BURST)
    ex.write_csv(tmp_path / "d.csv", ex.DETAIL_COLUMNS, BURST_DETAIL[:-1])
    with pytest.raises(ValueError, match="results_detail.csv is invalid"):
        tb.build_all(out, tmp_path / "noreal", None, None, configs=CFG, detail_path=tmp_path / "d.csv")
    text, _ = tb.build_all(out, tmp_path / "noreal", None, None, configs=CFG, detail_path=tmp_path / "absent.csv")  # absent: tables still build
    assert "Not measured (no results_detail.csv)" in text
