import json

from app import fix_eval_pacing as fx

JOB = {"scenario": "eval", "retrieval_setting": "A", "question_id": "Q05", "outcome": "ok_late", "arrival_s": 100.0, "first_send_s": 150.0,
       "end_s": 151.2, "queue_wait_s": 50.0, "latency_s": 51.2, "deadline_s": 45.0, "ttft_from_arrival_s": 50.9}


def test_pacing_wait_is_moved_out_of_latency_and_the_outcome_is_recomputed():
    f = fx.fix_line(JOB, 800.0)
    assert f["pacing_wait_s"] == 50.0 and f["arrival_s"] == 150.0 and f["latency_s"] == 1.2 and f["queue_wait_s"] == 0.0
    assert f["outcome"] == "ok" and f["ttft_from_arrival_s"] == 0.8 and "pacing_fix" in f


def test_a_really_slow_answer_stays_late():
    f = fx.fix_line(dict(JOB, end_s=200.0), 800.0)
    assert f["outcome"] == "ok_late"


def test_fixing_twice_changes_nothing_and_the_original_is_archived(tmp_path):
    jobs, att, arc = tmp_path / "j.jsonl", tmp_path / "a.jsonl", tmp_path / "arc" / "o.jsonl"
    jobs.write_text(json.dumps(JOB) + "\n")
    att.write_text(json.dumps({"retrieval_setting": "A", "question_id": "Q05", "result": {"status": "ok", "first_token_ms": 800.0}}) + "\n")
    fx.main(jobs_path=jobs, attempts_path=att, archive=arc)
    once = jobs.read_text()
    fx.main(jobs_path=jobs, attempts_path=att, archive=arc)
    assert jobs.read_text() == once and json.loads(arc.read_text())["outcome"] == "ok_late" and json.loads(once)["outcome"] == "ok"
