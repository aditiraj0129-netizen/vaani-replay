import json

from app import fix_eval_provenance as fp

JOB = {"scenario": "eval", "retrieval_setting": "A", "question_id": "Q10", "citation_status": "invalid", "cited": [3],
       "unsupported_numbers": [], "answer": "Pattern [0, 2, 2, 2] gives 240 ms [1].",
       "passages": [{"n": 1, "text": "we switch to a reduced latency of 240 ms with the pattern"}]}


def test_restates_the_flag_keeps_the_original_and_is_idempotent(tmp_path):
    jobs, arc = tmp_path / "j.jsonl", tmp_path / "a" / "o.jsonl"
    jobs.write_text(json.dumps(JOB) + "\n" + json.dumps({"scenario": "x"}) + "\n")
    fp.main(jobs_path=jobs, archive=arc)
    first = jobs.read_text()
    new = json.loads(first.splitlines()[0])
    assert new["citation_status"] == "cited" and new["cited"] == [1] and "provenance_fix" in new
    assert json.loads(arc.read_text().splitlines()[0])["citation_status"] == "invalid"
    fp.main(jobs_path=jobs, archive=arc)
    assert jobs.read_text() == first
