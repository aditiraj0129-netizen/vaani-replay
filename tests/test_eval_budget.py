from types import SimpleNamespace

from app import eval_budget as eb


def test_budget_arithmetic_and_upper_bound():
    b = eb.estimate_budget({"A": [1000, 1000], "B": [2000, 2000]}, scale=1.0)
    a, bb = b["configs"]["A"], b["configs"]["B"]
    assert a["per_request_typ"] == 1000 + eb.COMPLETION_TYPICAL and a["total_typ"] == 2 * (1000 + eb.COMPLETION_TYPICAL)
    assert bb["per_request_hi"] == 2000 * eb.PROMPT_HIGH_FACTOR + eb.COMPLETION_HIGH
    assert b["calls_min"] == 4
    assert b["total_typ"] == a["total_typ"] + bb["total_typ"]
    assert b["total_hi"] == (a["total_hi"] + bb["total_hi"]) * (1 + eb.RETRY_ALLOWANCE) and b["total_hi"] > b["total_typ"]


def test_scale_is_applied_to_prompts():
    assert eb.estimate_budget({"B": [1000]}, scale=2.0)["configs"]["B"]["prompt_typ"] == 2000


def test_prompt_estimate_counts_the_whole_prompt():
    chunk = SimpleNamespace(chunk_id="c", paper_key="attention", version=7, page_start=1, page_end=1, section="s", text="x" * 400)
    r = SimpleNamespace(retrieve=lambda q: [SimpleNamespace(chunk=chunk, score=1.0)])
    assert eb.prompt_estimate(r, "q?") > 100  # at least the 400-character passage (100 tokens)


def test_questions_file_has_15_rows_and_the_agreed_columns():
    rows = eb.load_questions(eb.Path("data/eval/questions.csv"))
    assert len(rows) == 16 and list(rows[0]) == ["id", "question", "kind", "paper", "expected_answer", "must_include", "evidence_page"]
    assert {r["kind"] for r in rows} <= {"answerable", "partial", "unanswerable"}
