from types import SimpleNamespace

from app import eval_verify as ev


def chunk(i, paper, text, section="s", pages=(3, 3)):
    return SimpleNamespace(chunk_id=f"{paper}-{i}", paper_key=paper, arxiv_id="1", version=7, section=section,
                           page_start=pages[0], page_end=pages[1], text=text)


def test_group_needs_every_term_and_terms_accept_alternatives():
    t = ev.norm("the dot  products grow;\nsoftmax has small gradients")
    assert ev.group_matches(t, ["dot product", "softmax"])
    assert not ev.group_matches(t, ["dot product", "cosine"])
    assert ev.group_matches(t, ["160 ms/small gradients"])
    assert ev.any_group(t, [["cosine"], ["softmax"]]) == 1


def test_scan_only_looks_inside_the_named_paper():
    cs = [chunk(0, "moshi", "latency is 160 ms"), chunk(1, "attention", "latency is 160 ms", pages=(7, 8))]
    found = ev.find_evidence(cs, "attention", [["160 ms"]])
    assert [f["chunk_id"] for f in found] == ["attention-1"] and found[0]["pages"] == (7, 8)


def test_retrieval_rank_is_the_first_hit_that_matches_a_group():
    hits = [SimpleNamespace(chunk=chunk(i, "x", t)) for i, t in enumerate(["nothing", "has softmax and values"])]
    assert ev.retrieval_rank(hits, [["softmax", "values"]]) == 2
    assert ev.retrieval_rank(hits, [["cosine"]]) is None


def test_labels_separate_missing_evidence_from_retrieval_failure():
    assert ev.label(False, {"A": 1}, {"A": 1}).startswith("NO MATCHING PASSAGE")
    assert ev.label(True, {"A": 1, "B": 2}, {}).startswith("ok")
    miss = ev.label(True, {"A": 1, "B": None}, {"B": 3})
    assert "RETRIEVAL FAILURE (configs B)" in miss and "crowd it out" in miss
    assert "crowd" not in ev.label(True, {"A": None}, {"A": None})


def test_excerpt_is_centred_on_the_matching_term():
    text = "x " * 200 + "the softmax saturates " + "y " * 200
    e = ev.excerpt(text, ["softmax"])
    assert "softmax" in e and len(e) <= 230


def test_proposal_file_covers_every_question_id():
    from app.eval_budget import load_questions
    from pathlib import Path
    ids = {r["id"] for r in load_questions(Path("data/eval/questions.csv"))}
    # an unanswerable question has no evidence terms on purpose: its absence is checked by app/check_absence.py
    answerable = {r["id"] for r in load_questions(Path("data/eval/questions.csv")) if r["kind"] != "unanswerable"}
    assert set(ev.load_groups(Path("data/eval/evidence_proposals.csv"))) == answerable
    assert ids - answerable == {"Q16"}


def test_terms_still_match_when_the_pdf_text_glued_words_together():
    t = ev.norm("we find Thesecond category and the dot products grow")
    assert ev.group_matches(t, ["the second category"]) and ev.group_matches(t, ["dotproducts"])
    assert not ev.group_matches(t, ["third category"])


def test_a_cross_paper_question_scans_every_paper_it_names():
    cs = [chunk(0, "moshi", "the first full-duplex system"), chunk(1, "dynamic_chunking", "hierarchical processing"),
          chunk(2, "attention", "full-duplex hierarchical")]
    found = ev.find_evidence(cs, "moshi|dynamic_chunking", [["full-duplex"], ["hierarchical"]], limit=0)
    assert sorted(f["chunk_id"] for f in found) == ["dynamic_chunking-1", "moshi-0"]
