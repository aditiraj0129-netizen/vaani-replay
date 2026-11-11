from types import SimpleNamespace

from app import retrieval_diagnose as rd
from app.config import RetrievalConfig
from app.retrieval import Retriever


def ch(i, paper, text, section="s", page=1):
    return SimpleNamespace(chunk_id=f"{paper}-{i}", paper_key=paper, arxiv_id="1", version=1, section=section,
                           page_start=page, page_end=page, text=text)


def retr(chunks, top_k=1, rerank=False, candidates=0):
    return Retriever(chunks, RetrievalConfig("T", chunk_words=50, overlap_words=0, top_k=top_k, rerank=rerank, candidates=candidates))


G = [["needle fact"]]
Q = "needle fact about widgets"
FILLER = [ch(i, "b", "widgets are common widgets everywhere") for i in range(4)]


def one(chunks, question=Q, groups=G, paper="a", **kw):
    return rd.diagnose_config(question, paper, groups, retr(chunks, **kw))


def test_ok_when_the_gold_piece_is_sent():
    d = one([ch(0, "a", "the needle fact about widgets")] + FILLER)
    assert d["cause"] == "OK" and d["final_rank"] == 1 and d["n_gold"] == 1


def test_no_gold_piece_means_the_evidence_terms_match_nothing():
    assert one([ch(0, "a", "nothing relevant")] + FILLER)["cause"] == "NO_GOLD_CHUNK"


def test_a_chunk_boundary_that_cuts_the_evidence_is_named():
    d = one([ch(0, "a", "ends with the needle"), ch(1, "a", "fact starts here")] + FILLER)
    assert d["cause"] == "SPLIT_AT_CHUNK_BOUNDARY"
    # neighbours from different papers do not count
    assert one([ch(0, "a", "the needle"), ch(1, "b", "fact here")])["cause"] == "NO_GOLD_CHUNK"


def test_other_papers_crowd_out_the_gold_piece_but_a_paper_restricted_search_finds_it():
    chunks = [ch(0, "b", "widgets widgets widgets about widgets"), ch(1, "a", "the needle fact about widgets")] + FILLER[:2]
    d = one(chunks, question="widgets widgets about")
    assert d["cause"] == "OTHER_PAPER_CROWDING" and d["bm25_rank"] > 1 and d["paper_rank"] == 1


def test_no_shared_question_word_is_reported_as_no_overlap():
    d = one([ch(0, "a", "the needle fact lives here")] + FILLER, question="zebra giraffe")
    assert d["cause"] == "NO_QUERY_TERM_OVERLAP" and d["bm25_rank"] is None


def test_wording_mismatch_when_the_gold_piece_has_few_question_words():
    chunks = [ch(0, "a", "the needle fact")] + [ch(i, "a", "alpha beta gamma delta epsilon zeta") for i in range(1, 4)]
    d = one(chunks, question="alpha beta gamma delta epsilon zeta needle")
    assert d["cause"] == "WORDING_MISMATCH" and d["coverage"] < 0.5 and "alpha" in d["missing_terms"]


def test_bm25_scoring_when_most_words_match_but_other_pieces_score_higher():
    chunks = [ch(0, "a", "needle fact about widgets plus a very long tail " + "filler " * 40),
              ch(1, "a", "needle fact about widgets")]
    d = one(chunks, groups=[["long tail"]])  # only chunk 0 is gold; chunk 1 is shorter so it scores higher
    assert d["cause"] == "BM25_SCORING" and d["coverage"] >= 0.5 and d["bm25_rank"] == 2


def test_rerank_that_does_not_promote_is_not_blamed_for_dropping():
    # the gold piece is BM25 rank 2 but top_k is 1: it never was inside the top k, so the re-ranker did not drop it
    cands = [ch(0, "a", "needle fact about widgets " + "pad " * 30, section="other"),
             ch(1, "a", "needle fact about widgets", section="needle fact widgets")]
    d = rd.diagnose_config(Q, "a", [["pad pad"]], retr(cands, top_k=1, rerank=True, candidates=2))
    assert d["bm25_rank"] == 2 and d["final_rank"] is None and d["cause"] == "RERANK_DID_NOT_PROMOTE" and d["rerank_rank"] == 2


def test_rerank_dropped_only_when_bm25_had_it_inside_the_top_k():
    # BM25 ranks the gold piece first (repeated "alpha"); the re-ranker prefers the piece with every question word, a phrase match and a matching section title
    cands = [ch(0, "a", "alpha alpha alpha beta"), ch(1, "a", "alpha beta gamma " + "zz " * 15, section="alpha beta gamma"),
             ch(2, "a", "gamma gamma " + "qq " * 40)]
    d = rd.diagnose_config("alpha beta gamma", "a", [["alpha alpha"]], retr(cands, top_k=1, rerank=True, candidates=3))
    assert d["bm25_rank"] == 1 and d["final_rank"] is None and d["cause"] == "RERANK_DROPPED" and d["rerank_rank"] == 2


def test_bibliography_pieces_among_the_sent_pieces_are_counted():
    chunks = [ch(0, "a", "needle fact about widgets", section="References")] + FILLER
    d = one(chunks, top_k=2)
    assert d["refs_sent"] == 1 and d["n_sent"] == 2
    text = rd.render(rd.run([{"id": "Q1", "question": Q, "kind": "answerable", "paper": "a"}], {"Q1": G}, {"X": retr(chunks, top_k=2)}))
    assert "X: 1 of 2 pieces sent" in text


def test_run_and_render_cover_unanswerable_questions_and_count_causes():
    qs = [{"id": "Q1", "question": Q, "kind": "answerable", "paper": "a"}, {"id": "Q2", "question": "x?", "kind": "unanswerable", "paper": "a"}]
    rep = rd.run(qs, {"Q1": G}, {"X": retr([ch(0, "a", "the needle fact about widgets")] + FILLER)})
    assert rep[0]["configs"]["X"]["cause"] == "OK" and rep[1]["configs"] == {}
    text = rd.render(rep)
    assert "no evidence terms" in text and "X: OK=1" in text
