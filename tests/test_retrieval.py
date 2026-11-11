import pytest

from app.config import RetrievalConfig
from app.retrieval import BM25Index, Retriever, tokenize
from tests.conftest import mk_chunk


def index(*texts):
    return BM25Index([mk_chunk(i, t) for i, t in enumerate(texts)])


def test_tokenize_lowercases_drops_stopwords_keeps_not():
    assert tokenize("What is the Scaled Dot-Product, not 8 heads?") == ["scaled", "dot", "product", "not", "8", "heads"]


def test_finds_the_chunk_that_contains_the_rare_term():
    idx = index("glimmer networks converge", "widget benchmark common", "zebra tokens are rare")
    hits = idx.search("zebra tokens", k=3)
    assert hits[0].chunk.chunk_id == "t-2" and hits[0].rank == 1
    assert all(h.score > 0 for h in hits)


def test_rare_term_outweighs_common_term():
    idx = index("widget a", "widget b", "widget c", "zebra d")
    assert idx.search("widget zebra", k=1)[0].chunk.chunk_id == "t-3"


def test_shorter_chunk_wins_when_term_count_is_equal():
    idx = index("alpha beta", "alpha beta gamma delta epsilon zeta")
    assert [h.chunk.chunk_id for h in idx.search("alpha", k=2)] == ["t-0", "t-1"]


def test_repeating_a_query_word_does_not_change_the_score():
    idx = index("alpha beta", "gamma")
    assert idx.search("alpha", 1)[0].score == pytest.approx(idx.search("alpha alpha alpha", 1)[0].score)


def test_stopword_only_and_unknown_queries_return_nothing():
    idx = index("alpha beta", "gamma delta")
    assert idx.search("what is the", k=3) == []
    assert idx.search("nonexistentterm", k=3) == []


def test_ties_break_by_chunk_order_and_k_limits_results():
    idx = index("alpha beta", "alpha beta", "alpha beta")
    assert [h.chunk.chunk_id for h in idx.search("alpha", k=2)] == ["t-0", "t-1"]


def test_hit_keeps_source_metadata():
    hit = index("alpha beta").search("alpha", 1)[0]
    assert (hit.chunk.arxiv_id, hit.chunk.version, hit.chunk.page_start) == ("0000.00000", 1, 1)


def test_empty_corpus_is_safe():
    assert BM25Index([]).search("anything", 3) == []


def test_retriever_uses_top_k_and_reranks_when_asked():
    chunks = [mk_chunk(i, "alpha beta") for i in range(6)]
    assert len(Retriever(chunks, RetrievalConfig("T", 10, 2, 4)).retrieve("alpha")) == 4
    out = Retriever(chunks, RetrievalConfig("T", 10, 2, 2, rerank=True, candidates=4)).retrieve("alpha")
    assert len(out) == 2 and [h.rank for h in out] == [1, 2]  # re-ranked from 4 candidates down to top_k = 2
