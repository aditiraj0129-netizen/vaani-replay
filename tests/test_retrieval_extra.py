from app.chunking import Chunk
from app.config import RetrievalConfig
from app.retrieval import BM25Index, Hit, Retriever, rerank


def chunk(i, paper, text, section="Intro"):
    return Chunk(chunk_id=f"{paper}-{i}", paper_key=paper, arxiv_id="1.1", version=1, title="t", ordinal=i,
                 page_start=1, page_end=1, section=section, text=text, n_words=len(text.split()))


CHUNKS = [
    chunk(0, "attention", "scaled dot product attention divides by the square root of the key dimension", "3.2.1 Scaled Dot-Product Attention"),
    chunk(1, "moshi", "attention attention attention layers in the temporal transformer attention"),
    chunk(2, "dynamic_chunking", "the routing module predicts chunk boundaries with cosine similarity", "2.2.1 Chunking Layer"),
    chunk(3, "attention", "attention is used in many places, attention attention"),
]


def test_paper_filter_restricts_results_to_the_named_paper():
    idx = BM25Index(CHUNKS)
    allp = idx.search("attention", 4)
    only = idx.search("attention", 4, papers=["attention"])
    assert {h.chunk.paper_key for h in allp} == {"attention", "moshi"}
    assert {h.chunk.paper_key for h in only} == {"attention"}
    assert idx.search("attention", 4, papers=[]) == allp  # an empty filter means no filter


def test_retriever_without_rerank_matches_the_plain_search():
    r = Retriever(CHUNKS, RetrievalConfig("T", chunk_words=10, overlap_words=2, top_k=2))
    assert [h.chunk.chunk_id for h in r.retrieve("attention")] == [h.chunk.chunk_id for h in r.index.search("attention", 2)]


def test_rerank_prefers_the_chunk_that_covers_more_of_the_question():
    q = "scaled dot product attention square root key dimension"
    cfg = RetrievalConfig("R", chunk_words=10, overlap_words=2, top_k=2, rerank=True, candidates=4)
    r = Retriever(CHUNKS, cfg)
    plain = r.index.search(q, 4)
    out = r.retrieve(q)
    assert out[0].chunk.chunk_id == "attention-0"  # full coverage, same word pairs, matching section title
    assert len(out) == 2 and [h.rank for h in out] == [1, 2]
    assert plain[0].chunk.chunk_id == "attention-0"


def test_rerank_can_overturn_a_bm25_ordering():
    # chunk 1 repeats one query word many times (BM25 likes that); chunk 0 contains the whole phrase once
    q = "routing module chunk boundaries"
    a = chunk(0, "dynamic_chunking", "the routing module predicts chunk boundaries")
    b = chunk(1, "dynamic_chunking", "routing routing routing routing routing routing routing routing and nothing else about it")
    idx = BM25Index([a, b] + [chunk(10 + i, "moshi", "filler text number %d" % i) for i in range(5)])
    plain = idx.search(q, 2)
    out = rerank(q, plain, 2)
    assert out[0].chunk.chunk_id == "dynamic_chunking-0"


def test_rerank_handles_empty_input_and_keeps_top_k():
    assert rerank("anything", [], 3) == []
    hits = [Hit(CHUNKS[i], 1.0 + i, i + 1) for i in range(4)]
    assert len(rerank("attention", hits, 2)) == 2


def test_config_c_is_no_longer_blocked():
    from app.config import RETRIEVAL_CONFIGS
    r = Retriever(CHUNKS, RETRIEVAL_CONFIGS["C"])
    assert r.retrieve("routing module") is not None


def _h(text, section="s", score=1.0, rank=1, i=[0]):
    i[0] += 1
    return Hit(chunk(i[0], "attention", text, section), score, rank)


def test_each_rerank_signal_alone_can_decide_the_order():
    # In every case both candidates tie on the other signals and the ORIGINAL order puts the weaker one first.
    q = "alpha beta gamma"
    cov_weak, cov_strong = _h("alpha x beta x x", rank=1), _h("alpha x beta x gamma", rank=2)
    assert rerank(q, [cov_weak, cov_strong], 2)[0].chunk is cov_strong.chunk  # coverage: 3 of 3 terms vs 2 of 3
    ph_weak, ph_strong = _h("gamma x beta x alpha", rank=1), _h("alpha beta gamma", rank=2)
    assert rerank(q, [ph_weak, ph_strong], 2)[0].chunk is ph_strong.chunk  # phrase: pairs next to each other
    sec_weak, sec_strong = _h("alpha beta gamma", "Other", rank=1), _h("alpha beta gamma", "alpha beta", rank=2)
    assert rerank(q, [sec_weak, sec_strong], 2)[0].chunk is sec_strong.chunk  # section title shares the query words
    bm_weak, bm_strong = _h("alpha beta gamma", score=5.0, rank=1), _h("alpha beta gamma", score=10.0, rank=2)
    assert rerank(q, [bm_weak, bm_strong], 2)[0].chunk is bm_strong.chunk  # original BM25 evidence still counts
