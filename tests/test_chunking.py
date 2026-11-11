import json

import pytest

from app.chunking import Word, build_corpus, clean_page_text, is_heading, make_chunks, paper_words
from app.config import RetrievalConfig
from tests.conftest import FIXTURE_PAGES


def words(n, page=1, section="s"):
    return [Word(f"w{i}", page, section) for i in range(n)]


def test_window_and_overlap_math(paper):
    chunks = make_chunks(paper, words(25), 10, 2, "T")
    assert [c.n_words for c in chunks] == [10, 10, 9]  # starts 0, 8, 16
    assert chunks[1].text.split()[:2] == ["w8", "w9"]  # overlap = last 2 words of chunk 0


def test_exact_fit_gives_one_chunk_and_empty_input_gives_none(paper):
    assert len(make_chunks(paper, words(10), 10, 3, "T")) == 1
    assert make_chunks(paper, [], 10, 3, "T") == []


def test_every_word_is_covered(paper):
    ws = words(137)
    covered = {w for c in make_chunks(paper, ws, 20, 5, "T") for w in c.text.split()}
    assert covered == {w.text for w in ws}


def test_invalid_sizes_rejected(paper):
    with pytest.raises(ValueError):
        make_chunks(paper, words(5), 10, 10, "T")


def test_page_range_spans_pages_and_ids_are_deterministic(paper):
    ws = words(6, page=1) + words(6, page=2)
    a = make_chunks(paper, ws, 8, 2, "B")
    assert (a[0].page_start, a[0].page_end) == (1, 2)
    assert a == make_chunks(paper, ws, 8, 2, "B")
    assert len({c.chunk_id for c in a}) == len(a) and a[0].chunk_id == "alpha-B-0000"


def test_clean_page_text_handles_hyphenation_and_ligatures():
    assert clean_page_text("trans-\nformer") == "transformer"
    assert clean_page_text("e\ufb03cient \ufb01ne") == "efficient fine"
    assert clean_page_text("state-of-the-art") == "state-of-the-art"


@pytest.mark.parametrize("line,expected", [
    ("3.2.1 Scaled Dot-Product Attention", True),
    ("1 Introduction", True),
    ("Abstract", True),
    ("References", True),
    ("Appendix A", True),
    ("The model uses 3 layers.", False),
    ("6 Base 65M 512 2048", False),
    ("2017 Conference on Neural Information Processing Systems", False),
    ("Abstract shows that results improve across many tasks today", False),
])
def test_is_heading(line, expected):
    assert is_heading(line) is expected


def test_sections_pages_and_dehyphenation_on_fixture_text(paper):
    ws = paper_words(list(enumerate(FIXTURE_PAGES, start=1)))
    assert ws[0].section == "(front matter)"
    by_word = {}
    for w in ws:
        by_word.setdefault(w.text, w)
    assert by_word["glimmer"].section == "Abstract" and by_word["glimmer"].page == 1
    assert by_word["transformer"].section == "1 Introduction" and by_word["transformer"].page == 2
    assert by_word["Zebra"].section == "3.2 Gizmo Attention" and by_word["Zebra"].page == 3


def test_build_corpus_requires_ingest_and_rejects_wrong_version(settings, paper, monkeypatch):
    config = RetrievalConfig("T", 10, 2, 3)
    with pytest.raises(FileNotFoundError, match="app.ingest"):
        build_corpus(settings, config)
    settings.processed_dir.mkdir(parents=True)
    (settings.processed_dir / "attention.json").write_text(
        json.dumps({"arxiv_id": "1706.03762", "version": 5, "pages": []}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="expected 1706.03762v7"):
        build_corpus(settings, config)


def test_heading_number_on_its_own_line_is_joined_with_the_title_line():
    page = "Abstract\nWe study things.\n3.1\nEncoder and Decoder Stacks\nEncoder: a stack of layers.\n3"
    ws = paper_words([(3, page)])
    by_word = {w.text: w for w in ws}
    assert by_word["study"].section == "Abstract"
    assert by_word["Encoder:"].section == "3.1 Encoder and Decoder Stacks"


def test_bare_number_before_lowercase_text_is_not_a_heading():
    ws = paper_words([(1, "Abstract\nWe study things.\n512\nlayers of the model\nmore text")])
    assert {w.section for w in ws} == {"Abstract"}


def test_printed_page_number_at_page_bottom_is_dropped_but_inner_numbers_stay():
    ws = paper_words([(2, "Hello\n7\nworld\n2")])
    assert [w.text for w in ws] == ["Hello", "7", "world"]
