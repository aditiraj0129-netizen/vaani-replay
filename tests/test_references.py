from app import chunking as ck
from app import reference_audit as ra
from app.chunking import Word, filter_reference_words, make_chunks
from app.config import Paper, RetrievalConfig

PAPER = Paper("alpha", "Alpha Paper", "1111.11111", 2)
BIB = ("Ashish Vaswani, Noam Shazeer, Niki Parmar. Attention is all you need. In Advances in Neural Information Processing Systems, "
       "pages 5998-6008, 2017. Tom B. Brown, Benjamin Mann. Language models are few-shot learners. arXiv preprint arXiv:2005.14165, 2020. "
       "Yann LeCun, Yoshua Bengio, and Geoffrey Hinton. Deep learning. Nature, 521(7553):436-444, 2015. Diederik P. Kingma and Jimmy Ba. "
       "Adam: A method for stochastic optimization. In International Conference on Learning Representations, 2015. ")
CITING_PROSE = ("As shown by Vaswani et al., 2017, the model uses attention (Brown et al., 2020). We follow Kingma and Ba (2015), e.g. the setup "
                "of Smith et al., 2019. In Appendix B we describe the boundaries between chunks and how the routing module predicts a "
                "probability for each position, see Figure 12 for details on the experiments with the ablation study. ")
BODY = "Body text about attention and boundaries and results. " * 8


def words(text, section, page):
    return [Word(w, page, section) for w in text.split()]


def doc(with_appendix=True):
    ws = words(BODY, "3 Method", 3) + words(BIB * 3, "References", 9)
    if with_appendix:
        ws += words(CITING_PROSE * 4, "References", 12)  # an appendix whose heading was not detected keeps the References label
    return ws


def text_of(ws):
    return " ".join(w.text for w in ws)


def test_bibliography_text_scores_high_and_prose_with_inline_citations_scores_low():
    n, density = ck.bibliography_score(BIB)
    assert n >= 10 and density > 8 and ck.looks_like_bibliography(BIB)
    assert not ck.looks_like_bibliography(CITING_PROSE)          # years and "et al." in running text are not enough
    assert not ck.looks_like_bibliography("")
    assert not ck.looks_like_bibliography("e.g. The, i.e. Some, results follow")  # lower-case initials are not author initials


def test_body_words_are_never_removed_and_the_reference_list_is_gone():
    kept, removed, flagged = filter_reference_words(doc())
    assert text_of(kept).startswith(BODY.strip()) and BODY.strip() in text_of(kept)   # every body word survives, boundary included
    assert removed and {w.section for w in removed} == {"References"} and {w.page for w in removed} == {9} and len(removed) <= len(words(BIB * 3, 'References', 9))
    assert "Language models are few-shot learners" not in text_of(kept) and "Language models" in text_of(removed)


def test_an_appendix_after_the_reference_list_is_kept_and_flagged_even_with_the_references_label():
    kept, removed, flagged = filter_reference_words(doc())
    assert text_of(kept).startswith(BODY.strip())
    assert text_of(kept).count("routing module predicts a probability") == 4        # all 4 appendix copies survive, none clipped
    assert text_of(kept).count("As shown by Vaswani") == 4
    assert flagged and all(f.page_start == 12 and f.markers < ck.BIB_MARKERS_MIN or f.per100 < ck.BIB_DENSITY_MIN for f in flagged)
    assert all(not ck.looks_like_bibliography(f.start) for f in flagged)
    assert len(kept) + len(removed) == len(doc())                                     # nothing disappears without being counted


def test_a_references_run_with_no_bibliography_text_removes_nothing():
    kept, removed, flagged = filter_reference_words(words(CITING_PROSE * 6, "References", 5))
    assert removed == [] and flagged and len(kept) == len(CITING_PROSE.split()) * 6


def test_weak_entries_between_bibliography_segments_are_removed_with_the_run():
    weak = ("Smith. A book about things. Publisher House, 2016. Jones. Another title here. Press Name, 2014. " * 2)  # few markers
    ws = words(BIB * 2 + weak + BIB * 2, "References", 9)
    kept, removed, flagged = filter_reference_words(ws)
    assert kept == [] and len(removed) == len(ws) and flagged == []


def test_text_outside_a_references_heading_is_untouched_even_if_it_looks_like_a_bibliography():
    ws = words(BIB * 3, "4 Experiments", 5)
    kept, removed, flagged = filter_reference_words(ws)
    assert removed == [] and flagged == [] and kept == ws


def test_label_rule_and_filter_off(monkeypatch):
    assert ck.has_reference_label("References") and ck.has_reference_label("Bibliography") and ck.has_reference_label("REFERENCES")
    assert not ck.has_reference_label("7 Conclusion") and not ck.has_reference_label("Related References in Practice")


def _patched(monkeypatch, ws):
    monkeypatch.setattr(ck, "PAPERS", [PAPER])
    monkeypatch.setattr(ck, "load_processed", lambda s, p: [(1, "x")])
    monkeypatch.setattr(ck, "paper_words", lambda pages: ws)


def test_build_corpus_removes_the_list_for_every_config_but_keeps_the_config_differences(monkeypatch):
    _patched(monkeypatch, doc())
    small = RetrievalConfig("S", chunk_words=40, overlap_words=10, top_k=3)
    large = RetrievalConfig("L", chunk_words=120, overlap_words=30, top_k=3)
    rs, rl = ck.build_corpus_report(None, small), ck.build_corpus_report(None, large)
    off = ck.build_corpus_report(None, small, exclude_references=False)
    assert off.chunks_removed == 0 and rs.chunks_removed > 0 and rl.chunks_removed > 0
    assert rs.chunks_before > rl.chunks_before                                  # piece size still differs between configs
    assert all("Language models are few-shot" not in c.text for c in rs.kept + rl.kept)
    assert any("routing module" in c.text for c in rs.kept) and any("routing module" in c.text for c in rl.kept)
    assert ck.build_corpus(None, small) == rs.kept
    p = rs.papers[0]
    assert p.words_removed > 0 and p.removed_pages == (9, 9) and p.chunks_after == len(rs.kept) and p.flagged


def test_a_paper_that_is_all_reference_list_yields_no_pieces_not_a_crash(monkeypatch):
    _patched(monkeypatch, words(BIB * 3, "References", 9))
    rep = ck.build_corpus_report(None, RetrievalConfig("S", chunk_words=40, overlap_words=10, top_k=3))
    assert rep.kept == [] and rep.papers[0].chunks_after == 0 and rep.papers[0].chunks_before > 0


def test_audit_report_lists_counts_pages_samples_and_flagged_segments(monkeypatch):
    _patched(monkeypatch, doc())
    reps = {c: ck.build_corpus_report(None, RetrievalConfig(c, chunk_words=40, overlap_words=10, top_k=3)) for c in ("A", "B")}
    text = ra.render(reps)
    assert "CONFIG A:" in text and "CONFIG B:" in text and "words removed" in text and "p.9-9" in text
    assert "first removed words" in text and "last removed words" in text and "FLAGGED kept p.12-12" in text
