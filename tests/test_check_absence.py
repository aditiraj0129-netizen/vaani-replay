from app import check_absence as ca


def test_clean_text_reports_zero_matches_and_lists_the_bleu_pairs_that_exist():
    res = ca.scan_pages([(8, "Table 2: BLEU scores\nEN-DE BLEU 28.4\nEN-FR BLEU 41.8", 0.0), (9, "nothing here", 0.0)])
    assert res["hits"]["romanian"] == [] and res["hits"]["wmt 2016"] == [] and res["hits"]["bleu"] == [8]
    assert [t for _, t in res["bleu_lines"]] == ["Table 2: BLEU scores", "EN-DE BLEU 28.4", "EN-FR BLEU 41.8"]
    assert "0 matches in the text layer" in ca.render("x.pdf", res)


def test_present_terms_are_found_in_their_spellings():
    res = ca.scan_pages([(3, "results on newstest2016 En-Ro and WMT  2016 English-Romanian", 0.0)])
    assert res["hits"]["romanian"] == [3] and res["hits"]["wmt 2016"] == [3] and res["hits"]["newstest2016"] == [3] and res["hits"]["en-ro / en\u2192ro"] == [3]
    assert "FOUND on pages [3]" in ca.render("x.pdf", res)


def test_picture_heavy_pages_are_listed_for_a_manual_look():
    res = ca.scan_pages([(1, "text", 0.05), (5, "a few words", 0.4)])
    assert res["look_at"] == [(5, 0.4, 3)]
    assert "p.5 (0.4, 3 words)" in ca.render("x.pdf", res)
