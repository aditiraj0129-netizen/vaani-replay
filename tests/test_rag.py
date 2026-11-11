import pytest

from app.rag import analyze, verify

P = ["The base Transformer encoder stack has N = 6 identical layers. Each layer has d_model = 512.",
     "The Mimi codec runs at a frame rate of 12.5 Hz and the theoretical latency is 160 ms.",
     "H-Net compresses 4.5 bytes per chunk."]


@pytest.mark.parametrize("question, papers", [
    ("Why does scaled dot-product attention divide by the square root of d_k?", ["attention"]),
    ("What is Inner Monologue in Moshi?", ["moshi"]),
    ("What frame rate does the Mimi codec operate at?", ["moshi"]),
    ("What does the routing module of H-Net compute?", ["dynamic_chunking"]),
    ("How does H-Net's dynamic chunking differ from the Transformer's multi-head attention?", ["attention", "dynamic_chunking"]),
    ("What is the capital of France?", []),
    ("How does attention work?", []),  # "attention" alone is not specific enough
])
def test_analyze_names_the_papers_a_question_is_about(question, papers):
    a = analyze(question)
    assert sorted(a.papers) == sorted(papers)
    assert a.single_paper == (len(papers) == 1)


def test_verify_accepts_a_properly_cited_supported_answer():
    v = verify("The encoder has 6 identical layers [1].", P)
    assert (v.status, v.cited, v.invalid, v.unsupported_numbers) == ("cited", [1], [], [])


def test_verify_flags_a_number_that_is_not_in_the_cited_passage():
    v = verify("The encoder has 8 identical layers [1].", P)
    assert v.status == "cited" and v.unsupported_numbers == ["8"]


def test_verify_checks_numbers_against_the_cited_passage_not_just_any_passage():
    v = verify("The codec runs at 12.5 Hz [1].", P)  # 12.5 is in passage 2, but passage 1 was cited
    assert v.unsupported_numbers == ["12.5"]
    assert verify("The codec runs at 12.5 Hz [2].", P).unsupported_numbers == []


def test_verify_number_matching_is_whole_number_not_substring():
    assert verify("It uses 4 bytes [3].", P).unsupported_numbers == ["4"]  # "4.5" does not support "4"
    assert verify("It uses 4.5 bytes [3].", P).unsupported_numbers == []
    assert verify("Latency is 16 ms [2].", P).unsupported_numbers == ["16"]  # "160" does not support "16"


def test_verify_ignores_the_citation_numbers_themselves():
    assert verify("The latency is 160 ms [2][3].", P).unsupported_numbers == []
    assert verify("The latency is 160 ms 【2】.", P).unsupported_numbers == []


def test_verify_uncited_answer_is_checked_against_all_passages():
    v = verify("The Mimi codec operates at 12.5 Hz", P)
    assert v.status == "uncited" and v.unsupported_numbers == []
    assert verify("It operates at 99 Hz", P).unsupported_numbers == ["99"]


def test_verify_invalid_citation_and_refusal():
    assert verify("Yes [9].", P).status == "invalid"
    v = verify("Not found in the provided passages.", P)
    assert v.status == "refusal" and v.unsupported_numbers == []
