import asyncio
from types import SimpleNamespace

from app.answer import (NOT_FOUND, SYSTEM_PROMPT, answer_question, build_messages, estimate_tokens,
                        extract_citations)
from app.providers.base import LLMResult


def hit(n, text="some passage text", p1=3, p2=3):
    chunk = SimpleNamespace(chunk_id=f"attention-B-{n:04d}", paper_key="attention", version=7,
                            page_start=p1, page_end=p2, section="3.2 Attention", text=text)
    return SimpleNamespace(chunk=chunk, score=1.0, rank=n)


class FakeRetriever:
    def __init__(self, hits):
        self.hits = hits
        self.config = SimpleNamespace(name="B")

    def retrieve(self, query):
        return self.hits


class FakeProvider:
    def __init__(self, result):
        self.result, self.requests = result, []

    async def generate(self, req, on_token=None):
        self.requests.append(req)
        return self.result


def ok_result(text):
    return LLMResult(status="ok", text=text, complete=True, http_status=200, finish_reason="stop",
                     prompt_tokens=100, completion_tokens=20)


def test_messages_number_passages_from_one_and_end_with_the_question():
    messages, passages = build_messages("What is X?", [hit(1, "alpha"), hit(2, "beta", 4, 5)])
    assert messages[0] == {"role": "system", "content": SYSTEM_PROMPT}
    user = messages[1]["content"]
    assert "[1] (attention v7, p.3, section: 3.2 Attention)\nalpha" in user
    assert "[2] (attention v7, p.4-5, section: 3.2 Attention)\nbeta" in user
    assert user.endswith("Question: What is X?")
    assert [p["n"] for p in passages] == [1, 2] and passages[1]["chunk_id"] == "attention-B-0002"
    assert passages[1]["pages"] == "p.4-5"


def test_system_prompt_demands_citations_and_an_exact_not_found_reply():
    assert "ONLY" in SYSTEM_PROMPT and NOT_FOUND in SYSTEM_PROMPT and "[1]" in SYSTEM_PROMPT


def test_extract_citations_valid_and_invalid():
    assert extract_citations("Yes [1][3]. Also [2, 3].", 3) == ([1, 2, 3], [])
    assert extract_citations("Claim [9] and [0] and [1].", 3) == ([1], [0, 9])
    assert extract_citations("no citations here", 3) == ([], [])
    assert extract_citations("array a[1] looks like a citation", 3) == ([1], [])  # known limitation, documented


def test_estimate_tokens_is_chars_over_four_rounded_up():
    assert estimate_tokens("") == 0 and estimate_tokens("abcd") == 1 and estimate_tokens("abcde") == 2


def test_answer_question_success_records_citations_and_sends_the_request_settings():
    provider = FakeProvider(ok_result("It is a weighted sum [1]. Also [7]."))
    rec = asyncio.run(answer_question(provider, FakeRetriever([hit(1), hit(2)]), "Q?", max_tokens=555, timeout_s=9.0))
    req = provider.requests[0]
    assert req.max_tokens == 555 and req.timeout_s == 9.0 and req.messages == rec.messages
    assert rec.cited == [1] and rec.invalid_cited == [7] and not rec.said_not_found
    assert rec.config == "B" and len(rec.passages) == 2 and rec.est_prompt_tokens > 0


def test_incomplete_answer_is_never_inspected_for_citations():
    failed = LLMResult(status="timeout", partial_text="half answer [1]", complete=False, error_kind="timeout")
    rec = asyncio.run(answer_question(FakeProvider(failed), FakeRetriever([hit(1)]), "Q?"))
    assert rec.cited == [] and rec.invalid_cited == [] and not rec.said_not_found and not rec.result.ok


def test_not_found_reply_is_detected():
    rec = asyncio.run(answer_question(FakeProvider(ok_result(NOT_FOUND)), FakeRetriever([hit(1)]), "Q?"))
    assert rec.said_not_found and rec.cited == []


def test_no_passages_means_no_request_is_spent():
    provider = FakeProvider(ok_result("should never be used"))
    rec = asyncio.run(answer_question(provider, FakeRetriever([]), "Q?"))
    assert provider.requests == [] and rec.result is None and rec.said_not_found


def test_full_width_citation_brackets_from_real_model_output_are_understood():
    # real Step B answer #4: "The Mimi codec operates at a frame rate of 12.5 Hz【3】"
    assert extract_citations("The Mimi codec operates at a frame rate of 12.5 Hz【3】", 5) == ([3], [])
    assert extract_citations("Both [1] and 【2, 4】.", 5) == ([1, 2, 4], [])
    assert extract_citations("Mixed bracket types [3】 are not a citation", 5) == ([3], [])  # lenient on purpose


def _record(text, cited, invalid=(), nf=False, ok=True):
    from app.answer import AnswerRecord
    from app.providers.base import LLMResult
    res = LLMResult(status="ok" if ok else "error", text=text, complete=ok)
    return AnswerRecord("q", "B", [], [], 0, res, list(cited), list(invalid), nf)


def test_citation_status_covers_every_case():
    from app.answer import AnswerRecord
    assert _record("x [1]", [1]).citation_status == "cited"
    assert _record("x", []).citation_status == "uncited"  # real Step B answer #9 looked like this
    assert _record("x [9]", [], [9]).citation_status == "invalid"
    assert _record("Not found in the provided passages.", [], nf=True).citation_status == "refusal"
    assert _record("", [], ok=False).citation_status == "n/a"
    assert AnswerRecord("q", "B", [], [], 0, None, [], [], True).citation_status == "n/a"


def test_bracketed_data_with_repeated_numbers_is_not_a_citation():
    assert extract_citations("A pattern of [0, 2, 2, 2, 2] gives 240 ms [3].", 5) == ([3], [])
    assert extract_citations("Pattern [0, 0, 0, 0] costs 80 ms [1][2].", 5) == ([1, 2], [])
    assert extract_citations("Claim [9] and [0] and [1].", 3) == ([1], [0, 9])      # a single out-of-range number is still invalid
