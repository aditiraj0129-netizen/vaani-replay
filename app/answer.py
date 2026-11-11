"""Turn a question + retrieved passages into an LLM request, and record what came back.

Provenance only: `extract_citations` checks that cited passage NUMBERS exist in the prompt.
That says nothing about whether the answer is factually correct.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.providers.base import LLMProvider, LLMRequest, LLMResult

NOT_FOUND = "Not found in the provided passages."

SYSTEM_PROMPT = (
    "You answer questions using ONLY the numbered passages provided. "
    "Cite the passages you use with their numbers in square brackets, like [1] or [2][3]. "
    f"If the passages do not contain the answer, reply exactly: {NOT_FOUND} "
    "Be concise: at most 4 sentences."
)

# The prompt asks for [1]; real model output sometimes uses full-width brackets 【1】, so both are accepted.
_CITATION = re.compile(r"[\[【](\d+(?:\s*,\s*\d+)*)[\]】]")


def estimate_tokens(text: str) -> int:
    """Rough guess: 1 token is about 4 characters. An ESTIMATE; we compare it with the measured prompt_tokens."""
    return math.ceil(len(text) / 4)


def _pages(chunk: Any) -> str:
    if chunk.page_start == chunk.page_end:
        return f"p.{chunk.page_start}"
    return f"p.{chunk.page_start}-{chunk.page_end}"


def build_messages(question: str, hits: Sequence[Any]) -> Tuple[List[Dict[str, str]], List[Dict[str, Any]]]:
    """Returns (chat messages, passage table). Passage numbers start at 1 and follow the hit order."""
    passages: List[Dict[str, Any]] = []
    blocks: List[str] = []
    for n, hit in enumerate(hits, 1):
        c = hit.chunk
        passages.append({
            "n": n, "chunk_id": c.chunk_id, "paper_key": c.paper_key, "version": c.version,
            "pages": _pages(c), "section": c.section, "text": c.text,
        })
        blocks.append(f"[{n}] ({c.paper_key} v{c.version}, {_pages(c)}, section: {c.section})\n{c.text}")
    user = "Passages:\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}"
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}], passages


def extract_citations(answer: str, n_passages: int) -> Tuple[List[int], List[int]]:
    """(valid cited numbers, invalid cited numbers), each sorted and unique."""
    found = set()
    for group in _CITATION.findall(answer):
        numbers = [int(x) for x in re.split(r"\s*,\s*", group)]
        if len(numbers) != len(set(numbers)):   # "[0, 2, 2, 2]" is data (a delay pattern), not a citation: a citation never repeats a number
            continue
        found.update(numbers)
    valid = sorted(n for n in found if 1 <= n <= n_passages)
    invalid = sorted(n for n in found if not 1 <= n <= n_passages)
    return valid, invalid


@dataclass
class AnswerRecord:
    question: str
    config: str
    passages: List[Dict[str, Any]]
    messages: List[Dict[str, str]]
    est_prompt_tokens: int
    result: Optional[LLMResult]  # None only when no passage matched and no call was made
    cited: List[int]
    invalid_cited: List[int]
    said_not_found: bool

    @property
    def citation_status(self) -> str:
        """n/a (no complete answer) | refusal | invalid (cites a passage that does not exist) | cited | uncited.

        'uncited' = the model gave a non-refusal answer with no valid citation (seen once in the real Step B run, #9).
        This is a PROVENANCE flag only; it says nothing about whether the answer is correct.
        """
        if self.result is None or not self.result.ok:
            return "n/a"
        if self.said_not_found:
            return "refusal"
        if self.invalid_cited:
            return "invalid"
        return "cited" if self.cited else "uncited"


async def answer_question(
    provider: LLMProvider,
    retriever: Any,
    question: str,
    *,
    max_tokens: int = 700,
    timeout_s: float = 30.0,
) -> AnswerRecord:
    hits = retriever.retrieve(question)
    messages, passages = build_messages(question, hits)
    est = sum(estimate_tokens(m["content"]) for m in messages)
    config = retriever.config.name
    if not hits:  # nothing to answer from: do not spend a request
        return AnswerRecord(question, config, passages, messages, est, None, [], [], True)
    result = await provider.generate(LLMRequest(messages=messages, max_tokens=max_tokens, timeout_s=timeout_s))
    cited: List[int] = []
    invalid: List[int] = []
    not_found = False
    if result.ok:  # only a complete answer is ever inspected
        cited, invalid = extract_citations(result.text, len(passages))
        not_found = NOT_FOUND.lower() in result.text.lower()
    return AnswerRecord(question, config, passages, messages, est, result, cited, invalid, not_found)
