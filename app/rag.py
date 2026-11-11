"""The local (zero-token) steps around the one model call.

Pipeline steps, in order:
  1. analyze   (local)  what is the question about? Which paper(s) does it name? -> a hint for retrieval
  2. retrieve  (tool)   BM25 search over the three papers, optionally re-ranked and restricted to the hinted paper
  3. draft     (model)  the one Groq call: a cited answer, or the exact refusal sentence
  4. verify    (local)  are the citations real? do the numbers in the answer appear in the cited passages?

Branches: a refusal is recognised and kept as a refusal; an answer with no valid citation or with numbers that do not
appear in its cited passages is FLAGGED (never silently trusted). Provenance checks only: they cannot prove an
answer is correct, and they cannot see numbers written as words ("six").
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

from app.answer import NOT_FOUND, extract_citations

# Whole-phrase cues that identify one paper. Deliberately specific: "attention" and "transformer" alone also appear
# in the other two papers (Moshi has an RQ-Transformer, H-Net discusses attention layers).
PAPER_CUES: Dict[str, Sequence[str]] = {
    "attention": ("attention is all you need", "scaled dot-product", "multi-head attention", "positional encoding",
                  "base transformer", "big transformer", "wmt 2014", "sinusoidal", "label smoothing", "self-attention",
                  "encoder stack", "decoder stack"),
    "moshi": ("moshi", "mimi", "helium", "inner monologue", "full-duplex", "rq-transformer", "depformer",
              "spoken dialogue", "semantic tokens", "acoustic tokens"),
    "dynamic_chunking": ("h-net", "dynamic chunking", "dechunking", "routing module", "smoothing module", "ratio loss",
                         "hierarchical network", "byte-level", "chunking layer"),
}


@dataclass
class Analysis:
    question: str
    papers: List[str] = field(default_factory=list)  # papers the question names; empty = unknown or several

    @property
    def single_paper(self) -> bool:
        return len(self.papers) == 1


def analyze(question: str) -> Analysis:
    q = " ".join(question.lower().split())
    papers = [key for key, cues in PAPER_CUES.items() if any(c in q for c in cues)]
    return Analysis(question=question.strip(), papers=papers)


_NUMBER = re.compile(r"(?<![\w.])\d+(?:\.\d+)?")
_CITE_BRACKETS = re.compile(r"[\[【]\d+(?:\s*,\s*\d+)*[\]】]")


def _numbers(text: str) -> List[str]:
    return sorted(set(_NUMBER.findall(_CITE_BRACKETS.sub(" ", text))))


def _contains_number(passage_text: str, number: str) -> bool:
    return re.search(rf"(?<![\d.]){re.escape(number)}(?!\d|\.\d)", passage_text) is not None


@dataclass
class Verification:
    status: str  # refusal | cited | uncited | invalid
    cited: List[int]
    invalid: List[int]
    unsupported_numbers: List[str]  # numbers in the answer found in none of the cited passages


def verify(answer: str, passages_text: Sequence[str]) -> Verification:
    """passages_text[i] is the full text of passage number i+1 as it was shown to the model."""
    cited, invalid = extract_citations(answer, len(passages_text))
    if NOT_FOUND.lower() in answer.lower():
        return Verification("refusal", cited, invalid, [])
    status = "invalid" if invalid else "cited" if cited else "uncited"
    evidence = " ".join(passages_text[n - 1] for n in cited) if cited else " ".join(passages_text)
    unsupported = [n for n in _numbers(answer) if not _contains_number(evidence, n)]
    return Verification(status, cited, invalid, unsupported)
