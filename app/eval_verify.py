"""FREE (zero API calls): check each evaluation question against the pinned papers' text, and separate
"the paper has it but retrieval missed it" from "no passage matches the proposed evidence terms".

    python -m app.eval_verify            # writes data/eval/verification_v2.txt and .json

Inputs: data/eval/questions.csv (your questions) and data/eval/evidence_proposals.csv (PROPOSED evidence terms).
For each question:
  1. SCAN every passage of the question's paper (config B pieces) for the proposed term groups, and print the section,
     PDF page (as stored by the chunker; confirm by opening the PDF) and a short excerpt.
  2. RETRIEVE with configs A, B and C, unfiltered and restricted to the paper, and report whether a passage that
     matches a term group was returned, and at which rank.
  3. LABEL (candidate only): evidence found + retrieved by every config = "ok"; evidence found but not retrieved =
     "RETRIEVAL FAILURE (config ...)" (and "cross-paper" if the paper-restricted search finds it); no evidence found =
     "NO MATCHING PASSAGE" (the paper may not contain it, OR the proposed terms are wrong, OR the PDF text extraction
     is broken: read the PDF).
A keyword match is NOT verification. A person must read each excerpt and decide that it really supports the answer.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

Group = List[str]  # terms; each term is "alt1/alt2"; every term must match


def norm(text: str) -> str:
    return " ".join(text.replace("\u00ad", "").lower().split())


def term_hit(text_norm: str, term: str) -> Optional[str]:
    """The alternative that matched, or None."""
    squashed = text_norm.replace(" ", "")  # the PDF text sometimes glues words ("Thesecond"), so also compare without spaces
    for alt in (a.strip().lower() for a in term.split("/")):
        if alt and (alt in text_norm or alt.replace(" ", "") in squashed):
            return alt
    return None


def group_matches(text_norm: str, group: Group) -> bool:
    return all(term_hit(text_norm, t) for t in group)


def any_group(text_norm: str, groups: Sequence[Group]) -> Optional[int]:
    for i, g in enumerate(groups):
        if group_matches(text_norm, g):
            return i
    return None


def excerpt(text: str, group: Group, width: int = 110) -> str:
    flat = " ".join(text.split())
    low = flat.lower()
    alt = term_hit(norm(text), group[0]) or ""
    pos = max(0, low.find(alt)) if alt else 0
    start = max(0, pos - width // 2)
    return flat[start:start + width * 2].strip()


def load_groups(path: Path) -> Dict[str, List[Group]]:
    out: Dict[str, List[Group]] = {}
    with Path(path).open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out.setdefault(row["id"], []).append([t.strip() for t in row["terms"].split("|") if t.strip()])
    return out


def find_evidence(chunks: Sequence[Any], paper: str, groups: Sequence[Group], limit: int = 3) -> List[Dict[str, Any]]:
    found: List[Dict[str, Any]] = []
    papers = set(paper.split("|"))  # a cross-paper question names several papers, joined by |
    for c in chunks:
        if c.paper_key not in papers:
            continue
        gi = any_group(norm(c.text), groups)
        if gi is not None:
            found.append({"group": gi, "chunk_id": c.chunk_id, "arxiv_id": c.arxiv_id, "version": c.version,
                          "section": c.section, "pages": (c.page_start, c.page_end),
                          "excerpt": excerpt(c.text, groups[gi])})
    return found[:limit] if limit else found


def retrieval_rank(hits: Sequence[Any], groups: Sequence[Group]) -> Optional[int]:
    for i, h in enumerate(hits, 1):
        if any_group(norm(h.chunk.text), groups) is not None:
            return i
    return None


def label(evidence_found: bool, ranks: Dict[str, Optional[int]], ranks_in_paper: Dict[str, Optional[int]]) -> str:
    if not evidence_found:
        return "NO MATCHING PASSAGE (paper lacks it, OR proposed terms are wrong, OR PDF text extraction is broken - read the PDF)"
    missed = [c for c, r in ranks.items() if r is None]
    if not missed:
        return "ok: evidence found and retrieved by every config"
    contaminated = [c for c in missed if ranks_in_paper.get(c) is not None]
    text = f"RETRIEVAL FAILURE (configs {', '.join(missed)})"
    if contaminated:
        text += f"; paper-restricted search finds it for {', '.join(contaminated)}, so other papers crowd it out"
    return text


def run(questions_path: Path, proposals_path: Path, out_txt: Path, out_json: Path) -> List[Dict[str, Any]]:
    from app.config import RETRIEVAL_CONFIGS, load_settings
    from app.eval_budget import load_questions
    from app.retrieval import Retriever

    settings = load_settings()
    qs = load_questions(questions_path)
    proposals = load_groups(proposals_path)
    retrievers = {k: Retriever.from_settings(settings, c) for k, c in RETRIEVAL_CONFIGS.items()}
    scan_chunks = retrievers["B"].index.chunks
    report: List[Dict[str, Any]] = []
    lines: List[str] = ["EVIDENCE CHECK - generated by app/eval_verify.py (free, no API calls)",
                        "Proposed terms come from data/eval/evidence_proposals.csv and are UNCONFIRMED.",
                        "A keyword match is not verification: read each excerpt and confirm it supports the answer.", ""]
    for q in qs:
        groups = proposals.get(q["id"], [])
        ev = find_evidence(scan_chunks, q["paper"], groups) if groups else []
        ranks = {k: retrieval_rank(r.retrieve(q["question"]), groups) for k, r in retrievers.items()} if groups else {}
        in_paper = ({k: retrieval_rank(r.retrieve(q["question"], q["paper"].split("|")), groups) for k, r in retrievers.items()}
                    if groups else {})
        verdict = label(bool(ev), ranks, in_paper) if groups else "NO TERMS PROPOSED"
        entry = {"id": q["id"], "question": q["question"], "kind_proposed": q["kind"], "paper": q["paper"],
                 "groups": groups, "evidence": ev, "rank_unfiltered": ranks, "rank_paper_restricted": in_paper, "label": verdict}
        report.append(entry)
        lines.append(f"{q['id']} [{q['kind']}] {q['question']}")
        lines.append(f"   term groups (any row, all terms): {' || '.join(' + '.join(g) for g in groups)}")
        for e in ev:
            lines.append(f"   EVIDENCE {e['arxiv_id']} v{e['version']} [{e['section']}] PDF p.{e['pages'][0]}"
                         + (f"-{e['pages'][1]}" if e['pages'][1] != e['pages'][0] else "") + f": ...{e['excerpt']}...")
        lines.append(f"   retrieved at rank (unfiltered)       : {ranks}")
        lines.append(f"   retrieved at rank (paper-restricted) : {in_paper}")
        lines.append(f"   => {verdict}\n")
    out_txt.write_text("\n".join(lines), encoding="utf-8")
    out_json.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print("\n".join(lines))
    print(f"saved {out_txt} and {out_json}")
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default="data/eval/questions.csv")
    ap.add_argument("--proposals", default="data/eval/evidence_proposals.csv")
    ap.add_argument("--out", default="data/eval/verification_v2.txt")
    args = ap.parse_args(argv)
    run(Path(args.questions), Path(args.proposals), Path(args.out), Path(args.out).with_suffix(".json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
