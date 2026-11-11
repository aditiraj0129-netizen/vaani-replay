"""FREE (zero API calls): find WHY a question's evidence was not retrieved, config by config.

    python -m app.retrieval_diagnose --evidence data/eval/evidence_core.csv --out data/eval/retrieval_diagnosis.txt

"Gold" passages are the pieces of the question's own paper(s) that match the evidence term groups (same rule as eval_verify).
They are used ONLY to measure; nothing here changes what retrieval returns. For every question and config A, B, C it reports where the
best gold piece ranks and assigns ONE cause, checked in this order:

  OK                       the gold piece is among the pieces the pipeline would send
  NO_GOLD_CHUNK            no piece of this config matches the evidence terms (the terms, or the PDF text, need a look)
  SPLIT_AT_CHUNK_BOUNDARY  no single piece matches, but two neighbouring pieces together do (chunk boundary cut the evidence)
  RERANK_DROPPED           C only: BM25 ranked it inside the top k, the re-ranker pushed it out
  RERANK_DID_NOT_PROMOTE   C only: BM25 ranked it beyond the top k but inside the candidates, and the re-ranker did not lift it into the top k
  OTHER_PAPER_CROWDING     it ranks inside the cut-off when only the right paper is searched, but not across all papers
  NO_QUERY_TERM_OVERLAP    the piece shares no question word at all (BM25 cannot find it: wording)
  WORDING_MISMATCH         the piece contains under half of the question's words (question and paper use different words)
  BM25_SCORING             the piece contains most question words but other pieces score higher (length, repeated terms, common words)

Metadata/paper filtering: the pipeline searches ALL papers (no filter), so OTHER_PAPER_CROWDING is a corpus-wide scoring effect, not a bug in a filter.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from app.eval_verify import any_group, group_matches, load_groups, norm
from app.retrieval import Retriever, rerank, tokenize

CAUSES = ("OK", "NO_GOLD_CHUNK", "SPLIT_AT_CHUNK_BOUNDARY", "RERANK_DROPPED", "RERANK_DID_NOT_PROMOTE", "OTHER_PAPER_CROWDING",
          "NO_QUERY_TERM_OVERLAP", "WORDING_MISMATCH", "BM25_SCORING")


def _rank_of(hits: Sequence[Any], gold: set) -> Optional[int]:
    for h in hits:
        if h.chunk.chunk_id in gold:
            return h.rank
    return None


def diagnose_config(question: str, paper: str, groups: Sequence[List[str]], retriever: Retriever) -> Dict[str, Any]:
    cfg, index = retriever.config, retriever.index
    papers = paper.split("|")
    chunks = index.chunks
    gold = {c.chunk_id for c in chunks if c.paper_key in papers and any_group(norm(c.text), groups) is not None}
    sent = retriever.retrieve(question)
    out: Dict[str, Any] = {"config": cfg.name, "n_gold": len(gold), "bm25_rank": None, "paper_rank": None, "rerank_rank": None,
                           "final_rank": None, "coverage": None, "missing_terms": [], "top1": None, "gold_score": None,
                           "top_score": None, "n_sent": len(sent),
                           "refs_sent": sum(1 for h in sent if h.chunk.section.lower().startswith("reference"))}
    if not gold:
        split = any(a.paper_key == b.paper_key and a.paper_key in papers and group_matches(norm(a.text + " " + b.text), g)
                    for a, b in zip(chunks, chunks[1:]) for g in groups)
        out["cause"] = "SPLIT_AT_CHUNK_BOUNDARY" if split else "NO_GOLD_CHUNK"
        return out
    ranking = index.search(question, len(chunks))
    in_paper = index.search(question, len(chunks), papers)
    out["bm25_rank"] = _rank_of(ranking, gold)
    out["paper_rank"] = _rank_of(in_paper, gold)
    out["final_rank"] = _rank_of(sent, gold)
    if cfg.rerank:  # where the re-ranker puts it among the candidates (not cut to top k)
        out["rerank_rank"] = _rank_of(rerank(question, index.search(question, cfg.candidates), cfg.candidates), gold)
    if ranking:
        top = ranking[0]
        out["top1"] = f"{top.chunk.paper_key} [{top.chunk.section}] p.{top.chunk.page_start}"
        out["top_score"] = round(top.score, 3)
    q_terms = set(tokenize(question))
    best = next((h for h in ranking if h.chunk.chunk_id in gold), None)
    gold_tokens = set(tokenize(best.chunk.text)) if best else set().union(*(set(tokenize(c.text)) for c in chunks if c.chunk_id in gold))
    out["coverage"] = round(len(q_terms & gold_tokens) / len(q_terms), 2) if q_terms else 0.0
    out["missing_terms"] = sorted(q_terms - gold_tokens)
    if best:
        out["gold_score"] = round(best.score, 3)
    if out["final_rank"] is not None:
        out["cause"] = "OK"
        return out
    limit = cfg.candidates if cfg.rerank else cfg.top_k
    if cfg.rerank and out["bm25_rank"] is not None and out["bm25_rank"] <= cfg.candidates:
        out["cause"] = "RERANK_DROPPED" if out["bm25_rank"] <= cfg.top_k else "RERANK_DID_NOT_PROMOTE"
    elif out["paper_rank"] is not None and out["paper_rank"] <= limit:
        out["cause"] = "OTHER_PAPER_CROWDING"
    elif out["bm25_rank"] is None:
        out["cause"] = "NO_QUERY_TERM_OVERLAP"
    elif out["coverage"] < 0.5:
        out["cause"] = "WORDING_MISMATCH"
    else:
        out["cause"] = "BM25_SCORING"
    return out


def run(questions: Sequence[Dict[str, str]], groups_by_id: Dict[str, List[List[str]]], retrievers: Dict[str, Retriever]) -> List[Dict[str, Any]]:
    report = []
    for q in questions:
        groups = groups_by_id.get(q["id"], [])
        if not groups:
            report.append({"id": q["id"], "question": q["question"], "kind": q["kind"], "configs": {},
                           "note": "no evidence terms (expected for an unanswerable question)"})
            continue
        report.append({"id": q["id"], "question": q["question"], "kind": q["kind"], "paper": q["paper"],
                       "configs": {k: diagnose_config(q["question"], q["paper"], groups, r) for k, r in retrievers.items()}})
    return report


def render(report: Sequence[Dict[str, Any]]) -> str:
    lines = ["RETRIEVAL DIAGNOSIS (free, no API calls). Gold = pieces matching the evidence terms; used only to measure.", ""]
    tally: Dict[str, Dict[str, int]] = {}
    for e in report:
        lines.append(f"{e['id']} [{e['kind']}] {e['question']}")
        if not e["configs"]:
            lines += [f"   {e['note']}", ""]
            continue
        for cfg, d in e["configs"].items():
            tally.setdefault(cfg, {}).setdefault(d["cause"], 0)
            tally[cfg][d["cause"]] += 1
            lines.append(f"   {cfg}: {d['cause']:<24} gold pieces={d['n_gold']} bm25_rank={d['bm25_rank']} in_paper_rank={d['paper_rank']} "
                         f"rerank_rank={d['rerank_rank']} sent_rank={d['final_rank']} question-words-in-gold={d['coverage']}")
            if d["cause"] != "OK" and d["n_gold"]:
                lines.append(f"      gold score {d['gold_score']} vs best piece {d['top_score']} ({d['top1']}); question words missing from gold: {d['missing_terms']}")
        lines.append("")
    lines.append("BIBLIOGRAPHY PIECES AMONG THE PIECES SENT (a reference list is never evidence for an answer)")
    refs: Dict[str, List[int]] = {}
    for e in report:
        for cfg, d in e["configs"].items():
            r = refs.setdefault(cfg, [0, 0])
            r[0] += d["refs_sent"]
            r[1] += d["n_sent"]
    for cfg, (a, b) in sorted(refs.items()):
        lines.append(f"   {cfg}: {a} of {b} pieces sent")
    lines.append("")
    lines.append("CAUSE COUNTS PER CONFIG")
    for cfg, counts in sorted(tally.items()):
        lines.append(f"   {cfg}: " + ", ".join(f"{c}={counts[c]}" for c in CAUSES if c in counts))
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    from app.config import RETRIEVAL_CONFIGS, load_settings
    from app.eval_budget import load_questions
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default="data/eval/questions.csv")
    ap.add_argument("--evidence", default="data/eval/evidence_core.csv")
    ap.add_argument("--out", default="data/eval/retrieval_diagnosis.txt")
    args = ap.parse_args(argv)
    settings = load_settings()
    retrievers = {k: Retriever.from_settings(settings, c) for k, c in RETRIEVAL_CONFIGS.items()}
    report = run(load_questions(Path(args.questions)), load_groups(Path(args.evidence)), retrievers)
    text = render(report)
    Path(args.out).write_text(text, encoding="utf-8")
    Path(args.out).with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
