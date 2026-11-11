"""FREE: turn the real evaluation answers into a reading sheet and an empty grades template. No API calls.

    python -m app.grading_sheet            # writes data/eval/grading_sheet.md and data/eval/grades_template.csv (never overwrites grades.csv)

You read grading_sheet.md, then copy grades_template.csv to data/eval/grades.csv and fill two columns per row:
  answer_grade      correct | wrong | refused_correctly | refused_wrongly
  citation_support  supported | unsupported      (does the cited passage really back the claim? a person decides)
Rules: Q16 (unanswerable) is refused_correctly when the answer says it is not in the passages, wrong if it gives a BLEU number as the answer.
'correct' needs: the content is right, the pipeline's citation provenance is 'cited', AND citation_support is 'supported'.
A request that failed or timed out gets no grade (leave the row out).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


def load_jobs(path: Path) -> List[Dict[str, Any]]:
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    return [r for r in rows if r.get("scenario") == "eval"]


def request_id(job: Dict[str, Any]) -> str:
    return f"real-{job['run_id']}-{int(job['job_id']):03d}"      # the same id export_csv builds


def load_questions(path: Path) -> Dict[str, Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return {r["id"]: r for r in csv.DictReader(fh)}


def render_sheet(jobs: Sequence[Dict[str, Any]], questions: Dict[str, Dict[str, str]]) -> str:
    out = ["# Grading sheet (real evaluation answers)", "",
           "For each answer: is it right? Does the cited passage back the claim? Fill data/eval/grades.csv.", ""]
    for j in sorted(jobs, key=lambda x: (x["question_id"], x["retrieval_setting"])):
        q = questions.get(j["question_id"], {})
        out += [f"## {j['question_id']} / setting {j['retrieval_setting']}  ({q.get('kind', '')})", f"request_id: `{request_id(j)}`",
                f"Q: {j['question']}", f"Expected terms: {q.get('must_include') or '-'}   Evidence page: {q.get('evidence_page') or '-'}",
                f"Outcome: {j['outcome']}; provenance: {j.get('citation_status')}; cited passages: {j.get('cited')}; "
                f"numbers not found in cited passages: {j.get('unsupported_numbers')}", "", f"ANSWER: {j.get('answer') or '(none)'}", ""]
        cited = set(j.get("cited") or [])
        for p in j.get("passages", []):
            mark = "CITED" if p["n"] in cited else "     "
            out.append(f"- [{p['n']}] {mark} {p['paper_key']} {p['pages']} [{p['section']}]: {' '.join(p['text'].split())[:420]}")
        out.append("")
    return "\n".join(out)


def template_rows(jobs: Sequence[Dict[str, Any]]) -> List[Dict[str, str]]:
    return [{"request_id": request_id(j), "answer_grade": "", "citation_support": "", "note": f"{j['question_id']} {j['retrieval_setting']}"}
            for j in sorted(jobs, key=lambda x: (x["retrieval_setting"], x["question_id"])) if j["outcome"] in ("ok", "ok_late")]


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", default="data/raw/real/eval_jobs.jsonl")
    ap.add_argument("--questions", default="data/eval/questions.csv")
    ap.add_argument("--sheet", default="data/eval/grading_sheet.md")
    ap.add_argument("--template", default="data/eval/grades_template.csv")
    args = ap.parse_args(argv)
    jobs = load_jobs(Path(args.jobs))
    Path(args.sheet).write_text(render_sheet(jobs, load_questions(Path(args.questions))), encoding="utf-8")
    with Path(args.template).open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["request_id", "answer_grade", "citation_support", "note"])
        w.writeheader()
        w.writerows(template_rows(jobs))
    print(f"{len(jobs)} real evaluation jobs -> {args.sheet} and {args.template}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
