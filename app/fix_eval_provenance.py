"""One-off, FREE: re-run the mechanical citation check on the saved real answers. No API calls.

extract_citations used to read bracketed data such as the delay pattern "[0, 2, 2, 2]" as a citation of passage 0, so two correct
Q10 answers were flagged 'invalid'. The rule now ignores a bracket group that repeats a number (a citation never does). This script
  * copies data/raw/real/eval_jobs.jsonl to data/raw/real_archive/eval_jobs_before_provenance_fix.jsonl (kept, never edited),
  * recomputes citation_status, cited and unsupported_numbers for every eval line from its saved answer and passages.
Running it twice changes nothing.   python -m app.fix_eval_provenance
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Optional, Sequence

from app.rag import verify

JOBS = Path("data/raw/real/eval_jobs.jsonl")
ARCHIVE = Path("data/raw/real_archive/eval_jobs_before_provenance_fix.jsonl")


def main(argv: Optional[Sequence[str]] = None, jobs_path: Path = JOBS, archive: Path = ARCHIVE) -> int:
    lines = [json.loads(x) for x in jobs_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    out, changed = [], []
    for j in lines:
        if j.get("scenario") == "eval" and j.get("answer"):
            v = verify(j["answer"], [p["text"] for p in j["passages"]])
            new = dict(j, citation_status=v.status, cited=v.cited, unsupported_numbers=v.unsupported_numbers)
            if new != j:
                changed.append((j, new))
                j = dict(new, provenance_fix="re-verified with the repeated-number rule; original kept in real_archive")
        out.append(j)
    if changed:
        archive.parent.mkdir(parents=True, exist_ok=True)
        if not archive.exists():
            shutil.copy2(jobs_path, archive)
        jobs_path.write_text("".join(json.dumps(j, ensure_ascii=False, sort_keys=True) + "\n" for j in out), encoding="utf-8")
    print(f"{len(changed)} of {len(lines)} job lines re-verified (original kept at {archive})")
    for old, new in changed:
        print(f"   {old['retrieval_setting']} {old['question_id']}: {old['citation_status']} -> {new['citation_status']}; "
              f"cited {old['cited']} -> {new['cited']}; unsupported numbers {old['unsupported_numbers']} -> {new['unsupported_numbers']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
