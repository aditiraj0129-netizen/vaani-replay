"""Free check (no API): do the terms behind our 'unanswerable' / 'partial' probe questions appear in the papers?

Run:  python -m app.check_probe_terms
Counts are of case-insensitive whole-word matches in the extracted text of each paper.
A count of 0 supports an 'unanswerable' label. It does not prove it (the PDF text can miss things).
"""
from __future__ import annotations

import json
import re
from typing import Dict, Iterable, List

from app.config import PAPERS, load_settings

TERMS = ["Russian", "Mandarin", "ImageNet", "dollar", "dollars", "cost", "energy", "Helium", "GPU", "P100"]


def count_terms(texts: Iterable[str], terms: List[str]) -> Dict[str, int]:
    joined = "\n".join(texts)
    return {t: len(re.findall(rf"\b{re.escape(t)}\b", joined, flags=re.IGNORECASE)) for t in terms}


def main() -> int:
    settings = load_settings()
    print(f"{'term':<10} " + " ".join(f"{p.key:>17}" for p in PAPERS))
    per_paper = {}
    for paper in PAPERS:
        data = json.loads((settings.processed_dir / f"{paper.key}.json").read_text(encoding="utf-8"))
        per_paper[paper.key] = count_terms((p["text"] for p in data["pages"]), TERMS)
    for term in TERMS:
        print(f"{term:<10} " + " ".join(f"{per_paper[p.key][term]:>17}" for p in PAPERS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
