"""FREE (zero API calls): show exactly what the reference-list filter removes, per paper and per retrieval config.

    python -m app.reference_audit --out data/eval/reference_audit.txt

For each config (A, B, C) and paper it prints the pieces before and after the filter, the words removed and their pages, the first and
last removed words (check that they really are a bibliography), and every FLAGGED segment: text under a "References" heading that did
NOT read like a bibliography and was therefore KEPT (typical cause: an appendix with a lettered heading after the reference list).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence

from app.chunking import BIB_DENSITY_MIN, BIB_MARKERS_MIN, SEGMENT_WORDS, CorpusReport


def render(reports: Dict[str, CorpusReport]) -> str:
    lines = ["REFERENCE-LIST FILTER AUDIT (free, no API calls)",
             f"Words under a References/Bibliography heading are cut into {SEGMENT_WORDS}-word segments. A segment is removed only if it has at least "
             f"{BIB_MARKERS_MIN} bibliography markers and {BIB_DENSITY_MIN} per 100 words (or sits between / right after bibliography segments).",
             "Flagged segments carry the References label but do not read like a bibliography: they are KEPT.", ""]
    for cfg, rep in reports.items():
        lines.append(f"CONFIG {cfg}: pieces before {rep.chunks_before}, after {rep.chunks_before - rep.chunks_removed}, removed {rep.chunks_removed}")
        for p in rep.papers:
            pages = f"p.{p.removed_pages[0]}-{p.removed_pages[1]}" if p.words_removed else "none"
            lines.append(f"   {p.paper_key:<18} pieces {p.chunks_before:>4} -> {p.chunks_after:>4} (removed {p.chunks_before - p.chunks_after:>3}); "
                         f"words removed {p.words_removed} of {p.words_before} ({pages}); flagged segments {len(p.flagged)}")
            if p.words_removed:
                lines.append(f"      first removed words: {p.first_removed}")
                lines.append(f"      last removed words : {p.last_removed}")
            for f in p.flagged:
                lines.append(f"      FLAGGED kept p.{f.page_start}-{f.page_end} markers={f.markers} per100={f.per100:.1f}: {f.start}")
        lines.append("")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    from app.chunking import build_corpus_report
    from app.config import RETRIEVAL_CONFIGS, load_settings
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/eval/reference_audit.txt")
    args = ap.parse_args(argv)
    settings = load_settings()
    text = render({k: build_corpus_report(settings, c) for k, c in RETRIEVAL_CONFIGS.items()})
    Path(args.out).write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
