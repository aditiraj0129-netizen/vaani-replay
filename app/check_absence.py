"""FREE (zero API calls): check that Q16's subject (a WMT 2016 English-Romanian result) is absent from the three pinned PDFs.

    python -m app.check_absence --pdfs data/pdfs --out data/eval/absence_check.txt

Text check: every page of every PDF is searched for Romanian / WMT 2016 / newstest2016 / En-Ro / BLEU, and every BLEU table line is listed,
so you can read which language pairs the Transformer paper really reports.
Image check: text search cannot see a table that is a picture. Pages that contain images are listed with how much of the page they
cover and how many words the page has; open those pages yourself. A clean text result is evidence of absence, not proof.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

TERMS = {"romanian": r"romanian", "wmt 2016": r"wmt\s*[-']?\s*2016", "newstest2016": r"newstest\s*2016",
         "en-ro / en→ro": r"\ben\s*[-\u2013\u2192>]+\s*ro\b", "bleu": r"\bbleu\b"}
IMAGE_COVER_FLAG = 0.15  # a page with images covering this share of it is listed for a manual look


def scan_pages(pages: Iterable[Tuple[int, str, float]]) -> Dict[str, Any]:
    """pages: (page number, text, share of the page covered by images). Returns hits, BLEU lines and pages to eyeball."""
    hits: Dict[str, List[int]] = {k: [] for k in TERMS}
    bleu_lines: List[Tuple[int, str]] = []
    look: List[Tuple[int, float, int]] = []
    for no, text, cover in pages:
        for name, pat in TERMS.items():
            if re.search(pat, text, flags=re.IGNORECASE):
                hits[name].append(no)
        for line in text.splitlines():
            if re.search(TERMS["bleu"], line, flags=re.IGNORECASE):
                bleu_lines.append((no, " ".join(line.split())))
        if cover >= IMAGE_COVER_FLAG:
            look.append((no, round(cover, 2), len(text.split())))
    return {"hits": hits, "bleu_lines": bleu_lines, "look_at": look}


def render(name: str, res: Dict[str, Any]) -> str:
    lines = [f"===== {name} ====="]
    for term, pgs in res["hits"].items():
        if term != "bleu":
            lines.append(f"  {term:<14}: " + (f"FOUND on pages {pgs}" if pgs else "0 matches in the text layer"))
    lines.append(f"  bleu          : on pages {res['hits']['bleu']}")
    for no, text in res["bleu_lines"][:40]:
        lines.append(f"     p.{no}: {text[:150]}")
    lines.append("  pages with images to open by eye (page, share of page covered, words of text): "
                 + (", ".join(f"p.{n} ({c}, {w} words)" for n, c, w in res["look_at"]) or "none"))
    return "\n".join(lines)


def pdf_pages(path: Path) -> List[Tuple[int, str, float]]:
    import pymupdf  # imported here so the checks above run without it
    pages = []
    with pymupdf.open(path) as doc:
        for no, page in enumerate(doc, 1):
            area = page.rect.width * page.rect.height or 1.0
            covered = 0.0
            for info in page.get_image_info():
                x0, y0, x1, y1 = info["bbox"]
                covered += max(x1 - x0, 0) * max(y1 - y0, 0)
            pages.append((no, page.get_text("text"), min(covered / area, 1.0)))
    return pages


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdfs", default="data/pdfs")
    ap.add_argument("--out", default="data/eval/absence_check.txt")
    args = ap.parse_args(argv)
    blocks = [render(p.name, scan_pages(pdf_pages(p))) for p in sorted(Path(args.pdfs).glob("*.pdf"))]
    if not blocks:
        print(f"no PDFs found in {args.pdfs}")
        return 1
    text = "\n\n".join(blocks)
    Path(args.out).write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
