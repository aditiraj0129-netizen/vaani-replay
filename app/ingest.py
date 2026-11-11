"""Step 1: fetch the exact arXiv versions, extract text page by page, write a manifest.

Run:  python -m app.ingest                      download + extract all three papers
      python -m app.ingest --only moshi         just one paper
      python -m app.ingest --inspect attention 3   print the extracted text of page 3
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional

import httpx
import pymupdf

from app.config import PAPERS, PAPERS_BY_KEY, Paper, Settings, load_settings

STAMP_RE = re.compile(r"arXiv\s*:\s*(\d{4}\.\d{4,5})\s*v\s*(\d+)")
MIN_CHARS_PER_PAGE = 50
POLITE_DELAY_S = 3.0


class IngestError(RuntimeError):
    """A paper could not be fetched, or it does not look like the requested version."""


def make_client() -> httpx.Client:
    return httpx.Client(
        follow_redirects=True,
        timeout=60.0,
        headers={"User-Agent": "relay-assignment/0.1 (coursework)"},
    )


def download_pdf(
    paper: Paper,
    pdf_dir: Path,
    client: httpx.Client,
    *,
    force: bool = False,
    attempts: int = 3,
    sleep: Callable[[float], None] = time.sleep,
) -> Path:
    """Download the exact version once. Existing files are reused unless force=True."""
    dest = pdf_dir / f"{paper.versioned_id}.pdf"
    if dest.exists() and not force:
        return dest
    pdf_dir.mkdir(parents=True, exist_ok=True)
    problem = "no attempt made"
    for attempt in range(1, attempts + 1):
        try:
            resp = client.get(paper.pdf_url)
        except httpx.HTTPError as exc:
            problem = f"network error: {type(exc).__name__}"
        else:
            if resp.status_code == 404:
                raise IngestError(f"{paper.pdf_url} returned 404: that exact version may not exist")
            if resp.status_code == 200 and resp.content.startswith(b"%PDF"):
                tmp = dest.with_suffix(".part")
                tmp.write_bytes(resp.content)
                tmp.replace(dest)  # never leave a half-written PDF behind
                return dest
            problem = f"HTTP {resp.status_code}, first bytes {resp.content[:16]!r}"
        if attempt < attempts:
            sleep(2.0 * attempt)
    raise IngestError(f"could not download {paper.pdf_url} after {attempts} attempts: {problem}")


def extract_pages(pdf_path: Path) -> List[str]:
    """Raw text of each page, in order. Page numbers are index + 1."""
    with pymupdf.open(str(pdf_path)) as doc:
        return [page.get_text("text") for page in doc]


def check_version_stamp(paper: Paper, first_page_text: str) -> Optional[bool]:
    """arXiv prints 'arXiv:<id>v<n>' in the margin of page 1.

    True  = stamp found and matches the requested version
    None  = no stamp found (cannot confirm; recorded in the manifest)
    Raises IngestError if the stamp shows a DIFFERENT version.
    """
    versions = [int(v) for pid, v in STAMP_RE.findall(first_page_text) if pid == paper.arxiv_id]
    if not versions:
        return None
    if paper.version in versions:
        return True
    raise IngestError(
        f"{paper.arxiv_id}: wanted v{paper.version} but page 1 is stamped v{sorted(set(versions))}"
    )


def process_paper(
    paper: Paper,
    settings: Settings,
    client: httpx.Client,
    *,
    force: bool = False,
    sleep: Callable[[float], None] = time.sleep,
) -> Dict[str, object]:
    pdf_path = download_pdf(paper, settings.pdf_dir, client, force=force, sleep=sleep)
    pages = extract_pages(pdf_path)
    if not pages:
        raise IngestError(f"{pdf_path} has no pages")
    stamp = check_version_stamp(paper, pages[0])
    thin = [i + 1 for i, text in enumerate(pages) if len(text.strip()) < MIN_CHARS_PER_PAGE]
    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "key": paper.key,
        "title": paper.title,
        "arxiv_id": paper.arxiv_id,
        "version": paper.version,
        "pages": [{"page": i + 1, "text": text} for i, text in enumerate(pages)],
    }
    (settings.processed_dir / f"{paper.key}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return {
        "key": paper.key,
        "title": paper.title,
        "arxiv_id": paper.arxiv_id,
        "version": paper.version,
        "url": paper.pdf_url,
        "pdf_file": pdf_path.name,
        "bytes": pdf_path.stat().st_size,
        "sha256": hashlib.sha256(pdf_path.read_bytes()).hexdigest(),
        "n_pages": len(pages),
        "n_chars": sum(len(t) for t in pages),
        "version_stamp_confirmed": stamp,
        "pages_with_little_text": thin,
        "processed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def update_manifest(settings: Settings, entries: List[Dict[str, object]]) -> None:
    """Merge entries into data/manifest.json by paper key."""
    existing: Dict[str, Dict[str, object]] = {}
    if settings.manifest_path.is_file():
        existing = json.loads(settings.manifest_path.read_text(encoding="utf-8"))
    for entry in entries:
        existing[str(entry["key"])] = entry
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.manifest_path.write_text(json.dumps(existing, indent=1, sort_keys=True), encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Download and extract the three arXiv papers.")
    parser.add_argument("--force", action="store_true", help="download again even if the PDF exists")
    parser.add_argument("--only", choices=sorted(PAPERS_BY_KEY), help="process a single paper")
    parser.add_argument("--inspect", nargs=2, metavar=("KEY", "PAGE"), help="print one extracted page")
    args = parser.parse_args(argv)
    settings = load_settings()

    if args.inspect:
        key, page_no = args.inspect[0], int(args.inspect[1])
        path = settings.processed_dir / f"{key}.json"
        if not path.is_file():
            print(f"{path} not found. Run: python -m app.ingest", file=sys.stderr)
            return 1
        pages = json.loads(path.read_text(encoding="utf-8"))["pages"]
        if not 1 <= page_no <= len(pages):
            print(f"{key} has pages 1..{len(pages)}", file=sys.stderr)
            return 1
        print(pages[page_no - 1]["text"])
        return 0

    papers = [PAPERS_BY_KEY[args.only]] if args.only else list(PAPERS)
    entries: List[Dict[str, object]] = []
    with make_client() as client:
        for paper in papers:
            was_cached = (settings.pdf_dir / f"{paper.versioned_id}.pdf").exists() and not args.force
            try:
                entry = process_paper(paper, settings, client, force=args.force)
            except IngestError as exc:
                print(f"FAILED {paper.key}: {exc}", file=sys.stderr)
                return 1
            entries.append(entry)
            print(
                f"{paper.key:17s} v{paper.version}  pages={entry['n_pages']}  chars={entry['n_chars']}  "
                f"stamp_confirmed={entry['version_stamp_confirmed']}  "
                f"thin_pages={entry['pages_with_little_text']}"
            )
            if not was_cached:
                time.sleep(POLITE_DELAY_S)  # be polite to arxiv.org
    update_manifest(settings, entries)
    print(f"manifest written to {settings.manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
