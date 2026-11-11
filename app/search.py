"""Manual retrieval inspector.   python -m app.search "your question" --config B"""
from __future__ import annotations

import argparse
from typing import List, Optional

from app.config import RETRIEVAL_CONFIGS, load_settings
from app.retrieval import Retriever


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Show what BM25 retrieves for a question.")
    parser.add_argument("query")
    parser.add_argument("--config", choices=sorted(RETRIEVAL_CONFIGS), default="B")
    parser.add_argument("--k", type=int, default=None, help="override top_k")
    args = parser.parse_args(argv)
    config = RETRIEVAL_CONFIGS[args.config]
    retriever = Retriever.from_settings(load_settings(), config)
    hits = retriever.index.search(args.query, args.k or config.top_k)
    if not hits:
        print("no passages matched")
    for h in hits:
        c = h.chunk
        pages = f"p.{c.page_start}" if c.page_start == c.page_end else f"p.{c.page_start}-{c.page_end}"
        print(f"#{h.rank}  score={h.score:.2f}  {c.chunk_id}  {c.paper_key} v{c.version} {pages}  [{c.section}]")
        print(f"    {c.text[:240]}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
