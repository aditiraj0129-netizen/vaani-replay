"""Shared fixtures. The PDFs here are SYNTHETIC test data, not the real arXiv papers."""
from __future__ import annotations

from typing import List

import pymupdf
import pytest

from app.chunking import Chunk
from app.config import Paper, Settings

FIXTURE_PAGES: List[str] = [
    "arXiv:1111.11111v2 [cs.LG] 1 Jan 2020\nAlpha Paper\nAbstract\n"
    "We study glimmer networks and show they converge quickly.",
    "1 Introduction\nGlimmer networks use a trans-\nformer backbone with gizmo attention. The widget\n"
    "benchmark is common in the field.",
    "3.2 Gizmo Attention\nGizmo attention costs quadratic memory. The widget\n"
    "benchmark is also used here. Zebra tokens are rare.",
]


def build_pdf_bytes(pages: List[str]) -> bytes:
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        leftover = page.insert_textbox(pymupdf.Rect(50, 50, 550, 780), text, fontsize=10)
        assert leftover >= 0, "fixture text does not fit on the page"
    data = doc.tobytes()
    doc.close()
    return data


@pytest.fixture
def paper() -> Paper:
    return Paper("alpha", "Alpha Paper", "1111.11111", 2)


@pytest.fixture
def pdf_bytes() -> bytes:
    return build_pdf_bytes(FIXTURE_PAGES)


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(data_dir=tmp_path)


def mk_chunk(i: int, text: str) -> Chunk:
    return Chunk(
        chunk_id=f"t-{i}", paper_key="t", arxiv_id="0000.00000", version=1, title="T",
        ordinal=i, page_start=1 + i, page_end=1 + i, section="s", text=text, n_words=len(text.split()),
    )
