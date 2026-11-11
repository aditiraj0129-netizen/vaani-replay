"""Step 2: turn page text into overlapping chunks that remember where they came from.

Design choice: chunks may cross page boundaries (a sentence can span two pages), so each
chunk stores page_start and page_end instead of being cut at the page edge.
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from typing import List, NamedTuple, Sequence, Tuple

from app.config import PAPERS, Paper, RetrievalConfig, Settings

FRONT_MATTER = "(front matter)"
_NUMBERED = re.compile(r"^\d{1,2}(?:\.\d{1,2})*\.?\s+[A-Z]\S*(?:\s+\S+)*$")
_NAMED = re.compile(r"^(?:Abstract|References|Acknowledge?ments?|Appendix(?:\s+[A-Z])?)\b.*$")
_BARE_NUMBER = re.compile(r"^\d{1,2}(?:\.\d{1,2})*$")  # a heading number alone on its line, e.g. "3.1"
_PAGE_NUMBER = re.compile(r"^\d{1,3}$")


class Word(NamedTuple):
    text: str
    page: int
    section: str


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    paper_key: str
    arxiv_id: str
    version: int
    title: str
    ordinal: int
    page_start: int
    page_end: int
    section: str
    text: str
    n_words: int


def clean_page_text(text: str) -> str:
    """Normalise ligatures (fi), drop soft hyphens, re-join words split across lines."""
    text = unicodedata.normalize("NFKC", text).replace("\u00ad", "")
    return re.sub(r"(\w)-\n(\w)", r"\1\2", text)


def _plausible_section_number(token: str) -> bool:
    """'3.2.1' yes. '1.05', '0.84', '2.95', '49.7', '93.0' (table cells) no.

    Real section numbers in these papers have no leading zeros and small parts (1 to 20).
    A trailing dot ("1.") marks a list item, not a heading.
    """
    if token.endswith("."):
        return False
    parts = token.split(".")
    for part in parts:
        if not part.isdigit() or (len(part) > 1 and part.startswith("0")) or not 1 <= int(part) <= 20:
            return False
    return True


def is_heading(line: str) -> bool:
    """Best-effort section-heading detector. Page numbers are reliable; section labels are not."""
    tokens = line.split()
    if not tokens or line.endswith((".", ",", ";")):
        return False
    if _NAMED.match(line):
        return len(tokens) <= 6
    if _NUMBERED.match(line) and len(tokens) <= 12:
        if not _plausible_section_number(tokens[0]) or ". " in line:  # table cell / list item / sentence
            return False
        numeric = sum(t.replace(".", "").isdigit() for t in tokens)
        return numeric <= 1  # table rows like "6 Base 65M 512 2048" have many numbers
    return False


def _top_number(heading: str, current: int) -> int:
    first = heading.split()[0]
    return int(first.split(".")[0]) if first[0].isdigit() else current


def _in_order(heading: str, top: int) -> bool:
    """Numbered headings must follow the paper's numbering: never backwards, never far ahead.

    Rejects table rows and list items that look like headings but are out of sequence
    ("1 Text delay" after section 3, "20.2 LlamaByte", a second plain "4 ..." heading).
    Before the first numbered heading anything plausible is accepted.
    """
    first = heading.split()[0]
    if not first[0].isdigit():  # Abstract / References / Appendix X
        return True
    number = int(first.split(".")[0])
    if top == 0:
        return True
    if number < top:
        return False
    if "." not in first:  # a plain top-level heading must be a NEW section, close to the last one
        return number in (top + 1, top + 2)
    return number <= top + 2


def paper_words(pages: Sequence[Tuple[int, str]]) -> List[Word]:
    """Flatten (page_number, raw_text) pairs into words tagged with page and current section.

    Two PDF quirks handled here (both seen in the real Attention paper):
    - a heading number and its title can sit on separate lines ("3.1" then "Encoder and ...");
    - the printed page number is the last line of the page and is not part of the text.
    """
    words: List[Word] = []
    section = FRONT_MATTER
    top = 0  # top-level section number reached so far; headings may not go backwards or skip ahead
    for page_no, raw in pages:
        lines = [ln.strip() for ln in clean_page_text(raw).splitlines() if ln.strip()]
        if lines and _PAGE_NUMBER.match(lines[-1]):
            lines = lines[:-1]
        i = 0
        while i < len(lines):
            line = lines[i]
            if _BARE_NUMBER.match(line) and i + 1 < len(lines) and is_heading(f"{line} {lines[i + 1]}"):
                line = f"{line} {lines[i + 1]}"
                i += 1
            if is_heading(line) and _in_order(line, top):
                section = line
                top = _top_number(line, top)
            words.extend(Word(w, page_no, section) for w in line.split())
            i += 1
    return words


def _main_section(window: Sequence[Word]) -> str:
    """The section most of the chunk's words belong to (ties: the earlier one).

    Labelling by the FIRST word hid short sections: a 100-word section never started a chunk.
    """
    counts: dict = {}
    for w in window:
        counts[w.section] = counts.get(w.section, 0) + 1
    best = max(counts.values())
    return next(w.section for w in window if counts[w.section] == best)


def make_chunks(
    paper: Paper, words: Sequence[Word], chunk_words: int, overlap_words: int, config_name: str
) -> List[Chunk]:
    if chunk_words <= 0 or not 0 <= overlap_words < chunk_words:
        raise ValueError("need chunk_words > 0 and 0 <= overlap_words < chunk_words")
    step = chunk_words - overlap_words
    chunks: List[Chunk] = []
    start = 0
    while start < len(words):
        window = words[start : start + chunk_words]
        ordinal = len(chunks)
        chunks.append(
            Chunk(
                chunk_id=f"{paper.key}-{config_name}-{ordinal:04d}",
                paper_key=paper.key,
                arxiv_id=paper.arxiv_id,
                version=paper.version,
                title=paper.title,
                ordinal=ordinal,
                page_start=window[0].page,
                page_end=window[-1].page,
                section=_main_section(window),
                text=" ".join(w.text for w in window),
                n_words=len(window),
            )
        )
        if start + chunk_words >= len(words):
            break  # the last window already reaches the end; no tiny duplicate tail
        start += step
    return chunks


def load_processed(settings: Settings, paper: Paper) -> List[Tuple[int, str]]:
    path = settings.processed_dir / f"{paper.key}.json"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found. Run: python -m app.ingest")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data["arxiv_id"] != paper.arxiv_id or data["version"] != paper.version:
        raise ValueError(f"{path} is for {data['arxiv_id']}v{data['version']}, expected {paper.versioned_id}")
    return [(p["page"], p["text"]) for p in data["pages"]]


# ---------------------------------------------------------------------------------------------- reference lists
# Reference lists are removed from the index (same rule for A, B and C) because a bibliography is never evidence for an answer and,
# full of titles, it can outrank real passages. The filter works on WORDS, before chunking, so body text can never be clipped:
#   * only words under a heading that starts with "References" / "Bibliography" (the chunker's heading detector) are considered;
#   * that run is cut into segments of SEGMENT_WORDS words; a segment is dropped only if it READS like a bibliography
#     (many entry markers: arXiv, "In Proceedings", pages, DOIs, volume(issue):pages ...), or it sits between two bibliography segments;
#     in the last bibliography segment the words after the final entry's full stop are kept (they may start an appendix);
#   * anything else under that heading is KEPT and reported as "flagged". Typical cause: a lettered appendix heading ("A Details") is not
#     recognised by the heading detector, so the appendix after the reference list keeps the label "References".
# Chunks are then built from the remaining words, so each config simply has fewer pieces (python -m app.reference_audit shows how many).
EXCLUDE_REFERENCE_LISTS = True
SEGMENT_WORDS = 60
BIB_MARKERS_MIN = 3      # a segment needs at least this many entry markers ...
BIB_DENSITY_MIN = 3.0    # ... and at least this many per 100 words
_REF_LABEL = re.compile(r"^\s*(?:references|bibliography)\b", re.IGNORECASE)
_BIB_MARKERS_CI = re.compile("|".join([
    r"\barxiv\b", r"\bpreprint\b", r"\bin proceedings\b", r"\bproc\.", r"\bproceedings of\b", r"\bpages?\s+\d", r"\bpp\.\s*\d",
    r"https?://", r"\bdoi\b", r"\bjournal of\b", r"\bconference on\b", r"\badvances in neural information\b", r"\b(?:ieee|acm)\b",
    r"\bvol(?:ume|\.)\s*\d", r"\d+\s*\(\d+\)\s*:\s*\d+", r"\b\d{1,4}\s?[–-]\s?\d{1,4},\s*(?:19|20)\d{2}\b",
    r",\s(?:19|20)\d{2}[a-z]?\.(?:\s|$)",
]), re.IGNORECASE)
_BIB_INITIALS = re.compile(r"\b[A-Z]\.\s?(?:[A-Z]\.\s?)?[A-Z][a-z]+,")  # "Y. Bengio," (case-sensitive: not "e.g. the,")


def bibliography_score(text: str) -> Tuple[int, float]:
    """(number of bibliography-entry markers, markers per 100 words)."""
    count = len(_BIB_MARKERS_CI.findall(text)) + len(_BIB_INITIALS.findall(text))
    words = len(text.split())
    return count, (100.0 * count / words if words else 0.0)


def has_reference_label(section: str) -> bool:
    return bool(_REF_LABEL.match(section))


def looks_like_bibliography(text: str) -> bool:
    count, density = bibliography_score(text)
    return count >= BIB_MARKERS_MIN and density >= BIB_DENSITY_MIN


@dataclass(frozen=True)
class Flagged:
    page_start: int
    page_end: int
    markers: int
    per100: float
    start: str


def _entry_end(seg: Sequence[Word]) -> int:
    """Index just after the last bibliography entry in a segment (a segment can end with a few non-bibliography words)."""
    text = " ".join(w.text for w in seg)
    ends = sorted([m.end() for m in _BIB_MARKERS_CI.finditer(text)] + [m.end() for m in _BIB_INITIALS.finditer(text)])
    # a lone marker (an inline "et al., 2019." in running text) does not count: a real entry has several markers close together
    ends = [e for i, e in enumerate(ends) if (i > 0 and e - ends[i - 1] <= 150) or (i + 1 < len(ends) and ends[i + 1] - e <= 150)]
    if not ends:
        return 0 if not looks_like_bibliography(text) else len(seg)
    last, pos, idx = max(ends), 0, 0
    for idx, w in enumerate(seg):
        pos += len(w.text) + 1
        if pos >= last and w.text.endswith("."):     # an entry ends at the first full stop after its last marker
            return idx + 1
    return len(seg)


def filter_reference_words(words: Sequence[Word]) -> Tuple[List[Word], List[Word], List[Flagged]]:
    """(kept words, removed words, flagged segments). See the block comment above for the rule."""
    kept: List[Word] = []
    removed: List[Word] = []
    flagged: List[Flagged] = []
    i, n = 0, len(words)
    while i < n:
        if not has_reference_label(words[i].section):
            kept.append(words[i])
            i += 1
            continue
        j = i
        while j < n and words[j].section == words[i].section:  # one run under one heading
            j += 1
        segs = [list(words[k:k + SEGMENT_WORDS]) for k in range(i, j, SEGMENT_WORDS)]
        bib = [looks_like_bibliography(" ".join(w.text for w in seg)) for seg in segs]
        drop = []
        for k, seg in enumerate(segs):
            between = 0 < k < len(segs) - 1 and bib[k - 1] and bib[k + 1]
            tail = k == len(segs) - 1 and k > 0 and len(seg) < SEGMENT_WORDS and bib[k - 1]   # short last segment: cut at its final entry
            drop.append(bib[k] or between or tail)
        for k, (seg, d) in enumerate(zip(segs, drop)):
            if d:
                nxt_kept = k + 1 < len(segs) and not drop[k + 1]
                tail = k == len(segs) - 1 and not bib[k]
                cut = _entry_end(seg) if (nxt_kept or tail) else len(seg)   # last bibliography segment: keep words after its final entry
                removed.extend(seg[:cut])
                kept.extend(seg[cut:])
            else:
                kept.extend(seg)
                text = " ".join(w.text for w in seg)
                count, dens = bibliography_score(text)
                flagged.append(Flagged(seg[0].page, seg[-1].page, count, dens, " ".join(text.split())[:120]))
        i = j
    return kept, removed, flagged


@dataclass(frozen=True)
class PaperFilter:
    paper_key: str
    words_before: int
    words_removed: int
    removed_pages: Tuple[int, int]      # (0, 0) when nothing was removed
    chunks_before: int
    chunks_after: int
    flagged: List[Flagged]
    first_removed: str
    last_removed: str


@dataclass(frozen=True)
class CorpusReport:
    kept: List[Chunk]                   # what the retriever indexes
    papers: List[PaperFilter]

    @property
    def chunks_before(self) -> int:
        return sum(p.chunks_before for p in self.papers)

    @property
    def chunks_removed(self) -> int:
        return sum(p.chunks_before - p.chunks_after for p in self.papers)


def build_corpus_report(settings: Settings, config: RetrievalConfig, exclude_references: bool = EXCLUDE_REFERENCE_LISTS) -> CorpusReport:
    kept_chunks: List[Chunk] = []
    papers: List[PaperFilter] = []
    for paper in PAPERS:
        words = paper_words(load_processed(settings, paper))
        before = make_chunks(paper, words, config.chunk_words, config.overlap_words, config.name)
        if exclude_references:
            kept, removed, flagged = filter_reference_words(words)
        else:
            kept, removed, flagged = list(words), [], []
        after = make_chunks(paper, kept, config.chunk_words, config.overlap_words, config.name) if kept else []
        kept_chunks.extend(after)
        papers.append(PaperFilter(
            paper_key=paper.key, words_before=len(words), words_removed=len(removed),
            removed_pages=(min(w.page for w in removed), max(w.page for w in removed)) if removed else (0, 0),
            chunks_before=len(before), chunks_after=len(after), flagged=flagged,
            first_removed=" ".join(w.text for w in removed[:14]), last_removed=" ".join(w.text for w in removed[-14:])))
    return CorpusReport(kept=kept_chunks, papers=papers)


def build_corpus(settings: Settings, config: RetrievalConfig) -> List[Chunk]:
    """The pieces a retrieval config indexes: every piece, built after reference lists are removed (EXCLUDE_REFERENCE_LISTS)."""
    return build_corpus_report(settings, config).kept
