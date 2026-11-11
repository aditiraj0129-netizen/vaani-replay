"""Central configuration: the paper list, retrieval presets and environment settings.

Nothing here is a measurement. Groq limits stay None ("unknown") until you copy them
from your own Groq account page or from the output of `python -m app.check_groq`.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Paper:
    key: str
    title: str
    arxiv_id: str
    version: int

    @property
    def versioned_id(self) -> str:
        return f"{self.arxiv_id}v{self.version}"

    @property
    def pdf_url(self) -> str:
        return f"https://arxiv.org/pdf/{self.versioned_id}"


PAPERS: Tuple[Paper, ...] = (
    Paper("attention", "Attention Is All You Need", "1706.03762", 7),
    Paper("moshi", "Moshi: a speech-text foundation model for real-time dialogue", "2410.00037", 1),
    Paper("dynamic_chunking", "Dynamic Chunking for End-to-End Hierarchical Sequence Modeling", "2507.07955", 2),
)
PAPERS_BY_KEY: Dict[str, Paper] = {p.key: p for p in PAPERS}


@dataclass(frozen=True)
class RetrievalConfig:
    """How papers are split and how much is fetched. Sizes are in WORDS (whitespace split)."""

    name: str
    chunk_words: int
    overlap_words: int
    top_k: int
    rerank: bool = False
    candidates: int = 0  # BM25 hits handed to the re-ranker (only used when rerank=True)

    def __post_init__(self) -> None:
        if self.chunk_words <= 0:
            raise ValueError("chunk_words must be positive")
        if not 0 <= self.overlap_words < self.chunk_words:
            raise ValueError("overlap_words must be >= 0 and smaller than chunk_words")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.rerank and self.candidates < self.top_k:
            raise ValueError("with rerank=True, candidates must be >= top_k")


# PROVISIONAL values. We finalise them after reading the extracted papers (see README).
RETRIEVAL_CONFIGS: Dict[str, RetrievalConfig] = {
    "A": RetrievalConfig("A", chunk_words=80, overlap_words=20, top_k=8),
    "B": RetrievalConfig("B", chunk_words=200, overlap_words=40, top_k=5),
    "C": RetrievalConfig("C", chunk_words=300, overlap_words=60, top_k=3, rerank=True, candidates=12),
}


@dataclass(frozen=True)
class Settings:
    data_dir: Path = PROJECT_ROOT / "data"
    seed: int = 1234
    groq_api_key: Optional[str] = field(default=None, repr=False)  # never printed
    groq_model: Optional[str] = None
    groq_base_url: str = "https://api.groq.com/openai/v1"
    # Account limits. None means "not known yet". We never guess these.
    groq_rpm: Optional[int] = None
    groq_tpm: Optional[int] = None
    groq_rpd: Optional[int] = None
    groq_tpd: Optional[int] = None

    @property
    def pdf_dir(self) -> Path:
        return self.data_dir / "pdfs"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def manifest_path(self) -> Path:
        return self.data_dir / "manifest.json"


def _load_dotenv(path: Path) -> None:
    """Tiny .env reader. Variables already set in the shell win."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        os.environ.setdefault(name.strip(), value.strip().strip("'\""))


def _opt_int(env: Mapping[str, str], name: str) -> Optional[int]:
    raw = env.get(name, "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"{name} must be a whole number, got {raw!r}") from None


def load_settings(env: Optional[Mapping[str, str]] = None) -> Settings:
    """Build Settings from the environment (and .env). Pass `env` in tests."""
    if env is None:
        _load_dotenv(PROJECT_ROOT / ".env")
        env = os.environ
    seed = _opt_int(env, "RELAY_SEED")
    return Settings(
        data_dir=Path(env.get("RELAY_DATA_DIR") or PROJECT_ROOT / "data"),
        seed=1234 if seed is None else seed,
        groq_api_key=env.get("GROQ_API_KEY") or None,
        groq_model=env.get("GROQ_MODEL") or None,
        groq_base_url=env.get("GROQ_BASE_URL") or "https://api.groq.com/openai/v1",
        groq_rpm=_opt_int(env, "GROQ_RPM"),
        groq_tpm=_opt_int(env, "GROQ_TPM"),
        groq_rpd=_opt_int(env, "GROQ_RPD"),
        groq_tpd=_opt_int(env, "GROQ_TPD"),
    )
