import pytest

from app.config import PAPERS, RETRIEVAL_CONFIGS, RetrievalConfig, Settings, load_settings


def test_exact_paper_versions_and_urls_are_pinned():
    urls = {p.key: p.pdf_url for p in PAPERS}
    assert urls == {
        "attention": "https://arxiv.org/pdf/1706.03762v7",
        "moshi": "https://arxiv.org/pdf/2410.00037v1",
        "dynamic_chunking": "https://arxiv.org/pdf/2507.07955v2",
    }


def test_api_key_never_appears_in_repr():
    s = Settings(groq_api_key="gsk_SECRET_VALUE")
    assert "gsk_SECRET_VALUE" not in repr(s)
    assert "gsk_SECRET_VALUE" not in str(s)


def test_load_settings_parses_env_and_leaves_unknown_limits_as_none():
    s = load_settings({"GROQ_MODEL": "m", "GROQ_RPM": "30", "RELAY_SEED": "0"})
    assert s.groq_model == "m" and s.groq_rpm == 30 and s.seed == 0
    assert s.groq_tpm is None and s.groq_rpd is None and s.groq_api_key is None


def test_non_numeric_limit_is_a_clear_error():
    with pytest.raises(ValueError, match="GROQ_RPM"):
        load_settings({"GROQ_RPM": "thirty"})


def test_retrieval_presets_are_valid_and_distinct():
    assert set(RETRIEVAL_CONFIGS) == {"A", "B", "C"}
    assert RETRIEVAL_CONFIGS["C"].rerank and not RETRIEVAL_CONFIGS["A"].rerank
    sizes = {c.chunk_words for c in RETRIEVAL_CONFIGS.values()}
    assert len(sizes) == 3


@pytest.mark.parametrize("kwargs", [
    dict(chunk_words=0, overlap_words=0, top_k=1),
    dict(chunk_words=10, overlap_words=10, top_k=1),
    dict(chunk_words=10, overlap_words=2, top_k=0),
    dict(chunk_words=10, overlap_words=2, top_k=5, rerank=True, candidates=3),
])
def test_invalid_retrieval_config_rejected(kwargs):
    with pytest.raises(ValueError):
        RetrievalConfig("X", **kwargs)
