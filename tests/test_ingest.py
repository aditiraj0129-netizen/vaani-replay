import json

import httpx
import pytest

from app.ingest import IngestError, check_version_stamp, download_pdf, extract_pages, process_paper, update_manifest

NO_SLEEP = lambda _seconds: None  # noqa: E731


def client_for(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)


def test_extract_pages_one_string_per_page_in_order(tmp_path, pdf_bytes):
    path = tmp_path / "x.pdf"
    path.write_bytes(pdf_bytes)
    pages = extract_pages(path)
    assert len(pages) == 3
    assert "Abstract" in pages[0] and "1 Introduction" in pages[1] and "Zebra" in pages[2]


def test_version_stamp_match_missing_and_mismatch(paper):
    assert check_version_stamp(paper, "arXiv:1111.11111v2 [cs.LG] 1 Jan 2020") is True
    assert check_version_stamp(paper, "arXiv : 1111.11111 v2") is True
    assert check_version_stamp(paper, "no stamp here") is None
    assert check_version_stamp(paper, "cites arXiv:9999.99999v1 only") is None
    with pytest.raises(IngestError, match="stamped"):
        check_version_stamp(paper, "arXiv:1111.11111v5 [cs.LG]")


def test_download_saves_valid_pdf_from_exact_url(tmp_path, paper, pdf_bytes):
    seen = []

    def handler(request):
        seen.append(str(request.url))
        return httpx.Response(200, content=pdf_bytes)

    path = download_pdf(paper, tmp_path, client_for(handler), sleep=NO_SLEEP)
    assert seen == ["https://arxiv.org/pdf/1111.11111v2"]
    assert path.name == "1111.11111v2.pdf" and path.read_bytes() == pdf_bytes
    assert not list(tmp_path.glob("*.part"))


def test_download_rejects_html_pretending_to_be_a_pdf(tmp_path, paper):
    client = client_for(lambda r: httpx.Response(200, content=b"<html>captcha</html>"))
    with pytest.raises(IngestError, match="3 attempts"):
        download_pdf(paper, tmp_path, client, sleep=NO_SLEEP)
    assert not list(tmp_path.glob("*.pdf"))


def test_download_404_fails_fast_without_retrying(tmp_path, paper):
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(404)

    with pytest.raises(IngestError, match="404"):
        download_pdf(paper, tmp_path, client_for(handler), sleep=NO_SLEEP)
    assert len(calls) == 1


def test_download_retries_transient_errors_then_succeeds(tmp_path, paper, pdf_bytes):
    answers = iter([httpx.Response(503), httpx.Response(200, content=pdf_bytes)])
    path = download_pdf(paper, tmp_path, client_for(lambda r: next(answers)), sleep=NO_SLEEP)
    assert path.exists()


def test_existing_pdf_is_reused_and_force_refetches(tmp_path, paper, pdf_bytes):
    dest = tmp_path / "1111.11111v2.pdf"
    dest.write_bytes(b"%PDF-old")

    def forbidden(request):
        raise AssertionError("network must not be used")

    assert download_pdf(paper, tmp_path, client_for(forbidden)).read_bytes() == b"%PDF-old"
    download_pdf(paper, tmp_path, client_for(lambda r: httpx.Response(200, content=pdf_bytes)), force=True)
    assert dest.read_bytes() == pdf_bytes


def test_process_paper_writes_pages_json_and_manifest(settings, paper, pdf_bytes):
    client = client_for(lambda r: httpx.Response(200, content=pdf_bytes))
    entry = process_paper(paper, settings, client, sleep=NO_SLEEP)
    assert entry["n_pages"] == 3 and entry["version_stamp_confirmed"] is True
    assert len(str(entry["sha256"])) == 64
    data = json.loads((settings.processed_dir / "alpha.json").read_text(encoding="utf-8"))
    assert [p["page"] for p in data["pages"]] == [1, 2, 3]
    assert data["arxiv_id"] == "1111.11111" and data["version"] == 2
    update_manifest(settings, [entry])
    update_manifest(settings, [{"key": "other", "n_pages": 1}])
    manifest = json.loads(settings.manifest_path.read_text(encoding="utf-8"))
    assert set(manifest) == {"alpha", "other"}
