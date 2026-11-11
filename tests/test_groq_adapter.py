"""Adapter tests use a fake HTTP server (httpx.MockTransport). They prove our parsing and
failure handling; they do NOT prove how the real Groq service behaves."""
import asyncio
import json

import httpx
import pytest

from app.config import Settings
from app.providers.base import LLMRequest
from app.providers.groq_adapter import GroqProvider, classify_limit, parse_duration, parse_ratelimit
from app.raw_log import RawLog, result_record

SECRET = "gsk_TEST_SECRET_123"
SETTINGS = Settings(groq_api_key=SECRET, groq_model="openai/gpt-oss-20b")
REQ = LLMRequest(messages=[{"role": "user", "content": "hi"}], max_tokens=300, timeout_s=5.0)
USAGE = {"queue_time": 0.3152, "prompt_tokens": 78, "completion_tokens": 5, "total_tokens": 83,
         "total_time": 0.0088, "completion_tokens_details": {"reasoning_tokens": 3}}
HEADERS = {"content-type": "text/event-stream", "x-ratelimit-limit-requests": "1000",
           "x-ratelimit-remaining-requests": "999", "x-ratelimit-limit-tokens": "8000",
           "x-ratelimit-remaining-tokens": "7917", "x-ratelimit-reset-requests": "1m26.4s",
           "x-ratelimit-reset-tokens": "622ms", "set-cookie": "should-not-be-kept=1"}


def chunk(content=None, finish=None, usage=None):
    if usage is not None:
        return {"choices": [], "x_groq": {"usage": usage}}
    delta = {} if content is None else {"content": content}
    return {"choices": [{"delta": delta, "finish_reason": finish}]}


def sse(obj):
    return b"data: " + json.dumps(obj).encode() + b"\n\n"


DONE = b"data: [DONE]\n\n"


class Script(httpx.AsyncByteStream):
    """Sends `parts` with a delay (seconds) before each one; optionally ends by raising `boom`."""

    def __init__(self, parts, delays=None, boom=None):
        self.parts, self.delays, self.boom = parts, delays or [0] * len(parts), boom

    async def __aiter__(self):
        for part, delay in zip(self.parts, self.delays):
            await asyncio.sleep(delay)
            yield part
        if self.boom:
            raise self.boom


def stream_response(parts, delays=None, boom=None, status=200, headers=HEADERS):
    return httpx.Response(status, headers=headers, stream=Script(parts, delays, boom))


async def call(handler, req=REQ, on_token=None, **kw):
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="https://api.groq.com/openai/v1")
    provider = GroqProvider(SETTINGS, client=client, **kw)
    try:
        return await provider.generate(req, on_token)
    finally:
        await client.aclose()


def run(coro):
    return asyncio.run(coro)


def good_stream():
    return [sse(chunk("")), sse(chunk("Hel")), sse(chunk("lo")), sse(chunk(None, "stop")), sse(chunk(usage=USAGE)), DONE]


# ---------- pure helpers ----------

@pytest.mark.parametrize("text,expected", [
    ("1m26.4s", 86.4), ("622ms", 0.622), ("7.66s", 7.66), ("2m59.56s", 179.56), ("1h2m3s", 3723.0),
    ("abc", None), ("", None), ("5", None), ("1x", None),
])
def test_parse_duration(text, expected):
    got = parse_duration(text)
    assert (got is None and expected is None) or got == pytest.approx(expected)


def test_parse_ratelimit_on_the_headers_from_the_real_probe():
    info = parse_ratelimit({
        "x-ratelimit-limit-requests": "1000", "x-ratelimit-limit-tokens": "8000",
        "x-ratelimit-remaining-requests": "999", "x-ratelimit-remaining-tokens": "7917",
        "x-ratelimit-reset-requests": "1m26.4s", "x-ratelimit-reset-tokens": "622ms",
    })
    assert (info.limit_requests, info.remaining_requests) == (1000, 999)
    assert (info.limit_tokens, info.remaining_tokens) == (8000, 7917)
    assert info.reset_requests_s == pytest.approx(86.4) and info.reset_tokens_s == pytest.approx(0.622)
    assert info.retry_after_s is None


@pytest.mark.parametrize("message,kind", [
    ("Rate limit reached ... on tokens per minute (TPM): Limit 8000", "tpm"),
    ("... on requests per minute (RPM): Limit 30", "rpm"),
    ("... on requests per day (RPD): Limit 1000", "rpd"),
    ("... on tokens per day (TPD): Limit 200000", "tpd"),
    ("something else entirely", "unknown"),
])
def test_classify_limit_wording_is_an_assumption_until_a_real_429(message, kind):
    assert classify_limit(message) == kind


# ---------- streaming success ----------

def test_success_measures_ttft_separately_from_total_and_reads_usage():
    seen = []
    result = run(call(
        lambda r: stream_response(good_stream(), delays=[0, 0.06, 0.06, 0, 0, 0]),
        on_token=seen.append,
    ))
    assert result.ok and result.text == "Hello" and result.partial_text == ""
    assert seen == ["Hel", "lo"]
    assert result.first_chunk_ms < 40  # the empty role chunk arrives immediately
    assert result.first_token_ms >= 55  # first ANSWER text comes later
    assert result.total_ms >= result.first_token_ms + 55
    assert (result.prompt_tokens, result.completion_tokens, result.reasoning_tokens) == (78, 5, 3)
    assert result.server_queue_s == pytest.approx(0.3152) and result.server_total_s == pytest.approx(0.0088)
    assert result.ratelimit.remaining_tokens == 7917 and result.ratelimit.reset_requests_s == pytest.approx(86.4)


def test_request_body_and_secret_handling():
    captured = {}

    def handler(request):
        captured["body"] = json.loads(request.content)
        captured["auth"] = request.headers["authorization"]
        captured["url"] = str(request.url)
        return stream_response(good_stream())

    result = run(call(handler))
    body = captured["body"]
    assert captured["url"] == "https://api.groq.com/openai/v1/chat/completions"
    assert body["model"] == "openai/gpt-oss-20b" and body["stream"] is True
    assert body["stream_options"] == {"include_usage": True}
    assert body["reasoning_effort"] == "low" and body["include_reasoning"] is False
    assert body["max_completion_tokens"] == 300 and "max_tokens" not in body
    assert captured["auth"] == f"Bearer {SECRET}"
    assert SECRET not in repr(result)
    assert set(result.raw_headers) <= {k for k in HEADERS if k.startswith("x-ratelimit")}  # only limit headers kept


def test_usage_missing_leaves_tokens_unknown_instead_of_inventing_them():
    parts = [sse(chunk("Hi")), sse(chunk(None, "stop")), DONE]
    result = run(call(lambda r: stream_response(parts)))
    assert result.ok and result.prompt_tokens is None and result.completion_tokens is None


def test_top_level_usage_field_is_also_understood():
    parts = [sse(chunk("Hi")), sse(chunk(None, "stop")), sse({"choices": [], "usage": USAGE}), DONE]
    assert run(call(lambda r: stream_response(parts))).completion_tokens == 5


# ---------- HTTP errors ----------

def error_response(status, message, headers=None):
    return httpx.Response(status, json={"error": {"message": message}}, headers=headers or {})


def test_429_is_rate_limited_with_retry_after_and_limit_kind():
    result = run(call(lambda r: error_response(
        429, "Rate limit reached on tokens per minute (TPM): Limit 8000",
        {"retry-after": "7", "x-ratelimit-remaining-tokens": "0"})))
    assert result.status == "rate_limited" and result.http_status == 429
    assert result.limit_kind == "tpm" and not result.is_daily_limit
    assert result.ratelimit.retry_after_s == 7.0 and result.ratelimit.remaining_tokens == 0
    assert not result.ok and result.text == ""


def test_daily_limit_is_flagged_so_the_scheduler_will_not_wait_for_it():
    result = run(call(lambda r: error_response(429, "limit on requests per day (RPD): Limit 1000")))
    assert result.limit_kind == "rpd" and result.is_daily_limit


@pytest.mark.parametrize("status,kind", [(503, "server"), (500, "server"), (401, "auth"), (400, "client"), (404, "client")])
def test_other_http_errors_get_a_kind(status, kind):
    result = run(call(lambda r: error_response(status, "boom")))
    assert result.status == "error" and result.error_kind == kind and result.http_status == status


# ---------- unsafe or incomplete streams must never be "ok" ----------

def test_timeout_keeps_partial_text_but_is_not_an_answer():
    parts = [sse(chunk("Hel")), sse(chunk("lo"))]
    req = LLMRequest(messages=REQ.messages, timeout_s=0.25)
    result = run(call(lambda r: stream_response(parts, delays=[0.0, 5.0]), req=req))
    assert result.status == "timeout" and result.error_kind == "timeout"
    assert result.first_token_ms is not None and result.first_token_ms < 150
    assert result.text == "" and result.partial_text == "Hel" and not result.complete and not result.ok
    assert 200 <= result.total_ms < 1500


def test_connection_dropped_mid_stream_is_interrupted_not_ok():
    parts = [sse(chunk("Hel"))]
    result = run(call(lambda r: stream_response(parts, boom=httpx.ReadError("dropped"))))
    assert result.status == "error" and result.error_kind == "stream_interrupted"
    assert result.text == "" and result.partial_text == "Hel" and not result.ok


def test_stream_that_just_ends_without_done_is_not_ok():
    result = run(call(lambda r: stream_response([sse(chunk("Hel")), sse(chunk("lo"))])))
    assert result.error_kind == "stream_interrupted" and result.text == "" and result.partial_text == "Hello"


def test_finish_reason_length_is_truncated_not_ok():
    parts = [sse(chunk("Half an ans")), sse(chunk(None, "length")), DONE]
    result = run(call(lambda r: stream_response(parts)))
    assert result.error_kind == "truncated" and result.text == "" and result.partial_text == "Half an ans"


def test_finished_without_any_answer_text_is_an_error():
    parts = [sse(chunk(None, "stop")), sse(chunk(usage=USAGE)), DONE]
    result = run(call(lambda r: stream_response(parts)))
    assert result.error_kind == "empty_answer" and not result.ok


def test_garbage_chunk_is_reported():
    result = run(call(lambda r: stream_response([b"data: {not json\n\n", DONE])))
    assert result.error_kind == "bad_stream" and not result.ok


def test_network_error_before_any_data():
    def handler(request):
        raise httpx.ConnectError("no route")

    result = run(call(handler))
    assert result.status == "error" and result.error_kind == "network"


def test_provider_needs_key_and_model():
    with pytest.raises(ValueError, match="GROQ_API_KEY"):
        GroqProvider(Settings(groq_model="m"))
    with pytest.raises(ValueError, match="GROQ_MODEL"):
        GroqProvider(Settings(groq_api_key="k"))


# ---------- raw evidence log ----------

def test_raw_log_writes_one_json_line_per_attempt_in_the_real_folder(tmp_path):
    settings = Settings(data_dir=tmp_path)
    log = RawLog.for_real(settings, "unit")
    result = run(call(lambda r: stream_response(good_stream())))
    for label in ("a", "b"):
        log.write(result_record(result, source="real", label=label, model="m", messages=REQ.messages))
    lines = log.path.read_text(encoding="utf-8").splitlines()
    assert log.path == tmp_path / "raw" / "real" / "unit.jsonl" and len(lines) == 2
    first = json.loads(lines[0])
    assert first["source"] == "real" and first["result"]["text"] == "Hello"
    assert first["result"]["raw_headers"]["x-ratelimit-remaining-tokens"] == "7917"
    assert SECRET not in log.path.read_text(encoding="utf-8")


def test_finish_reason_seen_but_no_done_marker_is_still_not_ok():
    parts = [sse(chunk("Hello")), sse(chunk(None, "stop"))]  # connection ends before [DONE]
    result = run(call(lambda r: stream_response(parts)))
    assert result.error_kind == "stream_interrupted" and result.text == "" and not result.ok
