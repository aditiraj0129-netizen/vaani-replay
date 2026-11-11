from app.check_groq import pick_headers


def test_pick_headers_keeps_only_rate_limit_headers_and_never_auth():
    headers = {
        "Authorization": "Bearer gsk_SECRET",
        "X-RateLimit-Remaining-Tokens": "100",
        "Retry-After": "7",
        "Content-Type": "application/json",
    }
    assert pick_headers(headers) == {"x-ratelimit-remaining-tokens": "100", "retry-after": "7"}
