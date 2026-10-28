import pytest

from groundcheck.edgar.client import SecClient, SecRequestError
from groundcheck.edgar.ratelimit import RateLimiter


class FakeClock:
    """A clock that only moves when sleep() is called, so tests run instantly."""

    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


class FakeResponse:
    def __init__(self, status_code, content=b"", headers=None):
        self.status_code = status_code
        self.content = content
        self.headers = headers or {}


class FakeSession:
    """Stands in for requests.Session: returns queued responses, records requests."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.headers = {}
        self.urls = []

    def get(self, url, timeout):
        self.urls.append(url)
        return self.responses.pop(0)


# --- RateLimiter -----------------------------------------------------------------


def test_rate_limiter_spaces_requests():
    clock = FakeClock()
    limiter = RateLimiter(max_per_second=5, clock=clock, sleep=clock.sleep)
    start_times = []
    for _ in range(4):
        limiter.wait()
        start_times.append(clock.now)
    gaps = [b - a for a, b in zip(start_times, start_times[1:], strict=False)]
    assert all(gap >= 0.2 - 1e-9 for gap in gaps)


def test_rate_limiter_does_not_sleep_when_requests_are_already_slow():
    clock = FakeClock()
    limiter = RateLimiter(max_per_second=5, clock=clock, sleep=clock.sleep)
    limiter.wait()
    clock.now += 1.0  # the caller spent a second doing other work
    limiter.wait()
    assert clock.sleeps == []


def test_rate_limiter_never_exceeds_five_per_second():
    clock = FakeClock()
    limiter = RateLimiter(max_per_second=5, clock=clock, sleep=clock.sleep)
    for _ in range(50):
        limiter.wait()
    # 50 requests at <=5/s need at least 49 * 0.2 = 9.8 seconds.
    assert clock.now >= 9.8 - 1e-9


def test_rate_limiter_rejects_nonpositive_rate():
    with pytest.raises(ValueError):
        RateLimiter(max_per_second=0)


# --- SecClient --------------------------------------------------------------------


def make_client(responses, **kwargs):
    clock = FakeClock()
    session = FakeSession(responses)
    client = SecClient(
        "Jane Doe jane@example.com",
        limiter=RateLimiter(max_per_second=5, clock=clock, sleep=clock.sleep),
        session=session,
        sleep=clock.sleep,
        **kwargs,
    )
    return client, session, clock


def test_client_sends_user_agent():
    client, session, _ = make_client([FakeResponse(200, b"ok")])
    assert client.get("https://example.test/a") == b"ok"
    assert session.headers["User-Agent"] == "Jane Doe jane@example.com"


def test_client_retries_server_errors_with_backoff():
    client, session, clock = make_client(
        [FakeResponse(503), FakeResponse(500), FakeResponse(200, b"ok")]
    )
    assert client.get("https://example.test/a") == b"ok"
    assert len(session.urls) == 3
    assert clock.sleeps[:1] == [2.0]  # first backoff
    assert 4.0 in clock.sleeps  # second backoff doubles


def test_client_honors_retry_after():
    client, _, clock = make_client(
        [FakeResponse(429, headers={"Retry-After": "7"}), FakeResponse(200)]
    )
    client.get("https://example.test/a")
    assert 7.0 in clock.sleeps


def test_client_does_not_retry_404():
    client, session, _ = make_client([FakeResponse(404)])
    with pytest.raises(SecRequestError, match="404"):
        client.get("https://example.test/missing")
    assert len(session.urls) == 1


def test_client_gives_up_after_max_retries():
    client, session, _ = make_client([FakeResponse(503)] * 3, max_retries=2)
    with pytest.raises(SecRequestError, match="503"):
        client.get("https://example.test/a")
    assert len(session.urls) == 3
