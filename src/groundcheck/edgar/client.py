"""HTTP client for SEC EDGAR that follows SEC's fair-access rules.

Every request:
- sends the User-Agent from SEC_USER_AGENT ("Name email"), which SEC requires;
- waits on the shared RateLimiter (at most 5 requests per second by default);
- retries on 429 (too many requests), 5xx (server errors) and network errors (timeouts,
  dropped connections) with exponential backoff, honoring the server's Retry-After
  header when it sends one.

Other errors (e.g. 404 for a wrong accession number) are raised immediately, because
retrying cannot fix them.
"""

import logging
import time
from collections.abc import Callable

import requests

from groundcheck.edgar.ratelimit import RateLimiter

logger = logging.getLogger(__name__)

RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class SecRequestError(Exception):
    """Raised when SEC returns an error we cannot (or should not) retry."""


class SecClient:
    def __init__(
        self,
        user_agent: str,
        limiter: RateLimiter | None = None,
        session: requests.Session | None = None,
        max_retries: int = 4,
        backoff_seconds: float = 2.0,
        timeout_seconds: float = 30.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.limiter = limiter or RateLimiter(max_per_second=5.0)
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"})
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self.timeout_seconds = timeout_seconds
        self._sleep = sleep

    def get(self, url: str) -> bytes:
        """GET a URL and return the response body, retrying transient failures."""
        for attempt in range(self.max_retries + 1):
            self.limiter.wait()
            try:
                response = self.session.get(url, timeout=self.timeout_seconds)
            except (requests.ConnectionError, requests.Timeout) as exc:
                if attempt == self.max_retries:
                    raise SecRequestError(
                        f"GET {url} failed: {exc}. Check your network connection; some "
                        "networks (e.g. sandboxed cloud environments) block sec.gov."
                    ) from exc
                delay = self.backoff_seconds * (2**attempt)
                logger.warning("GET %s failed (%s); retrying in %.1fs", url, exc, delay)
                self._sleep(delay)
                continue
            if response.status_code == 200:
                return response.content
            if response.status_code not in RETRYABLE_STATUS or attempt == self.max_retries:
                raise SecRequestError(f"GET {url} failed with HTTP {response.status_code}")
            delay = self._retry_delay(response, attempt)
            logger.warning(
                "GET %s returned HTTP %s; retrying in %.1fs", url, response.status_code, delay
            )
            self._sleep(delay)
        raise AssertionError("unreachable")  # the loop always returns or raises

    def _retry_delay(self, response: requests.Response, attempt: int) -> float:
        retry_after = response.headers.get("Retry-After")
        if retry_after is not None and retry_after.isdigit():
            return float(retry_after)
        return self.backoff_seconds * (2**attempt)  # 2s, 4s, 8s, 16s
