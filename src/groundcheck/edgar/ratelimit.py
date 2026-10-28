"""A minimal rate limiter: never start two requests closer together than 1/rate seconds.

SEC allows up to 10 requests per second per client. We default to 5 to leave margin.

Why a "minimum interval" limiter instead of a token bucket? A token bucket allows short
bursts (e.g. 5 requests at once, then a pause). We download one file at a time, so
bursts buy nothing, and an even spacing is simpler to reason about: if the rate is 5/s,
consecutive requests are always at least 0.2 s apart.

The clock and sleep functions are injectable so tests can check the timing instantly
with a fake clock instead of actually sleeping.
"""

import time
from collections.abc import Callable


class RateLimiter:
    def __init__(
        self,
        max_per_second: float = 5.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if max_per_second <= 0:
            raise ValueError("max_per_second must be positive")
        self.min_interval = 1.0 / max_per_second
        self._clock = clock
        self._sleep = sleep
        self._last_start: float | None = None

    def wait(self) -> None:
        """Block until it is OK to start the next request, then record its start."""
        now = self._clock()
        if self._last_start is not None:
            remaining = self._last_start + self.min_interval - now
            if remaining > 0:
                self._sleep(remaining)
                now = self._clock()
        self._last_start = now
