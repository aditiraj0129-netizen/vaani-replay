"""A clock that can run faster than real time, so load tests with minutes of simulated waiting take seconds.

All code that waits (the stand-in service now, the scheduler later) must use clock.sleep() and clock.now(),
never asyncio.sleep()/time directly, or simulated and real time would be mixed up.
speed=1.0 is real time; speed=100 means 100 simulated seconds pass per real second.
Keep speed <= 100: at 1000 one real millisecond of scheduling delay becomes a simulated second and inflates
measured latency (checked: median first token 1,056 ms at 20x, 1,126 ms at 100x, about 2,100 ms at 1000x; real: 1,142 ms).
"""
from __future__ import annotations

import asyncio
import time


class Clock:
    def __init__(self, speed: float = 1.0) -> None:
        if speed <= 0:
            raise ValueError("speed must be positive")
        self.speed = speed
        self._t0 = time.monotonic()

    def now(self) -> float:
        """Simulated seconds since this clock was created."""
        return (time.monotonic() - self._t0) * self.speed

    async def sleep(self, seconds: float) -> None:
        if seconds > 0:
            await asyncio.sleep(seconds / self.speed)
