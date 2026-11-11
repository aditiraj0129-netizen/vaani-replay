"""Token budget guard: the ONE gate every real API call must pass through (nothing here calls any API).

    guard = TokenBudget()                       # caps A 30,000 / B 41,000 / C 37,000 and a global ceiling of 110,000 tokens
    r = guard.reserve("B", worst_case_tokens)   # BEFORE sending: raises BudgetExceeded if it could break a cap or the ceiling
    ...send the request...
    guard.settle(r, actual_tokens)              # AFTER the answer: replaces the reservation by the real usage
    guard.settle(r, None)                       # usage unknown (timeout, error, 429 when rejected_counts): counts as FULLY spent

Rules (all enforced under one lock, so threads and asyncio tasks cannot race past a limit):
  * spent + reserved + the new reservation must stay <= the stage cap AND <= the global ceiling, otherwise BudgetExceeded and nothing is sent;
  * a retry is a new call, so it reserves again (retries cannot sneak past the limits);
  * reservations count as spent until settled, so concurrent requests are all covered;
  * if actual usage is above what was reserved, the overrun is recorded and the guard HALTS (every later reserve fails);
  * the ledger (JSON, written atomically) is reloaded on restart; a reservation left open by a crash counts as fully spent;
  * prior_spend lets a later run start from tokens already used in an earlier run.
The stage caps add up to 108,000, below the ceiling (110,000); the ceiling is a second, independent safety net that binds even if caps are changed.
"""
from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

GLOBAL_CEILING = 110_000
STAGE_CAPS: Dict[str, int] = {"A": 30_000, "B": 41_000, "C": 37_000}


class BudgetExceeded(RuntimeError):
    """Raised BEFORE a request is sent when it could break a stage cap or the global ceiling (or after an overrun halted the guard)."""


@dataclass(frozen=True)
class Reservation:
    id: int
    stage: str
    tokens: int


class TokenBudget:
    def __init__(self, caps: Optional[Dict[str, int]] = None, ceiling: int = GLOBAL_CEILING, prior_spend: int = 0,
                 ledger_path: Optional[Path] = None, rejected_counts: bool = True) -> None:
        self.caps = dict(STAGE_CAPS if caps is None else caps)
        self.ceiling = int(ceiling)
        self.rejected_counts = rejected_counts   # a rejected (429) attempt is counted as spent unless told otherwise
        self.ledger_path = Path(ledger_path) if ledger_path else None
        self._lock = threading.Lock()
        self._spent: Dict[str, int] = {s: 0 for s in self.caps}
        self._open: Dict[int, Reservation] = {}
        self._next = 1
        self._prior = int(prior_spend)
        self.halted = False
        self.halt_reason = ""
        self.overruns = []
        if self._prior > self.ceiling:
            raise BudgetExceeded(f"prior spend {self._prior} is already above the ceiling {self.ceiling}")
        if self.ledger_path and self.ledger_path.exists():
            self._load()

    # ------------------------------------------------------------------ views
    def spent(self, stage: Optional[str] = None) -> int:
        with self._lock:
            return self._spent_locked(stage)

    def committed(self, stage: Optional[str] = None) -> int:
        """spent + still-open reservations (what the limits are checked against)."""
        with self._lock:
            return self._committed_locked(stage)

    def remaining(self, stage: Optional[str] = None) -> int:
        with self._lock:
            if stage is None:
                return self.ceiling - self._committed_locked()
            return min(self.caps[stage] - self._committed_locked(stage), self.ceiling - self._committed_locked())

    def _spent_locked(self, stage: Optional[str]) -> int:
        return sum(self._spent.values()) + self._prior if stage is None else self._spent[stage]

    def _committed_locked(self, stage: Optional[str] = None) -> int:
        if stage is None:
            return sum(self._spent.values()) + self._prior + sum(r.tokens for r in self._open.values())
        return self._spent[stage] + sum(r.tokens for r in self._open.values() if r.stage == stage)

    # ------------------------------------------------------------------ the gate
    def reserve(self, stage: str, tokens: int) -> Reservation:
        if stage not in self.caps:
            raise KeyError(f"unknown stage {stage!r}")
        tokens = int(tokens)
        if tokens < 0:
            raise ValueError("tokens must be >= 0")
        with self._lock:
            if self.halted:
                raise BudgetExceeded(f"budget guard is halted: {self.halt_reason}")
            if self._committed_locked(stage) + tokens > self.caps[stage]:
                raise BudgetExceeded(f"stage {stage}: {self._committed_locked(stage)} committed + {tokens} would pass the cap {self.caps[stage]}")
            if self._committed_locked() + tokens > self.ceiling:
                raise BudgetExceeded(f"global: {self._committed_locked()} committed + {tokens} would pass the ceiling {self.ceiling}")
            r = Reservation(self._next, stage, tokens)
            self._next += 1
            self._open[r.id] = r
            self._save_locked()
            return r

    def settle(self, reservation: Reservation, actual: Optional[int]) -> int:
        """Replace the reservation by the real usage. actual=None (unknown) counts the whole reservation as spent. Returns tokens booked."""
        with self._lock:
            if reservation.id not in self._open:
                raise KeyError(f"reservation {reservation.id} is not open (settled twice?)")
            del self._open[reservation.id]
            booked = reservation.tokens if actual is None else int(actual)
            self._spent[reservation.stage] += booked
            if booked > reservation.tokens:
                self.overruns.append((reservation.stage, reservation.tokens, booked))
                self._halt_locked(f"stage {reservation.stage}: actual {booked} above reserved {reservation.tokens}")
            if self._spent[reservation.stage] > self.caps[reservation.stage] or self._committed_locked() > self.ceiling:
                self._halt_locked("a cap or the ceiling was passed by actual usage")
            self._save_locked()
            return booked

    def settle_rejected(self, reservation: Reservation) -> int:
        """A request the provider refused (HTTP 429). Counted as fully spent by default because that is the safe reading."""
        return self.settle(reservation, reservation.tokens if self.rejected_counts else 0)

    def halt(self, reason: str) -> None:
        with self._lock:
            self._halt_locked(reason)
            self._save_locked()

    def _halt_locked(self, reason: str) -> None:
        if not self.halted:
            self.halted, self.halt_reason = True, reason

    # ------------------------------------------------------------------ ledger
    def snapshot(self) -> Dict:
        return {"caps": self.caps, "ceiling": self.ceiling, "prior": self._prior, "spent": dict(self._spent),
                "open": [{"id": r.id, "stage": r.stage, "tokens": r.tokens} for r in self._open.values()],
                "next": self._next, "halted": self.halted, "halt_reason": self.halt_reason, "overruns": self.overruns}

    def _save_locked(self) -> None:
        if not self.ledger_path:
            return
        tmp = self.ledger_path.with_suffix(self.ledger_path.suffix + ".tmp")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(self.snapshot(), indent=2), encoding="utf-8")
        os.replace(tmp, self.ledger_path)   # atomic: a crash leaves the old or the new file, never half of one

    def _load(self) -> None:
        d = json.loads(self.ledger_path.read_text(encoding="utf-8"))
        for s, v in d.get("spent", {}).items():
            if s in self._spent:
                self._spent[s] += int(v)
        for r in d.get("open", []):   # open at crash time: the request may have been sent, so it counts as fully spent
            if r["stage"] in self._spent:
                self._spent[r["stage"]] += int(r["tokens"])
        self._prior += int(d.get("prior", 0))
        self._next = int(d.get("next", 1))
        if d.get("halted"):
            self.halted, self.halt_reason = True, d.get("halt_reason", "halted in an earlier run")
        if self._committed_locked() > self.ceiling:
            self._halt_locked("ledger already above the ceiling")


def upper_bound_tokens(prompt_tokens: float, completion_cap: int = 250, prompt_factor: float = 1.15) -> int:
    """Worst-case tokens for ONE request: the prompt (with a safety factor) plus the largest completion we allow. Use it for reserve()."""
    return int(round(prompt_tokens * prompt_factor)) + int(completion_cap)
