"""
Generation Lease: prevent stale replies.

Problem: while the model is generating a reply, the user may send a new message.
By the time the reply is ready, it is already answering an outdated question.

Solution:
  1. Every incoming message bumps a per-conversation counter (epoch).
  2. A generation records the epoch when it starts.
  3. While streaming and right before sending, it checks the epoch.
  4. If the epoch moved, abort — a newer message exists, don't send a stale reply.

Reference: Ecbot (github.com/blackmouse572/ecbot)
"""
import threading
import time
from typing import Optional


class GenerationLease:
    """Per-conversation generation lease with monotonic epoch counter."""

    def __init__(self, ttl_seconds: int = 3600):
        self._epochs: dict[str, int] = {}
        self._timestamps: dict[str, float] = {}
        self._ttl = ttl_seconds
        self._lock = threading.Lock()

    def bump(self, conversation_id: str) -> int:
        """
        Increment epoch on a new incoming message.
        Returns the new epoch value.
        """
        with self._lock:
            self._cleanup_locked()
            epoch = self._epochs.get(conversation_id, 0) + 1
            self._epochs[conversation_id] = epoch
            self._timestamps[conversation_id] = time.time()
            return epoch

    def current(self, conversation_id: str) -> int:
        """Get the current epoch for a conversation."""
        with self._lock:
            return self._epochs.get(conversation_id, 0)

    def is_current(self, conversation_id: str, epoch: int) -> bool:
        """
        Check whether a generation is still current.
        Returns False if a newer message has arrived since the generation started.
        """
        return self.current(conversation_id) == epoch

    def acquire(self, conversation_id: str) -> "LeaseHandle":
        """
        Acquire a generation lease at the current epoch.
        Use handle.is_current() to check freshness during generation.
        """
        epoch = self.current(conversation_id)
        return LeaseHandle(self, conversation_id, epoch)

    def _cleanup_locked(self):
        """Expire old entries."""
        now = time.time()
        expired = [
            cid for cid, ts in self._timestamps.items()
            if now - ts > self._ttl
        ]
        for cid in expired:
            self._epochs.pop(cid, None)
            self._timestamps.pop(cid, None)


class LeaseHandle:
    """A generation lease captured at a specific epoch."""

    def __init__(self, lease: GenerationLease, conversation_id: str, epoch: int):
        self._lease = lease
        self.conversation_id = conversation_id
        self.epoch = epoch

    def is_current(self) -> bool:
        """True if no newer message has arrived since this lease was acquired."""
        return self._lease.is_current(self.conversation_id, self.epoch)

    def check_or_abort(self) -> None:
        """Raise StaleGenerationError if the lease is no longer current."""
        if not self.is_current():
            raise StaleGenerationError(
                f"generation stale: epoch {self.epoch} "
                f"!= current {self._lease.current(self.conversation_id)}"
            )


class StaleGenerationError(Exception):
    """Raised when a generation is made obsolete by a newer message."""
    pass
