"""Local worker that advances agent sessions after labeling and export jobs finish."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from threading import Event
from uuid import UUID

from .agent import AgentService, AgentSessionView

logger = logging.getLogger("agentic_cv.worker")


@dataclass(frozen=True, kw_only=True)
class AgentWorker:
    service: AgentService

    def run_once(self, *, worker_id: str = "agent-worker") -> tuple[UUID, ...]:
        processed: list[UUID] = []
        for session in self.service.open_sessions():
            processed.append(session.session_id)
            try:
                self.service.advance(session.session_id)
            except Exception:
                logger.exception(
                    "agent session advance failed",
                    extra={
                        "worker_id": worker_id,
                        "session_id": str(session.session_id),
                        "status": session.status,
                    },
                )
        return tuple(processed)

    def run_forever(
        self,
        *,
        worker_id: str = "agent-worker",
        poll_interval: float = 1.0,
        stop_event: Event | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        if poll_interval < 0:
            raise ValueError("poll_interval must not be negative")
        stop_event = stop_event or Event()
        logger.info(
            "agent worker loop started",
            extra={"worker_id": worker_id, "poll_interval": poll_interval},
        )
        try:
            while not stop_event.is_set():
                self.run_once(worker_id=worker_id)
                if not stop_event.is_set() and poll_interval:
                    sleep_fn(poll_interval)
        finally:
            logger.info("agent worker loop stopped", extra={"worker_id": worker_id})
