"""Retry-aware delivery loop for the durable local outbox."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

from .outbox import Outbox, OutboxRecord


class EventPublisher(Protocol):
    def publish(self, event: OutboxRecord) -> None: ...


@dataclass(frozen=True, kw_only=True)
class OutboxDispatcher:
    outbox: Outbox
    publisher: EventPublisher

    def dispatch_once(
        self, *, limit: int = 100, retry_after_seconds: int = 1
    ) -> tuple[UUID, ...]:
        if retry_after_seconds < 0:
            raise ValueError("retry_after_seconds must not be negative")
        processed: list[UUID] = []
        for event in self.outbox.pending(limit=limit):
            processed.append(event.event_id)
            try:
                self.publisher.publish(event)
            except Exception as exc:
                self.outbox.mark_failed(
                    event.event_id, str(exc), retry_after_seconds=retry_after_seconds
                )
            else:
                self.outbox.mark_published(event.event_id)
        return tuple(processed)