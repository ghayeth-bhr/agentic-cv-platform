"""RabbitMQ publisher for durable outbox integration events."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Callable

from .outbox import OutboxRecord


@dataclass(frozen=True, kw_only=True)
class RabbitMqConfig:
    host: str = "localhost"
    port: int = 5672
    virtual_host: str = "/"
    username: str = "guest"
    password: str = "guest"
    exchange: str = "agentic-cv.events"
    exchange_type: str = "topic"

    @classmethod
    def from_environment(cls) -> "RabbitMqConfig":
        return cls(
            host=os.getenv("AGENTIC_CV_RABBITMQ_HOST", "localhost"),
            port=int(os.getenv("AGENTIC_CV_RABBITMQ_PORT", "5672")),
            virtual_host=os.getenv("AGENTIC_CV_RABBITMQ_VHOST", "/"),
            username=os.getenv("AGENTIC_CV_RABBITMQ_USER", "guest"),
            password=os.getenv("AGENTIC_CV_RABBITMQ_PASSWORD", "guest"),
            exchange=os.getenv("AGENTIC_CV_RABBITMQ_EXCHANGE", "agentic-cv.events"),
        )


class RabbitMqPublisher:
    """Publish durable outbox records to a topic exchange."""

    def __init__(
        self,
        config: RabbitMqConfig | None = None,
        *,
        connection_factory: Callable[[RabbitMqConfig], Any] | None = None,
    ) -> None:
        self.config = config or RabbitMqConfig()
        self._connection_factory = connection_factory or self._default_connection
        self._connection: Any | None = None
        self._channel: Any | None = None

    def publish(self, event: OutboxRecord) -> None:
        channel = self._get_channel()
        body = json.dumps(
            {
                "event_id": str(event.event_id),
                "event_type": event.event_type,
                "tenant_id": str(event.tenant_id),
                "aggregate_type": event.aggregate_type,
                "aggregate_id": str(event.aggregate_id),
                "payload": event.payload,
                "occurred_at": event.occurred_at,
                "idempotency_key": event.idempotency_key,
            },
            sort_keys=True,
        ).encode("utf-8")
        properties = self._basic_properties(event)
        channel.basic_publish(
            exchange=self.config.exchange,
            routing_key=event.event_type,
            body=body,
            properties=properties,
            mandatory=True,
        )

    def close(self) -> None:
        if self._connection is not None and not self._connection.is_closed:
            self._connection.close()
        self._connection = None
        self._channel = None

    def _get_channel(self) -> Any:
        if self._channel is None:
            self._connection = self._connection_factory(self.config)
            self._channel = self._connection.channel()
            self._channel.exchange_declare(
                exchange=self.config.exchange,
                exchange_type=self.config.exchange_type,
                durable=True,
            )
            confirm = getattr(self._channel, "confirm_delivery", None)
            if confirm is not None:
                confirm()
        return self._channel

    @staticmethod
    def _basic_properties(event: OutboxRecord) -> Any:
        import pika

        return pika.BasicProperties(
            content_type="application/json",
            delivery_mode=2,
            message_id=str(event.event_id),
            type=event.event_type,
            headers={"tenant_id": str(event.tenant_id)},
        )

    def _default_connection(self, config: RabbitMqConfig) -> Any:
        import pika

        credentials = pika.PlainCredentials(config.username, config.password)
        parameters = pika.ConnectionParameters(
            host=config.host,
            port=config.port,
            virtual_host=config.virtual_host,
            credentials=credentials,
        )
        return pika.BlockingConnection(parameters)
