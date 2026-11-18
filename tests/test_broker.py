import json
from uuid import uuid4

from agentic_cv.broker import RabbitMqConfig, RabbitMqPublisher
from agentic_cv.contracts import EventEnvelope
from agentic_cv.outbox import Outbox


class FakeChannel:
    def __init__(self):
        self.declarations = []
        self.messages = []
        self.confirmed = False

    def exchange_declare(self, **kwargs):
        self.declarations.append(kwargs)

    def confirm_delivery(self):
        self.confirmed = True

    def basic_publish(self, **kwargs):
        self.messages.append(kwargs)


class FakeConnection:
    def __init__(self):
        self.channel_value = FakeChannel()
        self.is_closed = False
        self.closed = False

    def channel(self):
        return self.channel_value

    def close(self):
        self.closed = True
        self.is_closed = True


def test_rabbit_publisher_declares_durable_exchange_and_persistent_message(tmp_path):
    outbox = Outbox(str(tmp_path / "events.sqlite3"))
    event = outbox.enqueue(EventEnvelope(
        event_type="DatasetExportCompleted", tenant_id=uuid4(), aggregate_type="job",
        aggregate_id=uuid4(), payload={"format": "coco"}, idempotency_key="event-1",
    ))
    connection = FakeConnection()
    publisher = RabbitMqPublisher(
        RabbitMqConfig(exchange="events"), connection_factory=lambda config: connection
    )

    publisher.publish(event)

    assert connection.channel_value.declarations == [{
        "exchange": "events", "exchange_type": "topic", "durable": True,
    }]
    assert connection.channel_value.confirmed
    message = connection.channel_value.messages[0]
    assert message["exchange"] == "events"
    assert message["routing_key"] == "DatasetExportCompleted"
    assert json.loads(message["body"])["event_id"] == str(event.event_id)
    assert message["properties"].delivery_mode == 2
    publisher.close()
    assert connection.closed
    outbox.close()


def test_rabbit_config_reads_environment(monkeypatch):
    monkeypatch.setenv("AGENTIC_CV_RABBITMQ_HOST", "rabbit")
    monkeypatch.setenv("AGENTIC_CV_RABBITMQ_PORT", "5673")
    monkeypatch.setenv("AGENTIC_CV_RABBITMQ_EXCHANGE", "custom.events")

    config = RabbitMqConfig.from_environment()

    assert config.host == "rabbit"
    assert config.port == 5673
    assert config.exchange == "custom.events"
