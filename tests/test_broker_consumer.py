import json

import pytest

from agentic_cv.broker import RabbitMqConfig, RabbitMqConsumer


class FakeChannel:
    def __init__(self):
        self.actions = []
        self.callback = None

    def exchange_declare(self, **kwargs): self.actions.append(("exchange", kwargs))
    def queue_declare(self, **kwargs): self.actions.append(("queue", kwargs))
    def queue_bind(self, **kwargs): self.actions.append(("bind", kwargs))
    def basic_qos(self, **kwargs): self.actions.append(("qos", kwargs))
    def basic_consume(self, **kwargs): self.callback = kwargs["on_message_callback"]
    def start_consuming(self): self.actions.append(("start", {}))
    def stop_consuming(self): self.actions.append(("stop", {}))
    def basic_ack(self, **kwargs): self.actions.append(("ack", kwargs))
    def basic_nack(self, **kwargs): self.actions.append(("nack", kwargs))


class FakeConnection:
    def __init__(self):
        self.channel_value = FakeChannel()
        self.is_closed = False

    def channel(self): return self.channel_value
    def close(self): self.is_closed = True


class Method:
    delivery_tag = 7


def test_consumer_declares_durable_queue_and_acknowledges_valid_messages():
    connection = FakeConnection()
    received = []
    consumer = RabbitMqConsumer(
        config=RabbitMqConfig(exchange="events"), queue="worker", handler=received.append,
        routing_keys=("DatasetExportCompleted",), prefetch_count=2,
        connection_factory=lambda config: connection,
    )

    consumer.run()
    connection.channel_value.callback(
        connection.channel_value, Method(), None, json.dumps({"event_id": "1"}).encode()
    )

    assert received == [{"event_id": "1"}]
    assert ("queue", {"queue": "worker", "durable": True}) in connection.channel_value.actions
    assert ("qos", {"prefetch_count": 2}) in connection.channel_value.actions
    assert ("ack", {"delivery_tag": 7}) in connection.channel_value.actions
    consumer.stop()


def test_consumer_requeues_handler_failures():
    connection = FakeConnection()
    consumer = RabbitMqConsumer(
        config=RabbitMqConfig(), queue="worker", handler=lambda payload: (_ for _ in ()).throw(RuntimeError("fail")),
        connection_factory=lambda config: connection,
    )
    consumer.run()
    connection.channel_value.callback(connection.channel_value, Method(), None, b"{}")

    assert ("nack", {"delivery_tag": 7, "requeue": True}) in connection.channel_value.actions
    consumer.stop()


def test_consumer_validates_queue_and_prefetch():
    with pytest.raises(ValueError, match="queue"):
        RabbitMqConsumer(config=RabbitMqConfig(), queue=" ", handler=lambda payload: None)
    with pytest.raises(ValueError, match="prefetch"):
        RabbitMqConsumer(config=RabbitMqConfig(), queue="worker", handler=lambda payload: None, prefetch_count=0)