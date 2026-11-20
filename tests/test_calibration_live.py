"""Phase 2b: background calibration, live progress, resume, threshold override."""

from __future__ import annotations

import threading
import time
from dataclasses import replace

from fastapi.testclient import TestClient

from agentic_cv.agent import RulesTaskPlanner
from agentic_cv.agent_store import AgentSessionStore
from agentic_cv.application import LocalAppConfig, create_local_app
from agentic_cv.calibration_validation import ValidatorSelection, any_prediction_selection

from test_agent_chat_e2e import FakeSam3, NoPredictionSam3, headers, upload_image, wait_for_session


def make_app(tmp_path, *, sam3=None, validator_factory=None, start_workers=True):
    return create_local_app(
        LocalAppConfig(
            data_dir=tmp_path / "data",
            sam3_provider=sam3 or FakeSam3(),
            agent_planner=RulesTaskPlanner(),
            calibration_timeout_sec=60,
            calibration_validator_factory=validator_factory
            or (lambda tenant_id, session_id: any_prediction_selection()),
            start_workers=start_workers,
            worker_poll_interval=0.05,
        )
    )


def plan(client, message="Create a segmentation dataset of workers."):
    asset_id = upload_image(client)
    response = client.post(
        "/v1/agent/sessions", headers=headers(), json={"asset_ids": [asset_id], "message": message}
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_confirm_returns_immediately_and_publishes_progress(tmp_path):
    gate = threading.Event()
    seen = []

    def validator(sample, predictions):
        seen.append(len(predictions))
        if len(seen) > 1:  # hold the second threshold until the test has looked
            assert gate.wait(timeout=10)
        return bool(predictions)

    factory = lambda tenant_id, session_id: ValidatorSelection(validator=validator, mode="custom")
    with TestClient(make_app(tmp_path, validator_factory=factory), raise_server_exceptions=False) as client:
        session = plan(client)
        started = time.monotonic()
        confirmed = client.post(f"/v1/agent/sessions/{session['session_id']}/confirm", headers=headers(), json={})
        assert confirmed.status_code == 200, confirmed.text
        assert confirmed.json()["status"] == "calibrating"
        assert time.monotonic() - started < 2

        live = wait_for_session(client, session["session_id"], statuses=("calibrating",))
        deadline = time.monotonic() + 10
        while not (live.get("calibration") or {}).get("progress") and time.monotonic() < deadline:
            time.sleep(0.05)
            live = client.get(f"/v1/agent/sessions/{session['session_id']}", headers=headers()).json()
        progress = live["calibration"]["progress"]
        assert live["status"] == "calibrating"
        assert live["calibration"]["status"] == "running"
        assert progress[0]["name"] == "workers"
        assert progress[0]["threshold"] == 0.3
        gate.set()

        done = wait_for_session(client, session["session_id"], statuses=("complete",))
        assert done["calibration"]["status"] == "calibrated"


def test_interrupted_calibration_resumes_after_restart(tmp_path):
    with TestClient(make_app(tmp_path, start_workers=False), raise_server_exceptions=False) as client:
        session = plan(client)
    # Simulate a crash mid-calibration: the session is left "calibrating".
    store = AgentSessionStore(str(tmp_path / "data" / "metadata" / "agent-sessions.sqlite3"))
    from uuid import UUID

    record = store.get_any(UUID(session["session_id"]))
    store.save(
        replace(
            record,
            status="calibrating",
            spec={**record.spec, "calibration_request": {"threshold": None, "thresholds": [0.3, 0.6], "minimum_agreement": 0.6, "class_thresholds": None}},
        )
    )
    store.close()

    with TestClient(make_app(tmp_path), raise_server_exceptions=False) as client:
        done = wait_for_session(client, session["session_id"], statuses=("complete", "failed"))
        assert done["status"] == "complete", done["messages"][-1]
        assert [attempt["threshold"] for attempt in done["calibration"]["attempts"]] == [0.3, 0.6]


def test_user_thresholds_per_class_after_calibration_needs_attention(tmp_path):
    with TestClient(make_app(tmp_path, sam3=NoPredictionSam3()), raise_server_exceptions=False) as client:
        session = plan(client, "detect helmets and vests")
        failed = client.post(
            f"/v1/agent/sessions/{session['session_id']}/confirm", headers=headers(), json={"wait": True}
        ).json()
        assert failed["status"] == "needs_attention"

        missing = client.post(
            f"/v1/agent/sessions/{session['session_id']}/confirm",
            headers=headers(),
            json={"class_thresholds": {"helmets": 0.4}, "wait": True},
        )
        assert missing.status_code == 422

        chosen = client.post(
            f"/v1/agent/sessions/{session['session_id']}/confirm",
            headers=headers(),
            json={"class_thresholds": {"helmets": 0.4, "vests": 0.55}, "wait": True},
        ).json()
        assert chosen["status"] == "labeling"
        assert chosen["calibration"]["status"] == "manual"
        assert chosen["calibration"]["class_thresholds"] == {"helmets": 0.4, "vests": 0.55}
        assert chosen["calibration"]["previous"]["status"] == "no_acceptable_threshold"
        assert "Using your thresholds: helmets 0.40, vests 0.55" in chosen["messages"][-1]["content"]
        wait_for_session(client, session["session_id"], statuses=("complete",))
