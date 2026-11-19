"""Phase 1: slot-filling dialogue, multi-asset/multi-class runs, prompt agent."""

from __future__ import annotations

import io
import zipfile
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from pycocotools.coco import COCO

from agentic_cv.agent import OllamaTaskPlanner, RulesTaskPlanner, _patch_from_model
from agentic_cv.agent_dialogue import (
    ClassSpec,
    DraftPatch,
    RulesDraftExtractor,
    TaskDraft,
    apply_patch,
    mark_asked,
    missing_slots,
    next_question,
    resolve_defaults,
    rules_prompt,
)
from agentic_cv.application import LocalAppConfig, create_local_app
from agentic_cv.calibration_validation import any_prediction_selection
from agentic_cv.prompt_enhancement import PromptEnhancement

from test_agent_chat_e2e import FakeSam3, headers, upload_image, wait_for_session


# --------------------------------------------------------------------------- unit


@pytest.mark.parametrize(
    ("message", "task_type", "classes", "export_format"),
    [
        ("Create a segmentation dataset of workers wearing yellow vests.", "segmentation",
         ["workers wearing yellow vests"], None),
        ("Detect helmets, safety vests and gloves, export YOLO", "detection",
         ["helmets", "safety vests", "gloves"], "yolo"),
        ("segment cars and trucks in coco format", "segmentation", ["cars", "trucks"], "coco"),
        ("make a dataset", None, None, None),
        ("I want to build a dataset for my project", None, None, None),
        ("boxes please", "detection", None, None),
        ("cardboard boxes", None, ["cardboard boxes"], None),
        ("find red apples using bounding boxes", "detection", ["red apples"], None),
        ("detect forklifts", "detection", ["forklifts"], None),
        ("label apples", None, ["apples"], None),
        ("build a forklift segmentation dataset", "segmentation", ["forklift"], None),
        ("I need masks of the forklifts and pallets on my warehouse footage, YOLO please",
         "segmentation", ["forklifts", "pallets"], "yolo"),
        ("give me boxes around the trucks in these videos", "detection", ["trucks"], None),
    ],
)
def test_rules_extractor(message, task_type, classes, export_format):
    patch = RulesDraftExtractor().extract(
        message, draft=TaskDraft(asset_ids=("a",)), pending_slot=None
    ).patch
    assert patch.task_type == task_type
    assert ([item.name for item in patch.classes] if patch.classes else None) == classes
    assert patch.export_format == export_format


def test_append_and_remove_classes():
    draft = TaskDraft(asset_ids=("a",), classes=(ClassSpec(name="helmets"), ClassSpec(name="vests")))
    extractor = RulesDraftExtractor()
    added = apply_patch(draft, extractor.extract("also gloves", draft=draft, pending_slot=None).patch)
    assert [item.name for item in added.classes] == ["helmets", "vests", "gloves"]
    removed = apply_patch(added, extractor.extract("remove vests", draft=added, pending_slot=None).patch)
    assert [item.name for item in removed.classes] == ["helmets", "gloves"]
    # Answering the task-type question does not wipe the class list.
    answered = apply_patch(
        removed, extractor.extract("Segmentation please", draft=removed, pending_slot="task_type").patch
    )
    assert answered.task_type == "segmentation"
    assert [item.name for item in answered.classes] == ["helmets", "gloves"]


def test_chained_append_words():
    draft = TaskDraft(asset_ids=("a",), classes=(ClassSpec(name="hard hat"),))
    patch = RulesDraftExtractor().extract("also add gloves", draft=draft, pending_slot=None).patch
    assert patch.classes_mode == "append"
    assert [item.name for item in patch.classes] == ["gloves"]


def test_planner_skips_a_dead_model_for_a_while():
    now = [100.0]
    client = ExtractingClient(error=ConnectionError("down"))
    planner = OllamaTaskPlanner(client=client, retry_after_sec=30, clock=lambda: now[0])
    planner.extract("segment cars", draft=TaskDraft(), pending_slot=None)
    planner.extract("segment cars", draft=TaskDraft(), pending_slot=None)
    assert len(client.contexts) == 1  # second turn did not wait on the model
    now[0] += 31
    planner.extract("segment cars", draft=TaskDraft(), pending_slot=None)
    assert len(client.contexts) == 2


def test_questions_follow_slot_order_and_task_type_is_assumed_after_one_ask():
    draft = TaskDraft()
    assert missing_slots(draft) == ("assets", "classes", "task_type")
    assert next_question(draft).slot == "assets"
    draft = TaskDraft(asset_ids=("a",), classes=(ClassSpec(name="cars"),))
    question = next_question(draft)
    assert question.slot == "task_type"
    assert question.quick_replies == ("Detection", "Segmentation")
    asked = mark_asked(draft, question)
    resolved = resolve_defaults(asked)
    assert resolved.task_type == "detection"
    assert resolved.export_format == "coco"
    assert any("Assumed detection" in text for text in resolved.assumptions)


def test_rules_prompt_adds_missing_attributes_only():
    assert rules_prompt(ClassSpec(name="worker", attributes=("yellow vest",))) == "worker with yellow vest"
    assert rules_prompt(ClassSpec(name="yellow vests", attributes=("yellow",))) == "yellow vests"


def test_model_patch_drops_classes_not_grounded_in_the_conversation():
    draft = TaskDraft(asset_ids=("a",), pending_slot="classes")
    payload = {
        "task_type": "unknown",
        "classes_changed": True,
        "classes": [
            {"name": "hard hat", "attributes": ["yellow"]},
            {"name": "person", "attributes": []},  # never mentioned
        ],
        "export_format": "unknown",
    }
    rules = RulesDraftExtractor().extract(
        "yellow hard hats, segmentation", draft=draft, pending_slot="classes"
    ).patch
    patch = _patch_from_model(payload, rules=rules, message="yellow hard hats, segmentation", draft=draft)
    assert [item.name for item in patch.classes] == ["hard hat"]
    assert patch.classes[0].attributes == ("yellow",)
    assert patch.task_type == "segmentation"  # backed up by the rules


class ExtractingClient:
    def __init__(self, payload=None, error=None):
        self.payload = payload
        self.error = error
        self.contexts = []

    def extract_draft(self, message, context):
        self.contexts.append(context)
        if self.error:
            raise self.error
        return self.payload

    def plan_task(self, message):  # pragma: no cover - not used when extract_draft exists
        raise AssertionError("plan_task should not be called")


def test_ollama_planner_uses_model_and_falls_back_to_rules():
    planner = OllamaTaskPlanner(
        client=ExtractingClient(
            payload={
                "task_type": "segmentation",
                "classes_changed": True,
                "classes": [{"name": "forklift", "attributes": []}],
                "export_format": "yolo",
            }
        )
    )
    result = planner.extract("label the forklifts as masks, yolo please", draft=TaskDraft(), pending_slot=None)
    assert result.mode == "ollama"
    assert [item.name for item in result.patch.classes] == ["forklift"]
    assert planner.client.contexts[0]["agent_just_asked_about"] == "nothing"

    failing = OllamaTaskPlanner(client=ExtractingClient(error=ConnectionError("down")))
    fallback = failing.extract("detect forklifts", draft=TaskDraft(), pending_slot=None)
    assert fallback.mode == "rules"
    assert "local rule-based planner" in fallback.note
    assert [item.name for item in fallback.patch.classes] == ["forklifts"]


# --------------------------------------------------------------------------- HTTP


class FakeEnhancer:
    def __init__(self, fail=False):
        self.fail = fail
        self.calls = []

    def enhance(self, description):
        self.calls.append(description)
        if self.fail:
            raise RuntimeError("ollama down")
        return PromptEnhancement(source_description=description, prompts=(f"{description} object",))


def make_app(tmp_path, *, planner=None, enhancer=None):
    return create_local_app(
        LocalAppConfig(
            data_dir=tmp_path / "data",
            sam3_provider=FakeSam3(),
            agent_planner=planner or RulesTaskPlanner(),
            agent_prompt_enhancer=enhancer,
            calibration_timeout_sec=60,
            calibration_validator_factory=lambda tenant_id, session_id: any_prediction_selection(),
            start_workers=True,
            worker_poll_interval=0.05,
        )
    )


def post_message(client, session_id, message, **extra):
    response = client.post(
        f"/v1/agent/sessions/{session_id}/messages",
        headers=headers(),
        json={"message": message, **extra},
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_agent_asks_for_missing_details_then_runs_two_classes(tmp_path):
    with TestClient(make_app(tmp_path), raise_server_exceptions=False) as client:
        asset_id = upload_image(client)
        created = client.post(
            "/v1/agent/sessions",
            headers=headers(),
            json={"asset_ids": [asset_id], "message": "make a dataset"},
        )
        assert created.status_code == 200, created.text
        session = created.json()
        assert session["status"] == "gathering"
        assert session["missing_slots"] == ["classes", "task_type"]
        assert session["pending_slot"] == "classes"
        assert "What should I label" in session["messages"][-1]["content"]
        assert session["spec"] is None

        early = client.post(f"/v1/agent/sessions/{session['session_id']}/confirm", headers=headers())
        assert early.status_code == 409

        session = post_message(client, session["session_id"], "hard hats, safety vests")
        assert session["status"] == "gathering"
        assert [item["name"] for item in session["classes"]] == ["hard hats", "safety vests"]
        assert session["pending_slot"] == "task_type"
        assert session["quick_replies"] == ["Detection", "Segmentation"]

        session = post_message(client, session["session_id"], "Segmentation")
        assert session["status"] == "awaiting_confirmation"
        assert session["task_type"] == "segmentation"
        assert session["export_format"] == "coco"
        assert session["spec"]["target_objects"] == ["hard hats", "safety vests"]
        assert "Here is the plan" in session["messages"][-1]["content"]
        assert any("COCO by default" in text for text in session["assumptions"])

        confirmed = client.post(f"/v1/agent/sessions/{session['session_id']}/confirm", headers=headers())
        assert confirmed.status_code == 200, confirmed.text
        assert confirmed.json()["status"] == "labeling"
        assert "Calibrated hard hats at" in confirmed.json()["messages"][-1]["content"]
        assert [item["name"] for item in confirmed.json()["calibration"]["classes"]] == [
            "hard hats", "safety vests",
        ]

        done = wait_for_session(client, session["session_id"])
        assert [item["name"] for item in done["dataset"]["categories"]] == ["hard hats", "safety vests"]
        assert done["dataset"]["annotation_count"] == 2

        download = client.get(done["download_path"], headers=headers())
        with zipfile.ZipFile(io.BytesIO(download.content)) as bundle:
            bundle.extractall(tmp_path / "coco")
        coco = COCO(str(tmp_path / "coco" / "annotations.json"))
        assert {category["name"] for category in coco.loadCats(coco.getCatIds())} == {
            "hard hats", "safety vests",
        }

        late = client.post(
            f"/v1/agent/sessions/{session['session_id']}/messages",
            headers=headers(),
            json={"message": "also gloves"},
        )
        assert late.status_code == 409


def test_one_message_with_everything_runs_on_two_assets_and_exports_yolo(tmp_path):
    with TestClient(make_app(tmp_path), raise_server_exceptions=False) as client:
        first = upload_image(client, name="site A")
        second = upload_image(client, name="site B")
        created = client.post(
            "/v1/agent/sessions",
            headers=headers(),
            json={
                "asset_ids": [first, second],
                "message": "Detect helmets and vests, export YOLO",
            },
        )
        session = created.json()
        assert session["status"] == "awaiting_confirmation", session["messages"]
        assert session["asset_ids"] == [first, second]
        assert session["asset_name"] == "site A, site B"
        assert session["export_format"] == "yolo"
        assert "(2 images)" in session["messages"][-1]["content"]

        confirmed = client.post(f"/v1/agent/sessions/{session['session_id']}/confirm", headers=headers())
        assert confirmed.status_code == 200, confirmed.text
        done = wait_for_session(client, session["session_id"])
        assert done["dataset"]["image_count"] == 2
        assert "YOLO detection dataset is ready" in done["messages"][-1]["content"]

        download = client.get(done["download_path"], headers=headers())
        with zipfile.ZipFile(io.BytesIO(download.content)) as bundle:
            data = bundle.read("data.yaml").decode()
            labels = [name for name in bundle.namelist() if name.startswith("labels/") and name.endswith(".txt")]
            classes = {line.split()[0] for name in labels for line in bundle.read(name).decode().splitlines()}
        assert "nc: 2" in data
        assert classes == {"0", "1"}

        datasets = client.get("/v1/datasets", headers=headers()).json()
        assert datasets[0]["source_asset"] == "site A, site B"


def test_plan_can_be_edited_before_confirmation(tmp_path):
    with TestClient(make_app(tmp_path), raise_server_exceptions=False) as client:
        asset_id = upload_image(client)
        session = client.post(
            "/v1/agent/sessions",
            headers=headers(),
            json={"asset_ids": [asset_id], "message": "segment helmets"},
        ).json()
        assert session["status"] == "awaiting_confirmation"
        session = post_message(client, session["session_id"], "also gloves")
        assert [item["name"] for item in session["classes"]] == ["helmets", "gloves"]
        session = post_message(client, session["session_id"], "make it yolo")
        assert session["export_format"] == "yolo"
        assert session["status"] == "awaiting_confirmation"


def test_agent_asks_for_assets_when_none_are_selected_or_they_are_empty(tmp_path):
    with TestClient(make_app(tmp_path), raise_server_exceptions=False) as client:
        session = client.post(
            "/v1/agent/sessions", headers=headers(), json={"message": "segment helmets"}
        ).json()
        assert session["status"] == "gathering"
        assert session["missing_slots"] == ["assets"]
        empty = client.post("/v1/assets", headers=headers(), json={"name": "empty"}).json()["asset_id"]
        session = post_message(client, session["session_id"], "segment helmets", asset_ids=[empty])
        assert session["status"] == "gathering"
        assert "can't label the selected assets yet" in session["messages"][-1]["content"]
        asset_id = upload_image(client)
        session = post_message(client, session["session_id"], "go ahead", asset_ids=[asset_id])
        assert session["status"] == "awaiting_confirmation"
        assert [item["name"] for item in session["classes"]] == ["helmets"]


def test_prompt_agent_proposes_phrases_and_keeps_the_rules_fallback(tmp_path):
    enhancer = FakeEnhancer()
    with TestClient(make_app(tmp_path, enhancer=enhancer), raise_server_exceptions=False) as client:
        asset_id = upload_image(client)
        session = client.post(
            "/v1/agent/sessions",
            headers=headers(),
            json={"asset_ids": [asset_id], "message": "segment forklifts"},
        ).json()
        assert enhancer.calls == ["forklifts"]
        assert session["classes"][0]["prompts"] == ["forklifts object", "forklifts"]
        assert session["prompt"] == "forklifts object"
        assert "SAM 3 prompt “forklifts object”" in session["messages"][-1]["content"]

    failing = FakeEnhancer(fail=True)
    with TestClient(make_app(tmp_path / "b", enhancer=failing), raise_server_exceptions=False) as client:
        asset_id = upload_image(client)
        session = client.post(
            "/v1/agent/sessions",
            headers=headers(),
            json={"asset_ids": [asset_id], "message": "segment forklifts"},
        ).json()
        assert session["classes"][0]["prompts"] == ["forklifts"]


def test_unknown_asset_is_a_404(tmp_path):
    with TestClient(make_app(tmp_path), raise_server_exceptions=False) as client:
        response = client.post(
            "/v1/agent/sessions",
            headers=headers(),
            json={"asset_ids": ["00000000-0000-0000-0000-00000000abcd"], "message": "segment cars"},
        )
        assert response.status_code == 404


# --------------------------------------------------------------------------- browser


def test_browser_conversation_with_quick_reply(tmp_path):
    playwright_api = pytest.importorskip("playwright.sync_api")
    from PIL import Image

    from test_agent_ui_playwright import FakeEmbeddingProvider, _serve, vlm_factory
    from test_agent_ui_playwright import FakeSam3 as UiFakeSam3

    image_path = tmp_path / "yard.png"
    Image.new("RGB", (96, 64), (30, 90, 150)).save(image_path)
    app = create_local_app(
        LocalAppConfig(
            data_dir=tmp_path / "data",
            embedding_provider=FakeEmbeddingProvider(),
            sam3_provider=UiFakeSam3(),
            agent_planner=RulesTaskPlanner(),
            calibration_validator_factory=vlm_factory(tmp_path),
            calibration_timeout_sec=30,
            start_workers=True,
            worker_poll_interval=0.05,
        )
    )
    expect = playwright_api.expect
    with _serve(app) as base_url:
        with playwright_api.sync_playwright() as playwright:
            try:
                browser = playwright.chromium.launch()
            except playwright_api.Error as exc:
                pytest.skip(f"Playwright Chromium is not installed: {exc}")
            page = browser.new_page(viewport={"width": 1440, "height": 1000})
            page.goto(base_url, wait_until="domcontentloaded")
            page.locator("#upload-toggle").click()
            page.locator("#asset-name").fill("Yard")
            page.locator("#asset-file").set_input_files(str(image_path))
            page.locator("#upload-button").click()
            expect(page.locator("#upload-status")).to_contain_text("indexed for search", timeout=30_000)

            page.locator(".nav-item[data-view='agent']").click()
            expect(page.locator("#agent-assets .agent-asset-option input").first).to_be_checked()
            page.locator("#agent-message").fill("make a dataset")
            page.locator("#agent-message").press("Enter")
            expect(page.locator("#agent-run-pill")).to_have_text("Needs details", timeout=10_000)
            expect(page.locator("#agent-messages")).to_contain_text("What should I label?")
            expect(page.locator("#agent-plan")).to_have_text("Send")

            page.locator("#agent-message").fill("hard hats, safety vests")
            page.locator("#agent-plan").click()
            chip = page.locator("#agent-quick-replies .agent-quick-reply", has_text="Segmentation")
            expect(chip).to_be_visible(timeout=10_000)
            expect(page.locator("#agent-spec")).to_contain_text("hard hats")
            expect(page.locator("#agent-spec")).to_contain_text("safety vests")
            chip.click()

            expect(page.locator("#agent-run-pill")).to_have_text("Awaiting confirmation", timeout=10_000)
            expect(page.locator("#agent-quick-replies")).to_be_hidden()
            expect(page.locator("#agent-spec")).to_contain_text("segmentation")
            page.locator("#agent-confirm").click()
            expect(page.locator("#agent-run-pill")).to_have_text("Dataset ready", timeout=30_000)
            expect(page.locator("#agent-calibration")).to_contain_text("hard hats")
            expect(page.locator("#agent-calibration")).to_contain_text("safety vests")
            expect(page.locator("#agent-progress")).to_contain_text("COCO archive ready to download")
            browser.close()
