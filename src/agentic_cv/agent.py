"""Agent chat: plan a task, calibrate it, label an asset, and package a dataset."""

from __future__ import annotations

import logging
import os
import json
import queue
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Protocol
from uuid import UUID

from PIL import Image

from .agent_store import (
    AgentMessage,
    AgentSessionBackend,
    AgentSessionRecord,
    AgentSessionStatus,
    new_session_record,
)
from .asset_registry import AssetRegistryBackend
from .calibration import (
    CalibrationCancelled,
    CalibrationResult,
    CalibrationSample,
    Sam3Calibrator,
    Validator,
)
from .agent_dialogue import (
    ClassSpec,
    DraftPatch,
    ExtractionResult,
    Question,
    RulesDraftExtractor,
    Slot,
    TaskDraft,
    apply_patch,
    clean_class_phrase,
    draft_from_spec,
    join_names,
    mark_asked,
    missing_slots,
    next_question,
    resolve_defaults,
    rules_prompt,
)
from .batch_labeling import LabelClass, mask_polygons
from .calibration_samples import select_samples_for_target
from .dataset import DatasetCategory
from .prompt_enhancement import OllamaPromptEnhancer, PromptEnhancer
from .calibration_validation import ValidatorSelection, any_prediction_selection
from .contracts import TaskSpec
from .orchestrator import TaskOrchestrationRequest, TaskOrchestrator
from .dataset_builder import DatasetBuilder
from .dataset_export_workflow import DatasetExportWorkflow
from .errors import ModelUnavailableError
from .job_store import JobStoreBackend
from .label_images import LabelImageResolver
from .label_workflow import DurableLabelWorkflow
from .ollama_text import OllamaTextClient
from .sam3 import Sam3Provider
from .semantic_search import VectorIndex

logger = logging.getLogger("agentic_cv.agent")

_DEFAULT_THRESHOLDS = (0.3, 0.45, 0.6, 0.75, 0.85)
_TASK_LEAD = re.compile(
    r"^\s*(?:please\s+)?(?:can\s+you\s+|could\s+you\s+)?(?:i\s+want\s+to\s+)?"
    r"(?:create|build|make|generate|produce|prepare)\s+"
    r"(?:me\b\s*)?(?:(?:a|an|the)\b)?\s*"
    r"(?:(?:detection|segmentation)\s+)?"
    r"(?:dataset\s+)?"
    r"(?:(?:of|for|with|showing|that\s+(?:labels|detects|segments))\b)?\s*",
    re.IGNORECASE,
)
_VERB_LEAD = re.compile(
    r"^\s*(?:please\s+)?(?:segment|detect|label|find|locate|count)\s+(?:the\s+)?",
    re.IGNORECASE,
)
_OVERLAY_ID = re.compile(r"^[0-9a-f]{16,64}\.png$")
_CALIBRATION_THREAD_GRACE_SEC = 2.0


class AgentConflict(RuntimeError):
    """Raised when a session cannot accept the requested transition."""


class CalibrationTimeoutError(RuntimeError):
    """Raised when threshold calibration exceeds the configured budget."""


@dataclass(frozen=True, kw_only=True)
class PlannerResult:
    spec: TaskSpec
    mode: str
    note: str | None = None


class TaskPlanner(Protocol):
    def plan(
        self,
        message: str,
        *,
        task_type: str | None = None,
        target_objects: tuple[str, ...] = (),
        visual_attributes: tuple[str, ...] = (),
        sample_size: int | None = None,
        max_iterations: int | None = None,
    ) -> PlannerResult: ...


class RulesTaskPlanner:
    """Deterministic local planner used when no LLM is available."""

    def __init__(self, orchestrator: TaskOrchestrator | None = None) -> None:
        self.orchestrator = orchestrator or TaskOrchestrator()

    def plan(
        self,
        message: str,
        *,
        task_type: str | None = None,
        target_objects: tuple[str, ...] = (),
        visual_attributes: tuple[str, ...] = (),
        sample_size: int | None = None,
        max_iterations: int | None = None,
    ) -> PlannerResult:
        cleaned = clean_target_phrase(message)
        targets = target_objects or ((cleaned,) if cleaned else ())
        spec = self.orchestrator.orchestrate(
            TaskOrchestrationRequest(
                message=message,
                task_type=task_type,  # type: ignore[arg-type]
                target_objects=targets,
                visual_attributes=visual_attributes,
                sample_size=sample_size or 5,
                max_iterations=max_iterations or 5,
            )
        )
        return PlannerResult(spec=spec, mode="rules")

    def extract(
        self, message: str, *, draft: TaskDraft, pending_slot: Slot | None
    ) -> ExtractionResult:
        return RulesDraftExtractor().extract(message, draft=draft, pending_slot=pending_slot)


class OllamaTaskPlanner:
    """Try a local text model, then fall back to the deterministic planner.

    After a failure the model is skipped for ``retry_after_sec`` so every chat
    turn does not wait on a dead endpoint (on Windows ``localhost`` tries IPv6
    then IPv4, about 4 s per refused request).
    """

    def __init__(
        self,
        client: OllamaTextClient,
        fallback: TaskPlanner | None = None,
        *,
        retry_after_sec: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.client = client
        self.fallback = fallback or RulesTaskPlanner()
        self.retry_after_sec = retry_after_sec
        self._clock = clock
        self._down_until = 0.0

    def plan(
        self,
        message: str,
        *,
        task_type: str | None = None,
        target_objects: tuple[str, ...] = (),
        visual_attributes: tuple[str, ...] = (),
        sample_size: int | None = None,
        max_iterations: int | None = None,
    ) -> PlannerResult:
        try:
            spec = self.client.plan_task(message)
        except Exception:
            logger.warning("agent planner unavailable; using local rules", exc_info=True)
            result = self.fallback.plan(
                message,
                task_type=task_type,
                target_objects=target_objects,
                visual_attributes=visual_attributes,
                sample_size=sample_size,
                max_iterations=max_iterations,
            )
            return PlannerResult(
                spec=result.spec,
                mode="rules",
                note="Planning model unavailable; using the local rule-based planner.",
            )
        spec = _apply_overrides(
            spec,
            task_type=task_type,
            target_objects=target_objects,
            visual_attributes=visual_attributes,
            sample_size=sample_size,
            max_iterations=max_iterations,
        )
        return PlannerResult(spec=spec, mode="ollama")

    def extract(
        self, message: str, *, draft: TaskDraft, pending_slot: Slot | None
    ) -> ExtractionResult:
        """LLM slot extraction, checked against deterministic rules.

        The rules extractor always runs: it backs up fields the model left
        "unknown", and it is the whole answer when the model is unreachable.
        """

        rules = RulesDraftExtractor().extract(message, draft=draft, pending_slot=pending_slot)
        unavailable = "Planning model unavailable; using the local rule-based planner."
        if self._clock() < self._down_until:
            return ExtractionResult(patch=rules.patch, mode="rules", note=unavailable)
        try:
            if hasattr(self.client, "extract_draft"):
                payload = self.client.extract_draft(
                    message, _draft_context(draft, pending_slot)
                )
                patch = _patch_from_model(payload, rules=rules.patch, message=message, draft=draft)
            else:
                spec = self.client.plan_task(message)
                lifted = draft_from_spec(spec, asset_ids=draft.asset_ids)
                patch = DraftPatch(
                    task_type=spec.task_type,
                    classes=lifted.classes,
                    export_format=rules.patch.export_format,
                )
        except Exception as exc:
            self._down_until = self._clock() + self.retry_after_sec
            logger.warning("agent planner unavailable; using local rules: %s", exc)
            return ExtractionResult(patch=rules.patch, mode="rules", note=unavailable)
        return ExtractionResult(patch=patch, mode="ollama")


def _draft_context(draft: TaskDraft, pending_slot: Slot | None) -> dict[str, Any]:
    return {
        "task_type": draft.task_type or "unknown",
        "classes": [
            {"name": item.name, "attributes": list(item.attributes)} for item in draft.classes
        ],
        "export_format": draft.export_format or "unknown",
        "agent_just_asked_about": pending_slot or "nothing",
    }


def _stems(text: str) -> set[str]:
    return {word[:4] for word in re.findall(r"[a-z0-9]+", text.lower()) if len(word) >= 3}


def _class_key(item: ClassSpec) -> str:
    """Order-free, singular-ish key: "orange safety vest" == "safety vest" + [orange]."""

    words = re.findall(r"[a-z0-9]+", " ".join((*item.attributes, item.name)).lower())
    return " ".join(sorted(word[:-1] if word.endswith("s") and len(word) > 3 else word for word in words))


def _patch_from_model(
    payload: dict[str, Any],
    *,
    rules: DraftPatch,
    message: str,
    draft: TaskDraft,
) -> DraftPatch:
    """Validate the model's JSON; code decides what actually changed.

    Measured on qwen3-vl:4b: the model extracts good class names but sets
    ``classes_changed`` to false almost every time, invents a task type for
    messages that never mention one, and repeats a class with and without its
    attribute. So:

    * ``classes_changed`` is ignored: the model's list is compared with the
      current draft, and an identical list means "no change";
    * a class must share a word stem with the user's message or the current
      classes (no invented classes), and near-duplicates are merged;
    * the model's task type is used only when the message hints at one or
      the agent just asked for it;
    * when the model changes nothing, the deterministic rules (which handle
      "also", "remove", "only") decide.
    """

    hinted = rules.task_type is not None or draft.pending_slot == "task_type"
    task_type = payload.get("task_type")
    if task_type not in ("detection", "segmentation") or not hinted:
        task_type = None
    task_type = rules.task_type or task_type
    export_format = payload.get("export_format")
    export_format = export_format if export_format in ("coco", "yolo") else rules.export_format

    classes = None
    if isinstance(payload.get("classes"), list):
        grounding = _stems(message) | _stems(" ".join(item.name for item in draft.classes))
        parsed: list[ClassSpec] = []
        keys: set[str] = set()
        for item in payload["classes"][:10]:
            if not isinstance(item, dict) or not isinstance(item.get("name"), str):
                continue
            name = clean_class_phrase(" ".join(item["name"].split()))
            if not name or not (_stems(name) & grounding):
                continue
            attributes = tuple(
                " ".join(value.split())
                for value in item.get("attributes", [])
                if isinstance(value, str) and value.strip()
            )[:5]
            candidate = ClassSpec(name=name, attributes=attributes)
            key = _class_key(candidate)
            # "safety vest" + [orange] after "orange safety vest": same class.
            if key in keys or any(set(key.split()) <= set(seen.split()) for seen in keys):
                continue
            keys.add(key)
            parsed.append(candidate)
        current = {_class_key(item) for item in draft.classes}
        if parsed and {_class_key(item) for item in parsed} != current:
            classes = tuple(parsed)
    if classes is None and rules.classes is not None:
        return replace(rules, task_type=task_type, export_format=export_format)
    return DraftPatch(task_type=task_type, classes=classes, export_format=export_format)


def pending_slot_is_classes(draft: TaskDraft) -> bool:
    return draft.pending_slot == "classes"


def planner_from_environment() -> TaskPlanner:
    if os.getenv("AGENTIC_CV_AGENT_PLANNER", "").strip().lower() == "rules":
        return RulesTaskPlanner()
    try:
        timeout = float(os.getenv("AGENTIC_CV_OLLAMA_TIMEOUT_SEC", "30"))
    except ValueError:
        timeout = 30.0
    client = OllamaTextClient(
        model=os.getenv("AGENTIC_CV_TEXT_MODEL", "qwen3-vl:4b"),
        base_url=os.getenv("AGENTIC_CV_OLLAMA_URL", "http://localhost:11434"),
        timeout_sec=timeout,
    )
    return OllamaTaskPlanner(client=client, fallback=RulesTaskPlanner())


def prompt_enhancer_from_environment() -> PromptEnhancer | None:
    """The Ollama SAM 3 prompt agent, unless rules-only mode is configured.

    ``AGENTIC_CV_AGENT_PROMPTS=rules`` keeps planning on the LLM but skips
    prompt rewriting.
    """

    if os.getenv("AGENTIC_CV_AGENT_PLANNER", "").strip().lower() == "rules":
        return None
    if os.getenv("AGENTIC_CV_AGENT_PROMPTS", "").strip().lower() == "rules":
        return None
    try:
        timeout = float(os.getenv("AGENTIC_CV_OLLAMA_TIMEOUT_SEC", "30"))
    except ValueError:
        timeout = 30.0
    return OllamaPromptEnhancer(
        model=os.getenv("AGENTIC_CV_TEXT_MODEL", "qwen3-vl:4b"),
        base_url=os.getenv("AGENTIC_CV_OLLAMA_URL", "http://localhost:11434"),
        timeout_sec=timeout,
    )


def clean_target_phrase(message: str) -> str:
    text = re.sub(r"\s+", " ", message or "").strip()
    if not text:
        return ""
    text = _TASK_LEAD.sub("", text)
    text = _VERB_LEAD.sub("", text)
    return text.strip(" .,:;!-")


def prompt_for_spec(spec: TaskSpec) -> str:
    targets = [item.strip() for item in spec.target_objects if item.strip()]
    prompt = " and ".join(targets)
    if spec.visual_attributes:
        target_text = prompt.lower()
        attributes = ", ".join(
            item.strip()
            for item in spec.visual_attributes
            if item.strip() and item.strip().lower() not in target_text
        )
        if attributes:
            prompt = f"{prompt} with {attributes}"
    return prompt or "object"


@dataclass(frozen=True, kw_only=True)
class AgentServiceDependencies:
    store: AgentSessionBackend
    registry: AssetRegistryBackend
    labels: DurableLabelWorkflow
    jobs: JobStoreBackend
    dataset_builder: DatasetBuilder
    export_workflow: DatasetExportWorkflow
    images: LabelImageResolver
    sam3_provider: Sam3Provider
    planner: TaskPlanner
    calibration_timeout_sec: float = 120.0
    thresholds: tuple[float, ...] = _DEFAULT_THRESHOLDS
    minimum_agreement: float = 0.6
    validator: Validator | None = None
    validator_factory: Callable[[UUID, UUID], ValidatorSelection] | None = None
    vectors: VectorIndex | None = None
    overlay_root: Path | None = None
    # SAM 3 prompt agent. None keeps the deterministic noun-phrase rules.
    prompt_enhancer: PromptEnhancer | None = None
    # Text embedding for target-aware calibration sampling (SigLIP text tower).
    text_embedder: Callable[[str], Any] | None = None
    # Unloads a managed model by name ("siglip2", "sam3") so calibration can
    # keep at most one large model on the GPU at a time.
    release_models: Callable[[str], Any] | None = None


@dataclass(frozen=True, kw_only=True)
class AgentSessionView:
    record: AgentSessionRecord
    label_status: str | None
    export_status: str | None
    dataset: dict[str, Any] | None
    download_path: str | None

    @property
    def draft(self) -> TaskDraft:
        return session_draft(self.record)

    @property
    def quick_replies(self) -> tuple[str, ...]:
        spec = self.record.spec or {}
        return tuple(spec.get("quick_replies", ()))


_MAX_PROMPTS_TRIED = 3
_OPEN_FOR_MESSAGES = ("gathering", "awaiting_confirmation", "needs_attention")
_NIL_ASSET = UUID(int=0)


def session_draft(record: AgentSessionRecord) -> TaskDraft:
    """The session's draft; sessions created before drafts existed are lifted from v1."""

    spec = record.spec or {}
    if "draft" in spec:
        return TaskDraft.from_dict(spec["draft"])
    if "target_objects" not in spec:
        return TaskDraft(asset_ids=(str(record.asset_id),))
    task = TaskSpec.from_dict(spec)
    draft = draft_from_spec(task, asset_ids=(str(record.asset_id),))
    if len(draft.classes) == 1 and record.prompt:
        draft = replace(draft, classes=(replace(draft.classes[0], prompts=(record.prompt,)),))
    return replace(draft, export_format="coco")


def export_format_for(draft: TaskDraft) -> str:
    if draft.export_format == "yolo":
        return "yolo-segmentation" if draft.task_type == "segmentation" else "yolo-detection"
    return "coco"


def _format_label(export_format: str) -> str:
    return {
        "coco": "COCO",
        "yolo-detection": "YOLO detection",
        "yolo-segmentation": "YOLO segmentation",
    }.get(export_format, export_format)


class AgentService:
    def __init__(self, dependencies: AgentServiceDependencies) -> None:
        self.dependencies = dependencies
        self._calibration_lock = threading.Lock()
        self._calibration_cancels: dict[UUID, threading.Event] = {}
        self._enhancer_down_until = 0.0
        # Sessions whose calibration runs in this process. A ``calibrating``
        # session missing from here (e.g. after a restart) is resumed by
        # ``advance``.
        self._inflight: set[UUID] = set()
        self._calibration_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="agent-calibration-run"
        )
        if dependencies.calibration_timeout_sec <= 0:
            raise ValueError("calibration_timeout_sec must be greater than 0")
        if not dependencies.thresholds:
            raise ValueError("at least one calibration threshold is required")
        if not 0 < dependencies.minimum_agreement <= 1:
            raise ValueError("minimum_agreement must be between 0 and 1")

    # ------------------------------------------------------------------ dialogue

    def create_session(
        self,
        *,
        tenant_id: UUID,
        message: str,
        asset_id: UUID | None = None,
        asset_ids: tuple[UUID, ...] = (),
        task_type: str | None = None,
        target_objects: tuple[str, ...] = (),
        visual_attributes: tuple[str, ...] = (),
        sample_size: int | None = None,
        max_iterations: int | None = None,
    ) -> AgentSessionView:
        if not message.strip():
            raise ValueError("message must not be empty")
        selected = _dedupe_assets(((asset_id,) if asset_id else ()) + tuple(asset_ids))
        names = self._asset_names(tenant_id=tenant_id, asset_ids=selected)
        overrides = _override_patch(
            task_type=task_type,
            target_objects=target_objects,
            visual_attributes=visual_attributes,
            sample_size=sample_size,
            max_iterations=max_iterations,
        )
        record = new_session_record(
            tenant_id=tenant_id,
            asset_id=selected[0] if selected else _NIL_ASSET,
            asset_name=", ".join(names) or "No asset selected",
            message=message.strip(),
            spec={"draft": TaskDraft(asset_ids=tuple(str(item) for item in selected)).to_dict()},
            planner_mode="rules",
            planner_note=None,
            prompt=None,
            messages=(),
        )
        record = self._turn(
            tenant_id=tenant_id,
            record=record,
            message=message.strip(),
            overrides=overrides,
        )
        return self.view(self.dependencies.store.create(record))

    def add_message(
        self,
        *,
        tenant_id: UUID,
        session_id: UUID,
        message: str,
        asset_ids: tuple[UUID, ...] | None = None,
    ) -> AgentSessionView:
        """One more chat turn: answer a question or change the plan before confirming."""

        if not message.strip():
            raise ValueError("message must not be empty")
        record = self.dependencies.store.get(tenant_id=tenant_id, session_id=session_id)
        if record.status not in _OPEN_FOR_MESSAGES:
            raise AgentConflict(
                f"session is {record.status}; start a new task to change the plan"
            )
        if asset_ids is not None:
            selected = _dedupe_assets(tuple(asset_ids))
            names = self._asset_names(tenant_id=tenant_id, asset_ids=selected)
            draft = replace(session_draft(record), asset_ids=tuple(str(item) for item in selected))
            record = replace(
                record,
                asset_id=selected[0] if selected else _NIL_ASSET,
                asset_name=", ".join(names) or "No asset selected",
                spec={**(record.spec or {}), "draft": draft.to_dict()},
            )
        record = self._turn(tenant_id=tenant_id, record=record, message=message.strip())
        return self.view(self.dependencies.store.save(record))

    def _turn(
        self,
        *,
        tenant_id: UUID,
        record: AgentSessionRecord,
        message: str,
        overrides: DraftPatch | None = None,
    ) -> AgentSessionRecord:
        draft = session_draft(record)
        extraction = self._extract(message, draft)
        draft = apply_patch(draft, extraction.patch)
        if overrides is not None:
            draft = apply_patch(draft, overrides)
        draft = resolve_defaults(draft)
        question = next_question(draft)
        image_count = 0
        if question is None:
            try:
                image_count = len(
                    self.dependencies.images.resolve_many(
                        tenant_id=tenant_id, asset_ids=draft.asset_ids
                    )
                )
            except ValueError as exc:
                question = Question(
                    slot="assets",
                    text=(
                        f"I can't label the selected assets yet: {exc}. Select assets that "
                        "contain images or videos, then send your message again."
                    ),
                )
        draft = mark_asked(draft, question)
        messages = record.messages + (AgentMessage.create("user", message),)
        if question is not None:
            reply = question.text
            spec: dict[str, Any] = {
                "draft": draft.to_dict(),
                "quick_replies": list(question.quick_replies),
            }
            status: AgentSessionStatus = "gathering"
            prompt = None
        else:
            draft = self._prepare_prompts(draft)
            reply = self._plan_summary(record, draft, image_count)
            spec = {**draft.task_spec().to_dict(), "draft": draft.to_dict(), "quick_replies": []}
            status = "awaiting_confirmation"
            prompt = "; ".join(item.prompt for item in draft.classes)
        if extraction.note:
            reply = f"{reply}\n\n{extraction.note}"
        return replace(
            record,
            status=status,
            spec=spec,
            prompt=prompt,
            planner_mode=extraction.mode,
            planner_note=extraction.note,
            threshold=None,
            calibration=None,
            error=None,
            messages=messages + (AgentMessage.create("assistant", reply),),
        )

    def _extract(self, message: str, draft: TaskDraft) -> ExtractionResult:
        planner = self.dependencies.planner
        extract = getattr(planner, "extract", None)
        if callable(extract):
            return extract(message, draft=draft, pending_slot=draft.pending_slot)
        # A planner with only plan(): treat its TaskSpec as the whole turn.
        planned = planner.plan(message)
        lifted = draft_from_spec(planned.spec, asset_ids=draft.asset_ids)
        return ExtractionResult(
            patch=DraftPatch(task_type=planned.spec.task_type, classes=lifted.classes),
            mode=planned.mode,
            note=planned.note,
        )

    def _prepare_prompts(self, draft: TaskDraft) -> TaskDraft:
        """Prompt agent: give each class short SAM 3 noun phrases.

        The enhancer (local LLM) proposes up to three phrases; the deterministic
        rules phrase is always kept as the last alternative so calibration has
        a safe fallback. Without an enhancer the rules phrase is used alone.
        """

        enhancer = self.dependencies.prompt_enhancer
        if enhancer is not None and time.monotonic() < self._enhancer_down_until:
            enhancer = None
        classes: list[ClassSpec] = []
        for item in draft.classes:
            if item.prompts:
                classes.append(item)
                continue
            fallback = rules_prompt(item)
            # The user's own phrase leads: on qwen3-vl:4b the enhancer drifts
            # ("yellow-orange safety vest with reflective strips") or drops
            # meaning ("yellow vest workers"). Its phrases are kept only as
            # short alternatives that still name the object, for calibration
            # to compare.
            prompts: list[str] = [fallback]
            head = _stems(item.name.split()[-1]) if item.name.split() else set()
            if enhancer is not None:
                try:
                    enhanced = enhancer.enhance(fallback)
                    prompts.extend(
                        phrase for phrase in enhanced.prompts
                        if 0 < len(phrase.split()) <= 4 and (not head or _stems(phrase) & head)
                    )
                except Exception as exc:
                    # Skip the enhancer for a while instead of paying its
                    # timeout on every class and every turn.
                    self._enhancer_down_until = time.monotonic() + 30.0
                    enhancer = None
                    logger.warning("prompt enhancement failed; using the class phrase: %s", exc)
            unique: list[str] = []
            for phrase in prompts:
                if _class_key(ClassSpec(name=phrase)) not in {
                    _class_key(ClassSpec(name=seen)) for seen in unique
                }:
                    unique.append(phrase)
            classes.append(replace(item, prompts=tuple(unique[:4])))
        return replace(draft, classes=tuple(classes))

    def _plan_summary(
        self, record: AgentSessionRecord, draft: TaskDraft, image_count: int
    ) -> str:
        task = "segmentation (pixel masks)" if draft.task_type == "segmentation" else "detection (bounding boxes)"
        lines = [
            f"Here is the plan: {task} on “{record.asset_name}” "
            f"({image_count} image{'' if image_count == 1 else 's'}).",
        ]
        for item in draft.classes:
            line = f"• {item.name}"
            if item.prompt.lower() != item.name.lower():
                line += f" — SAM 3 prompt “{item.prompt}”"
            lines.append(line)
        lines.append(
            f"I'll calibrate on up to {draft.sample_size} sample image"
            f"{'' if draft.sample_size == 1 else 's'} per class, label every image, and "
            f"package {_format_label(export_format_for(draft))}. "
            "Confirm to start, or tell me what to change."
        )
        lines.extend(draft.assumptions)
        return "\n".join(lines)

    def get_session(self, *, tenant_id: UUID, session_id: UUID) -> AgentSessionView:
        return self.view(self.dependencies.store.get(tenant_id=tenant_id, session_id=session_id))

    def list_sessions(
        self, *, tenant_id: UUID, limit: int = 50
    ) -> list[AgentSessionView]:
        return [
            self.view(record)
            for record in self.dependencies.store.list(tenant_id=tenant_id, limit=limit)
        ]

    # ------------------------------------------------------------------ run

    def confirm(
        self,
        *,
        tenant_id: UUID,
        session_id: UUID,
        threshold: float | None = None,
        thresholds: tuple[float, ...] | None = None,
        minimum_agreement: float | None = None,
        class_thresholds: dict[str, float] | None = None,
        wait: bool = False,
    ) -> AgentSessionView:
        """Start calibration and whole-asset labeling.

        By default this returns at once with status ``calibrating``; the run
        continues on the service's calibration executor (one at a time: the
        GPU is shared) and publishes progress on the session. ``wait=True``
        runs it inline and raises errors (scripts, HTTP tests). ``threshold``
        or ``class_thresholds`` skip calibration and label at fixed values.
        """

        record = self.dependencies.store.get(tenant_id=tenant_id, session_id=session_id)
        if record.status not in ("awaiting_confirmation", "needs_attention"):
            raise AgentConflict(
                f"session is {record.status}; only a planned session can be confirmed"
            )
        draft = session_draft(record)
        missing = missing_slots(draft)
        if record.spec is None or missing:
            raise AgentConflict(
                "the task is not fully specified yet; still needed: " + ", ".join(missing)
            )
        if threshold is not None and not 0 <= threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")
        agreement = (
            minimum_agreement
            if minimum_agreement is not None
            else self.dependencies.minimum_agreement
        )
        if not 0 < agreement <= 1:
            raise ValueError("minimum_agreement must be between 0 and 1")
        selected_thresholds = thresholds or self.dependencies.thresholds
        if any(not 0 <= value <= 1 for value in selected_thresholds):
            raise ValueError("thresholds must be between 0 and 1")
        if class_thresholds is not None:
            names = {item.name for item in draft.classes}
            if set(class_thresholds) != names:
                raise ValueError("class_thresholds must give a threshold for every class: " + ", ".join(sorted(names)))
            if any(not 0 <= float(value) <= 1 for value in class_thresholds.values()):
                raise ValueError("class thresholds must be between 0 and 1")

        request = {
            "threshold": threshold,
            "thresholds": list(selected_thresholds),
            "minimum_agreement": agreement,
            "class_thresholds": (
                {name: float(value) for name, value in class_thresholds.items()}
                if class_thresholds is not None
                else None
            ),
        }
        record = replace(
            record,
            status="calibrating",
            error=None,
            spec={**(record.spec or {}), "calibration_request": request},
        )
        record = self.dependencies.store.save(record)
        with self._calibration_lock:
            self._inflight.add(session_id)
        if wait:
            return self._run_calibration(tenant_id, session_id, request, raise_errors=True)
        self._calibration_executor.submit(
            self._run_calibration, tenant_id, session_id, request, raise_errors=False
        )
        return self.view(record)

    def _run_calibration(
        self,
        tenant_id: UUID,
        session_id: UUID,
        request: dict[str, Any],
        *,
        raise_errors: bool,
    ) -> AgentSessionView:
        try:
            return self._calibrate_and_submit(tenant_id, session_id, request, raise_errors=raise_errors)
        except Exception:
            if raise_errors:
                raise
            logger.exception("background calibration failed", extra={"session_id": str(session_id)})
            return self.view(self.dependencies.store.get(tenant_id=tenant_id, session_id=session_id))
        finally:
            with self._calibration_lock:
                self._inflight.discard(session_id)

    def _calibrate_and_submit(
        self,
        tenant_id: UUID,
        session_id: UUID,
        request: dict[str, Any],
        *,
        raise_errors: bool,
    ) -> AgentSessionView:
        record = self.dependencies.store.get(tenant_id=tenant_id, session_id=session_id)
        if record.status != "calibrating":
            return self.view(record)
        draft = session_draft(record)
        threshold = request.get("threshold")
        selected_thresholds = tuple(request.get("thresholds") or self.dependencies.thresholds)
        agreement = float(request.get("minimum_agreement") or self.dependencies.minimum_agreement)
        class_thresholds = request.get("class_thresholds")
        fixed = threshold is not None or class_thresholds is not None
        classes = draft.classes
        categories = [
            DatasetCategory(category_id=index, name=item.name[:80])
            for index, item in enumerate(classes, start=1)
        ]
        selection: ValidatorSelection | None = None
        try:
            selection = self._select_validator(
                tenant_id=tenant_id, session_id=session_id
            )
            initial_notice = selection.current_notice()
            if initial_notice:
                record = self._append(record, "assistant", initial_notice)
                record = self.dependencies.store.save(record)
            calibration_started = time.perf_counter()
            all_images = self.dependencies.images.resolve_many(
                tenant_id=tenant_id, asset_ids=draft.asset_ids
            )
            if not all_images:
                raise ValueError("asset has no image or video frames to label")
            total_images = len(all_images)
            sample_ids: set[str] = set()
            if not fixed:
                cancel_event = threading.Event()
                with self._calibration_lock:
                    self._calibration_cancels[session_id] = cancel_event
                image_cache: dict[str, Image.Image] = {}
                progress: list[dict[str, Any]] = []

                def load(item) -> CalibrationSample:
                    if item.image_id not in image_cache:
                        image_cache[item.image_id] = self._load_image(item.path)
                    return CalibrationSample(image_id=item.image_id, image=image_cache[item.image_id])

                def publish(name: str, prompt: str, attempt) -> None:
                    progress.append(
                        {"name": name, "prompt": prompt, **_attempt_payload(attempt)}
                    )
                    self._save_progress(tenant_id, session_id, progress)

                # Pick every class's samples first (this uses the SigLIP text
                # tower), then free SigLIP so SAM 3 and the VLM share the GPU.
                class_samples = {
                    item.name: self._sample_images(
                        images=all_images,
                        sample_size=draft.sample_size,
                        text=item.prompt,
                    )
                    for item in classes
                }
                self._release_model("siglip2")

                def calibrate_all() -> list[tuple[str, CalibrationResult, list[CalibrationResult]]]:
                    """Phased rounds so SAM 3 and the VLM are never resident together.

                    Each round runs SAM 3 for every class still pending (one
                    prompt each), unloads SAM 3, then lets the validator judge.
                    A class whose prompt failed moves to its next prompt in the
                    next round, after the VLM is unloaded. On a 6 GB GPU the two
                    models together paged to system memory: ~45 s per VLM
                    verdict instead of ~4 s (measured).
                    """

                    calibrator = Sam3Calibrator(
                        self.dependencies.sam3_provider,
                        selection.validator,
                        minimum_agreement=agreement,
                        max_iterations=draft.max_iterations,
                    )
                    samples_for = {
                        index: [load(image) for image in class_samples[item.name]]
                        for index, item in enumerate(classes)
                    }
                    for chosen in samples_for.values():
                        sample_ids.update(sample.image_id for sample in chosen)
                    queue = {
                        index: list((item.prompts or (item.prompt,))[:_MAX_PROMPTS_TRIED])
                        for index, item in enumerate(classes)
                    }
                    tried: dict[int, list[CalibrationResult]] = {index: [] for index in queue}
                    first_round = True
                    while queue:
                        if not first_round and selection.release is not None:
                            selection.release()  # free the VLM before SAM 3 runs again
                        first_round = False
                        round_prompts = {index: prompts.pop(0) for index, prompts in queue.items()}
                        raw = {
                            index: calibrator.infer(
                                samples_for[index],
                                prompt=prompt,
                                thresholds=list(selected_thresholds),
                                should_cancel=cancel_event.is_set,
                            )
                            for index, prompt in round_prompts.items()
                        }
                        self._release_model("sam3")
                        for index, prompt in round_prompts.items():
                            self._save_preview(
                                tenant_id,
                                session_id,
                                index=index,
                                name=classes[index].name,
                                prompt=prompt,
                                samples=samples_for[index],
                                raw=raw[index],
                            )
                        for index, prompt in round_prompts.items():
                            item = classes[index]
                            result = calibrator.score(
                                samples_for[index],
                                raw[index],
                                prompt=prompt,
                                thresholds=list(selected_thresholds),
                                should_cancel=cancel_event.is_set,
                                on_attempt=lambda attempt, name=item.name, prompt=prompt: publish(name, prompt, attempt),
                            )
                            tried[index].append(result)
                            if result.status in ("calibrated", "no_target_in_samples") or not queue[index]:
                                del queue[index]
                    outcomes = []
                    for index in range(len(classes)):
                        results = tried[index]
                        best = next(
                            (result for result in results if result.status == "calibrated"),
                            max(results, key=lambda result: max(a.agreement for a in result.attempts)),
                        )
                        outcomes.append((best.prompt, best, results))
                    return outcomes

                try:
                    outcomes = self._calibrate_with_timeout(
                        calibrate_all,
                        cancel_event=cancel_event,
                        budget=self.dependencies.calibration_timeout_sec * len(classes),
                    )
                finally:
                    with self._calibration_lock:
                        self._calibration_cancels.pop(session_id, None)
                timing = {
                    "calibration_sec": round(time.perf_counter() - calibration_started, 2)
                }
                final_notice = selection.current_notice()
                if final_notice and final_notice != initial_notice:
                    record = self._append(record, "assistant", final_notice)
                    record = self.dependencies.store.save(record)
                record, release_timing = self._release_validator(selection, record)
                timing.update(release_timing)
                calibration = _combined_calibration(
                    [
                        {
                            **_calibration_payload(result, selection, timings=timing),
                            "name": item.name,
                            "category_id": category.category_id,
                            "prompts_tried": [tried_result.prompt for tried_result in tried],
                        }
                        for item, category, (_, result, tried) in zip(classes, categories, outcomes)
                    ]
                )
                absent = [
                    item for item, (_, result, _) in zip(classes, outcomes)
                    if result.status == "no_target_in_samples"
                ]
                failed = [
                    item for item, (_, result, _) in zip(classes, outcomes)
                    if result.status != "calibrated" or result.threshold is None
                ]
                if failed:
                    target = join_names(tuple(failed))
                    if absent and len(absent) == len(failed):
                        explanation = (
                            f"None of the sample images shows {target}, so I could not "
                            "calibrate. Check that the selected assets contain it, or "
                            "describe it differently."
                        )
                    else:
                        explanation = (
                            f"I can't segment reliably from these images for {target}: no "
                            "threshold met the sample agreement with any of the prompts I "
                            "tried. You can describe the object differently (for example "
                            "“hard hat” instead of “helmet”), or run at a fixed threshold "
                            "such as 0.50."
                        )
                    record = replace(
                        record,
                        status="needs_attention",
                        calibration=calibration,
                        error=explanation,
                    )
                    record = self._append(record, "assistant", explanation)
                    return self.view(self.dependencies.store.save(record))
                chosen = [float(result.threshold) for _, result, _ in outcomes]  # type: ignore[arg-type]
                chosen_prompts = [prompt for prompt, _, _ in outcomes]
            else:
                if class_thresholds is not None:
                    chosen = [float(class_thresholds[item.name]) for item in classes]
                else:
                    chosen = [float(threshold) for _ in classes]
                # Keep the prompts that calibration chose when the user only
                # overrides thresholds after a calibration run.
                previous = {
                    item.get("name"): item.get("prompt")
                    for item in ((record.calibration or {}).get("classes") or [])
                }
                chosen_prompts = [previous.get(item.name) or item.prompt for item in classes]
                calibration = _manual_calibration_payload(
                    prompt=", ".join(chosen_prompts), threshold=chosen[0]
                )
                calibration["class_thresholds"] = {
                    item.name: value for item, value in zip(classes, chosen)
                }
                if record.calibration and record.calibration.get("classes"):
                    calibration["previous"] = record.calibration
                record, _ = self._release_validator(selection, record)

            # A cancel request can arrive while calibration is running. Do not
            # start labeling after the user has already cancelled the run.
            current = self.dependencies.store.get(
                tenant_id=tenant_id, session_id=session_id
            )
            if current.status != "calibrating":
                return self.view(current)

            label_classes = [
                LabelClass(prompt=prompt, threshold=value, category=category)
                for prompt, value, category in zip(chosen_prompts, chosen, categories)
            ]
            job = self.dependencies.labels.submit(
                tenant_id=tenant_id,
                asset_id=draft.asset_ids[0],
                prompt=label_classes[0].prompt,
                threshold=label_classes[0].threshold,
                task=draft.task_type,  # type: ignore[arg-type]
                category_id=categories[0].category_id,
                category_name=categories[0].name,
                idempotency_key=f"agent-label:{record.session_id}",
                asset_ids=list(draft.asset_ids),
                classes=label_classes,
            )
            record = replace(
                record,
                status="labeling",
                threshold=chosen[0],
                calibration=calibration,
                label_job_id=job.job_id,
                error=None,
            )
            plural = "" if total_images == 1 else "s"
            if not fixed:
                sample_count = len(sample_ids)
                if len(classes) == 1:
                    calibrated = f"Calibrated at {chosen[0]:.2f}"
                else:
                    calibrated = "Calibrated " + ", ".join(
                        f"{item.name} at {value:.2f}" for item, value in zip(classes, chosen)
                    )
                message = (
                    f"{calibrated} on {sample_count} sample image"
                    f"{'' if sample_count == 1 else 's'}. Labeling {total_images} image"
                    f"{plural} in the background."
                )
            elif class_thresholds is not None and len(classes) > 1:
                message = (
                    "Using your thresholds: "
                    + ", ".join(f"{item.name} {value:.2f}" for item, value in zip(classes, chosen))
                    + f". Labeling {total_images} image{plural} in the background."
                )
            else:
                message = (
                    f"Using the fixed threshold {chosen[0]:.2f}. Labeling "
                    f"{total_images} image{plural} in the background."
                )
            record = self._append(record, "assistant", message)
            record = self.dependencies.store.save(record)
            return self.view(record)
        except CalibrationTimeoutError as exc:
            if selection is not None:
                record, _ = self._release_validator(selection, record)
            record = replace(record, status="failed", error=str(exc))
            record = self._append(record, "assistant", str(exc))
            saved = self.dependencies.store.save(record)
            if raise_errors:
                raise
            return self.view(saved)
        except CalibrationCancelled as exc:
            if selection is not None:
                record, _ = self._release_validator(selection, record)
            current = self.dependencies.store.get(
                tenant_id=tenant_id, session_id=session_id
            )
            if current.status == "cancelled":
                return self.view(current)
            record = replace(record, status="failed", error=str(exc))
            record = self._append(record, "assistant", str(exc))
            saved = self.dependencies.store.save(record)
            if raise_errors:
                raise
            return self.view(saved)
        except ModelUnavailableError as exc:
            if selection is not None:
                record, _ = self._release_validator(selection, record)
            record = replace(record, status="failed", error=str(exc))
            record = self._append(record, "assistant", str(exc))
            saved = self.dependencies.store.save(record)
            if raise_errors:
                raise
            return self.view(saved)
        except Exception as exc:
            if selection is not None:
                record, _ = self._release_validator(selection, record)
            record = replace(record, status="failed", error=str(exc))
            record = self._append(record, "assistant", f"The labeling run could not start: {exc}")
            saved = self.dependencies.store.save(record)
            if raise_errors:
                raise
            return self.view(saved)

    def cancel(self, *, tenant_id: UUID, session_id: UUID) -> AgentSessionView:
        record = self.dependencies.store.get(tenant_id=tenant_id, session_id=session_id)
        if record.status in ("complete", "failed", "cancelled"):
            raise AgentConflict(f"session is already {record.status}")
        if record.status == "labeling" and record.label_job_id is not None:
            try:
                self.dependencies.jobs.request_cancel(record.label_job_id)
            except ValueError as exc:
                raise AgentConflict(str(exc)) from exc
        if record.status == "exporting" and record.export_job_id is not None:
            try:
                self.dependencies.jobs.request_cancel(record.export_job_id)
            except ValueError as exc:
                raise AgentConflict(str(exc)) from exc
        if record.status == "calibrating":
            with self._calibration_lock:
                cancel_event = self._calibration_cancels.get(session_id)
            if cancel_event is not None:
                cancel_event.set()
        record = replace(record, status="cancelled", error=None)
        record = self._append(
            record,
            "assistant",
            "Run cancelled. Partial annotations are not packaged into a dataset.",
        )
        record = self.dependencies.store.save(record)
        return self.view(record)

    def open_sessions(self) -> list[AgentSessionRecord]:
        return self.dependencies.store.list_open()

    def resume_calibration(self, record: AgentSessionRecord) -> bool:
        """Restart a calibration that no thread in this process owns."""

        if record.status != "calibrating":
            return False
        with self._calibration_lock:
            if record.session_id in self._inflight:
                return False
            self._inflight.add(record.session_id)
        request = (record.spec or {}).get("calibration_request") or {}
        logger.info("resuming interrupted calibration", extra={"session_id": str(record.session_id)})
        self._calibration_executor.submit(
            self._run_calibration, record.tenant_id, record.session_id, request, raise_errors=False
        )
        return True

    def shutdown(self) -> None:
        for event in list(self._calibration_cancels.values()):
            event.set()
        self._calibration_executor.shutdown(wait=False, cancel_futures=True)

    def advance(self, session_id: UUID) -> AgentSessionView:
        record = self.dependencies.store.get_any(session_id)
        if record.status == "calibrating":
            self.resume_calibration(record)
            return self.view(record)
        if record.status == "labeling" and record.label_job_id is not None:
            job = self.dependencies.jobs.get(record.label_job_id)
            if job.status == "failed":
                record = replace(record, status="failed", error=job.error or "labeling failed")
                record = self._append(
                    record, "assistant", f"Labeling failed: {job.error or 'unknown error'}"
                )
                return self.view(self.dependencies.store.save(record))
            if job.status == "cancelled":
                record = replace(record, status="cancelled")
                record = self._append(record, "assistant", "Labeling was cancelled.")
                return self.view(self.dependencies.store.save(record))
            if job.status == "succeeded":
                try:
                    draft = session_draft(record)
                    class_names = ", ".join(item.name for item in draft.classes)
                    result = self.dependencies.dataset_builder.build_from_job(
                        tenant_id=record.tenant_id,
                        job_id=job.job_id,
                        name=f"{record.asset_name} — {class_names or record.prompt or 'dataset'}",
                        idempotency_key=f"agent-dataset:{record.session_id}",
                    )
                except Exception as exc:
                    logger.exception("agent dataset creation failed")
                    record = replace(record, status="failed", error=str(exc))
                    record = self._append(
                        record, "assistant", f"The dataset could not be built: {exc}"
                    )
                    return self.view(self.dependencies.store.save(record))
                current = self.dependencies.store.get_any(session_id)
                if current.status != "labeling":
                    return self.view(current)
                try:
                    export_format = export_format_for(session_draft(record))
                    export = self.dependencies.export_workflow.submit(
                        tenant_id=record.tenant_id,
                        version_id=result.record.version_id,
                        format=export_format,  # type: ignore[arg-type]
                        idempotency_key=f"agent-export:{record.session_id}",
                    )
                except Exception as exc:
                    logger.exception("agent export creation failed")
                    record = replace(record, status="failed", error=str(exc))
                    record = self._append(
                        record, "assistant", f"The dataset export could not be started: {exc}"
                    )
                    return self.view(self.dependencies.store.save(record))
                record = replace(
                    record,
                    status="exporting",
                    dataset_version_id=result.record.version_id,
                    export_job_id=export.job_id,
                    error=None,
                )
                record = self._append(
                    record,
                    "assistant",
                    f"Built a dataset with {result.image_count} image"
                    f"{'' if result.image_count == 1 else 's'} and "
                    f"{result.annotation_count} annotation"
                    f"{'' if result.annotation_count == 1 else 's'}. "
                    f"Packaging {_format_label(export_format)} now.",
                )
                return self.view(self.dependencies.store.save(record))
        if record.status == "exporting" and record.export_job_id is not None:
            job = self.dependencies.jobs.get(record.export_job_id)
            if job.status == "failed":
                record = replace(record, status="failed", error=job.error or "export failed")
                record = self._append(
                    record, "assistant", f"Dataset export failed: {job.error or 'unknown error'}"
                )
            elif job.status == "cancelled":
                record = replace(record, status="cancelled")
                record = self._append(record, "assistant", "Dataset export was cancelled.")
            elif job.status == "succeeded":
                record = replace(record, status="complete", error=None)
                record = self._append(
                    record,
                    "assistant",
                    f"The {_format_label(export_format_for(session_draft(record)))} dataset is "
                    "ready. Download the archive from the run panel.",
                )
            else:
                return self.view(record)
            current = self.dependencies.store.get_any(session_id)
            if current.status != "exporting":
                return self.view(current)
            return self.view(self.dependencies.store.save(record))
        return self.view(record)

    def view(self, record: AgentSessionRecord) -> AgentSessionView:
        label_status = None
        if record.label_job_id is not None:
            try:
                label_status = self.dependencies.jobs.get(record.label_job_id).status
            except KeyError:
                label_status = None
        export_status = None
        if record.export_job_id is not None:
            try:
                export_status = self.dependencies.jobs.get(record.export_job_id).status
            except KeyError:
                export_status = None
        dataset = None
        if record.dataset_version_id is not None:
            try:
                version = self.dependencies.export_workflow.dependencies.versions.get(
                    tenant_id=record.tenant_id, version_id=record.dataset_version_id
                )
                dataset = {
                    "name": version.name,
                    "image_count": len(version.dataset.images),
                    "annotation_count": len(version.dataset.annotations),
                    "categories": [
                        {"category_id": category.category_id, "name": category.name}
                        for category in version.dataset.categories
                    ],
                }
            except KeyError:
                dataset = None
        download_path = (
            f"/v1/dataset-exports/{record.export_job_id}/download"
            if record.export_job_id is not None and export_status == "succeeded"
            else None
        )
        return AgentSessionView(
            record=record,
            label_status=label_status,
            export_status=export_status,
            dataset=dataset,
            download_path=download_path,
        )

    def _asset_names(self, *, tenant_id: UUID, asset_ids: tuple[UUID, ...]) -> list[str]:
        if not asset_ids:
            return []
        known = {
            asset.asset_id: asset.name
            for asset in self.dependencies.registry.list_assets(tenant_id=tenant_id)
        }
        missing = [str(item) for item in asset_ids if item not in known]
        if missing:
            raise KeyError(f"unknown asset: {missing[0]}")
        return [known[item] for item in asset_ids]

    def _release_model(self, name: str) -> None:
        """Unload a managed model; it reloads lazily on its next use."""

        release = self.dependencies.release_models
        if release is None:
            return
        try:
            release(name)
        except Exception:
            logger.warning("could not release model %s", name, exc_info=True)

    def _preview_path(self, tenant_id: UUID, session_id: UUID) -> Path | None:
        root = self.dependencies.overlay_root
        if root is None:
            return None
        return root / str(tenant_id) / str(session_id) / "preview.json"

    def _save_preview(
        self,
        tenant_id: UUID,
        session_id: UUID,
        *,
        index: int,
        name: str,
        prompt: str,
        samples: list[CalibrationSample],
        raw: list[list[Any]],
    ) -> None:
        """Keep SAM 3's raw predictions so the browser can re-threshold them live.

        Each prediction is stored as score, box, and simplified polygons, so a
        threshold slider redraws instantly without running the model again.
        """

        path = self._preview_path(tenant_id, session_id)
        if path is None:
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"classes": []}
        except (OSError, ValueError):
            data = {"classes": []}
        entry = {
            "index": index,
            "name": name,
            "category_id": index + 1,
            "prompt": prompt,
            "samples": [
                {
                    "image_id": sample.image_id,
                    "width": sample.image.width,
                    "height": sample.image.height,
                    "predictions": [
                        {
                            "score": round(float(prediction.score), 4),
                            "box": [round(float(value), 1) for value in prediction.box],
                            "polygons": [
                                [round(value, 1) for value in polygon]
                                for polygon in (
                                    mask_polygons(prediction.mask) if prediction.mask is not None else ()
                                )
                            ],
                        }
                        for prediction in predictions
                    ],
                }
                for sample, predictions in zip(samples, raw)
            ],
        }
        classes = [item for item in data.get("classes", []) if item.get("index") != index]
        classes.append(entry)
        classes.sort(key=lambda item: item.get("index", 0))
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"classes": classes}), encoding="utf-8")
            tmp.replace(path)
        except OSError:
            logger.warning("could not save the calibration preview", exc_info=True)

    def calibration_preview(self, *, tenant_id: UUID, session_id: UUID) -> dict[str, Any]:
        self.dependencies.store.get(tenant_id=tenant_id, session_id=session_id)
        path = self._preview_path(tenant_id, session_id)
        if path is None or not path.is_file():
            raise KeyError("calibration preview not found")
        return json.loads(path.read_text(encoding="utf-8"))

    def calibration_sample_path(
        self, *, tenant_id: UUID, session_id: UUID, image_id: str
    ) -> Path:
        """The source image of one calibration sample (only samples in the preview)."""

        preview = self.calibration_preview(tenant_id=tenant_id, session_id=session_id)
        known = {
            sample["image_id"] for item in preview.get("classes", []) for sample in item.get("samples", [])
        }
        if image_id not in known:
            raise KeyError("calibration sample not found")
        record = self.dependencies.store.get(tenant_id=tenant_id, session_id=session_id)
        for image in self.dependencies.images.resolve_many(
            tenant_id=tenant_id, asset_ids=session_draft(record).asset_ids
        ):
            if image.image_id == image_id:
                return image.path
        raise KeyError("calibration sample not found")

    def _save_progress(
        self, tenant_id: UUID, session_id: UUID, attempts: list[dict[str, Any]]
    ) -> None:
        """Publish calibration attempts as they finish, for the live view.

        Re-reads the session so a concurrent cancel is never overwritten.
        """

        try:
            current = self.dependencies.store.get(tenant_id=tenant_id, session_id=session_id)
            if current.status != "calibrating":
                return
            self.dependencies.store.save(
                replace(current, calibration={"status": "running", "progress": list(attempts)})
            )
        except Exception:
            logger.warning("could not save calibration progress", exc_info=True)

    def _sample_images(
        self,
        *,
        tenant_id: UUID | None = None,
        asset_ids: tuple[str, ...] = (),
        sample_size: int,
        images: list | None = None,
        text: str | None = None,
    ):
        if images is None:
            images = self.dependencies.images.resolve_many(
                tenant_id=tenant_id, asset_ids=asset_ids
            )
        if not images:
            raise ValueError("asset has no image or video frames to label")
        count = max(1, min(sample_size, len(images)))
        if count >= len(images):
            return images
        get_embedding = getattr(self.dependencies.vectors, "get_embedding", None)
        embeddings: dict[str, Any] = {}
        if get_embedding is not None:
            for image in images:
                try:
                    vector = get_embedding(image.image_id)
                except Exception:
                    logger.warning(
                        "could not read a stored embedding for calibration sampling",
                        exc_info=True,
                    )
                    vector = None
                if vector is not None:
                    embeddings[image.image_id] = vector
        text_vector = None
        embed_text = self.dependencies.text_embedder
        if text and embed_text is not None and len(embeddings) == len(images):
            try:
                text_vector = embed_text(text)
            except Exception:
                logger.warning("could not embed the class text for sampling", exc_info=True)
        return select_samples_for_target(
            images, count=count, embeddings=embeddings, text_embedding=text_vector
        )

    def _release_validator(
        self, selection: ValidatorSelection, record: AgentSessionRecord
    ) -> tuple[AgentSessionRecord, dict[str, Any]]:
        release = selection.release
        if release is None:
            return record, {"vlm_unloaded": False, "validator_unload_sec": 0.0}
        started = time.perf_counter()
        try:
            unloaded = bool(release())
        except Exception:
            logger.warning("validator unload after calibration failed", exc_info=True)
            unloaded = False
        timing = {
            "vlm_unloaded": unloaded,
            "validator_unload_sec": round(time.perf_counter() - started, 2),
        }
        if unloaded:
            record = self._append(
                record,
                "assistant",
                "Unloaded the vision validator after calibration; whole-asset labeling "
                "continues with SAM 3 only.",
            )
        return record, timing

    def _select_validator(
        self, *, tenant_id: UUID, session_id: UUID
    ) -> ValidatorSelection:
        if self.dependencies.validator is not None:
            return ValidatorSelection(
                validator=self.dependencies.validator,
                mode="custom",
            )
        if self.dependencies.validator_factory is not None:
            return self.dependencies.validator_factory(tenant_id, session_id)
        return any_prediction_selection()

    def overlay_path(
        self, *, tenant_id: UUID, session_id: UUID, overlay_id: str
    ) -> Path:
        if self.dependencies.overlay_root is None or not _OVERLAY_ID.fullmatch(overlay_id):
            raise KeyError("calibration overlay not found")
        self.dependencies.store.get(tenant_id=tenant_id, session_id=session_id)
        path = (
            self.dependencies.overlay_root
            / str(tenant_id)
            / str(session_id)
            / overlay_id
        )
        if not path.is_file():
            raise KeyError("calibration overlay not found")
        return path

    def _calibrate_with_timeout(
        self,
        calibrate,
        *,
        cancel_event: threading.Event | None = None,
        budget: float | None = None,
    ):
        timeout = budget or self.dependencies.calibration_timeout_sec
        result_queue: queue.Queue[tuple[bool, Any]] = queue.Queue(maxsize=1)

        def target() -> None:
            try:
                result_queue.put((True, calibrate()))
            except BaseException as exc:  # surfaced to the caller below
                result_queue.put((False, exc))

        thread = threading.Thread(target=target, name="agent-calibration", daemon=True)
        thread.start()
        thread.join(timeout)
        if thread.is_alive():
            if cancel_event is not None:
                cancel_event.set()
            thread.join(_CALIBRATION_THREAD_GRACE_SEC)
            if thread.is_alive():
                logger.warning(
                    "calibration worker is still inside an inference call after timeout"
                )
            raise CalibrationTimeoutError(
                f"Calibration timed out after {timeout:g}s. Try fewer sample images, "
                "a longer timeout, or confirm with a fixed threshold."
            )
        succeeded, value = result_queue.get_nowait()
        if not succeeded:
            raise value
        return value

    @staticmethod
    def _load_image(path) -> Image.Image:
        with Image.open(path) as source:
            return source.convert("RGB")

    @staticmethod
    def _append(
        record: AgentSessionRecord, role: str, content: str
    ) -> AgentSessionRecord:
        message = AgentMessage.create(role, content)  # type: ignore[arg-type]
        return replace(record, messages=record.messages + (message,))


def _calibration_payload(
    result: CalibrationResult,
    selection: ValidatorSelection,
    *,
    timings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    timings = timings or {}
    return {
        "status": result.status,
        "prompt": result.prompt,
        "threshold": result.threshold,
        "validator_mode": selection.mode,
        "validator_model": selection.model,
        "notice": selection.current_notice(),
        "attempts": [_attempt_payload(attempt) for attempt in result.attempts],
        "forward_passes": result.forward_passes,
        "reason": result.reason,
        "timings": timings,
        "vlm_unloaded_after_calibration": bool(timings.get("vlm_unloaded", False)),
    }


def _attempt_payload(attempt) -> dict[str, Any]:
    return {
        "threshold": attempt.threshold,
        "sample_count": attempt.sample_count,
        "accepted_count": attempt.accepted_count,
        "positive_count": attempt.positive_count,
        "agreement": attempt.agreement,
        "verdicts": [
            {
                "image_id": verdict.image_id,
                "accepted": verdict.accepted,
                "confidence": verdict.confidence,
                "reasoning": verdict.reasoning,
                "overlay_id": verdict.overlay_id,
                "prediction_count": verdict.prediction_count,
                "target_present": verdict.target_present,
            }
            for verdict in attempt.verdicts
        ],
    }


def _combined_calibration(per_class: list[dict[str, Any]]) -> dict[str, Any]:
    """One class keeps the historical payload; several get an overall summary.

    The top level mirrors the first class so older clients still render it;
    ``classes`` carries every class's attempts and verdicts.
    """

    if len(per_class) == 1:
        return {**per_class[0], "classes": per_class}
    statuses = {item["status"] for item in per_class}
    return {
        **per_class[0],
        "status": "calibrated" if statuses == {"calibrated"} else "no_acceptable_threshold",
        "prompt": ", ".join(str(item["prompt"]) for item in per_class),
        "classes": per_class,
    }


def _dedupe_assets(values: tuple[UUID, ...]) -> tuple[UUID, ...]:
    seen: list[UUID] = []
    for value in values:
        if value not in seen:
            seen.append(value)
    return tuple(seen)


def _override_patch(
    *,
    task_type: str | None,
    target_objects: tuple[str, ...],
    visual_attributes: tuple[str, ...],
    sample_size: int | None,
    max_iterations: int | None,
) -> DraftPatch | None:
    """Explicit API fields (v1 clients) win over what the message says."""

    if not any((task_type, target_objects, visual_attributes, sample_size, max_iterations)):
        return None
    classes = None
    if target_objects:
        classes = tuple(
            ClassSpec(name=name, attributes=tuple(visual_attributes)) for name in target_objects
        )
    return DraftPatch(
        task_type=task_type,  # type: ignore[arg-type]
        classes=classes,
        sample_size=sample_size,
        max_iterations=max_iterations,
    )


def _manual_calibration_payload(*, prompt: str, threshold: float) -> dict[str, Any]:
    return {
        "status": "manual",
        "prompt": prompt,
        "threshold": threshold,
        "validator_mode": "manual",
        "validator_model": None,
        "notice": None,
        "attempts": [],
        "reason": "threshold selected by the user",
    }


def _apply_overrides(
    spec: TaskSpec,
    *,
    task_type: str | None,
    target_objects: tuple[str, ...],
    visual_attributes: tuple[str, ...],
    sample_size: int | None,
    max_iterations: int | None,
) -> TaskSpec:
    return TaskSpec(
        target_objects=target_objects or spec.target_objects,
        task_type=task_type or spec.task_type,  # type: ignore[arg-type]
        visual_attributes=visual_attributes or spec.visual_attributes,
        sample_size=sample_size or spec.sample_size,
        max_iterations=max_iterations or spec.max_iterations,
    )
