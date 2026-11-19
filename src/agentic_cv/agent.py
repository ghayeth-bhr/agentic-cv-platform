"""Agent chat: plan a task, calibrate it, label an asset, and package a dataset."""

from __future__ import annotations

import logging
import os
import queue
import re
import threading
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
from .calibration_samples import select_calibration_images
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
    r"(?:me\s+)?(?:a|an|the)?\s*"
    r"(?:(?:detection|segmentation)\s+)?"
    r"(?:dataset\s+)?"
    r"(?:of|for|with|showing|that\s+(?:labels|detects|segments))?\s*",
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


class OllamaTaskPlanner:
    """Try a local text model, then fall back to the deterministic planner."""

    def __init__(self, client: OllamaTextClient, fallback: TaskPlanner | None = None) -> None:
        self.client = client
        self.fallback = fallback or RulesTaskPlanner()

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


@dataclass(frozen=True, kw_only=True)
class AgentSessionView:
    record: AgentSessionRecord
    label_status: str | None
    export_status: str | None
    dataset: dict[str, Any] | None
    download_path: str | None


class AgentService:
    def __init__(self, dependencies: AgentServiceDependencies) -> None:
        self.dependencies = dependencies
        self._calibration_lock = threading.Lock()
        self._calibration_cancels: dict[UUID, threading.Event] = {}
        if dependencies.calibration_timeout_sec <= 0:
            raise ValueError("calibration_timeout_sec must be greater than 0")
        if not dependencies.thresholds:
            raise ValueError("at least one calibration threshold is required")
        if not 0 < dependencies.minimum_agreement <= 1:
            raise ValueError("minimum_agreement must be between 0 and 1")

    def create_session(
        self,
        *,
        tenant_id: UUID,
        asset_id: UUID,
        message: str,
        task_type: str | None = None,
        target_objects: tuple[str, ...] = (),
        visual_attributes: tuple[str, ...] = (),
        sample_size: int | None = None,
        max_iterations: int | None = None,
    ) -> AgentSessionView:
        if not message.strip():
            raise ValueError("message must not be empty")
        asset_name = self._asset_name(tenant_id=tenant_id, asset_id=asset_id)
        planned = self.dependencies.planner.plan(
            message,
            task_type=task_type,
            target_objects=target_objects,
            visual_attributes=visual_attributes,
            sample_size=sample_size,
            max_iterations=max_iterations,
        )
        prompt = prompt_for_spec(planned.spec)
        reply = (
            f"I read this as a {planned.spec.task_type} task for “{prompt}”. "
            f"Confirm to calibrate on up to {planned.spec.sample_size} sample image"
            f"{'' if planned.spec.sample_size == 1 else 's'}, then label “{asset_name}”."
        )
        if planned.note:
            reply = f"{reply}\n\n{planned.note}"
        record = new_session_record(
            tenant_id=tenant_id,
            asset_id=asset_id,
            asset_name=asset_name,
            message=message.strip(),
            spec=planned.spec.to_dict(),
            planner_mode=planned.mode,
            planner_note=planned.note,
            prompt=prompt,
            messages=(
                AgentMessage.create("user", message.strip()),
                AgentMessage.create("assistant", reply),
            ),
        )
        record = self.dependencies.store.create(record)
        return self.view(record)

    def get_session(self, *, tenant_id: UUID, session_id: UUID) -> AgentSessionView:
        return self.view(self.dependencies.store.get(tenant_id=tenant_id, session_id=session_id))

    def list_sessions(
        self, *, tenant_id: UUID, limit: int = 50
    ) -> list[AgentSessionView]:
        return [
            self.view(record)
            for record in self.dependencies.store.list(tenant_id=tenant_id, limit=limit)
        ]

    def confirm(
        self,
        *,
        tenant_id: UUID,
        session_id: UUID,
        threshold: float | None = None,
        thresholds: tuple[float, ...] | None = None,
        minimum_agreement: float | None = None,
    ) -> AgentSessionView:
        record = self.dependencies.store.get(tenant_id=tenant_id, session_id=session_id)
        if record.status not in ("awaiting_confirmation", "needs_attention"):
            raise AgentConflict(
                f"session is {record.status}; only a planned session can be confirmed"
            )
        if record.spec is None:
            raise AgentConflict("session has no task specification")
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

        record = replace(record, status="calibrating", error=None)
        record = self.dependencies.store.save(record)
        try:
            selection = self._select_validator(
                tenant_id=tenant_id, session_id=session_id
            )
            initial_notice = selection.current_notice()
            if initial_notice:
                record = self._append(record, "assistant", initial_notice)
                record = self.dependencies.store.save(record)
            label_images = self._sample_images(
                tenant_id=tenant_id,
                asset_id=record.asset_id,
                sample_size=TaskSpec.from_dict(record.spec).sample_size,
            )
            total_images = len(
                self.dependencies.images.resolve(
                    tenant_id=tenant_id, asset_id=str(record.asset_id)
                )
            )
            if threshold is None:
                samples = [
                    CalibrationSample(
                        image_id=item.image_id,
                        image=self._load_image(item.path),
                    )
                    for item in label_images
                ]
                cancel_event = threading.Event()
                with self._calibration_lock:
                    self._calibration_cancels[session_id] = cancel_event
                try:
                    result = self._calibrate_with_timeout(
                        lambda: Sam3Calibrator(
                            self.dependencies.sam3_provider,
                            selection.validator,
                            minimum_agreement=agreement,
                            max_iterations=TaskSpec.from_dict(record.spec).max_iterations,
                        ).calibrate(
                            samples,
                            prompt=record.prompt or "",
                            thresholds=list(selected_thresholds),
                            should_cancel=cancel_event.is_set,
                        ),
                        cancel_event=cancel_event,
                    )
                finally:
                    with self._calibration_lock:
                        self._calibration_cancels.pop(session_id, None)
                calibration = _calibration_payload(result, selection)
                final_notice = selection.current_notice()
                if final_notice and final_notice != initial_notice:
                    record = self._append(record, "assistant", final_notice)
                    record = self.dependencies.store.save(record)
                if result.status != "calibrated" or result.threshold is None:
                    record = replace(
                        record,
                        status="needs_attention",
                        calibration=calibration,
                        error=(
                            "Calibration did not find a threshold with enough sample agreement. "
                            "Retry with different samples, or run with a fixed threshold."
                        ),
                    )
                    record = self._append(
                        record,
                        "assistant",
                        "Calibration did not reach agreement. I can retry, or you can run at "
                        "a fixed threshold such as 0.50.",
                    )
                    return self.view(self.dependencies.store.save(record))
                chosen_threshold = result.threshold
            else:
                chosen_threshold = threshold
                calibration = _manual_calibration_payload(
                    prompt=record.prompt or "", threshold=threshold
                )
                samples = []

            # A cancel request can arrive while calibration is running. Do not
            # start labeling after the user has already cancelled the run.
            current = self.dependencies.store.get(
                tenant_id=tenant_id, session_id=session_id
            )
            if current.status != "calibrating":
                return self.view(current)

            job = self.dependencies.labels.submit(
                tenant_id=tenant_id,
                asset_id=str(record.asset_id),
                prompt=record.prompt or "object",
                threshold=chosen_threshold,
                task=TaskSpec.from_dict(record.spec).task_type,
                category_id=1,
                category_name=(record.prompt or "object")[:80],
                idempotency_key=f"agent-label:{record.session_id}",
            )
            record = replace(
                record,
                status="labeling",
                threshold=chosen_threshold,
                calibration=calibration,
                label_job_id=job.job_id,
                error=None,
            )
            if threshold is None:
                sample_count = len(samples)
                message = (
                    f"Calibrated at {chosen_threshold:.2f} on {sample_count} sample image"
                    f"{'' if sample_count == 1 else 's'}. Labeling {total_images} image"
                    f"{'' if total_images == 1 else 's'} in the background."
                )
            else:
                message = (
                    f"Using the fixed threshold {chosen_threshold:.2f}. Labeling "
                    f"{total_images} image{'' if total_images == 1 else 's'} in the background."
                )
            record = self._append(record, "assistant", message)
            record = self.dependencies.store.save(record)
            return self.view(record)
        except CalibrationTimeoutError as exc:
            record = replace(record, status="failed", error=str(exc))
            record = self._append(record, "assistant", str(exc))
            self.dependencies.store.save(record)
            raise
        except CalibrationCancelled as exc:
            current = self.dependencies.store.get(
                tenant_id=tenant_id, session_id=session_id
            )
            if current.status == "cancelled":
                return self.view(current)
            record = replace(record, status="failed", error=str(exc))
            record = self._append(record, "assistant", str(exc))
            self.dependencies.store.save(record)
            raise
        except ModelUnavailableError as exc:
            record = replace(record, status="failed", error=str(exc))
            record = self._append(record, "assistant", str(exc))
            self.dependencies.store.save(record)
            raise
        except Exception as exc:
            record = replace(record, status="failed", error=str(exc))
            record = self._append(record, "assistant", f"The labeling run could not start: {exc}")
            self.dependencies.store.save(record)
            raise

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

    def advance(self, session_id: UUID) -> AgentSessionView:
        record = self.dependencies.store.get_any(session_id)
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
                    result = self.dependencies.dataset_builder.build_from_job(
                        tenant_id=record.tenant_id,
                        job_id=job.job_id,
                        name=f"{record.asset_name} — {record.prompt or 'dataset'}",
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
                    export = self.dependencies.export_workflow.submit(
                        tenant_id=record.tenant_id,
                        version_id=result.record.version_id,
                        format="coco",
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
                    f"{'' if result.annotation_count == 1 else 's'}. Packaging COCO now.",
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
                    "The COCO dataset is ready. Download the archive from the run panel.",
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

    def _asset_name(self, *, tenant_id: UUID, asset_id: UUID) -> str:
        for asset in self.dependencies.registry.list_assets(tenant_id=tenant_id):
            if asset.asset_id == asset_id:
                return asset.name
        raise KeyError(f"unknown asset: {asset_id}")

    def _sample_images(self, *, tenant_id: UUID, asset_id: UUID, sample_size: int):
        images = self.dependencies.images.resolve(
            tenant_id=tenant_id, asset_id=str(asset_id)
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
        return select_calibration_images(images, count=count, embeddings=embeddings)

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
        self, calibrate, *, cancel_event: threading.Event | None = None
    ):
        timeout = self.dependencies.calibration_timeout_sec
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
    result: CalibrationResult, selection: ValidatorSelection
) -> dict[str, Any]:
    return {
        "status": result.status,
        "prompt": result.prompt,
        "threshold": result.threshold,
        "validator_mode": selection.mode,
        "validator_model": selection.model,
        "notice": selection.current_notice(),
        "attempts": [
            {
                "threshold": attempt.threshold,
                "sample_count": attempt.sample_count,
                "accepted_count": attempt.accepted_count,
                "agreement": attempt.agreement,
                "verdicts": [
                    {
                        "image_id": verdict.image_id,
                        "accepted": verdict.accepted,
                        "confidence": verdict.confidence,
                        "reasoning": verdict.reasoning,
                        "overlay_id": verdict.overlay_id,
                    }
                    for verdict in attempt.verdicts
                ],
            }
            for attempt in result.attempts
        ],
        "reason": result.reason,
    }


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
