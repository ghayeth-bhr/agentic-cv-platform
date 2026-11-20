"""Labeling health monitor: rolling metrics, rules, and pause-and-ask alerts.

Calibration sees a handful of samples; the whole asset can behave differently
(a prompt that drifts on other scenes, a threshold too permissive for crowded
frames). The monitor watches every labeled image and, when a rule fires,
pauses the job so the agent can ask the user in chat instead of silently
producing a bad dataset. Rules are deterministic; an acknowledged alert does
not fire again for the same job.

Health is kept as a small JSON file per job (local-first; one host).
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import UUID

logger = logging.getLogger("agentic_cv.monitor")
_ATTEMPTS = 20
_RETRY_SEC = 0.02
MIN_TOTAL_FOR_ALERTS = 12
MIN_SHARE_BEFORE_ALERTS = 0.15
MIN_IMAGES_BEFORE_ALERTS = 8
MOSTLY_EMPTY_RATE = 0.7
CROWDED_MEAN = 25.0


@dataclass
class _ClassStats:
    name: str
    detections: int = 0
    images_with: int = 0
    score_sum: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "detections": self.detections,
            "images_with": self.images_with,
            "mean_score": round(self.score_sum / self.detections, 3) if self.detections else None,
        }


@dataclass
class LabelHealthMonitor:
    """Feed it one record per labeled image; ask it whether to pause."""

    total: int
    class_names: dict[int, str]
    acknowledged: set[str] = field(default_factory=set)
    processed: int = 0
    empty_images: int = 0
    started: float = field(default_factory=time.monotonic)
    resumed_from: int = 0
    classes: dict[int, _ClassStats] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.classes = {
            category_id: _ClassStats(name=name) for category_id, name in self.class_names.items()
        }

    def record(self, detections: dict[int, list[float]]) -> None:
        """``detections`` maps category id to the scores found on one image."""

        self.processed += 1
        found = 0
        for category_id, scores in detections.items():
            stats = self.classes.setdefault(category_id, _ClassStats(name=str(category_id)))
            if scores:
                stats.images_with += 1
                stats.detections += len(scores)
                stats.score_sum += float(sum(scores))
                found += len(scores)
        if not found:
            self.empty_images += 1

    def check(self) -> dict[str, Any] | None:
        """The first rule that fires and has not been acknowledged, or None."""

        if self.total < MIN_TOTAL_FOR_ALERTS:
            return None
        needed = max(MIN_IMAGES_BEFORE_ALERTS, math.ceil(self.total * MIN_SHARE_BEFORE_ALERTS))
        if self.processed < needed:
            return None
        empty_rate = self.empty_images / self.processed
        candidates: list[dict[str, Any]] = []
        if empty_rate > MOSTLY_EMPTY_RATE:
            candidates.append({
                "code": "mostly_empty",
                "message": (
                    f"{round(empty_rate * 100)}% of the first {self.processed} images have no "
                    "detections at all. The calibration samples may not represent this asset, or "
                    "the prompt does not fit these scenes."
                ),
            })
        # "X is missing" only says something new when other classes are being
        # found; an all-empty run is already the "mostly empty" alert.
        any_found = any(stats.detections for stats in self.classes.values())
        for stats in self.classes.values():
            if stats.detections == 0 and any_found:
                candidates.append({
                    "code": f"class_missing:{stats.name}",
                    "message": (
                        f"“{stats.name}” has not been found in any of the first "
                        f"{self.processed} images."
                    ),
                })
        for stats in self.classes.values():
            if stats.images_with and stats.detections / stats.images_with > CROWDED_MEAN:
                candidates.append({
                    "code": f"crowded:{stats.name}",
                    "message": (
                        f"“{stats.name}” averages {stats.detections / stats.images_with:.0f} "
                        "detections per image, which usually means the threshold is too low "
                        "and background is being labeled."
                    ),
                })
        for alert in candidates:
            if alert["code"] not in self.acknowledged:
                return alert
        return None

    def snapshot(self, *, status: str, alert: dict[str, Any] | None = None) -> dict[str, Any]:
        elapsed = max(1e-6, time.monotonic() - self.started)
        done_now = self.processed - self.resumed_from
        rate = done_now / elapsed if done_now > 0 else 0.0
        remaining = max(0, self.total - self.processed)
        return {
            "status": status,
            "processed": self.processed,
            "total": self.total,
            "images_per_sec": round(rate, 3),
            "eta_sec": round(remaining / rate, 1) if rate > 0 else None,
            "empty_images": self.empty_images,
            "empty_rate": round(self.empty_images / self.processed, 3) if self.processed else 0.0,
            "classes": [stats.as_dict() for stats in self.classes.values()],
            "alert": alert,
            "acknowledged": sorted(self.acknowledged),
        }


class LabelHealthStore:
    """One JSON file per label job under ``<data>/label-health``."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._lock = threading.Lock()

    def _path(self, job_id: UUID) -> Path:
        return self.root / f"{job_id}.json"

    def read(self, job_id: UUID) -> dict[str, Any] | None:
        for attempt in range(_ATTEMPTS):
            try:
                return json.loads(self._path(job_id).read_text(encoding="utf-8"))
            except FileNotFoundError:
                return None
            except (OSError, ValueError):
                # A writer is replacing the file (Windows); try again shortly.
                time.sleep(_RETRY_SEC)
        return None

    def write(self, job_id: UUID, health: dict[str, Any]) -> bool:
        """Best effort: health is advisory and must never fail a labeling run.

        On Windows a file cannot be replaced while a reader has it open, so
        the atomic rename is retried briefly before giving up.
        """

        with self._lock:
            try:
                self.root.mkdir(parents=True, exist_ok=True)
                path = self._path(job_id)
                tmp = path.with_suffix(f".{threading.get_ident()}.tmp")
                tmp.write_text(json.dumps(health), encoding="utf-8")
            except OSError:
                logger.warning("could not write labeling health", exc_info=True)
                return False
            for attempt in range(_ATTEMPTS):
                try:
                    tmp.replace(path)
                    return True
                except PermissionError:
                    time.sleep(_RETRY_SEC)
                except OSError:
                    break
            logger.warning("could not publish labeling health for job %s", job_id)
            try:
                tmp.unlink()
            except OSError:
                pass
            return False

    def acknowledge(self, job_id: UUID, code: str) -> None:
        health = self.read(job_id) or {}
        acknowledged = set(health.get("acknowledged", []))
        acknowledged.add(code)
        health["acknowledged"] = sorted(acknowledged)
        health["alert"] = None
        health["status"] = "resuming"
        self.write(job_id, health)

    def acknowledged(self, job_id: UUID) -> set[str]:
        return set((self.read(job_id) or {}).get("acknowledged", []))
