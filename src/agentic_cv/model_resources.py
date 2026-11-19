"""Serialized access to local GPU models with memory-pressure unloading.

The providers remain independently usable. When the application creates them it
passes one shared manager so that:

* only one model runs inference at a time (no concurrent CUDA allocations),
* before a model loads, an already-loaded model is closed when the requested
  estimate plus headroom does not fit in free device memory,
* a model that has been idle beyond the configured TTL is closed on the next
  model lease,
* ``/api-info`` can report which models are resident and what the device says.

No target is invented here: measurements live in ``docs/PERFORMANCE.md``.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Protocol


class ManagedModel(Protocol):
    def close(self) -> None: ...


MemoryProbe = Callable[[], tuple[float, float] | None]
EmptyCache = Callable[[], None]


@dataclass
class _Entry:
    name: str
    provider: ManagedModel
    estimated_mb: int
    last_used: float | None = None
    loaded_at: float | None = None


@dataclass(frozen=True, kw_only=True)
class ModelResourceStatus:
    device: str
    locked: bool
    free_mb: float | None
    total_mb: float | None
    models: dict[str, dict[str, Any]]
    events: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, Any]:
        return {
            "device": self.device,
            "locked": self.locked,
            "free_mb": self.free_mb,
            "total_mb": self.total_mb,
            "models": self.models,
            "events": list(self.events),
        }


class ModelResourceManager:
    """One lock + memory-aware unload policy shared by local model providers."""

    def __init__(
        self,
        *,
        device: str = "cuda",
        headroom_mb: int = 512,
        idle_unload_sec: float = 300.0,
        estimated_default_mb: int = 2048,
        memory_probe: MemoryProbe | None = None,
        empty_cache: EmptyCache | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if headroom_mb < 0:
            raise ValueError("headroom_mb must not be negative")
        if idle_unload_sec < 0:
            raise ValueError("idle_unload_sec must not be negative")
        self.device = device
        self.headroom_mb = headroom_mb
        self.idle_unload_sec = idle_unload_sec
        self.estimated_default_mb = estimated_default_mb
        self._memory_probe = memory_probe
        self._empty_cache = empty_cache
        self._clock = clock
        self._lock = threading.RLock()
        self._entries: dict[str, _Entry] = {}
        self._events: list[str] = []
        self._active = 0

    @property
    def lock(self) -> threading.RLock:
        return self._lock

    def register(
        self,
        name: str,
        provider: ManagedModel,
        *,
        estimated_mb: int | None = None,
    ) -> None:
        if not name.strip():
            raise ValueError("model name must not be empty")
        estimate = self.estimated_default_mb if estimated_mb is None else estimated_mb
        if estimate <= 0:
            raise ValueError("estimated_mb must be positive")
        with self._lock:
            self._entries[name] = _Entry(
                name=name, provider=provider, estimated_mb=int(estimate)
            )

    def unregister(self, name: str) -> None:
        with self._lock:
            self._entries.pop(name, None)

    @contextmanager
    def lease(self, name: str) -> Iterator[None]:
        """Hold the shared GPU lock for one inference and manage residency."""

        with self._lock:
            entry = self._entries.get(name)
            if entry is None:
                raise KeyError(f"unknown managed model: {name}")
            self._sweep(except_name=name)
            self._make_room(entry)
            self._active += 1
            try:
                yield
            finally:
                self._active = max(0, self._active - 1)
                entry.last_used = self._clock()

    def close(self, name: str, *, reason: str = "close") -> bool:
        with self._lock:
            entry = self._entries.get(name)
            if entry is None:
                return False
            was_loaded = _loaded(entry.provider)
            try:
                entry.provider.close()
            finally:
                entry.last_used = None
                entry.loaded_at = None
            if was_loaded:
                self._record(f"{name}: closed ({reason})")
            return was_loaded

    def close_all(self, *, reason: str = "shutdown") -> None:
        with self._lock:
            for name in tuple(self._entries):
                self.close(name, reason=reason)
            self._empty()

    def status(self) -> ModelResourceStatus:
        with self._lock:
            self._sweep(except_name=None)
            free_mb, total_mb = self._memory()
            now = self._clock()
            models = {}
            for name, entry in self._entries.items():
                last_used = entry.last_used
                models[name] = {
                    "loaded": _loaded(entry.provider),
                    "estimated_mb": entry.estimated_mb,
                    "load_seconds": _load_seconds(entry.provider),
                    "idle_sec": (
                        round(now - last_used, 2) if last_used is not None else None
                    ),
                }
            return ModelResourceStatus(
                device=self.device,
                locked=self._active > 0,
                free_mb=free_mb,
                total_mb=total_mb,
                models=models,
                events=tuple(self._events[-20:]),
            )

    def _make_room(self, target: _Entry) -> None:
        free_mb, _ = self._memory()
        if free_mb is None:
            return
        required = target.estimated_mb + self.headroom_mb
        if free_mb >= required:
            return
        for name, entry in self._entries.items():
            if name == target.name or not _loaded(entry.provider):
                continue
            self.close(name, reason=f"make room for {target.name}")
        self._empty()

    def _sweep(self, *, except_name: str | None) -> None:
        if self.idle_unload_sec <= 0:
            return
        now = self._clock()
        for name, entry in self._entries.items():
            if name == except_name or not _loaded(entry.provider):
                continue
            if entry.last_used is None:
                continue
            if now - entry.last_used >= self.idle_unload_sec:
                self.close(name, reason="idle TTL")

    def _memory(self) -> tuple[float | None, float | None]:
        if not self.device.startswith("cuda"):
            return None, None
        if self._memory_probe is not None:
            value = self._memory_probe()
            return value if value is not None else (None, None)
        try:
            import torch
        except ImportError:
            return None, None
        if not torch.cuda.is_available():
            return None, None
        free, total = torch.cuda.mem_get_info()
        return round(free / 1024**2, 1), round(total / 1024**2, 1)

    def _empty(self) -> None:
        if self._empty_cache is not None:
            self._empty_cache()
            return
        if not self.device.startswith("cuda"):
            return
        try:
            import torch
        except ImportError:
            return
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _record(self, event: str) -> None:
        self._events.append(f"{round(self._clock(), 3)} {event}")
        del self._events[:-20]


def _load_seconds(provider: Any) -> float:
    value = getattr(provider, "load_seconds", None)
    if value is None:
        return 0.0
    try:
        return round(float(value), 2)
    except (TypeError, ValueError):
        return 0.0


def _loaded(provider: Any) -> bool:
    value = getattr(provider, "is_loaded", None)
    if value is not None:
        try:
            return bool(value() if callable(value) else value)
        except Exception:
            return False
    return getattr(provider, "_model", None) is not None
