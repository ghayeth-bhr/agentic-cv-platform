"""CPU video frame extraction with timestamp and provenance records."""

from __future__ import annotations

import json
import logging
import os
import shutil
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger("agentic_cv.video")

_CAP_PROP_POS_MSEC = 0
_CAP_PROP_FPS = 5
_MANIFEST = "frames.json"
_MANIFEST_VERSION = 1
_cache_locks: dict[str, threading.Lock] = {}
_cache_locks_guard = threading.Lock()


@dataclass(frozen=True, kw_only=True)
class FrameExtractionConfig:
    frame_stride: int = 1
    max_frames: int | None = None
    # When set, sample by time: the stride becomes round(video_fps / target_fps).
    target_fps: float | None = None

    def __post_init__(self) -> None:
        if self.frame_stride < 1:
            raise ValueError("frame_stride must be at least 1")
        if self.max_frames is not None and self.max_frames < 1:
            raise ValueError("max_frames must be at least 1 when provided")
        if self.target_fps is not None and self.target_fps <= 0:
            raise ValueError("target_fps must be greater than 0 when provided")


def default_video_config() -> FrameExtractionConfig:
    """Time-based sampling shared by search indexing and labeling.

    ``AGENTIC_CV_VIDEO_SAMPLE_FPS`` (default 1 frame per second) and
    ``AGENTIC_CV_VIDEO_MAX_FRAMES`` (default 1000 per video; 0 = unlimited).
    """

    try:
        fps = float(os.getenv("AGENTIC_CV_VIDEO_SAMPLE_FPS", "1.0"))
    except ValueError:
        fps = 1.0
    try:
        max_frames = int(os.getenv("AGENTIC_CV_VIDEO_MAX_FRAMES", "1000"))
    except ValueError:
        max_frames = 1000
    return FrameExtractionConfig(
        frame_stride=30,  # used only when the container reports no fps
        target_fps=fps if fps > 0 else 1.0,
        max_frames=max_frames if max_frames > 0 else None,
    )


@dataclass(frozen=True, kw_only=True)
class ExtractedFrame:
    frame_index: int
    timestamp_ms: float
    image_path: Path


CaptureFactory = Callable[[str], Any]
ImageWriter = Callable[[str, Any], bool]


def _default_capture_factory(path: str) -> Any:
    import cv2

    return cv2.VideoCapture(path)


def _default_image_writer(path: str, frame: Any) -> bool:
    import cv2

    return bool(cv2.imwrite(path, frame))


def extract_frames(
    video_path: Path,
    output_dir: Path,
    *,
    config: FrameExtractionConfig | None = None,
    capture_factory: CaptureFactory | None = None,
    image_writer: ImageWriter | None = None,
) -> list[ExtractedFrame]:
    config = config or FrameExtractionConfig()
    capture = (capture_factory or _default_capture_factory)(str(video_path))
    writer = image_writer or _default_image_writer
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"unable to open video: {video_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    extracted: list[ExtractedFrame] = []
    frame_index = 0
    stride = _effective_stride(capture, config)
    truncated = False
    try:
        while True:
            if config.max_frames is not None and len(extracted) >= config.max_frames:
                truncated = True
                break
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % stride == 0:
                image_path = output_dir / f"frame_{frame_index:08d}.jpg"
                if not writer(str(image_path), frame):
                    raise IOError(f"unable to write extracted frame: {image_path}")
                timestamp_ms = float(capture.get(_CAP_PROP_POS_MSEC))
                extracted.append(
                    ExtractedFrame(
                        frame_index=frame_index,
                        timestamp_ms=timestamp_ms,
                        image_path=image_path,
                    )
                )
            frame_index += 1
    finally:
        capture.release()
    if truncated:
        logger.warning(
            "video sampling reached max_frames; any later frames are not included",
            extra={"video": str(video_path), "max_frames": config.max_frames, "stride": stride},
        )
    return extracted


def _effective_stride(capture: Any, config: FrameExtractionConfig) -> int:
    if config.target_fps is None:
        return config.frame_stride
    try:
        fps = float(capture.get(_CAP_PROP_FPS))
    except Exception:
        fps = 0.0
    if not fps or fps != fps or fps <= 0 or fps > 1000:
        return config.frame_stride
    return max(1, round(fps / config.target_fps))


def frames_dir_for(data_dir: Path, asset_key: str, source_key: str) -> Path:
    """One frame cache per video source, shared by indexing and labeling."""

    return data_dir / "derived" / "frames" / asset_key / source_key


def extract_frames_cached(
    video_path: Path,
    output_dir: Path,
    *,
    config: FrameExtractionConfig | None = None,
    capture_factory: CaptureFactory | None = None,
    image_writer: ImageWriter | None = None,
) -> list[ExtractedFrame]:
    """Extract once; reuse the frames while the video and config are unchanged.

    Label image ids embed the frame index, so search vectors, calibration
    samples, label runs, and previews must all see the same frames. The cache
    also stops the label resolver from decoding the video on every call.
    """

    config = config or default_video_config()
    stat = video_path.stat()
    key = {
        "version": _MANIFEST_VERSION,
        "video_size": stat.st_size,
        "video_mtime_ns": stat.st_mtime_ns,
        "config": asdict(config),
    }
    manifest_path = output_dir / _MANIFEST
    with _lock_for(output_dir):
        cached = _read_manifest(manifest_path, key)
        if cached is not None:
            return cached
        if output_dir.exists():
            shutil.rmtree(output_dir)
        frames = extract_frames(
            video_path,
            output_dir,
            config=config,
            capture_factory=capture_factory,
            image_writer=image_writer,
        )
        manifest_path.write_text(
            json.dumps(
                {
                    **key,
                    "frames": [
                        {
                            "frame_index": frame.frame_index,
                            "timestamp_ms": frame.timestamp_ms,
                            "file": frame.image_path.name,
                        }
                        for frame in frames
                    ],
                }
            ),
            encoding="utf-8",
        )
        return frames


def _lock_for(directory: Path) -> threading.Lock:
    with _cache_locks_guard:
        return _cache_locks.setdefault(str(directory.resolve()), threading.Lock())


def _read_manifest(path: Path, key: dict) -> list[ExtractedFrame] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if any(value.get(name) != expected for name, expected in key.items()):
        return None
    frames = [
        ExtractedFrame(
            frame_index=int(item["frame_index"]),
            timestamp_ms=float(item["timestamp_ms"]),
            image_path=path.parent / item["file"],
        )
        for item in value.get("frames", [])
    ]
    if not all(frame.image_path.is_file() for frame in frames):
        return None
    return frames
