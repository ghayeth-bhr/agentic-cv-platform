"""CPU video frame extraction with timestamp and provenance records."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True, kw_only=True)
class FrameExtractionConfig:
    frame_stride: int = 1
    max_frames: int | None = None

    def __post_init__(self) -> None:
        if self.frame_stride < 1:
            raise ValueError("frame_stride must be at least 1")
        if self.max_frames is not None and self.max_frames < 1:
            raise ValueError("max_frames must be at least 1 when provided")


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
    try:
        while config.max_frames is None or len(extracted) < config.max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % config.frame_stride == 0:
                image_path = output_dir / f"frame_{frame_index:08d}.jpg"
                if not writer(str(image_path), frame):
                    raise IOError(f"unable to write extracted frame: {image_path}")
                timestamp_ms = float(capture.get(0))
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
    return extracted
