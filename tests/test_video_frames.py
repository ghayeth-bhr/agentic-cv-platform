from pathlib import Path

import pytest

from agentic_cv.video_frames import FrameExtractionConfig, extract_frames


class FakeCapture:
    def __init__(self, frames):
        self.frames = list(frames)
        self.index = 0
        self.released = False

    def isOpened(self):
        return True

    def read(self):
        if self.index == len(self.frames):
            return False, None
        frame = self.frames[self.index]
        self.index += 1
        return True, frame

    def get(self, property_id):
        return float((self.index - 1) * 100)

    def release(self):
        self.released = True


def test_extractor_preserves_stride_timestamps_and_paths(tmp_path):
    capture = FakeCapture(["f0", "f1", "f2", "f3", "f4"])
    written = []

    frames = extract_frames(
        Path("input.mp4"),
        tmp_path / "frames",
        config=FrameExtractionConfig(frame_stride=2),
        capture_factory=lambda path: capture,
        image_writer=lambda path, frame: written.append((path, frame)) or True,
    )

    assert [frame.frame_index for frame in frames] == [0, 2, 4]
    assert [frame.timestamp_ms for frame in frames] == [0.0, 200.0, 400.0]
    assert [Path(path).name for path, _ in written] == [
        "frame_00000000.jpg", "frame_00000002.jpg", "frame_00000004.jpg"
    ]
    assert capture.released


def test_extractor_honors_max_frames_and_writer_failure(tmp_path):
    capture = FakeCapture(["f0", "f1", "f2"])
    frames = extract_frames(
        Path("input.mp4"),
        tmp_path / "frames",
        config=FrameExtractionConfig(max_frames=2),
        capture_factory=lambda path: capture,
        image_writer=lambda path, frame: True,
    )
    assert len(frames) == 2
    assert capture.index == 2

    with pytest.raises(IOError, match="unable to write"):
        extract_frames(
            Path("input.mp4"),
            tmp_path / "failed",
            capture_factory=lambda path: FakeCapture(["f0"]),
            image_writer=lambda path, frame: False,
        )


def test_config_and_unopened_video_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="frame_stride"):
        FrameExtractionConfig(frame_stride=0)

    class ClosedCapture(FakeCapture):
        def isOpened(self):
            return False

    with pytest.raises(ValueError, match="unable to open"):
        extract_frames(
            Path("missing.mp4"),
            tmp_path / "frames",
            capture_factory=lambda path: ClosedCapture([]),
        )
