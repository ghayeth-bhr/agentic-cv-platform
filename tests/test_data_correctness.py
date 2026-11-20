"""Phase 0 regressions: run-scoped annotations, tombstones, splits, exports, video cache."""

import hashlib
import json
import sqlite3
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from agentic_cv.annotation_sets import in_set, job_set_id, scoped_id
from agentic_cv.annotation_store import AnnotationRevisionStore
from agentic_cv.api import (
    ApiDependencies,
    DatasetApiDependencies,
    RefinementApiDependencies,
    create_app,
)
from agentic_cv.asset_registry import AssetRegistry
from agentic_cv.batch_labeling import Sam3BatchLabeler, mask_polygons
from agentic_cv.click_refinement import ClickPoint, ClickRefinementService
from agentic_cv.dataset import (
    Annotation,
    DatasetCategory,
    DatasetImage,
    DatasetVersion,
    assign_splits,
    export_coco,
    merge_polygons,
)
from agentic_cv.dataset_builder import DatasetBuilder
from agentic_cv.dataset_export import DatasetExportService
from agentic_cv.dataset_store import DatasetVersionStore
from agentic_cv.job_store import JobStore
from agentic_cv.label_images import LabelImageResolver
from agentic_cv.label_workflow import DurableLabelWorkflow, LabelWorkflowDependencies
from agentic_cv.object_storage import LocalObjectStorage
from agentic_cv.outbox import Outbox
from agentic_cv.sam3 import Sam3Prediction
from agentic_cv.sqlite_utils import connect_sqlite
from agentic_cv.uploads import ResumableUploadService
from agentic_cv.video_frames import FrameExtractionConfig, extract_frames, extract_frames_cached


def _box_mask(x1, y1, x2, y2, size=(20, 20)):
    mask = np.zeros(size, dtype=bool)
    mask[y1:y2, x1:x2] = True
    return mask


class ScriptedProvider:
    """Returns a fixed list of boxes per call, so each run can differ."""

    def __init__(self, boxes):
        self.boxes = boxes

    def predict(self, image, prompt, *, threshold=None, mask_threshold=None):
        return [
            Sam3Prediction(score=0.9, box=box, mask=_box_mask(*map(int, box)))
            for box in self.boxes
        ]


class Harness:
    def __init__(self, tmp_path: Path):
        self.tmp = tmp_path
        self.tenant_id = uuid4()
        self.registry = AssetRegistry(str(tmp_path / "assets.sqlite3"))
        self.annotations = AnnotationRevisionStore(str(tmp_path / "annotations.sqlite3"))
        self.jobs = JobStore(str(tmp_path / "jobs.sqlite3"))
        self.versions = DatasetVersionStore(str(tmp_path / "versions.sqlite3"))
        self.outbox = Outbox(str(tmp_path / "events.sqlite3"))
        self.resolver = LabelImageResolver(registry=self.registry, data_dir=tmp_path / "data")
        self.builder = DatasetBuilder(
            jobs=self.jobs,
            annotations=self.annotations,
            versions=self.versions,
            images=self.resolver,
            images_root=tmp_path / "dataset-images",
        )
        self.asset = self.registry.create_asset(tenant_id=self.tenant_id, name="yard")
        image_path = tmp_path / "frame.png"
        Image.new("RGB", (20, 20), (255, 255, 255)).save(image_path)
        self.registry.add_source(
            tenant_id=self.tenant_id,
            asset_id=self.asset.asset_id,
            filename=image_path.name,
            media_type="image/png",
            object_uri=image_path.resolve().as_uri(),
            content_sha256=hashlib.sha256(image_path.read_bytes()).hexdigest(),
        )

    def run(self, boxes, *, prompt, key, before_execute=None):
        workflow = DurableLabelWorkflow(
            LabelWorkflowDependencies(
                jobs=self.jobs,
                annotations=self.annotations,
                outbox=self.outbox,
                labeler=Sam3BatchLabeler(ScriptedProvider(boxes)),
            )
        )
        job = workflow.submit(
            tenant_id=self.tenant_id,
            asset_id=str(self.asset.asset_id),
            prompt=prompt,
            threshold=0.5,
            task="segmentation",
            category_id=1,
            category_name=prompt,
            idempotency_key=key,
        )
        if before_execute is not None:
            before_execute(job)
        workflow.execute(
            job.job_id,
            image_loader=lambda: self.resolver.resolve(
                tenant_id=self.tenant_id, asset_id=str(self.asset.asset_id)
            ),
            tenant_id=self.tenant_id,
            prompt=prompt,
            threshold=0.5,
            task="segmentation",
            category=DatasetCategory(category_id=1, name=prompt),
        )
        return job

    def build(self, job, key):
        return self.builder.build_from_job(
            tenant_id=self.tenant_id, job_id=job.job_id, name=key, idempotency_key=key
        ).record


def test_second_run_on_same_asset_does_not_inherit_first_run_labels(tmp_path):
    harness = Harness(tmp_path)
    cars = harness.run([(0, 0, 4, 4), (5, 5, 9, 9), (10, 10, 14, 14)], prompt="car", key="run-1")
    people = harness.run([(2, 2, 8, 8)], prompt="person", key="run-2")

    first = harness.build(cars, "cars")
    second = harness.build(people, "people")

    assert len(first.dataset.annotations) == 3
    assert len(second.dataset.annotations) == 1
    assert second.dataset.annotations[0].bbox == (2.0, 2.0, 6.0, 6.0)
    assert second.dataset.annotation_set == job_set_id(people.job_id)
    assert second.dataset.source_asset_ids == (str(harness.asset.asset_id),)


def test_empty_second_run_produces_an_empty_dataset(tmp_path):
    harness = Harness(tmp_path)
    harness.run([(0, 0, 4, 4)], prompt="car", key="run-1")
    nothing = harness.run([], prompt="bicycle", key="run-2")

    assert harness.build(nothing, "bikes").dataset.annotations == ()


def test_job_retry_retires_leftover_annotations_from_an_earlier_attempt(tmp_path):
    harness = Harness(tmp_path)
    image_id = harness.resolver.resolve(
        tenant_id=harness.tenant_id, asset_id=str(harness.asset.asset_id)
    )[0].image_id

    def earlier_attempt(job):
        # An attempt that lost its lease had already written two annotations.
        for index, box in enumerate([(0, 0, 4, 4), (5, 5, 4, 4)]):
            harness.annotations.save(
                tenant_id=harness.tenant_id,
                annotation=Annotation(
                    annotation_id=scoped_id(job_set_id(job.job_id), f"{image_id}:{index}"),
                    image_id=image_id,
                    category_id=1,
                    bbox=box,
                ),
                author_id="sam3-batch",
                expected_revision=None,
            )

    job = harness.run([(0, 0, 4, 4)], prompt="car", key="run-1", before_execute=earlier_attempt)

    dataset = harness.build(job, "cars").dataset
    assert [item.annotation_id for item in dataset.annotations] == [
        scoped_id(job_set_id(job.job_id), f"{image_id}:0")
    ]


def test_legacy_versions_without_a_set_only_read_unscoped_annotations(tmp_path):
    harness = Harness(tmp_path)
    image = DatasetImage(image_id="legacy-1", file_name="images/a.png", width=20, height=20)
    legacy = Annotation(annotation_id="legacy-1:0", image_id="legacy-1", category_id=1, bbox=(1, 1, 3, 3))
    scoped = Annotation(
        annotation_id=scoped_id("job-other", "legacy-1:0"),
        image_id="legacy-1",
        category_id=1,
        bbox=(5, 5, 3, 3),
    )
    for annotation in (legacy, scoped):
        harness.annotations.save(
            tenant_id=harness.tenant_id, annotation=annotation, author_id="t", expected_revision=0
        )
    record = harness.versions.create(
        tenant_id=harness.tenant_id,
        name="legacy",
        dataset=DatasetVersion(
            images=(image,),
            categories=(DatasetCategory(category_id=1, name="car"),),
            annotations=(legacy,),
        ),
        idempotency_key="legacy",
    )
    rebuilt = harness.builder.rebuild_from_version(
        tenant_id=harness.tenant_id, version_id=record.version_id, idempotency_key="legacy-2"
    )
    assert [item.annotation_id for item in rebuilt.record.dataset.annotations] == ["legacy-1:0"]


class _FakeRefiner:
    def __init__(self, box):
        self.box = box

    def refine_with_prompt(self, image, points, prompt):
        return [Sam3Prediction(score=0.8, box=self.box, mask=_box_mask(*map(int, self.box)))]


def _review_app(harness: Harness, refiner):
    service = ClickRefinementService(refiner)
    dependencies = ApiDependencies(
        registry=harness.registry,
        uploads=ResumableUploadService(harness.registry, LocalObjectStorage(harness.tmp / "objects")),
        outbox=harness.outbox,
        refinement=RefinementApiDependencies(service=service, annotations=harness.annotations),
        datasets=DatasetApiDependencies(
            builder=harness.builder,
            versions=harness.versions,
            images_root=harness.tmp / "dataset-images",
        ),
    )
    return service, TestClient(create_app(dependencies))


def test_delete_annotation_tombstones_and_rebuild_drops_it(tmp_path):
    harness = Harness(tmp_path)
    job = harness.run([(0, 0, 4, 4), (10, 10, 14, 14)], prompt="car", key="run-1")
    version = harness.build(job, "cars")
    image_id = version.dataset.images[0].image_id
    false_positive = version.dataset.annotations[1].annotation_id
    _, client = _review_app(harness, _FakeRefiner((0, 0, 4, 4)))
    headers = {"X-Tenant-ID": str(harness.tenant_id)}

    deleted = client.delete(
        f"/v1/datasets/{version.version_id}/images/{image_id}/annotations/{false_positive}",
        headers=headers,
    )
    assert deleted.status_code == 200, deleted.text
    assert deleted.json()["deleted"] is True
    again = client.delete(
        f"/v1/datasets/{version.version_id}/images/{image_id}/annotations/{false_positive}",
        headers=headers,
    )
    assert again.status_code == 409
    foreign = client.delete(
        f"/v1/datasets/{version.version_id}/images/{image_id}/annotations/job-x/other:0",
        headers=headers,
    )
    assert foreign.status_code == 404

    rebuilt = client.post(
        f"/v1/datasets/{version.version_id}/rebuild",
        headers=headers,
        json={"idempotency_key": "rebuild-1"},
    )
    assert rebuilt.status_code == 201, rebuilt.text
    assert rebuilt.json()["annotation_count"] == 1
    assert rebuilt.json()["source_asset"] == "yard"


def test_accepting_a_click_without_a_target_adds_a_new_annotation(tmp_path):
    harness = Harness(tmp_path)
    job = harness.run([(0, 0, 4, 4)], prompt="car", key="run-1")
    version = harness.build(job, "cars")
    image = version.dataset.images[0]
    existing = version.dataset.annotations[0]
    service, client = _review_app(harness, _FakeRefiner((10, 10, 16, 16)))
    snapshot = service.create(
        image_id=image.image_id,
        image=Image.new("RGB", (20, 20)),
        tenant_id=harness.tenant_id,
        version_id=version.version_id,
        prompt="car",
        category_id=1,
    )
    snapshot = service.click(
        snapshot.session_id, expected_revision=0, point=ClickPoint(x=12, y=12, label="positive")
    )
    headers = {"X-Tenant-ID": str(harness.tenant_id)}

    accepted = client.post(
        f"/v1/refinements/{snapshot.session_id}/accept",
        headers=headers,
        json={"prediction_index": 0, "refinement_revision": snapshot.revision},
    )
    assert accepted.status_code == 200, accepted.text
    new_id = accepted.json()["annotation_id"]
    assert new_id != existing.annotation_id
    assert in_set(new_id, version.dataset.annotation_set)

    rebuilt = harness.builder.rebuild_from_version(
        tenant_id=harness.tenant_id, version_id=version.version_id, idempotency_key="r1"
    ).record
    boxes = sorted(item.bbox for item in rebuilt.dataset.annotations)
    assert boxes == [(0.0, 0.0, 4.0, 4.0), (10.0, 10.0, 6.0, 6.0)]


def test_splits_keep_video_frames_together_and_are_persisted(tmp_path):
    images = tuple(
        DatasetImage(image_id=f"asset:video{video}:frame:{frame}", file_name=f"v{video}_{frame}.jpg", width=4, height=4)
        for video in range(6)
        for frame in range(5)
    ) + tuple(
        DatasetImage(image_id=f"asset:photo{index}:image", file_name=f"p{index}.jpg", width=4, height=4)
        for index in range(20)
    )
    splits = assign_splits(images)
    for video in range(6):
        assert len({splits[f"asset:video{video}:frame:{frame}"] for frame in range(5)}) == 1
    counts = {name: list(splits.values()).count(name) for name in ("train", "validation", "test")}
    assert counts["train"] >= 35 and counts["validation"] >= 3 and counts["test"] >= 3
    assert assign_splits(images) == splits

    store = DatasetVersionStore(str(tmp_path / "versions.sqlite3"))
    dataset = DatasetVersion(
        images=images, categories=(DatasetCategory(category_id=0, name="x"),), annotations=()
    ).with_splits()
    record = store.create(tenant_id=uuid4(), name="v", dataset=dataset, idempotency_key="k")
    assert {image.image_id: image.split for image in record.dataset.images} == splits


def _export_dataset(tmp_path, count=12):
    tenant = uuid4()
    store = DatasetVersionStore(str(tmp_path / "versions.sqlite3"))
    images = tuple(
        DatasetImage(image_id=f"a:s{index}:image", file_name=f"images/f{index}.png", width=20, height=10)
        for index in range(count)
    )
    annotations = tuple(
        Annotation(
            annotation_id=f"set/a:s{index}:image:0",
            image_id=f"a:s{index}:image",
            category_id=7,
            bbox=(2, 2, 4, 4),
            segmentation=((2, 2, 6, 2, 6, 6, 2, 6), (10, 2, 12, 2, 12, 4)),
            score=0.9,
        )
        for index in range(count)
    )
    record = store.create(
        tenant_id=tenant,
        name="v",
        dataset=DatasetVersion(
            images=images,
            categories=(DatasetCategory(category_id=7, name="helmet"),),
            annotations=annotations,
            annotation_set="set",
        ).with_splits(),
        idempotency_key="v",
    )
    root = tmp_path / "dataset-images" / str(tenant) / str(record.version_id) / "images"
    root.mkdir(parents=True)
    for index in range(count):
        Image.new("RGB", (20, 10)).save(root / f"f{index}.png")
    service = DatasetExportService(store, tmp_path / "exports", images_root=tmp_path / "dataset-images")
    return service, tenant, record


def test_yolo_export_has_zero_based_classes_and_real_split_folders(tmp_path):
    service, tenant, record = _export_dataset(tmp_path)
    result = service.export(tenant_id=tenant, version_id=record.version_id, format="yolo-segmentation")
    out = result.output_dir

    data = (out / "data.yaml").read_text()
    assert "train: images/train" in data and "val: images/val" in data
    assert "nc: 1" in data and '0: "helmet"' in data
    for split, folder in (("train", "train"), ("validation", "val"), ("test", "test")):
        names = {image.file_name.split("/")[-1] for image in record.dataset.images if image.split == split}
        assert names, split
        assert {path.name for path in (out / "images" / folder).iterdir()} == names
        assert {path.stem for path in (out / "labels" / folder).iterdir()} == {Path(n).stem for n in names}
    label = next((out / "labels" / "train").iterdir()).read_text().strip().split("\n")
    assert len(label) == 1  # one line per instance even with two polygon parts
    fields = label[0].split()
    assert fields[0] == "0"
    assert all(0.0 <= float(value) <= 1.0 for value in fields[1:])


def test_reexport_removes_files_from_an_older_layout(tmp_path):
    service, tenant, record = _export_dataset(tmp_path)
    stale = tmp_path / "exports" / str(tenant) / str(record.version_id) / "coco" / "stale.txt"
    stale.parent.mkdir(parents=True)
    stale.write_text("old")
    service.export(tenant_id=tenant, version_id=record.version_id, format="coco")
    assert not stale.exists()


def test_coco_export_uses_integer_ids_and_per_split_files(tmp_path):
    service, tenant, record = _export_dataset(tmp_path)
    result = service.export(tenant_id=tenant, version_id=record.version_id, format="coco")
    combined = json.loads((result.output_dir / "annotations.json").read_text())
    assert [image["id"] for image in combined["images"]] == list(range(1, 13))
    assert all(isinstance(item["image_id"], int) for item in combined["annotations"])
    assert {image["split"] for image in combined["images"]} == {"train", "validation", "test"}
    total = 0
    for split in ("train", "validation", "test"):
        part = json.loads((result.output_dir / "annotations" / f"instances_{split}.json").read_text())
        ids = {image["id"] for image in part["images"]}
        assert all(item["image_id"] in ids for item in part["annotations"])
        total += len(part["images"])
    assert total == 12
    pycocotools = pytest.importorskip("pycocotools.coco")
    pycocotools.COCO(str(result.output_dir / "annotations" / "instances_train.json"))


def test_coco_area_uses_polygon_area():
    dataset = DatasetVersion(
        images=(DatasetImage(image_id="i", file_name="i.png", width=10, height=10),),
        categories=(DatasetCategory(category_id=1, name="x"),),
        annotations=(
            Annotation(
                annotation_id="a", image_id="i", category_id=1, bbox=(0, 0, 4, 4),
                segmentation=((0, 0, 4, 0, 0, 4),),
            ),
        ),
    )
    assert export_coco(dataset)["annotations"][0]["area"] == 8.0


def test_merge_polygons_visits_every_vertex_and_closes():
    first = (0, 0, 4, 0, 4, 4, 0, 4)
    second = (10, 0, 14, 0, 14, 4)
    third = (20, 0, 24, 0, 24, 4, 20, 4)
    merged = merge_polygons((first, second, third))
    points = list(zip(merged[0::2], merged[1::2]))
    for polygon in (first, second, third):
        for vertex in zip(polygon[0::2], polygon[1::2]):
            assert vertex in points
    assert points[0] == points[-1]
    assert merge_polygons((first,)) == first


def test_mask_polygons_simplify_and_drop_speckles():
    mask = np.zeros((200, 200), dtype=bool)
    yy, xx = np.mgrid[:200, :200]
    mask[(yy - 100) ** 2 + (xx - 100) ** 2 < 70 ** 2] = True
    mask[5, 5] = True  # one-pixel speckle

    polygons = mask_polygons(mask)
    assert len(polygons) == 1
    vertices = len(polygons[0]) // 2
    assert 8 <= vertices <= 80

    import cv2

    rendered = np.zeros((200, 200), dtype=np.uint8)
    cv2.fillPoly(rendered, [np.array(polygons[0], dtype=np.int32).reshape(-1, 2)], 1)
    circle = mask.copy()
    circle[5, 5] = False
    iou = (rendered.astype(bool) & circle).sum() / (rendered.astype(bool) | circle).sum()
    assert iou > 0.95


def test_tiny_mask_still_yields_a_polygon():
    mask = np.zeros((20, 20), dtype=bool)
    mask[3:5, 3:5] = True
    assert len(mask_polygons(mask)) == 1


class _FpsCapture:
    def __init__(self, count, fps):
        self.count = count
        self.fps = fps
        self.index = 0

    def isOpened(self):
        return True

    def read(self):
        if self.index >= self.count:
            return False, None
        self.index += 1
        return True, "frame"

    def get(self, prop):
        if prop == 5:
            return self.fps
        return float((self.index - 1) * 1000 / self.fps)

    def release(self):
        pass


def test_time_based_sampling_covers_the_whole_video(tmp_path):
    frames = extract_frames(
        Path("v.mp4"),
        tmp_path / "frames",
        config=FrameExtractionConfig(target_fps=1.0, frame_stride=30, max_frames=None),
        capture_factory=lambda path: _FpsCapture(count=600 * 25, fps=25.0),
        image_writer=lambda path, frame: True,
    )
    assert len(frames) == 600  # ten minutes at 1 fps, not capped at 120
    assert frames[1].frame_index == 25


def test_frame_cache_extracts_once_and_reextracts_on_config_change(tmp_path):
    video = tmp_path / "v.mp4"
    video.write_bytes(b"not really a video")
    calls = []

    def factory(path):
        calls.append(path)
        return _FpsCapture(count=50, fps=10.0)

    def writer(path, frame):
        Path(path).write_bytes(b"jpg")
        return True

    config = FrameExtractionConfig(target_fps=2.0)
    first = extract_frames_cached(video, tmp_path / "cache", config=config, capture_factory=factory, image_writer=writer)
    second = extract_frames_cached(video, tmp_path / "cache", config=config, capture_factory=factory, image_writer=writer)
    assert len(calls) == 1
    assert [frame.frame_index for frame in second] == [frame.frame_index for frame in first]
    assert len(first) == 10

    extract_frames_cached(
        video, tmp_path / "cache", config=FrameExtractionConfig(target_fps=1.0),
        capture_factory=factory, image_writer=writer,
    )
    assert len(calls) == 2


def test_sqlite_stores_use_wal_and_busy_timeout(tmp_path):
    connection = connect_sqlite(tmp_path / "x.sqlite3")
    assert connection.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    assert connection.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
    connection.close()
    memory = connect_sqlite(":memory:")
    assert isinstance(memory, sqlite3.Connection)
    memory.close()
