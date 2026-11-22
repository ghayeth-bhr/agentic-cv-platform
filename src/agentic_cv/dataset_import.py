"""Import an external COCO dataset as a dataset version.

An imported version is an ordinary version: its images live in the version's
image folder and its annotations are revisions in the version's own
annotation set. The gallery, click refinement, approval, rebuild, and export
therefore work on it unchanged, and nothing else (assets, label jobs, the
agent) is touched.

The upload is one ZIP holding a COCO ``.json`` file and the images. Images are
matched to COCO ``file_name`` entries by base name, so any folder layout works.
"""

from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from uuid import UUID, uuid4

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

from .annotation_sets import scoped_id
from .annotation_store import AnnotationRevisionBackend
from .dataset import Annotation, DatasetCategory, DatasetImage, DatasetVersion
from .dataset_store import DatasetVersionBackend, DatasetVersionRecord

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MAX_IMAGES = 5000
MAX_ANNOTATIONS = 200_000
MAX_UNCOMPRESSED_BYTES = 4 * 1024**3
MAX_JSON_BYTES = 512 * 1024**2
IMPORT_AUTHOR = "dataset-import"


class DatasetImportError(ValueError):
    """The upload is not a usable COCO dataset; the message says why."""


@dataclass(frozen=True, kw_only=True)
class DatasetImportResult:
    record: DatasetVersionRecord
    image_count: int
    annotation_count: int
    skipped_annotations: int
    missing_images: tuple[str, ...]


class DatasetImporter:
    def __init__(
        self,
        *,
        annotations: AnnotationRevisionBackend,
        versions: DatasetVersionBackend,
        images_root: Path,
    ) -> None:
        self.annotations = annotations
        self.versions = versions
        self.images_root = images_root

    def import_coco_zip(
        self,
        *,
        tenant_id: UUID,
        archive: Path | io.BufferedIOBase,
        name: str,
        author_id: str = IMPORT_AUTHOR,
    ) -> DatasetImportResult:
        if not name.strip():
            raise DatasetImportError("give the imported dataset a name")
        try:
            bundle = zipfile.ZipFile(archive)
        except zipfile.BadZipFile as exc:
            raise DatasetImportError("the upload is not a ZIP file") from exc
        with bundle:
            members = _safe_members(bundle)
            coco = _load_coco(bundle, members)
            image_members = {
                PurePosixPath(info.filename).name: info
                for info in members
                if PurePosixPath(info.filename).suffix.lower() in IMAGE_SUFFIXES
            }
            return self._import(
                tenant_id=tenant_id,
                name=name.strip(),
                coco=coco,
                read_image=lambda base: bundle.read(image_members[base]),
                available=set(image_members),
                author_id=author_id,
            )

    def _import(self, *, tenant_id, name, coco, read_image, available, author_id) -> DatasetImportResult:
        categories, category_map = _categories(coco)
        set_key = uuid4().hex[:12]
        set_id = f"import-{set_key}"

        images: list[DatasetImage] = []
        payloads: dict[str, bytes] = {}
        image_map: dict[object, DatasetImage] = {}
        missing: list[str] = []
        used_names: set[str] = set()
        for index, entry in enumerate(coco["images"], start=1):
            base = PurePosixPath(str(entry.get("file_name", "")).replace("\\", "/")).name
            if not base or base not in available:
                missing.append(base or f"image {entry.get('id')}")
                continue
            data = read_image(base)
            try:
                with Image.open(io.BytesIO(data)) as probe:
                    width, height = probe.size
            except (UnidentifiedImageError, OSError) as exc:
                raise DatasetImportError(f"{base} is not a readable image") from exc
            file_name = base if base not in used_names else f"{index}-{base}"
            used_names.add(file_name)
            image = DatasetImage(
                image_id=f"{set_id}-{index}",
                file_name=f"images/{file_name}",
                width=width,
                height=height,
            )
            images.append(image)
            payloads[file_name] = data
            image_map[entry.get("id")] = image
        if not images:
            raise DatasetImportError(
                "none of the images listed in the COCO file were found in the ZIP"
            )

        annotations: list[Annotation] = []
        skipped = 0
        for index, item in enumerate(coco.get("annotations", []), start=1):
            image = image_map.get(item.get("image_id"))
            category_id = category_map.get(item.get("category_id"))
            if image is None or category_id is None or item.get("iscrowd"):
                skipped += 1
                continue
            annotation = _annotation(
                item,
                image=image,
                category_id=category_id,
                annotation_id=scoped_id(set_id, str(index)),
            )
            if annotation is None:
                skipped += 1
                continue
            annotations.append(annotation)

        # Revisions first, then the version: a version only ever points at
        # annotations that already exist in its set.
        for annotation in annotations:
            self.annotations.save(
                tenant_id=tenant_id,
                annotation=annotation,
                author_id=author_id,
                expected_revision=0,
            )
        dataset = DatasetVersion(
            images=tuple(images),
            categories=categories,
            annotations=tuple(annotations),
            annotation_set=set_id,
        ).with_splits()
        record = self.versions.create(
            tenant_id=tenant_id,
            name=name,
            dataset=dataset,
            idempotency_key=set_id,
        )
        target = self.images_root / str(tenant_id) / str(record.version_id) / "images"
        target.mkdir(parents=True, exist_ok=True)
        for file_name, data in payloads.items():
            (target / file_name).write_bytes(data)
        return DatasetImportResult(
            record=record,
            image_count=len(images),
            annotation_count=len(annotations),
            skipped_annotations=skipped,
            missing_images=tuple(missing),
        )


def _safe_members(bundle: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    members = [info for info in bundle.infolist() if not info.is_dir()]
    if sum(info.file_size for info in members) > MAX_UNCOMPRESSED_BYTES:
        raise DatasetImportError("the ZIP is too large once unpacked (limit 4 GB)")
    if len(members) > MAX_IMAGES + 100:
        raise DatasetImportError(f"the ZIP holds too many files (limit {MAX_IMAGES} images)")
    # Members are only ever read into memory by base name, never extracted
    # to their stored path, so a "../" name cannot write outside the version.
    return [
        info
        for info in members
        if not PurePosixPath(info.filename).name.startswith(".")
        and "__MACOSX" not in PurePosixPath(info.filename).parts
    ]


def _load_coco(bundle: zipfile.ZipFile, members: list[zipfile.ZipInfo]) -> dict:
    candidates = [info for info in members if info.filename.lower().endswith(".json")]
    if not candidates:
        raise DatasetImportError("the ZIP has no COCO .json annotation file")
    for info in sorted(candidates, key=lambda item: item.file_size, reverse=True):
        if info.file_size > MAX_JSON_BYTES:
            continue
        try:
            value = json.loads(bundle.read(info))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(value, dict) and isinstance(value.get("images"), list):
            if len(value["images"]) > MAX_IMAGES:
                raise DatasetImportError(f"too many images (limit {MAX_IMAGES})")
            if len(value.get("annotations", [])) > MAX_ANNOTATIONS:
                raise DatasetImportError(f"too many annotations (limit {MAX_ANNOTATIONS})")
            if not isinstance(value.get("categories"), list) or not value["categories"]:
                raise DatasetImportError("the COCO file has no categories")
            return value
    raise DatasetImportError("no .json file in the ZIP is a COCO dataset (it needs an images list)")


def _categories(coco: dict) -> tuple[tuple[DatasetCategory, ...], dict[object, int]]:
    categories: list[DatasetCategory] = []
    mapping: dict[object, int] = {}
    for position, entry in enumerate(coco["categories"]):
        label = str(entry.get("name", "")).strip()
        if not label:
            raise DatasetImportError("every COCO category needs a name")
        # Dense ids keep YOLO export class numbers compact and stable.
        mapping[entry.get("id")] = position
        categories.append(DatasetCategory(category_id=position, name=label))
    return tuple(categories), mapping


def _annotation(item: dict, *, image: DatasetImage, category_id: int, annotation_id: str) -> Annotation | None:
    polygons = _polygons(item.get("segmentation"), image)
    bbox = _bbox(item.get("bbox"), polygons, image)
    if bbox is None:
        return None
    score = item.get("score")
    if not isinstance(score, (int, float)) or not 0 <= score <= 1:
        score = None
    annotation = Annotation(
        annotation_id=annotation_id,
        image_id=image.image_id,
        category_id=category_id,
        bbox=bbox,
        segmentation=polygons,
        score=float(score) if score is not None else None,
    )
    try:
        annotation.validate(image)
    except ValueError:
        return None
    return annotation


def _clamp_polygon(values, image: DatasetImage) -> tuple[float, ...] | None:
    try:
        numbers = [float(value) for value in values]
    except (TypeError, ValueError):
        return None
    if len(numbers) < 6 or len(numbers) % 2:
        return None
    clamped = []
    for position, value in enumerate(numbers):
        limit = image.width if position % 2 == 0 else image.height
        clamped.append(round(min(max(value, 0.0), float(limit)), 2))
    return tuple(clamped)


def _polygons(segmentation, image: DatasetImage) -> tuple[tuple[float, ...], ...]:
    if isinstance(segmentation, list):
        polygons = (_clamp_polygon(polygon, image) for polygon in segmentation)
        return tuple(polygon for polygon in polygons if polygon is not None)
    if isinstance(segmentation, dict) and "counts" in segmentation:
        mask = _decode_rle(segmentation)
        if mask is None or mask.shape != (image.height, image.width):
            return ()
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polygons = (
            _clamp_polygon(contour.reshape(-1).tolist(), image)
            for contour in contours
            if len(contour) >= 3
        )
        return tuple(polygon for polygon in polygons if polygon is not None)
    return ()


def _bbox(value, polygons, image: DatasetImage) -> tuple[float, float, float, float] | None:
    if polygons:
        xs = [x for polygon in polygons for x in polygon[0::2]]
        ys = [y for polygon in polygons for y in polygon[1::2]]
        left, top, right, bottom = min(xs), min(ys), max(xs), max(ys)
    elif isinstance(value, list) and len(value) == 4:
        try:
            x, y, width, height = (float(item) for item in value)
        except (TypeError, ValueError):
            return None
        left, top, right, bottom = x, y, x + width, y + height
    else:
        return None
    left, top = max(left, 0.0), max(top, 0.0)
    right, bottom = min(right, float(image.width)), min(bottom, float(image.height))
    if right - left < 1 or bottom - top < 1:
        return None
    return (round(left, 2), round(top, 2), round(right - left, 2), round(bottom - top, 2))


def _decode_rle(rle: dict) -> np.ndarray | None:
    """Decode a COCO RLE (list or compressed string counts) to a boolean mask."""

    size = rle.get("size")
    counts = rle.get("counts")
    if not (isinstance(size, list) and len(size) == 2):
        return None
    height, width = int(size[0]), int(size[1])
    if isinstance(counts, str):
        counts = _uncompress_counts(counts)
    if not isinstance(counts, list):
        return None
    flat = np.zeros(height * width, dtype=bool)
    position, value = 0, False
    for run in counts:
        run = int(run)
        if run < 0 or position + run > flat.size:
            return None
        if value:
            flat[position : position + run] = True
        position += run
        value = not value
    # COCO masks are stored column-major.
    return flat.reshape((width, height)).T


def _uncompress_counts(text: str) -> list[int] | None:
    """pycocotools' LEB128-like string encoding of RLE counts."""

    counts: list[int] = []
    position = 0
    data = text.encode("ascii", errors="ignore")
    while position < len(data):
        value, shift, more = 0, 0, True
        while more:
            if position >= len(data):
                return None
            chunk = data[position] - 48
            value |= (chunk & 0x1F) << (5 * shift)
            more = bool(chunk & 0x20)
            position += 1
            shift += 1
            if not more and chunk & 0x10:
                value |= -1 << (5 * shift)
        if len(counts) > 2:
            value += counts[-2]
        counts.append(value)
    return counts
