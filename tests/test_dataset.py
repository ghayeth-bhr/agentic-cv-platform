from pathlib import Path

import pytest

from agentic_cv.dataset import (
    Annotation,
    DatasetCategory,
    DatasetImage,
    DatasetVersion,
    export_coco,
    export_yolo,
)


def make_dataset():
    image = DatasetImage(image_id="image-1", file_name="frame.jpg", width=100, height=50)
    category = DatasetCategory(category_id=2, name="bear")
    annotation = Annotation(
        annotation_id="annotation-1",
        image_id="image-1",
        category_id=2,
        bbox=(10, 5, 40, 20),
        segmentation=((10, 5, 50, 5, 50, 25, 10, 25),),
        score=0.95,
    )
    return DatasetVersion(images=(image,), categories=(category,), annotations=(annotation,))


def test_coco_export_contains_detection_and_segmentation_geometry():
    exported = export_coco(make_dataset())

    annotation = exported["annotations"][0]
    assert annotation["bbox"] == [10, 5, 40, 20]
    assert annotation["area"] == 800
    assert annotation["segmentation"] == [[10, 5, 50, 5, 50, 25, 10, 25]]


def test_yolo_detection_and_segmentation_exports_are_normalized():
    dataset = make_dataset()

    detection = export_yolo(dataset, "detection")["frame.txt"].split()
    segmentation = export_yolo(dataset, "segmentation")["frame.txt"].split()

    assert detection == ["2", "0.3", "0.3", "0.4", "0.4"]
    assert segmentation == ["2", "0.1", "0.1", "0.5", "0.1", "0.5", "0.5", "0.1", "0.5"]


def test_invalid_geometry_and_missing_segmentation_fail():
    image = DatasetImage(image_id="image-1", file_name="frame.jpg", width=10, height=10)
    category = DatasetCategory(category_id=0, name="object")
    invalid = Annotation(
        annotation_id="a", image_id="image-1", category_id=0, bbox=(0, 0, 20, 1)
    )

    with pytest.raises(ValueError, match="bbox"):
        DatasetVersion(images=(image,), categories=(category,), annotations=(invalid,))

    no_polygon = Annotation(
        annotation_id="a", image_id="image-1", category_id=0, bbox=(0, 0, 1, 1)
    )
    dataset = DatasetVersion(images=(image,), categories=(category,), annotations=(no_polygon,))
    with pytest.raises(ValueError, match="polygon"):
        export_yolo(dataset, "segmentation")
