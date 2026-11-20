"""Phase 4a: dataset review — live annotations, approval, class edits, approved-only rebuilds."""

from __future__ import annotations

from urllib.parse import quote

from fastapi.testclient import TestClient

from agentic_cv.application import LocalAppConfig, create_local_app
from agentic_cv.dataset import DatasetCategory, DatasetImage, DatasetVersion

from test_assets_ui_playwright import FakeEmbeddingProvider
from test_label_execution_e2e import FakeSam3, headers, submit_label_job, wait_for
from test_search_selection import upload_asset


def make_client(tmp_path):
    return TestClient(
        create_local_app(
            LocalAppConfig(
                data_dir=tmp_path / "data",
                embedding_provider=FakeEmbeddingProvider(),
                sam3_provider=FakeSam3(),
                start_workers=True,
                worker_poll_interval=0.05,
            )
        ),
        raise_server_exceptions=False,
    )


def labeled_dataset(client):
    asset_id = upload_asset(client, "Yard", ["red", "orange", "magenta"])
    job_id = submit_label_job(client, asset_id)
    wait_for(client, f"/v1/label-jobs/{job_id}")
    created = client.post(
        "/v1/datasets", headers=headers(),
        json={"job_id": job_id, "name": "yard-v1", "idempotency_key": "dataset-1"},
    )
    assert created.status_code == 201, created.text
    return created.json()["version_id"]


def test_review_page_starts_pending_and_shows_live_annotations(tmp_path):
    with make_client(tmp_path) as client:
        version_id = labeled_dataset(client)
        page = client.get(f"/v1/datasets/{version_id}/review", headers=headers()).json()
        assert page["counts"] == {"total": 3, "pending": 3, "approved": 0, "rejected": 0}
        assert all(image["status"] == "pending" for image in page["images"])
        assert all(len(image["annotations"]) == 1 for image in page["images"])
        first = page["images"][0]
        annotation = first["annotations"][0]

        # A deletion shows up in the review view at once (no rebuild needed).
        deleted = client.delete(
            f"/v1/datasets/{version_id}/images/{quote(first['image_id'], safe='')}/annotations/{annotation['annotation_id']}",
            headers=headers(),
        )
        assert deleted.status_code == 200, deleted.text
        page = client.get(f"/v1/datasets/{version_id}/review", headers=headers()).json()
        assert next(image for image in page["images"] if image["image_id"] == first["image_id"])["annotations"] == []


def test_approve_selected_reject_and_approve_all_pending(tmp_path):
    with make_client(tmp_path) as client:
        version_id = labeled_dataset(client)
        ids = [image["image_id"] for image in client.get(f"/v1/datasets/{version_id}/review", headers=headers()).json()["images"]]

        approved = client.post(
            f"/v1/datasets/{version_id}/review", headers=headers(),
            json={"status": "approved", "image_ids": ids[:1]},
        ).json()
        assert approved["counts"]["approved"] == 1
        client.post(
            f"/v1/datasets/{version_id}/review", headers=headers(),
            json={"status": "rejected", "image_ids": [ids[1]]},
        )
        rest = client.post(
            f"/v1/datasets/{version_id}/review", headers=headers(),
            json={"status": "approved", "all": True, "only_status": "pending"},
        ).json()
        assert rest["updated"] == 1
        assert rest["counts"] == {"total": 3, "pending": 0, "approved": 2, "rejected": 1}

        rejected = client.get(f"/v1/datasets/{version_id}/review?status=rejected", headers=headers()).json()
        assert [image["image_id"] for image in rejected["images"]] == [ids[1]]

        reset = client.post(
            f"/v1/datasets/{version_id}/review", headers=headers(),
            json={"status": "pending", "image_ids": [ids[1]]},
        ).json()
        assert reset["counts"]["pending"] == 1

        bad = client.post(
            f"/v1/datasets/{version_id}/review", headers=headers(),
            json={"status": "approved", "image_ids": ["not-in-dataset"]},
        )
        empty = client.post(f"/v1/datasets/{version_id}/review", headers=headers(), json={"status": "approved"})
        assert bad.status_code == 404
        assert empty.status_code == 422


def test_rebuild_approved_keeps_only_approved_images_and_their_corrections(tmp_path):
    with make_client(tmp_path) as client:
        version_id = labeled_dataset(client)
        images = client.get(f"/v1/datasets/{version_id}/review", headers=headers()).json()["images"]
        none = client.post(
            f"/v1/datasets/{version_id}/rebuild-approved", headers=headers(),
            json={"idempotency_key": "approved-0"},
        )
        assert none.status_code == 422

        keep = [images[0]["image_id"], images[2]["image_id"]]
        client.post(
            f"/v1/datasets/{version_id}/review", headers=headers(),
            json={"status": "approved", "image_ids": keep},
        )
        # Fix one approved image: remove its (wrong) mask.
        wrong = images[2]["annotations"][0]["annotation_id"]
        client.delete(
            f"/v1/datasets/{version_id}/images/{quote(images[2]['image_id'], safe='')}/annotations/{wrong}",
            headers=headers(),
        )
        rebuilt = client.post(
            f"/v1/datasets/{version_id}/rebuild-approved", headers=headers(),
            json={"idempotency_key": "approved-1"},
        )
        assert rebuilt.status_code == 201, rebuilt.text
        body = rebuilt.json()
        assert body["image_count"] == 2
        assert body["annotation_count"] == 1
        assert body["name"] == "yard-v1 — approved"

        # Approvals belong to the annotation set, so the new version keeps them.
        page = client.get(f"/v1/datasets/{body['version_id']}/review", headers=headers()).json()
        assert page["counts"] == {"total": 2, "pending": 0, "approved": 2, "rejected": 0}

        with_pending = client.post(
            f"/v1/datasets/{version_id}/rebuild-approved", headers=headers(),
            json={"idempotency_key": "approved-2", "include_pending": True},
        ).json()
        assert with_pending["image_count"] == 3


def test_change_class_creates_a_revision(tmp_path):
    with make_client(tmp_path) as client:
        version_id = labeled_dataset(client)
        image = client.get(f"/v1/datasets/{version_id}/review", headers=headers()).json()["images"][0]
        annotation = image["annotations"][0]
        path = (
            f"/v1/datasets/{version_id}/images/{quote(image['image_id'], safe='')}"
            f"/annotations/{annotation['annotation_id']}"
        )
        unknown_class = client.patch(path, headers=headers(), json={"category_id": 7})
        assert unknown_class.status_code == 422
        same = client.patch(path, headers=headers(), json={"category_id": 0, "expected_revision": annotation["revision"]})
        assert same.status_code == 200, same.text
        assert same.json()["revision"] == annotation["revision"] + 1
        stale = client.patch(path, headers=headers(), json={"category_id": 0, "expected_revision": annotation["revision"]})
        assert stale.status_code == 409


def test_legacy_version_reviews_by_version_key(tmp_path):
    with make_client(tmp_path) as client:
        # A version created before annotation sets existed (no set id).
        from uuid import UUID

        from agentic_cv.dataset_store import DatasetVersionStore

        store = DatasetVersionStore(str(tmp_path / "data" / "metadata" / "versions.sqlite3"))
        record = store.create(
            tenant_id=UUID(headers()["X-Tenant-ID"]),
            name="legacy",
            dataset=DatasetVersion(
                images=(DatasetImage(image_id="legacy-1", file_name="images/a.png", width=8, height=8),),
                categories=(DatasetCategory(category_id=1, name="thing"),),
                annotations=(),
            ),
            idempotency_key="legacy",
        )
        store.close()
        updated = client.post(
            f"/v1/datasets/{record.version_id}/review", headers=headers(),
            json={"status": "approved", "all": True},
        ).json()
        assert updated["counts"]["approved"] == 1
