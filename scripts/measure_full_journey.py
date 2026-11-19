"""Measure the full real-model HTTP journey on the configured device.

This is a manual performance harness. It is intentionally not collected by
pytest: it downloads model weights, starts workers, and can take minutes. It
prints machine-readable JSON with checkpoint timings and real CUDA memory.

Journey: upload images -> indexed semantic asset -> agent plan -> confirmation
-> calibration -> whole-asset labeling -> COCO download -> one click refinement
-> rebuild -> refined export download.

The measured numbers belong in ``docs/PERFORMANCE.md``; do not replace them with
targets.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import io
import json
import shutil
import socket
import subprocess
import threading
import time
import zipfile
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image, ImageDraw

TENANT_ID = "00000000-0000-0000-0000-000000000001"


class NvidiaSmiSampler:
    """Sample whole-GPU used memory, including Ollama's separate process."""

    def __init__(self, interval_sec: float = 0.25) -> None:
        self.executable = shutil.which("nvidia-smi")
        self.interval_sec = interval_sec
        self.latest_used_mb: float | None = None
        self.total_mb: float | None = None
        self.peak_used_mb: float | None = None
        self.window_baseline_mb: float | None = None
        self.window_peak_mb: float | None = None
        self.calibration_baseline_mb: float | None = None
        self.calibration_peak_mb: float | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self.executable is None:
            return
        self._stop.clear()
        used = self._query_used()
        if used is not None:
            self.window_baseline_mb = used
            self.window_peak_mb = used
        self._thread = threading.Thread(
            target=self._run, name="nvidia-smi-sampler", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3)

    def mark(self) -> None:
        used = self._query_used()
        self.calibration_baseline_mb = used
        self.calibration_peak_mb = used

    def report(self) -> dict[str, object]:
        return {
            "available": self.executable is not None,
            "peak_used_mb": self.peak_used_mb,
            "total_mb": self.total_mb,
            "window_baseline_mb": self.window_baseline_mb,
            "window_peak_mb": self.window_peak_mb,
            "window_growth_mb": (
                round(self.window_peak_mb - self.window_baseline_mb, 1)
                if self.window_baseline_mb is not None
                and self.window_peak_mb is not None
                else None
            ),
            "calibration_baseline_mb": self.calibration_baseline_mb,
            "calibration_peak_mb": self.calibration_peak_mb,
            "calibration_growth_mb": (
                round(self.calibration_peak_mb - self.calibration_baseline_mb, 1)
                if self.calibration_baseline_mb is not None
                and self.calibration_peak_mb is not None
                else None
            ),
        }

    def _run(self) -> None:
        while not self._stop.wait(self.interval_sec):
            self._query_used()

    def _query_used(self) -> float | None:
        if self.executable is None:
            return None
        try:
            completed = subprocess.run(
                [
                    self.executable,
                    "--query-gpu=memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=3,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if completed.returncode != 0 or not completed.stdout.strip():
            return None
        first = completed.stdout.strip().splitlines()[0]
        parts = [part.strip() for part in first.split(",")]
        if len(parts) != 2:
            return None
        try:
            used = float(parts[0])
            total = float(parts[1])
        except ValueError:
            return None
        self.latest_used_mb = used
        self.total_mb = total
        self.peak_used_mb = used if self.peak_used_mb is None else max(self.peak_used_mb, used)
        if self.window_peak_mb is not None:
            self.window_peak_mb = max(self.window_peak_mb, used)
        if self.calibration_peak_mb is not None:
            self.calibration_peak_mb = max(self.calibration_peak_mb, used)
        return used


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@contextmanager
def _serve(app):
    import uvicorn

    port = _free_port()
    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=port,
        log_level="warning",
        lifespan="on",
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 30
    while not server.started and time.monotonic() < deadline:
        if not thread.is_alive():
            raise RuntimeError("uvicorn stopped before startup")
        time.sleep(0.05)
    if not server.started:
        raise RuntimeError("uvicorn did not start")
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=30)


def _synthetic_images(root: Path, count: int) -> list[Path]:
    root.mkdir(parents=True, exist_ok=True)
    paths = []
    for index in range(count):
        image = Image.new("RGB", (720, 1280), (48, 92, 48))
        draw = ImageDraw.Draw(image)
        x = 120 + (index * 31) % 360
        y = 260 + (index * 47) % 620
        draw.rectangle((x, y, x + 210, y + 430), fill=(190, 150, 100))
        draw.ellipse((x + 40, y - 80, x + 170, y + 40), fill=(220, 185, 155))
        draw.rectangle((x + 20, y + 420, x + 90, y + 620), fill=(70, 45, 30))
        draw.rectangle((x + 120, y + 420, x + 190, y + 620), fill=(70, 45, 30))
        path = root / f"frame-{index:03d}.jpg"
        image.save(path, quality=90)
        paths.append(path)
    return paths


def _images_from_source(source: Path, root: Path, count: int) -> list[Path]:
    candidates = sorted(
        path
        for path in source.rglob("*")
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not candidates:
        return _synthetic_images(root, count)
    root.mkdir(parents=True, exist_ok=True)
    paths = []
    for index in range(count):
        source_path = candidates[index % len(candidates)]
        target = root / f"frame-{index:03d}.jpg"
        with Image.open(source_path) as image:
            image = image.convert("RGB")
            draw = ImageDraw.Draw(image)
            # Keep duplicate source frames distinct so uploads are not deduped.
            draw.point((index % image.width, index % image.height), fill=(255, 0, 0))
            image.save(target, quality=92)
        paths.append(target)
    return paths


def _peak(torch) -> dict[str, object]:
    if not torch.cuda.is_available():
        return {"cuda_available": False}
    return {
        "cuda_available": True,
        "peak_allocated_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 1),
        "peak_reserved_mb": round(torch.cuda.max_memory_reserved() / 1024**2, 1),
    }


async def _wait_index_job(client, headers, job: dict, *, timeout: float) -> dict:
    deadline = time.monotonic() + timeout
    current = job
    while time.monotonic() < deadline:
        if current["status"] in {"succeeded", "failed", "cancelled"}:
            if current["status"] != "succeeded":
                raise RuntimeError(f"index job failed: {current}")
            return current
        await asyncio.sleep(0.2)
        response = await client.get(
            f"/v1/index-jobs/{current['job_id']}", headers=headers
        )
        response.raise_for_status()
        current = response.json()
    raise TimeoutError(f"index job did not finish: {current}")


async def _wait_session(client, headers, session_id: str, *, timeout: float) -> dict:
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        response = await client.get(f"/v1/agent/sessions/{session_id}", headers=headers)
        response.raise_for_status()
        body = response.json()
        last = body
        if body["status"] in {"complete", "failed", "cancelled", "needs_attention"}:
            return body
        await asyncio.sleep(0.25)
    raise TimeoutError(f"agent session did not finish: {last}")


async def _upload_images(client, headers, paths: list[Path], name: str) -> str:
    response = await client.post("/v1/assets", headers=headers, json={"name": name})
    response.raise_for_status()
    asset_id = response.json()["asset_id"]
    for index, path in enumerate(paths):
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        media_type = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
        begin = await client.post(
            f"/v1/assets/{asset_id}/uploads",
            headers=headers,
            json={
                "filename": path.name,
                "media_type": media_type,
                "expected_sha256": digest,
            },
        )
        begin.raise_for_status()
        upload_id = begin.json()["upload_id"]
        part = await client.put(
            f"/v1/uploads/{upload_id}/parts/1",
            files={"part": (path.name, payload, media_type)},
        )
        part.raise_for_status()
        complete = await client.post(f"/v1/uploads/{upload_id}/complete")
        complete.raise_for_status()
        if (index + 1) % 5 == 0:
            print(f"uploaded {index + 1}/{len(paths)}", flush=True)
    return asset_id


async def _download(client, headers, path: str, target: Path) -> list[str]:
    response = await client.get(path, headers=headers)
    response.raise_for_status()
    target.write_bytes(response.content)
    with zipfile.ZipFile(io.BytesIO(response.content)) as bundle:
        return bundle.namelist()


async def _wait_export(client, headers, job_id: str, *, timeout: float) -> dict:
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        response = await client.get(f"/v1/dataset-exports/{job_id}", headers=headers)
        response.raise_for_status()
        body = response.json()
        last = body
        if body["status"] in {"succeeded", "failed", "cancelled"}:
            return body
        await asyncio.sleep(0.2)
    raise TimeoutError(f"export did not finish: {last}")


async def _run(args) -> dict:
    import httpx
    import torch
    import websockets

    from agentic_cv.agent import RulesTaskPlanner
    from agentic_cv.application import LocalAppConfig, create_local_app

    with TemporaryDirectory(ignore_cleanup_errors=True) as temp:
        root = Path(temp)
        source = Path(args.source).expanduser().resolve() if args.source else None
        images = (
            _images_from_source(source, root / "images", args.images)
            if source is not None
            else _synthetic_images(root / "images", args.images)
        )
        print(f"prepared {len(images)} images", flush=True)
        app = create_local_app(
            LocalAppConfig(
                data_dir=root / "data",
                agent_planner=RulesTaskPlanner(),
                calibration_timeout_sec=args.calibration_timeout,
                start_workers=True,
                worker_poll_interval=0.1,
            )
        )
        headers = {"X-Tenant-ID": TENANT_ID}
        report: dict[str, object] = {
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "gpu": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None
            ),
            "image_count": len(images),
            "prompt": args.prompt,
            "checkpoints": {},
        }
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        sampler = NvidiaSmiSampler()
        sampler.start()

        def checkpoint(name: str, started: float) -> None:
            elapsed = round(time.perf_counter() - started, 2)
            report["checkpoints"][name] = {  # type: ignore[index]
                "wall_sec": elapsed,
                **_peak(torch),
            }
            print(f"{name}: {elapsed}s", flush=True)

        with _serve(app) as base_url:
            async with httpx.AsyncClient(
                base_url=base_url,
                timeout=httpx.Timeout(args.calibration_timeout + 180, connect=30),
            ) as client:
                started = time.perf_counter()
                asset_id = await _upload_images(
                    client, headers, images, "Measured asset"
                )
                checkpoint("upload_and_embed", started)

                started = time.perf_counter()
                index = await client.post(
                    f"/v1/assets/{asset_id}/index-jobs", headers=headers
                )
                index.raise_for_status()
                await _wait_index_job(
                    client, headers, index.json(), timeout=args.timeout
                )
                checkpoint("index_job", started)
                api_info = (await client.get("/api-info")).json()
                report["api_info"] = api_info

                started = time.perf_counter()
                session = (
                    await client.post(
                        "/v1/agent/sessions",
                        headers=headers,
                        json={"asset_id": asset_id, "message": args.message},
                    )
                ).json()
                if session["status"] != "awaiting_confirmation":
                    raise RuntimeError(f"unexpected agent status: {session}")
                sampler.mark()
                confirmed = await client.post(
                    f"/v1/agent/sessions/{session['session_id']}/confirm",
                    headers=headers,
                    json={},
                )
                if confirmed.status_code >= 400:
                    raise RuntimeError(
                        f"confirmation failed: {confirmed.status_code} {confirmed.text}"
                    )
                confirmed_body = confirmed.json()
                calibration = confirmed_body.get("calibration") or {}
                verdicts = [
                    verdict
                    for attempt in calibration.get("attempts") or []
                    for verdict in attempt.get("verdicts") or []
                ]
                report["calibration"] = {
                    "status": calibration.get("status"),
                    "threshold": calibration.get("threshold"),
                    "validator_mode": calibration.get("validator_mode"),
                    "validator_model": calibration.get("validator_model"),
                    "notice": calibration.get("notice"),
                    "thresholds_tried": [
                        attempt.get("threshold")
                        for attempt in calibration.get("attempts") or []
                    ],
                    "attempts": len(calibration.get("attempts") or []),
                    "verdicts": len(verdicts),
                    "accepted_verdicts": sum(
                        1 for verdict in verdicts if verdict.get("accepted")
                    ),
                    "overlay_ids": [
                        verdict.get("overlay_id")
                        for verdict in verdicts
                        if verdict.get("overlay_id")
                    ],
                }
                if args.require_vlm and calibration.get("validator_mode") != "vlm":
                    raise RuntimeError(
                        "VLM validation was required but the run used "
                        f"{calibration.get('validator_mode')!r}: "
                        f"{calibration.get('notice')!r}"
                    )
                if args.require_vlm and (
                    not verdicts
                    or calibration.get("notice")
                    or any(not verdict.get("overlay_id") for verdict in verdicts)
                ):
                    raise RuntimeError(
                        "VLM validation was required but produced no visual "
                        f"verdict evidence: {report['calibration']}"
                    )
                completed = await _wait_session(
                    client,
                    headers,
                    session["session_id"],
                    timeout=args.timeout,
                )
                checkpoint("calibrate_label_export", started)
                if completed["status"] != "complete":
                    raise RuntimeError(f"agent failed: {completed}")
                download_path = completed["download_path"]
                names = await _download(
                    client, headers, download_path, root / "agent.zip"
                )
                if not any(name.endswith("annotations.json") for name in names):
                    raise RuntimeError(f"COCO export missing: {names}")
                report["agent"] = {
                    "image_count": completed["dataset"]["image_count"],
                    "annotation_count": completed["dataset"]["annotation_count"],
                    "download_bytes": (root / "agent.zip").stat().st_size,
                }

                version_id = completed["dataset_version_id"]
                listing = (
                    await client.get(
                        f"/v1/datasets/{version_id}/images", headers=headers
                    )
                ).json()["images"]
                target = next(
                    (item for item in listing if item.get("annotations")), None
                )
                if target is None:
                    report["refinement"] = {"status": "skipped_no_annotations"}
                else:
                    started = time.perf_counter()
                    bbox = target["annotations"][0]["bbox"]
                    x = float(bbox[0]) + max(1.0, float(bbox[2]) / 2)
                    y = float(bbox[1]) + max(1.0, float(bbox[3]) / 2)
                    ws_url = (
                        f"ws://127.0.0.1:{base_url.rsplit(':', 1)[1]}"
                        f"/v1/refinement/{version_id}/{target['image_id']}"
                        f"?tenant_id={TENANT_ID}"
                    )
                    async with websockets.connect(
                        ws_url, open_timeout=30, max_size=None
                    ) as socket:
                        initial = json.loads(await socket.recv())
                        await socket.send(
                            json.dumps(
                                {
                                    "type": "click",
                                    "expected_revision": initial["revision"],
                                    "x": x,
                                    "y": y,
                                    "label": "positive",
                                }
                            )
                        )
                        snapshot = json.loads(await socket.recv())
                    checkpoint("refinement_click", started)
                    if not snapshot.get("predictions"):
                        report["refinement"] = {
                            "status": "click_returned_no_prediction"
                        }
                    else:
                        accepted = await client.post(
                            f"/v1/refinements/{snapshot['session_id']}/accept",
                            headers=headers,
                            json={
                                "annotation_id": target["annotations"][0][
                                    "annotation_id"
                                ],
                                "prediction_index": 0,
                                "refinement_revision": snapshot["revision"],
                            },
                        )
                        accepted.raise_for_status()
                        rebuilt = await client.post(
                            f"/v1/datasets/{version_id}/rebuild",
                            headers=headers,
                            json={
                                "name": "Measured refined",
                                "idempotency_key": f"measure-refine-{time.time_ns()}",
                            },
                        )
                        rebuilt.raise_for_status()
                        refined_id = rebuilt.json()["version_id"]
                        export = await client.post(
                            "/v1/dataset-exports",
                            headers=headers,
                            json={
                                "version_id": refined_id,
                                "format": "coco",
                                "idempotency_key": f"measure-export-{time.time_ns()}",
                            },
                        )
                        export.raise_for_status()
                        export_job = await _wait_export(
                            client, headers, export.json()["job_id"], timeout=args.timeout
                        )
                        refined_names = await _download(
                            client,
                            headers,
                            f"/v1/dataset-exports/{export_job['job_id']}/download",
                            root / "refined.zip",
                        )
                        report["refinement"] = {
                            "status": "accepted_rebuilt_exported",
                            "revision": accepted.json()["revision"],
                            "annotations": len(
                                [n for n in refined_names if n.endswith(".json")]
                            ),
                            "download_bytes": (root / "refined.zip").stat().st_size,
                        }
                        checkpoint("refined_export", started)

                report["api_info_end"] = (await client.get("/api-info")).json()
        sampler.stop()
        report["peak"] = _peak(torch)
        report["nvidia_smi"] = sampler.report()
        print(json.dumps(report, indent=2), flush=True)
        if args.json_out is not None:
            args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=int, default=20)
    parser.add_argument("--source", default=None)
    parser.add_argument(
        "--message",
        default="Create a segmentation dataset of animals.",
    )
    parser.add_argument("--prompt", default="animals")
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--calibration-timeout", type=float, default=300)
    parser.add_argument("--require-vlm", action="store_true")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()
    asyncio.run(_run(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
