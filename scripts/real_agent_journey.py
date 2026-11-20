"""Real-model journey: LLM dialogue -> per-class VLM calibration -> SAM 3 labeling -> YOLO export.

Uses the production wiring (LocalAppConfig.from_environment) with real Ollama,
real SAM 3, real SigLIP indexing, and the real VLM validator. Not collected by
pytest. Prints one JSON line per step and a final ``REPORT {...}`` line.

    python scripts/real_agent_journey.py <repo> <folder of .jpg images> <fresh data dir>

The chat answers are scripted for an elephants/tusks wildlife set; edit
``say(...)`` calls for other data.
"""

import hashlib
import io
import json
import os
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path

repo = Path(sys.argv[1])
images_dir = Path(sys.argv[2])
data_dir = Path(sys.argv[3])
sys.path.insert(0, str(repo / "src"))
os.environ.setdefault("AGENTIC_CV_CALIBRATION_TIMEOUT_SEC", "900")

from fastapi.testclient import TestClient  # noqa: E402

from agentic_cv.application import LocalAppConfig, create_local_app  # noqa: E402

TENANT = {"X-Tenant-ID": "00000000-0000-0000-0000-000000000001"}
report: dict = {"steps": []}
peak = {"mb": 0}
stop = threading.Event()


def gpu_used() -> int:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        return int(out.splitlines()[0])
    except Exception:
        return -1


def watch_gpu():
    while not stop.is_set():
        peak["mb"] = max(peak["mb"], gpu_used())
        time.sleep(1)


def log(name, started, **extra):
    entry = {"step": name, "sec": round(time.perf_counter() - started, 2), **extra}
    report["steps"].append(entry)
    print(json.dumps(entry, ensure_ascii=False), flush=True)


def upload(client, name, files):
    asset = client.post("/v1/assets", headers=TENANT, json={"name": name}).json()["asset_id"]
    for path in files:
        payload = path.read_bytes()
        up = client.post(
            f"/v1/assets/{asset}/uploads", headers=TENANT,
            json={"filename": path.name, "media_type": "image/jpeg",
                  "expected_sha256": hashlib.sha256(payload).hexdigest()},
        ).json()["upload_id"]
        client.put(f"/v1/uploads/{up}/parts/1", files={"part": (path.name, io.BytesIO(payload), "image/jpeg")})
        assert client.post(f"/v1/uploads/{up}/complete").status_code == 200
    return asset


def say(client, session, message):
    started = time.perf_counter()
    if session is None:
        response = client.post("/v1/agent/sessions", headers=TENANT,
                               json={"asset_ids": ASSETS, "message": message})
    else:
        response = client.post(f"/v1/agent/sessions/{session}/messages", headers=TENANT,
                               json={"message": message})
    body = response.json()
    log("chat", started, user=message, status=body.get("status"), planner=body.get("planner_mode"),
        classes=[(c["name"], c["prompts"]) for c in body.get("classes", [])],
        task_type=body.get("task_type"), export=body.get("export_format"),
        reply=body["messages"][-1]["content"])
    return body


files = sorted(images_dir.glob("*.jpg"))
threading.Thread(target=watch_gpu, daemon=True).start()
report["gpu_before_mb"] = gpu_used()
app = create_local_app(LocalAppConfig.from_environment(data_dir))
with TestClient(app) as client:
    t = time.perf_counter()
    ASSETS = [upload(client, "Savanna clip A", files[:5]), upload(client, "Savanna clip B", files[5:])]
    log("upload", t, images=len(files))
    t = time.perf_counter()
    for asset in ASSETS:
        response = client.post(f"/v1/assets/{asset}/index", headers=TENANT)
        assert response.status_code == 200, response.text
    log("index (SigLIP)", t)

    s = say(client, None, "I want to build a dataset from these wildlife clips")
    sid = s["session_id"]
    if s["status"] == "gathering":
        s = say(client, sid, "elephants and their tusks")
    if s["status"] == "gathering":
        s = say(client, sid, "Segmentation, and export it as YOLO")
    report["plan"] = {"status": s["status"], "classes": s["classes"], "task_type": s["task_type"],
                      "export_format": s["export_format"]}
    assert s["status"] == "awaiting_confirmation", s["messages"][-1]

    t = time.perf_counter()
    # wait=True: run calibration inline (the default returns at once since Phase 2b).
    confirmed = client.post(f"/v1/agent/sessions/{sid}/confirm", headers=TENANT, json={"wait": True})
    body = confirmed.json()
    calibration = body.get("calibration") or {}
    log("confirm+calibration", t, http=confirmed.status_code, status=body.get("status"),
        validator=calibration.get("validator_mode"),
        per_class=[{"name": c.get("name"), "prompt": c.get("prompt"), "status": c.get("status"), "prompts_tried": c.get("prompts_tried"), "forward_passes": c.get("forward_passes"),
                    "threshold": c.get("threshold"),
                    "attempts": [(a["threshold"], f'{a["accepted_count"]}/{a["sample_count"]}') for a in c.get("attempts", [])],
                    "verdicts_at_choice": [(v.get("image_id","").split(":")[1][:8], v.get("accepted"), v.get("prediction_count"), v.get("target_present"), (v.get("reasoning") or "")[:90]) for a in c.get("attempts", []) if a["threshold"] == c.get("threshold") for v in a.get("verdicts", [])]}
                   for c in calibration.get("classes", [])],
        last_message=body["messages"][-1]["content"] if body.get("messages") else None)

    if body.get("status") == "labeling":
        t = time.perf_counter()
        while True:
            body = client.get(f"/v1/agent/sessions/{sid}", headers=TENANT).json()
            if body["status"] in ("complete", "failed", "cancelled", "needs_attention"):
                break
            time.sleep(1)
        log("label+build+export", t, status=body["status"], dataset=body.get("dataset"),
            last_message=body["messages"][-1]["content"])
        if body.get("download_path"):
            archive = client.get(body["download_path"], headers=TENANT).content
            with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
                names = bundle.namelist()
                labels = [n for n in names if n.startswith("labels/") and n.endswith(".txt")]
                lines = [line for n in labels for line in bundle.read(n).decode().splitlines()]
                report["archive"] = {
                    "bytes": len(archive),
                    "data_yaml": bundle.read("data.yaml").decode(),
                    "label_files": len(labels),
                    "instances_per_class": {k: sum(1 for l in lines if l.split()[0] == k) for k in sorted({l.split()[0] for l in lines})},
                    "splits": sorted({n.split("/")[1] for n in names if n.startswith("images/")}),
                }
    report["transcript"] = [(m["role"], m["content"]) for m in body.get("messages", [])]
stop.set()
report["gpu_peak_mb"] = peak["mb"]
print("REPORT " + json.dumps(report, ensure_ascii=False), flush=True)
