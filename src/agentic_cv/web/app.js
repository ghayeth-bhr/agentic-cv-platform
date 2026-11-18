(function () {
  "use strict";

  const $ = (id) => document.getElementById(id);

  function tenantId() {
    return $("tenant-id").value.trim();
  }

  function headersJSON(withTenant) {
    const h = { "Content-Type": "application/json" };
    if (withTenant) h["X-Tenant-ID"] = tenantId();
    return h;
  }

  async function request(path, options = {}) {
    const response = await fetch(path, options);
    const text = await response.text();
    let body;
    try { body = text ? JSON.parse(text) : null; } catch { body = { detail: text || response.statusText }; }
    if (!response.ok) {
      const detail = body && (body.detail || body.message) ? (body.detail || body.message) : `Request failed (${response.status})`;
      throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
    }
    return body;
  }

  function setStatus(id, message, kind) {
    const el = $(id);
    if (!el) return;
    el.textContent = message || "";
    if (kind) el.dataset.kind = kind; else delete el.dataset.kind;
  }

  function showDetails(id, entries) {
    const el = $(id);
    if (!el) return;
    el.innerHTML = "";
    if (!entries || !entries.length) { el.hidden = true; return; }
    for (const [k, v] of entries) {
      const dt = document.createElement("dt"); dt.textContent = k;
      const dd = document.createElement("dd"); dd.textContent = v == null ? "—" : String(v);
      el.append(dt, dd);
    }
    el.hidden = false;
  }

  function setBusy(button, busy, label) {
    if (!button) return;
    button.disabled = busy;
    if (label) button.dataset.label = label;
    if (busy) { button.dataset.orig = button.textContent; button.textContent = "Working…"; }
    else if (button.dataset.orig) { button.textContent = button.dataset.orig; delete button.dataset.orig; }
  }

  // Runtime check
  async function checkRuntime() {
    const chip = $("runtime-chip");
    try {
      const info = await request("/api-info");
      setStatus("runtime-status", info && info.service ? `Ready · ${info.service}` : "Runtime ready");
      if (chip) { chip.classList.add("ready"); chip.classList.remove("error"); }
    } catch {
      setStatus("runtime-status", "Runtime unavailable");
      if (chip) { chip.classList.add("error"); chip.classList.remove("ready"); }
    }
  }

  // File info helpers
  function bindFileInfo(inputId, infoId) {
    const input = $(inputId), info = $(infoId);
    if (!input || !info) return;
    input.addEventListener("change", () => {
      const f = input.files && input.files[0];
      if (!f) { info.textContent = "No file selected."; return; }
      const kb = (f.size / 1024).toFixed(1);
      info.textContent = `${f.name} · ${kb} KB · ${f.type || "unknown type"}`;
    });
  }
  bindFileInfo("asset-file", "asset-file-info");

  async function sha256Hex(buffer) {
    const hash = await crypto.subtle.digest("SHA-256", buffer);
    return Array.from(new Uint8Array(hash)).map((b) => b.toString(16).padStart(2, "0")).join("");
  }

  // 1) Ingest
  $("upload-button").addEventListener("click", async () => {
    const btn = $("upload-button");
    const file = $("asset-file").files && $("asset-file").files[0];
    const name = $("asset-name").value.trim() || "Untitled collection";
    if (!file) { setStatus("upload-status", "Choose an image file first.", "error"); return; }
    setBusy(btn, true);
    showDetails("upload-details", null);
    try {
      setStatus("upload-status", "Hashing file…", "busy");
      const bytes = await file.arrayBuffer();
      const digest = await sha256Hex(bytes);
      const h = headersJSON(true);

      setStatus("upload-status", "Creating asset…", "busy");
      const asset = await request("/v1/assets", { method: "POST", headers: h, body: JSON.stringify({ name }) });

      setStatus("upload-status", "Starting upload…", "busy");
      const upload = await request(`/v1/assets/${asset.asset_id}/uploads`, {
        method: "POST", headers: h,
        body: JSON.stringify({ filename: file.name, media_type: file.type || "application/octet-stream", expected_sha256: digest })
      });

      setStatus("upload-status", "Uploading bytes…", "busy");
      await request(`/v1/uploads/${upload.upload_id}/parts/1`, { method: "PUT", body: file });

      setStatus("upload-status", "Completing upload…", "busy");
      const source = await request(`/v1/uploads/${upload.upload_id}/complete`, { method: "POST" });

      setStatus("upload-status", "Upload complete.", "ok");
      showDetails("upload-details", [
        ["Asset", asset.asset_id],
        ["Upload", upload.upload_id],
        ["Source", source.source_id],
        ["SHA-256", (source.content_sha256 || digest).slice(0, 16) + "…"],
      ]);
    } catch (err) {
      setStatus("upload-status", err.message || String(err), "error");
    } finally {
      setBusy(btn, false);
    }
  });

  // 2) SAM3 prediction with box overlay
  function drawPredictions(canvas, img, predictions) {
    const ctx = canvas.getContext("2d");
    canvas.width = img.naturalWidth || img.width;
    canvas.height = img.naturalHeight || img.height;
    ctx.drawImage(img, 0, 0);
    const colors = ["#00c2d1", "#ff6b6b", "#ffd166", "#7bd389", "#c792ea", "#4dabf7"];
    (predictions || []).forEach((p, i) => {
      const color = colors[i % colors.length];
      const box = p.box || [0, 0, 0, 0];
      const [x1, y1, x2, y2] = box;
      const w = Math.max(0, x2 - x1), h = Math.max(0, y2 - y1);
      ctx.lineWidth = Math.max(2, Math.round(canvas.width / 300));
      ctx.strokeStyle = color;
      ctx.strokeRect(x1, y1, w, h);
      const label = `#${i + 1} ${(p.score != null ? Number(p.score).toFixed(3) : "")}`;
      ctx.font = `${Math.max(12, Math.round(canvas.width / 50))}px "IBM Plex Mono", monospace`;
      const pad = 4;
      const tw = ctx.measureText(label).width;
      const th = Math.max(14, Math.round(canvas.width / 45));
      const ly = Math.max(th + pad, y1);
      ctx.fillStyle = color;
      ctx.fillRect(x1, ly - th - pad, tw + pad * 2, th + pad);
      ctx.fillStyle = "#0f1418";
      ctx.fillText(label, x1 + pad, ly - pad / 2);
    });
  }

  $("predict-button").addEventListener("click", async () => {
    const btn = $("predict-button");
    const file = $("predict-file").files && $("predict-file").files[0];
    const prompt = $("predict-prompt").value.trim();
    const thresholdRaw = $("predict-threshold").value;
    if (!file) { setStatus("predict-status", "Choose an image file first.", "error"); return; }
    if (!prompt) { setStatus("predict-status", "Enter a prompt.", "error"); return; }
    setBusy(btn, true);
    setStatus("predict-status", "Running prediction…", "busy");
    $("predict-output").hidden = true;
    try {
      const form = new FormData();
      form.append("image", file);
      form.append("prompt", prompt);
      if (thresholdRaw !== "") form.append("threshold", thresholdRaw);
      const result = await request("/v1/predict", { method: "POST", body: form });

      const predictions = (result && result.predictions) || [];
      const img = new Image();
      const url = URL.createObjectURL(file);
      await new Promise((resolve, reject) => { img.onload = resolve; img.onerror = reject; img.src = url; });
      drawPredictions($("predict-canvas"), img, predictions);
      URL.revokeObjectURL(url);

      const summary = $("predict-summary");
      summary.innerHTML = "";
      if (!predictions.length) {
        summary.textContent = "No predictions returned.";
      } else {
        const intro = document.createElement("div");
        intro.textContent = `${predictions.length} prediction${predictions.length === 1 ? "" : "s"} for “${result.prompt || prompt}”.`;
        summary.appendChild(intro);
        predictions.forEach((p, i) => {
          const pill = document.createElement("span");
          pill.className = "pill";
          const box = (p.box || []).map((n) => Number(n).toFixed(1)).join(", ");
          pill.textContent = `#${i + 1} score ${Number(p.score).toFixed(3)} · [${box}]`;
          summary.appendChild(pill);
        });
      }
      $("predict-raw").textContent = JSON.stringify(result, null, 2);
      $("predict-output").hidden = false;
      setStatus("predict-status", predictions.length ? "Prediction complete." : "Prediction complete (no detections).", "ok");
    } catch (err) {
      setStatus("predict-status", err.message || String(err), "error");
    } finally {
      setBusy(btn, false);
    }
  });

  // 3) Semantic search
  function renderSearchResults(results) {
    const container = $("search-results");
    container.innerHTML = "";
    if (!results || !results.length) {
      const empty = document.createElement("div");
      empty.className = "results-empty";
      empty.textContent = "No results.";
      container.appendChild(empty);
      return;
    }
    const grid = document.createElement("div");
    grid.className = "result-gallery";
    results.forEach((r) => {
      const score = Number(r.score);
      const pct = Number.isFinite(score) ? Math.max(0, Math.min(100, score * 100)) : 0;
      const card = document.createElement("article");
      card.className = "result-card";

      const head = document.createElement("div");
      head.className = "result-card-head";
      const id = document.createElement("div");
      id.className = "result-id";
      id.textContent = r.record_id;
      const scoreEl = document.createElement("div");
      scoreEl.className = "result-score";
      scoreEl.textContent = Number.isFinite(score) ? score.toFixed(4) : "—";
      head.append(id, scoreEl);

      const meta = document.createElement("p");
      meta.className = "result-meta";
      meta.append("asset ");
      const asset = document.createElement("span");
      asset.textContent = r.asset_id || "—";
      meta.appendChild(asset);

      const bar = document.createElement("div");
      bar.className = "score-bar";
      const fill = document.createElement("i");
      fill.style.width = pct.toFixed(1) + "%";
      bar.appendChild(fill);

      card.append(head, meta, bar);
      grid.appendChild(card);
    });
    container.appendChild(grid);
  }

  function searchPayload() {
    const query = $("search-query").value.trim();
    const limitVal = parseInt($("search-limit").value, 10);
    const assetIdsRaw = $("search-assets").value.trim();
    const payload = { query, limit: Number.isFinite(limitVal) && limitVal > 0 ? limitVal : 10 };
    if (assetIdsRaw) {
      payload.asset_ids = assetIdsRaw.split(",").map((s) => s.trim()).filter(Boolean);
    }
    return payload;
  }

  $("search-button").addEventListener("click", async () => {
    const btn = $("search-button");
    const payload = searchPayload();
    if (!payload.query) { setStatus("search-status", "Enter a search query.", "error"); return; }
    setBusy(btn, true);
    setStatus("search-status", "Searching…", "busy");
    setStatus("subasset-status", "");
    try {
      const result = await request("/v1/search", { method: "POST", headers: headersJSON(false), body: JSON.stringify(payload) });
      const results = (result && result.results) || [];
      renderSearchResults(results);
      setStatus("search-status", `${results.length} result${results.length === 1 ? "" : "s"}.`, "ok");
    } catch (err) {
      $("search-results").innerHTML = "";
      setStatus("search-status", err.message || String(err), "error");
    } finally {
      setBusy(btn, false);
    }
  });

  $("subasset-button").addEventListener("click", async () => {
    const btn = $("subasset-button");
    const payload = searchPayload();
    if (!payload.query) { setStatus("subasset-status", "Enter a search query first.", "error"); return; }
    setBusy(btn, true);
    setStatus("subasset-status", "Creating sub-asset from search…", "busy");
    try {
      const record = await request("/v1/sub-assets/from-search", {
        method: "POST", headers: headersJSON(true), body: JSON.stringify(payload)
      });
      setStatus("subasset-status", `Sub-asset created: ${record.sub_asset_id} (${(record.result_ids || []).length} results).`, "ok");
    } catch (err) {
      setStatus("subasset-status", err.message || String(err), "error");
    } finally {
      setBusy(btn, false);
    }
  });

  // 4) Dataset export
  function renderExportJob(job) {
    if (!job) { showDetails("export-details", null); return; }
    showDetails("export-details", [
      ["Job", job.job_id],
      ["Kind", job.kind],
      ["Status", job.status],
      ["Version", job.version_id],
      ["Format", job.format],
      ["Created", job.created_at],
      ["Updated", job.updated_at],
    ]);
  }

  $("export-button").addEventListener("click", async () => {
    const btn = $("export-button");
    const versionId = $("version-id").value.trim();
    const format = $("export-format").value;
    const idempotencyKey = $("export-key").value.trim() || "browser-export";
    if (!versionId) { setStatus("export-status", "Enter a dataset version ID.", "error"); return; }
    setBusy(btn, true);
    setStatus("export-status", "Queuing export…", "busy");
    try {
      const job = await request("/v1/dataset-exports", {
        method: "POST", headers: headersJSON(true),
        body: JSON.stringify({ version_id: versionId, format, idempotency_key: idempotencyKey })
      });
      renderExportJob(job);
      $("job-id").value = job.job_id || "";
      setStatus("export-status", `Export job ${job.status}.`, "ok");
    } catch (err) {
      setStatus("export-status", err.message || String(err), "error");
    } finally {
      setBusy(btn, false);
    }
  });

  $("job-status-button").addEventListener("click", async () => {
    const btn = $("job-status-button");
    const jobId = $("job-id").value.trim();
    if (!jobId) { setStatus("export-status", "Enter a job ID.", "error"); return; }
    setBusy(btn, true);
    setStatus("export-status", "Fetching job status…", "busy");
    try {
      const job = await request(`/v1/dataset-exports/${encodeURIComponent(jobId)}`, { headers: { "X-Tenant-ID": tenantId() } });
      renderExportJob(job);
      setStatus("export-status", `Job is ${job.status}.`, "ok");
    } catch (err) {
      setStatus("export-status", err.message || String(err), "error");
    } finally {
      setBusy(btn, false);
    }
  });

  $("job-cancel-button").addEventListener("click", async () => {
    const btn = $("job-cancel-button");
    const jobId = $("job-id").value.trim();
    if (!jobId) { setStatus("export-status", "Enter a job ID.", "error"); return; }
    setBusy(btn, true);
    setStatus("export-status", "Requesting cancellation…", "busy");
    try {
      const job = await request(`/v1/dataset-exports/${encodeURIComponent(jobId)}/cancel`, {
        method: "POST", headers: { "X-Tenant-ID": tenantId() }
      });
      renderExportJob(job);
      setStatus("export-status", `Job is ${job.status}.`, "ok");
    } catch (err) {
      setStatus("export-status", err.message || String(err), "error");
    } finally {
      setBusy(btn, false);
    }
  });

  // 5) Preview gallery
  function parseSseEvents(text) {
    const events = [];
    const blocks = String(text || "").split(/\n\n+/);
    for (const block of blocks) {
      if (!block.trim()) continue;
      let eventId = null;
      let data = null;
      for (const line of block.split(/\n/)) {
        if (line.startsWith("id:")) {
          const n = Number(line.slice(3).trim());
          if (Number.isFinite(n)) eventId = n;
        } else if (line.startsWith("data:")) {
          const raw = line.slice(5).trim();
          try { data = JSON.parse(raw); } catch { data = null; }
        }
      }
      if (data) {
        if (eventId != null && data.event_id == null) data.event_id = eventId;
        events.push(data);
      }
    }
    return events;
  }

  function renderGallery(events) {
    const grid = $("gallery-grid");
    const summary = $("gallery-summary");
    grid.innerHTML = "";
    if (!events || !events.length) {
      summary.hidden = true;
      summary.innerHTML = "";
      return;
    }
    const last = events[events.length - 1];
    const total = Number(last.total) || 0;
    const processed = Number(last.processed) || 0;
    summary.hidden = false;
    summary.innerHTML = "";
    const parts = [
      ["Events", String(events.length)],
      ["Processed", total ? `${processed} / ${total}` : String(processed)],
      ["Latest status", String(last.status || "—")],
      ["Job", String(last.job_id || "—")],
    ];
    for (const [k, v] of parts) {
      const span = document.createElement("span");
      span.append(`${k} `);
      const strong = document.createElement("strong");
      strong.textContent = v;
      span.appendChild(strong);
      summary.appendChild(span);
    }

    for (const ev of events) {
      const evTotal = Number(ev.total) || 0;
      const evProcessed = Number(ev.processed) || 0;
      const pct = evTotal > 0 ? Math.max(0, Math.min(100, (evProcessed / evTotal) * 100)) : 0;
      const status = ev.status || "unknown";

      const card = document.createElement("article");
      card.className = "gallery-card";
      card.dataset.status = status;

      const head = document.createElement("div");
      head.className = "gallery-card-head";
      const title = document.createElement("div");
      title.className = "gallery-card-title";
      title.textContent = ev.image_id || "unknown image";
      const sub = document.createElement("div");
      sub.className = "gallery-card-sub";
      sub.textContent = ev.event_id != null ? `event #${ev.event_id}` : "event";
      head.append(title, sub);

      const body = document.createElement("div");
      body.className = "gallery-card-body";
      const pill = document.createElement("span");
      pill.className = "status-pill";
      pill.dataset.status = status;
      pill.textContent = status;
      const kv = document.createElement("dl");
      kv.className = "gallery-kv";
      const rows = [
        ["Progress", evTotal ? `${evProcessed} / ${evTotal}` : String(evProcessed)],
        ["Created", ev.created_at || "—"],
      ];
      for (const [k, v] of rows) {
        const dt = document.createElement("dt"); dt.textContent = k;
        const dd = document.createElement("dd"); dd.textContent = v;
        kv.append(dt, dd);
      }
      body.append(pill, kv);

      const foot = document.createElement("div");
      foot.className = "gallery-card-foot";
      const bar = document.createElement("div");
      bar.className = "gallery-progress";
      const fill = document.createElement("i");
      fill.style.width = `${pct.toFixed(1)}%`;
      bar.appendChild(fill);
      foot.appendChild(bar);

      card.append(head, body, foot);
      grid.appendChild(card);
    }
  }

  $("gallery-load-button").addEventListener("click", async () => {
    const btn = $("gallery-load-button");
    const jobId = $("gallery-job-id").value.trim();
    const afterRaw = $("gallery-after").value.trim();
    if (!jobId) { setStatus("gallery-status", "Enter a job ID.", "error"); return; }
    const params = new URLSearchParams();
    if (afterRaw !== "") params.set("after_event_id", afterRaw);
    const qs = params.toString();
    const path = `/v1/jobs/${encodeURIComponent(jobId)}/preview${qs ? `?${qs}` : ""}`;
    setBusy(btn, true);
    setStatus("gallery-status", "Loading preview events…", "busy");
    try {
      const response = await fetch(path, { headers: { "X-Tenant-ID": tenantId() } });
      const text = await response.text();
      if (!response.ok) {
        let body = null;
        try { body = text ? JSON.parse(text) : null; } catch { body = null; }
        const detail = body && body.detail ? body.detail : (text || `Request failed (${response.status})`);
        throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
      }
      const events = parseSseEvents(text);
      renderGallery(events);
      if (!events.length) {
        setStatus("gallery-status", "No preview events found for this job and tenant. The job may not emit previews, or none match the cursor.", "ok");
      } else {
        setStatus("gallery-status", `Loaded ${events.length} preview event${events.length === 1 ? "" : "s"}.`, "ok");
      }
    } catch (err) {
      renderGallery([]);
      setStatus("gallery-status", err.message || String(err), "error");
    } finally {
      setBusy(btn, false);
    }
  });

  $("gallery-use-export-job-button").addEventListener("click", () => {
    const exportJobId = $("job-id").value.trim();
    if (!exportJobId) { setStatus("gallery-status", "No export job ID to copy.", "error"); return; }
    $("gallery-job-id").value = exportJobId;
    setStatus("gallery-status", "Copied export job ID into gallery. Note: export jobs may not emit preview events.", "ok");
  });

  $("gallery-clear-button").addEventListener("click", () => {
    $("gallery-grid").innerHTML = "";
    $("gallery-summary").hidden = true;
    $("gallery-summary").innerHTML = "";
    setStatus("gallery-status", "Gallery cleared.", "ok");
  });

  checkRuntime();
})();
