(function () {
  "use strict";

  const $ = (id) => document.getElementById(id);

  const VIEW_META = {
    gallery: { title: "Gallery", eyebrow: "Assets", desc: "Manage uploaded assets and their representative media." },
    upload: { title: "Upload", eyebrow: "Assets", desc: "Create an asset and upload an image or video source." },
    predict: { title: "Predict / Label", eyebrow: "Agent", desc: "Run prompted prediction or labeling on selected assets and frames." },
    search: { title: "Search", eyebrow: "Agent", desc: "Find indexed frames by description to scope assets for labeling." },
    exports: { title: "Exports", eyebrow: "Jobs & Exports", desc: "Export labeled datasets (available after labeling/dataset versions are ready)." },
    preview: { title: "Label jobs", eyebrow: "Jobs & Exports", desc: "Replay durable label-job preview events and progress." },
  };

  function tenantId() {
    const el = $("tenant-id");
    return el ? el.value.trim() : "";
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

  function setBusy(button, busy) {
    if (!button) return;
    button.disabled = !!busy;
    if (busy) {
      if (!button.dataset.orig) button.dataset.orig = button.textContent;
      button.textContent = "Working…";
    } else if (button.dataset.orig) {
      button.textContent = button.dataset.orig;
      delete button.dataset.orig;
    }
  }

  function shortId(id) {
    const s = String(id || "");
    return s.length > 12 ? s.slice(0, 8) + "…" : s;
  }

  function formatDate(iso) {
    if (!iso) return "—";
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return String(iso);
    return d.toLocaleString([], { year: "numeric", month: "short", day: "2-digit", hour: "2-digit", minute: "2-digit" });
  }

  // Navigation
  function showView(name) {
    const meta = VIEW_META[name] || VIEW_META.gallery;
    document.querySelectorAll(".nav-item").forEach((btn) => {
      const active = btn.dataset.view === name;
      btn.classList.toggle("is-active", active);
      if (active) btn.setAttribute("aria-current", "page"); else btn.removeAttribute("aria-current");
    });
    document.querySelectorAll(".view").forEach((section) => {
      const active = section.id === `view-${name}`;
      section.classList.toggle("is-active", active);
      section.hidden = !active;
    });
    if ($("view-title")) $("view-title").textContent = meta.title;
    if ($("view-eyebrow")) $("view-eyebrow").textContent = meta.eyebrow;
    if ($("view-desc")) $("view-desc").textContent = meta.desc;
    if (name === "gallery") loadAssets();
    if (name === "search") loadAssets({ picker: true, quiet: true });
    if (name === "predict") loadAssets({ picker: true, quiet: true });
  }

  document.querySelectorAll(".nav-item").forEach((btn) => {
    btn.addEventListener("click", () => showView(btn.dataset.view));
  });

  // Runtime check
  async function checkRuntime() {
    const chip = $("runtime-chip");
    try {
      const info = await request("/api-info");
      setStatus("runtime-status", info && info.name ? `Ready · ${info.name}` : "Runtime ready");
      if (chip) { chip.classList.add("ready"); chip.classList.remove("error"); }
    } catch {
      setStatus("runtime-status", "Runtime unavailable");
      if (chip) { chip.classList.add("error"); chip.classList.remove("ready"); }
    }
  }

  // Assets: load, render gallery and picker
  let cachedAssets = [];

  async function loadAssets(opts = {}) {
    const picker = !!opts.picker;
    const quiet = !!opts.quiet;
    const meta = $("gallery-meta");
    if (!picker && meta && !quiet) meta.textContent = "Loading assets…";
    try {
      const data = await request("/v1/assets", { headers: { "X-Tenant-ID": tenantId() } });
      cachedAssets = (data && data.assets) || [];
      renderAssetGallery(cachedAssets);
      renderAssetPicker(cachedAssets);
      if (meta) {
        const n = cachedAssets.length;
        meta.textContent = n ? `${n} asset${n === 1 ? "" : "s"}` : "No assets yet";
      }
      return cachedAssets;
    } catch (err) {
      if (!picker && meta) meta.textContent = "Could not load assets";
      if (!quiet) setStatus("upload-status", "", null);
      renderAssetGallery([]);
      renderAssetPicker([]);
      if (!quiet && $("asset-gallery")) {
        $("asset-gallery").innerHTML = "";
        const msg = document.createElement("div");
        msg.className = "muted";
        msg.textContent = `Could not load assets: ${err.message || err}`;
        $("asset-gallery").appendChild(msg);
      }
      return [];
    }
  }

  function renderAssetGallery(assets) {
    const grid = $("asset-gallery");
    if (!grid) return;
    grid.innerHTML = "";
    if (!assets || !assets.length) return;
    for (const asset of assets) {
      const card = document.createElement("article");
      card.className = "asset-card";

      const thumb = document.createElement("div");
      thumb.className = "asset-thumb";
      if (asset.thumbnail_url) {
        const img = document.createElement("img");
        img.src = asset.thumbnail_url;
        img.alt = asset.name || "Asset thumbnail";
        img.loading = "lazy";
        img.onerror = () => {
          thumb.innerHTML = "";
          thumb.appendChild(thumbFallback(asset));
        };
        thumb.appendChild(img);
      } else {
        thumb.appendChild(thumbFallback(asset));
      }
      const rep = (asset.sources || [])[0];
      if (rep && rep.media_type) {
        const badge = document.createElement("span");
        badge.className = "thumb-badge";
        badge.textContent = (rep.media_type || "").split("/")[0] || "media";
        thumb.appendChild(badge);
      }

      const body = document.createElement("div");
      body.className = "asset-body";
      const name = document.createElement("div");
      name.className = "asset-name";
      name.textContent = asset.name || "Untitled asset";
      name.title = asset.name || "";
      const meta = document.createElement("div");
      meta.className = "asset-meta";
      const srcCount = (asset.sources || []).length;
      meta.textContent = `${srcCount} source${srcCount === 1 ? "" : "s"} · ${formatDate(asset.created_at)}`;
      const id = document.createElement("div");
      id.className = "asset-id";
      id.textContent = asset.asset_id;
      id.title = asset.asset_id;

      const actions = document.createElement("div");
      actions.className = "asset-actions";
      const copyBtn = document.createElement("button");
      copyBtn.type = "button";
      copyBtn.className = "btn btn-secondary btn-xs";
      copyBtn.textContent = "Copy ID";
      copyBtn.addEventListener("click", async () => {
        try { await navigator.clipboard.writeText(String(asset.asset_id)); copyBtn.textContent = "Copied"; setTimeout(() => (copyBtn.textContent = "Copy ID"), 1200); }
        catch { copyBtn.textContent = asset.asset_id; }
      });
      const indexBtn = document.createElement("button");
      indexBtn.type = "button";
      indexBtn.className = "btn btn-secondary btn-xs";
      indexBtn.textContent = "Index for search";
      indexBtn.addEventListener("click", async () => {
        setBusy(indexBtn, true);
        setStatus("gallery-status", `Indexing ${asset.name || "asset"} for semantic search…`, "busy");
        try {
          const result = await request(`/v1/assets/${encodeURIComponent(asset.asset_id)}/index`, {
            method: "POST",
            headers: { "X-Tenant-ID": tenantId() },
          });
          setStatus("gallery-status", result.detail || `Indexed ${result.indexed_records || 0} record(s).`, "ok");
        } catch (err) {
          setStatus("gallery-status", err.message || String(err), "error");
        } finally {
          setBusy(indexBtn, false);
        }
      });
      actions.append(copyBtn, indexBtn);

      body.append(name, meta, id, actions);
      card.append(thumb, body);
      grid.appendChild(card);
    }
  }

  function thumbFallback(asset) {
    const wrap = document.createElement("div");
    wrap.className = "thumb-fallback";
    const rep = (asset.sources || [])[0];
    const kind = rep ? (rep.media_type || "source") : "no source";
    wrap.textContent = kind;
    return wrap;
  }

  function renderAssetPicker(assets) {
    const picker = $("asset-picker");
    if (picker) {
      const prev = new Set(selectedAssetIds());
      picker.innerHTML = "";
      if (assets && assets.length) {
        for (const asset of assets) {
          const label = document.createElement("label");
          label.className = "pick-item";
          const cb = document.createElement("input");
          cb.type = "checkbox";
          cb.value = asset.asset_id;
          if (prev.has(String(asset.asset_id))) cb.checked = true;
          label.appendChild(cb);

          if (asset.thumbnail_url) {
            const img = document.createElement("img");
            img.className = "pick-thumb";
            img.src = asset.thumbnail_url;
            img.alt = "";
            img.loading = "lazy";
            img.onerror = () => { img.replaceWith(pickThumbFallback()); };
            label.appendChild(img);
          } else {
            label.appendChild(pickThumbFallback());
          }

          const text = document.createElement("div");
          text.className = "pick-text";
          const name = document.createElement("div");
          name.className = "pick-name";
          name.textContent = asset.name || "Untitled asset";
          const id = document.createElement("div");
          id.className = "pick-id";
          id.textContent = shortId(asset.asset_id);
          id.title = asset.asset_id;
          text.append(name, id);
          label.appendChild(text);
          picker.appendChild(label);
        }
      }
    }
    renderPredictAssetSelect(assets || []);
  }

  function renderPredictAssetSelect(assets) {
    const select = $("predict-asset");
    if (!select) return;
    const prev = select.value;
    select.innerHTML = "";
    const none = document.createElement("option");
    none.value = "";
    none.textContent = assets && assets.length ? "Use uploaded image file" : "No assets available (upload a file)";
    select.appendChild(none);
    for (const asset of (assets || [])) {
      if (!asset || !asset.asset_id) continue;
      const opt = document.createElement("option");
      opt.value = asset.asset_id;
      const kind = asset.sources && asset.sources[0] ? ((asset.sources[0].media_type || "").split("/")[0] || "asset") : "asset";
      const count = (asset.sources || []).length;
      opt.textContent = `${asset.name || "Untitled asset"} (${count} image${count === 1 ? "" : "s"}/${kind})`;
      if (asset.thumbnail_url) opt.dataset.thumb = asset.thumbnail_url;
      select.appendChild(opt);
    }
    if (prev && Array.from(select.options).some((o) => o.value === prev)) select.value = prev;
    renderPredictSourceSelect(select.value);
  }

  function renderPredictSourceSelect(assetId) {
    const select = $("predict-source");
    if (!select) return;
    const prev = select.value;
    select.innerHTML = "";
    if (!assetId) {
      const opt = document.createElement("option");
      opt.value = "";
      opt.textContent = "Not applicable (upload a file)";
      select.appendChild(opt);
      select.disabled = true;
      return;
    }
    select.disabled = false;
    const asset = (cachedAssets || []).find((a) => String(a.asset_id) === String(assetId));
    const sources = asset && Array.isArray(asset.sources) ? asset.sources : [];
    const auto = document.createElement("option");
    auto.value = "";
    auto.textContent = sources.length > 1 ? "Representative (auto)" : "Selected source";
    select.appendChild(auto);
    sources.forEach((s, i) => {
      const opt = document.createElement("option");
      opt.value = s.source_id || "";
      const label = s.filename || s.source_id || `Source ${i + 1}`;
      const type = s.media_type ? ` · ${s.media_type}` : "";
      opt.textContent = `${i + 1}. ${label}${type}`;
      select.appendChild(opt);
    });
    if (prev && Array.from(select.options).some((o) => o.value === prev)) select.value = prev;
  }

  if ($("predict-asset")) $("predict-asset").addEventListener("change", (e) => {
    renderPredictSourceSelect(e.target.value);
  });

  function pickThumbFallback() {
    const d = document.createElement("div");
    d.className = "pick-thumb-fallback";
    d.textContent = "—";
    return d;
  }

  function selectedAssetIds() {
    const picker = $("asset-picker");
    if (!picker) return [];
    return Array.from(picker.querySelectorAll('input[type="checkbox"]:checked')).map((cb) => cb.value);
  }

  // File info helper
  function bindFileInfo(inputId, infoId) {
    const input = $(inputId), info = $(infoId);
    if (!input || !info) return;
    input.addEventListener("change", () => {
      const f = input.files && input.files[0];
      if (!f) { info.textContent = "No file selected."; return; }
      const mb = f.size / (1024 * 1024);
      const size = mb >= 1 ? `${mb.toFixed(2)} MB` : `${(f.size / 1024).toFixed(1)} KB`;
      info.textContent = `${f.name} · ${size} · ${f.type || "unknown type"}`;
    });
  }
  bindFileInfo("asset-file", "asset-file-info");

  async function sha256Hex(buffer) {
    const hash = await crypto.subtle.digest("SHA-256", buffer);
    return Array.from(new Uint8Array(hash)).map((b) => b.toString(16).padStart(2, "0")).join("");
  }

  // Upload
  function renderUploadResult(asset, source) {
    const container = $("upload-result");
    if (!container) return;
    container.innerHTML = "";
    container.classList.remove("muted");
    const card = document.createElement("div");
    card.className = "upload-result-card";

    const thumb = document.createElement("div");
    thumb.className = "asset-thumb";
    if (asset && asset.asset_id) {
      const img = document.createElement("img");
      img.src = `/v1/assets/${asset.asset_id}/thumbnail?tenant_id=${encodeURIComponent(tenantId())}`;
      img.alt = asset.name || "Uploaded asset";
      img.onerror = () => { thumb.innerHTML = ""; thumb.appendChild(thumbFallback({ sources: source ? [source] : [] })); };
      thumb.appendChild(img);
    } else {
      thumb.appendChild(thumbFallback({ sources: [] }));
    }

    const body = document.createElement("div");
    body.className = "asset-body";
    const name = document.createElement("div");
    name.className = "asset-name";
    name.textContent = (asset && asset.name) || "Uploaded asset";
    const id = document.createElement("div");
    id.className = "asset-id";
    id.textContent = asset ? asset.asset_id : "—";
    const meta = document.createElement("div");
    meta.className = "asset-meta";
    meta.textContent = source ? `${source.filename || "source"} · ${(source.media_type || "")}` : "Completed";
    body.append(name, meta, id);
    card.append(thumb, body);
    container.appendChild(card);
  }

  if ($("upload-button")) $("upload-button").addEventListener("click", async () => {
    const btn = $("upload-button");
    const file = $("asset-file").files && $("asset-file").files[0];
    const name = ($("asset-name").value || "").trim() || "Untitled asset";
    if (!file) { setStatus("upload-status", "Choose a file to upload.", "error"); return; }
    setBusy(btn, true);
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
      const partData = new FormData();
      partData.append("part", file, file.name);
      await request(`/v1/uploads/${upload.upload_id}/parts/1`, { method: "PUT", body: partData });

      setStatus("upload-status", "Completing upload…", "busy");
      const source = await request(`/v1/uploads/${upload.upload_id}/complete`, { method: "POST", headers: h });

      renderUploadResult(asset, { filename: file.name, media_type: file.type });
      await loadAssets({ quiet: true });

      setStatus("upload-status", "Upload complete. Indexing for semantic search…", "busy");
      try {
        const idx = await request(`/v1/assets/${encodeURIComponent(asset.asset_id)}/index`, {
          method: "POST",
          headers: { "X-Tenant-ID": tenantId() },
        });
        setStatus("upload-status", `Upload complete. ${idx.detail || "Indexed for search."}`, "ok");
      } catch (idxErr) {
        setStatus("upload-status", `Upload complete. Semantic indexing unavailable: ${idxErr.message || idxErr}`, "error");
      }
    } catch (err) {
      setStatus("upload-status", err.message || String(err), "error");
    } finally {
      setBusy(btn, false);
    }
  });

  if ($("gallery-refresh")) $("gallery-refresh").addEventListener("click", () => loadAssets());

  // Predict
  function drawPredictions(canvas, img, predictions, predSize) {
    const ctx = canvas.getContext("2d");
    canvas.width = img.naturalWidth || img.width;
    canvas.height = img.naturalHeight || img.height;
    ctx.drawImage(img, 0, 0);
    const predW = predSize && Number.isFinite(predSize.w) && predSize.w > 0 ? predSize.w : canvas.width;
    const predH = predSize && Number.isFinite(predSize.h) && predSize.h > 0 ? predSize.h : canvas.height;
    const sx = canvas.width / predW;
    const sy = canvas.height / predH;
    const colors = ["#18b6c4", "#ff6b6b", "#ffd166", "#7bd389", "#c792ea", "#4dabf7"];
    (predictions || []).forEach((p, i) => {
      const color = colors[i % colors.length];
      const box = p.box || [0, 0, 0, 0];
      const x1 = Number(box[0] || 0) * sx;
      const y1 = Number(box[1] || 0) * sy;
      const x2 = Number(box[2] || 0) * sx;
      const y2 = Number(box[3] || 0) * sy;
      const w = Math.max(0, x2 - x1), h = Math.max(0, y2 - y1);
      ctx.lineWidth = Math.max(2, Math.round(canvas.width / 300));
      ctx.strokeStyle = color;
      ctx.strokeRect(x1, y1, w, h);
      const label = `#${i + 1} ${(p.score != null ? Number(p.score).toFixed(3) : "")}`;
      ctx.font = `${Math.max(12, Math.round(canvas.width / 55))}px "IBM Plex Mono", monospace`;
      const pad = 4;
      const tw = ctx.measureText(label).width;
      const th = Math.max(14, Math.round(canvas.width / 45));
      const ly = Math.max(th + pad, y1);
      ctx.fillStyle = color;
      ctx.fillRect(x1, ly - th - pad, tw + pad * 2, th + pad);
      ctx.fillStyle = "#041316";
      ctx.fillText(label, x1 + pad, ly - pad / 2);
    });
  }

  async function loadImageForCanvas(src) {
    const img = new Image();
    await new Promise((resolve, reject) => {
      img.onload = resolve;
      img.onerror = () => reject(new Error("Could not load image for prediction preview"));
      img.src = src;
    });
    return img;
  }

  if ($("predict-refresh-assets")) $("predict-refresh-assets").addEventListener("click", () => loadAssets({ picker: true }));

  if ($("predict-button")) $("predict-button").addEventListener("click", async () => {
    const btn = $("predict-button");
    const assetId = ($("predict-asset") && $("predict-asset").value) || "";
    const file = $("predict-file").files && $("predict-file").files[0];
    const prompt = ($("predict-prompt").value || "").trim();
    const thresholdRaw = $("predict-threshold").value;
    if (!assetId && !file) { setStatus("predict-status", "Choose an existing asset or upload an image file.", "error"); return; }
    if (!prompt) { setStatus("predict-status", "Enter a prompt.", "error"); return; }
    setBusy(btn, true);
    setStatus("predict-status", assetId ? "Running prediction on selected asset…" : "Running prediction on uploaded image…", "busy");
    if ($("predict-output")) $("predict-output").hidden = true;
    try {
      let result;
      let img;
      let objectUrl = null;
      let predSize = null;
      if (assetId) {
        const payload = { prompt };
        if (thresholdRaw !== "") payload.threshold = Number(thresholdRaw);
        const sourceId = ($("predict-source") && $("predict-source").value) || "";
        if (sourceId) payload.source_id = sourceId;
        result = await request(`/v1/assets/${encodeURIComponent(assetId)}/predict`, {
          method: "POST", headers: headersJSON(true), body: JSON.stringify(payload)
        });
        const imageUrl = result && result.image_url
          ? result.image_url
          : `/v1/assets/${encodeURIComponent(assetId)}/image?tenant_id=${encodeURIComponent(tenantId())}`;
        if (result && Number.isFinite(result.image_width) && Number.isFinite(result.image_height)) {
          predSize = { w: Number(result.image_width), h: Number(result.image_height) };
        }
        img = await loadImageForCanvas(imageUrl);
      } else {
        const form = new FormData();
        form.append("image", file);
        form.append("prompt", prompt);
        if (thresholdRaw !== "") form.append("threshold", thresholdRaw);
        result = await request("/v1/predict", { method: "POST", body: form });
        objectUrl = URL.createObjectURL(file);
        img = await loadImageForCanvas(objectUrl);
        predSize = { w: img.naturalWidth || img.width, h: img.naturalHeight || img.height };
      }

      const predictions = (result && result.predictions) || [];
      drawPredictions($("predict-canvas"), img, predictions, predSize);
      if (objectUrl) URL.revokeObjectURL(objectUrl);

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

  // Search
  function renderSearchResults(results) {
    const container = $("search-results");
    if (!container) return;
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

      if (r.image_url) {
        const thumb = document.createElement("div");
        thumb.className = "result-thumb";
        const img = document.createElement("img");
        img.src = r.image_url;
        img.alt = r.record_id || "Search result";
        img.loading = "lazy";
        img.onerror = () => { thumb.remove(); };
        thumb.appendChild(img);
        card.appendChild(thumb);
      }

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
    const query = ($("search-query").value || "").trim();
    const limitVal = parseInt($("search-limit").value, 10);
    const payload = { query, limit: Number.isFinite(limitVal) && limitVal > 0 ? limitVal : 10 };
    const ids = selectedAssetIds();
    if (ids.length) payload.asset_ids = ids;
    return payload;
  }

  if ($("search-select-all")) $("search-select-all").addEventListener("click", () => {
    document.querySelectorAll("#asset-picker input[type='checkbox']").forEach((cb) => (cb.checked = true));
  });
  if ($("search-select-none")) $("search-select-none").addEventListener("click", () => {
    document.querySelectorAll("#asset-picker input[type='checkbox']").forEach((cb) => (cb.checked = false));
  });
  if ($("search-refresh-assets")) $("search-refresh-assets").addEventListener("click", () => loadAssets({ picker: true }));

  if ($("search-button")) $("search-button").addEventListener("click", async () => {
    const btn = $("search-button");
    const payload = searchPayload();
    if (!payload.query) { setStatus("search-status", "Enter a search query.", "error"); return; }
    setBusy(btn, true);
    setStatus("search-status", "Searching…", "busy");
    setStatus("subasset-status", "");
    try {
      const result = await request("/v1/search", { method: "POST", headers: headersJSON(true), body: JSON.stringify(payload) });
      const results = (result && result.results) || [];
      renderSearchResults(results);
      setStatus("search-status", `${results.length} result${results.length === 1 ? "" : "s"}.`, "ok");
    } catch (err) {
      if ($("search-results")) $("search-results").innerHTML = "";
      setStatus("search-status", err.message || String(err), "error");
    } finally {
      setBusy(btn, false);
    }
  });

  if ($("subasset-button")) $("subasset-button").addEventListener("click", async () => {
    const btn = $("subasset-button");
    const payload = searchPayload();
    if (!payload.query) { setStatus("subasset-status", "Enter a search query first.", "error"); return; }
    setBusy(btn, true);
    setStatus("subasset-status", "Creating sub-asset from search results…", "busy");
    try {
      const record = await request("/v1/sub-assets/from-search", {
        method: "POST", headers: headersJSON(true), body: JSON.stringify(payload)
      });
      await loadAssets({ quiet: true });
      if (record.asset_id) {
        const count = Number.isFinite(record.source_count) ? record.source_count : (record.result_ids || []).length;
        setStatus("subasset-status", `Created gallery asset “${record.asset_name || "Sub-asset"}” (${record.asset_id}) with ${count} image source${count === 1 ? "" : "s"}.`, "ok");
        showView("gallery");
        setStatus("gallery-status", `Sub-asset created from “${record.query || payload.query}” and added to the gallery.`, "ok");
      } else {
        setStatus("subasset-status", `Sub-asset record created: ${record.sub_asset_id} (${(record.result_ids || []).length} results).`, "ok");
      }
    } catch (err) {
      setStatus("subasset-status", err.message || String(err), "error");
    } finally {
      setBusy(btn, false);
    }
  });

  // Exports
  function renderExportJob(job) {
    if (!job) { showDetails("export-details", null); return; }
    showDetails("export-details", [
      ["Job", job.job_id],
      ["Kind", job.kind],
      ["Status", job.status],
      ["Attempts", job.attempts],
      ["Error", job.error],
      ["Tenant", job.tenant_id],
    ]);
  }

  if ($("export-button")) $("export-button").addEventListener("click", async () => {
    const btn = $("export-button");
    const versionId = ($("version-id").value || "").trim();
    const format = $("export-format").value;
    const idempotencyKey = ($("export-key").value || "").trim() || "browser-export";
    if (!versionId) { setStatus("export-status", "Enter a dataset version ID.", "error"); return; }
    setBusy(btn, true);
    setStatus("export-status", "Queuing export…", "busy");
    try {
      const job = await request("/v1/dataset-exports", {
        method: "POST", headers: headersJSON(true),
        body: JSON.stringify({ version_id: versionId, format, idempotency_key: idempotencyKey })
      });
      renderExportJob(job);
      if ($("job-id")) $("job-id").value = job.job_id || "";
      setStatus("export-status", `Export job ${job.status}.`, "ok");
    } catch (err) {
      setStatus("export-status", err.message || String(err), "error");
    } finally {
      setBusy(btn, false);
    }
  });

  if ($("job-status-button")) $("job-status-button").addEventListener("click", async () => {
    const btn = $("job-status-button");
    const jobId = ($("job-id").value || "").trim();
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

  if ($("job-cancel-button")) $("job-cancel-button").addEventListener("click", async () => {
    const btn = $("job-cancel-button");
    const jobId = ($("job-id").value || "").trim();
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

  // Preview gallery (label job events)
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
    if (!grid || !summary) return;
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

  if ($("gallery-load-button")) $("gallery-load-button").addEventListener("click", async () => {
    const btn = $("gallery-load-button");
    const jobId = ($("gallery-job-id").value || "").trim();
    const afterRaw = ($("gallery-after").value || "").trim();
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
        setStatus("gallery-status", "No preview events found for this job and tenant.", "ok");
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

  if ($("gallery-use-export-job-button")) $("gallery-use-export-job-button").addEventListener("click", () => {
    const exportJobId = ($("job-id") && $("job-id").value.trim()) || "";
    if (!exportJobId) { setStatus("gallery-status", "No export job ID to copy.", "error"); return; }
    $("gallery-job-id").value = exportJobId;
    setStatus("gallery-status", "Copied export job ID. Export jobs may not emit preview events.", "ok");
  });

  if ($("gallery-clear-button")) $("gallery-clear-button").addEventListener("click", () => {
    if ($("gallery-grid")) $("gallery-grid").innerHTML = "";
    if ($("gallery-summary")) { $("gallery-summary").hidden = true; $("gallery-summary").innerHTML = ""; }
    setStatus("gallery-status", "Gallery cleared.", "ok");
  });

  // Tenant change: reload assets when relevant
  if ($("tenant-id")) $("tenant-id").addEventListener("change", () => {
    loadAssets({ quiet: true });
  });

  // Init
  checkRuntime();
  loadAssets({ quiet: true });
})();
