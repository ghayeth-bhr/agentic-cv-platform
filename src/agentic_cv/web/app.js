(function () {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const DEFAULT_TENANT = "00000000-0000-0000-0000-000000000001";
  const UUID_RE = /[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}/gi;
  const CHUNK_SIZE = 5 * 1024 * 1024;

  const VIEW_META = {
    assets: {
      title: "Assets",
      eyebrow: "Workspace",
      desc: "Upload photos and videos, inspect what is inside, and search across the library.",
    },
    agent: {
      title: "Agent",
      eyebrow: "Workspace",
      desc: "Turn a natural-language task into a calibrated, whole-asset labeling run.",
    },
    datasets: {
      title: "Datasets",
      eyebrow: "Library",
      desc: "Dataset versions built from label revisions, ready to export.",
    },
    jobs: {
      title: "Jobs",
      eyebrow: "Library",
      desc: "Background indexing, labeling, and export activity.",
    },
  };

  const state = {
    assets: [],
    selectedAsset: null,
    lineage: new Map(),
    searchPayload: null,
    searchResults: [],
    uploadFiles: [],
    agentSession: null,
    agentPollToken: 0,
  };

  function tenantHeaders(json) {
    const headers = { "X-Tenant-ID": DEFAULT_TENANT };
    if (json) headers["Content-Type"] = "application/json";
    return headers;
  }

  function cleanMessage(message) {
    return String(message == null ? "" : message)
      .replace(UUID_RE, "this item")
      .replace(/\s+/g, " ")
      .trim();
  }

  async function request(path, options = {}) {
    const response = await fetch(path, options);
    const text = await response.text();
    let body = null;
    try { body = text ? JSON.parse(text) : null; } catch { body = { detail: text || response.statusText }; }
    if (!response.ok) {
      const detail = body && (body.detail || body.message) ? (body.detail || body.message) : `Request failed (${response.status})`;
      throw new Error(cleanMessage(typeof detail === "string" ? detail : JSON.stringify(detail)));
    }
    return body;
  }

  function setStatus(id, message, kind) {
    const el = $(id);
    if (!el) return;
    el.textContent = message || "";
    if (kind) el.dataset.kind = kind; else delete el.dataset.kind;
  }

  function setBusy(button, busy, label) {
    if (!button) return;
    button.disabled = !!busy;
    if (busy) {
      if (!button.dataset.orig) button.dataset.orig = button.textContent;
      button.textContent = label || "Working…";
    } else if (button.dataset.orig) {
      button.textContent = button.dataset.orig;
      delete button.dataset.orig;
    }
  }

  function formatDate(iso) {
    if (!iso) return "—";
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return String(iso);
    return d.toLocaleString([], {
      year: "numeric", month: "short", day: "2-digit",
      hour: "2-digit", minute: "2-digit",
    });
  }

  function formatBytes(bytes) {
    const n = Number(bytes) || 0;
    if (n >= 1024 * 1024) return `${(n / (1024 * 1024)).toFixed(2)} MB`;
    if (n >= 1024) return `${(n / 1024).toFixed(1)} KB`;
    return `${n} B`;
  }

  function node(tag, className, text) {
    const el = document.createElement(tag);
    if (className) el.className = className;
    if (text != null) el.textContent = text;
    return el;
  }

  function sleep(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  async function downloadWithTenant(path, filename) {
    const response = await fetch(path, { headers: tenantHeaders(false) });
    if (!response.ok) {
      const text = await response.text();
      let detail = `Download failed (${response.status})`;
      try {
        const body = text ? JSON.parse(text) : null;
        if (body && (body.detail || body.message)) detail = body.detail || body.message;
      } catch {
        if (text) detail = text;
      }
      throw new Error(cleanMessage(detail));
    }
    const blob = await response.blob();
    const objectUrl = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = objectUrl;
    anchor.download = filename || "dataset.zip";
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    setTimeout(() => URL.revokeObjectURL(objectUrl), 2000);
  }

  // Navigation
  function showView(name) {
    const meta = VIEW_META[name] || VIEW_META.assets;
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
    if (name === "assets") loadAssets();
    if (name === "agent") initAgentView();
    if (name === "datasets") loadDatasets();
    if (name === "jobs") loadJobs();
  }

  document.querySelectorAll(".nav-item").forEach((btn) => {
    btn.addEventListener("click", () => showView(btn.dataset.view));
  });

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

  // Assets
  async function loadAssets(opts = {}) {
    const quiet = !!opts.quiet;
    const meta = $("assets-meta");
    if (meta && !quiet) meta.textContent = "Loading assets…";
    try {
      const [assetData, lineageData] = await Promise.all([
        request("/v1/assets", { headers: tenantHeaders(false) }),
        request("/v1/sub-assets", { headers: tenantHeaders(false) }),
      ]);
      state.assets = (assetData && assetData.assets) || [];
      state.lineage = new Map(
        ((lineageData || [])
          .filter((item) => item && item.asset_id)
          .map((item) => [String(item.asset_id), item]))
      );
      if (state.selectedAsset) {
        state.selectedAsset = state.assets.find(
          (asset) => String(asset.asset_id) === String(state.selectedAsset.asset_id)
        ) || null;
      }
      renderAssets();
      renderAssetDetail();
      syncSearchScopeOptions();
      const n = state.assets.length;
      if (meta) meta.textContent = n ? `${n} asset${n === 1 ? "" : "s"}` : "No assets yet";
      return state.assets;
    } catch (err) {
      if (meta) meta.textContent = "Could not load assets";
      if (!quiet) setStatus("gallery-status", err.message || String(err), "error");
      state.assets = [];
      state.lineage = new Map();
      renderAssets();
      renderAssetDetail();
      syncSearchScopeOptions();
      return [];
    }
  }

  function assetName(asset) {
    return (asset && asset.name) || "Untitled asset";
  }

  function mediaKind(asset) {
    const source = asset && Array.isArray(asset.sources) ? asset.sources[0] : null;
    if (!source) return "no source";
    return (source.media_type || "media").split("/")[0] || "media";
  }

  function renderAssets() {
    const grid = $("asset-gallery");
    if (!grid) return;
    const filter = (($("asset-filter") && $("asset-filter").value) || "").trim().toLowerCase();
    const assets = state.assets.filter((asset) => {
      if (!filter) return true;
      return assetName(asset).toLowerCase().includes(filter);
    });
    grid.innerHTML = "";
    if (!assets.length) {
      const empty = node("div", "empty-state");
      empty.append(
        node("strong", null, state.assets.length ? "No matching assets." : "No assets yet."),
        node("p", null, state.assets.length
          ? "Try a different filter, or clear it to see the whole library."
          : "Use Upload media to add your first photos or videos.")
      );
      grid.appendChild(empty);
      return;
    }
    assets.forEach((asset) => {
      const index = state.assets.indexOf(asset);
      const card = node("button", "asset-card");
      card.type = "button";
      card.dataset.assetIndex = String(index);
      card.setAttribute("aria-label", `Open ${assetName(asset)}`);
      if (state.selectedAsset && String(state.selectedAsset.asset_id) === String(asset.asset_id)) {
        card.classList.add("is-selected");
      }

      const thumb = node("div", "asset-thumb");
      if (asset.thumbnail_url) {
        const img = document.createElement("img");
        img.src = asset.thumbnail_url;
        img.alt = "";
        img.loading = "lazy";
        img.onerror = () => {
          thumb.innerHTML = "";
          thumb.appendChild(node("div", "thumb-fallback", mediaKind(asset)));
        };
        thumb.appendChild(img);
      } else {
        thumb.appendChild(node("div", "thumb-fallback", mediaKind(asset)));
      }
      const badge = node("span", "thumb-badge", mediaKind(asset));
      thumb.appendChild(badge);
      if (state.lineage.has(String(asset.asset_id))) {
        thumb.appendChild(node("span", "thumb-badge thumb-badge-sub", "sub-asset"));
      }

      const body = node("div", "asset-body");
      const name = node("div", "asset-name", assetName(asset));
      const sourceCount = (asset.sources || []).length;
      const meta = node("div", "asset-meta", `${sourceCount} source${sourceCount === 1 ? "" : "s"} · ${formatDate(asset.created_at)}`);
      body.append(name, meta);
      card.append(thumb, body);
      card.addEventListener("click", () => selectAsset(index));
      grid.appendChild(card);
    });
  }

  function selectAsset(index) {
    const asset = state.assets[index];
    if (!asset) return;
    state.selectedAsset = asset;
    renderAssets();
    renderAssetDetail();
    syncSearchScopeOptions();
  }

  function selectAssetById(assetId) {
    const index = state.assets.findIndex((asset) => String(asset.asset_id) === String(assetId));
    if (index >= 0) selectAsset(index);
    return index >= 0;
  }

  function assetImageUrl(asset, source) {
    let url = `/v1/assets/${encodeURIComponent(asset.asset_id)}/image?tenant_id=${encodeURIComponent(DEFAULT_TENANT)}`;
    if (source && source.source_id) {
      url += `&source_id=${encodeURIComponent(source.source_id)}`;
    }
    return url;
  }

  function renderAssetDetail() {
    const content = $("asset-detail-content");
    const placeholder = $("asset-detail-placeholder");
    if (!content || !placeholder) return;
    const asset = state.selectedAsset;
    if (!asset) {
      content.hidden = true;
      content.innerHTML = "";
      placeholder.hidden = false;
      return;
    }
    placeholder.hidden = true;
    content.hidden = false;
    content.innerHTML = "";

    const head = node("div", "detail-head");
    const thumb = node("div", "detail-thumb");
    if (asset.thumbnail_url) {
      const img = document.createElement("img");
      img.src = asset.thumbnail_url;
      img.alt = "";
      img.onerror = () => { thumb.innerHTML = ""; thumb.appendChild(node("div", "thumb-fallback", mediaKind(asset))); };
      thumb.appendChild(img);
    } else {
      thumb.appendChild(node("div", "thumb-fallback", mediaKind(asset)));
    }
    const heading = node("div", "detail-heading");
    heading.append(
      node("p", "detail-eyebrow", "Asset"),
      node("h2", "detail-title", assetName(asset)),
      node("p", "detail-meta", `${(asset.sources || []).length} source${(asset.sources || []).length === 1 ? "" : "s"} · added ${formatDate(asset.created_at)}`)
    );
    head.append(thumb, heading);

    const actions = node("div", "detail-actions");
    const indexButton = node("button", "btn btn-secondary btn-small", "Index for search");
    indexButton.type = "button";
    indexButton.addEventListener("click", () => submitIndexJob(asset, { button: indexButton, source: "detail" }));
    const searchButton = node("button", "btn btn-secondary btn-small", "Search this asset");
    searchButton.type = "button";
    searchButton.addEventListener("click", () => {
      if ($("search-scope")) $("search-scope").value = "selected";
      syncSearchScopeOptions();
      if ($("search-query")) $("search-query").focus();
    });
    actions.append(indexButton, searchButton);

    content.append(head, actions);

    const lineage = state.lineage.get(String(asset.asset_id));
    if (lineage) {
      const box = node("div", "lineage-box");
      box.append(
        node("div", "lineage-label", "Sub-asset lineage"),
        node("p", "lineage-query", `Created from search: “${lineage.query}”`),
        node("p", "lineage-meta", `Parent assets: ${(lineage.parent_asset_names || []).join(", ") || "Source assets"}`),
        node("p", "lineage-meta", `${lineage.source_count || (lineage.result_ids || []).length} matched image${(lineage.source_count || (lineage.result_ids || []).length) === 1 ? "" : "s"}`)
      );
      content.appendChild(box);
    }

    const sources = node("div", "source-section");
    sources.appendChild(node("h3", "detail-section-title", "Sources"));
    const list = node("div", "source-list");
    const sourceItems = asset.sources || [];
    if (!sourceItems.length) {
      list.appendChild(node("p", "muted", "No source media has finished uploading."));
    }
    sourceItems.forEach((source, index) => {
      const row = node("div", "source-row");
      const image = document.createElement("img");
      image.className = "source-thumb";
      image.src = assetImageUrl(asset, source);
      image.alt = "";
      image.loading = "lazy";
      image.onerror = () => { image.replaceWith(node("div", "source-thumb source-thumb-fallback", "—")); };
      const text = node("div", "source-text");
      text.append(
        node("div", "source-name", source.filename || `Source ${index + 1}`),
        node("div", "source-meta", `${source.media_type || "media"} · added ${formatDate(source.created_at)}`)
      );
      row.append(image, text);
      list.appendChild(row);
    });
    sources.appendChild(list);
    content.appendChild(sources);
  }

  function syncSearchScopeOptions() {
    const select = $("search-scope");
    const chip = $("search-scope-chip");
    if (!select) return;
    const selectedOption = select.querySelector('option[value="selected"]');
    if (selectedOption) {
      selectedOption.disabled = !state.selectedAsset;
      selectedOption.textContent = state.selectedAsset
        ? `Selected: ${assetName(state.selectedAsset)}`
        : "Selected asset (none)";
    }
    if (!state.selectedAsset && select.value === "selected") select.value = "all";
    if (chip) chip.textContent = select.value === "selected" && state.selectedAsset
      ? assetName(state.selectedAsset)
      : "All assets";
  }

  // Upload
  function renderUploadFiles() {
    const info = $("asset-file-info");
    if (!info) return;
    info.innerHTML = "";
    if (!state.uploadFiles.length) {
      info.textContent = "No files selected.";
      return;
    }
    state.uploadFiles.forEach((file) => {
      const chip = node("span", "file-chip");
      chip.append(node("strong", null, file.name), node("span", null, ` · ${formatBytes(file.size)}`));
      info.appendChild(chip);
    });
  }

  if ($("asset-file")) $("asset-file").addEventListener("change", (event) => {
    state.uploadFiles = Array.from(event.target.files || []);
    renderUploadFiles();
  });

  function showUploadPanel(show) {
    const panel = $("upload-panel");
    const toggle = $("upload-toggle");
    if (!panel) return;
    panel.hidden = !show;
    if (toggle) toggle.setAttribute("aria-expanded", show ? "true" : "false");
  }

  if ($("upload-toggle")) $("upload-toggle").addEventListener("click", () => {
    const panel = $("upload-panel");
    showUploadPanel(!panel || panel.hidden);
  });
  if ($("upload-close")) $("upload-close").addEventListener("click", () => showUploadPanel(false));

  async function sha256Hex(bytes) {
    if (!crypto || !crypto.subtle) {
      throw new Error("This browser cannot hash files for upload. Use a recent Chrome, Edge, or Firefox.");
    }
    const hash = await crypto.subtle.digest("SHA-256", bytes);
    return Array.from(new Uint8Array(hash)).map((b) => b.toString(16).padStart(2, "0")).join("");
  }

  if ($("upload-button")) $("upload-button").addEventListener("click", async () => {
    const button = $("upload-button");
    const files = state.uploadFiles.slice();
    const name = (($("asset-name") && $("asset-name").value) || "").trim()
      || (files.length ? files[0].name.replace(/\.[^.]+$/, "") : "Untitled asset");
    if (!files.length) {
      setStatus("upload-status", "Choose at least one image or video.", "error");
      return;
    }
    setBusy(button, true, "Uploading…");
    let activeAsset = null;
    try {
      setStatus("upload-status", "Creating asset…", "busy");
      activeAsset = await request("/v1/assets", {
        method: "POST",
        headers: tenantHeaders(true),
        body: JSON.stringify({ name }),
      });

      for (let fileIndex = 0; fileIndex < files.length; fileIndex += 1) {
        const file = files[fileIndex];
        setStatus("upload-status", `Hashing ${file.name} (${fileIndex + 1}/${files.length})…`, "busy");
        const digest = await sha256Hex(await file.arrayBuffer());
        const upload = await request(`/v1/assets/${encodeURIComponent(activeAsset.asset_id)}/uploads`, {
          method: "POST",
          headers: tenantHeaders(true),
          body: JSON.stringify({
            filename: file.name,
            media_type: file.type || "application/octet-stream",
            expected_sha256: digest,
          }),
        });
        const totalParts = Math.max(1, Math.ceil(file.size / CHUNK_SIZE));
        for (let part = 1; part <= totalParts; part += 1) {
          const start = (part - 1) * CHUNK_SIZE;
          const chunk = file.slice(start, Math.min(start + CHUNK_SIZE, file.size));
          const form = new FormData();
          form.append("part", chunk, file.name);
          setStatus("upload-status", `Uploading ${file.name} · ${fileIndex + 1}/${files.length} · part ${part}/${totalParts}`, "busy");
          await request(`/v1/uploads/${encodeURIComponent(upload.upload_id)}/parts/${part}`, {
            method: "PUT",
            headers: tenantHeaders(false),
            body: form,
          });
        }
        await request(`/v1/uploads/${encodeURIComponent(upload.upload_id)}/complete`, {
          method: "POST",
          headers: tenantHeaders(false),
        });
      }

      await loadAssets({ quiet: true });
      selectAssetById(activeAsset.asset_id);
      setStatus("upload-status", `Uploaded ${files.length} file${files.length === 1 ? "" : "s"}. Queuing search indexing…`, "ok");
      await submitIndexJob(activeAsset, { source: "upload" });
      state.uploadFiles = [];
      if ($("asset-file")) $("asset-file").value = "";
      renderUploadFiles();
      showUploadPanel(false);
    } catch (err) {
      const message = cleanMessage(err.message || err);
      if (activeAsset) {
        await loadAssets({ quiet: true });
        selectAssetById(activeAsset.asset_id);
        setStatus("upload-status", `Upload incomplete: ${message}`, "error");
      } else {
        setStatus("upload-status", message, "error");
      }
    } finally {
      setBusy(button, false);
    }
  });

  async function submitIndexJob(asset, opts = {}) {
    const button = opts.button || null;
    setBusy(button, true, "Indexing…");
    setStatus("gallery-status", `Queuing search indexing for ${assetName(asset)}…`, "busy");
    try {
      let job = await request(`/v1/assets/${encodeURIComponent(asset.asset_id)}/index-jobs`, {
        method: "POST",
        headers: tenantHeaders(false),
      });
      for (let attempt = 0; attempt < 60; attempt += 1) {
        if (job.status === "succeeded" || job.status === "failed" || job.status === "cancelled") break;
        await sleep(500);
        job = await request(`/v1/index-jobs/${encodeURIComponent(job.job_id)}`, {
          headers: tenantHeaders(false),
        });
      }
      if (job.status === "succeeded") {
        setStatus("gallery-status", `${assetName(asset)} is indexed for search.`, "ok");
        if (opts.source === "upload") {
          setStatus("upload-status", `Upload complete. ${assetName(asset)} is indexed for search.`, "ok");
        }
      } else if (job.status === "failed") {
        const detail = cleanMessage(job.error || "search indexing is unavailable");
        setStatus("gallery-status", `Indexing failed: ${detail}`, "error");
        if (opts.source === "upload") {
          setStatus("upload-status", `Upload complete. Search indexing unavailable: ${detail}`, "error");
        }
      } else {
        setStatus("gallery-status", `Indexing ${job.status || "queued"}…`, "busy");
      }
      if ($("view-jobs") && !$("view-jobs").hidden) loadJobs();
      return job;
    } catch (err) {
      const detail = cleanMessage(err.message || err);
      setStatus("gallery-status", `Indexing failed: ${detail}`, "error");
      if (opts.source === "upload") {
        setStatus("upload-status", `Upload complete. Search indexing unavailable: ${detail}`, "error");
      }
      return null;
    } finally {
      setBusy(button, false);
    }
  }

  if ($("assets-refresh")) $("assets-refresh").addEventListener("click", () => loadAssets());
  if ($("asset-filter")) $("asset-filter").addEventListener("input", renderAssets);

  // Search
  function currentSearchPayload() {
    const query = (($("search-query") && $("search-query").value) || "").trim();
    const payload = { query, limit: 20 };
    const scope = $("search-scope") ? $("search-scope").value : "all";
    if (scope === "selected" && state.selectedAsset) {
      payload.asset_ids = [state.selectedAsset.asset_id];
    }
    return payload;
  }

  function assetsById() {
    return new Map(state.assets.map((asset) => [String(asset.asset_id), asset]));
  }

  function renderSearchResults() {
    const container = $("search-results");
    if (!container) return;
    container.innerHTML = "";
    if (!state.searchResults.length) {
      container.appendChild(node("div", "results-empty", "No results yet. Try a description of the people, objects, or scene."));
      return;
    }
    const byId = assetsById();
    const grid = node("div", "result-gallery");
    state.searchResults.forEach((result, index) => {
      const score = Number(result.score);
      const pct = Number.isFinite(score) ? Math.max(0, Math.min(100, score * 100)) : 0;
      const card = node("button", "result-card");
      card.type = "button";
      card.setAttribute("aria-label", `Open match ${index + 1}`);
      if (result.image_url) {
        const thumb = node("div", "result-thumb");
        const img = document.createElement("img");
        img.src = result.image_url;
        img.alt = "";
        img.loading = "lazy";
        thumb.appendChild(img);
        card.appendChild(thumb);
      }
      const head = node("div", "result-card-head");
      head.append(
        node("span", "result-rank", `Match ${index + 1}`),
        node("span", "result-score", Number.isFinite(score) ? `${pct.toFixed(0)}%` : "—")
      );
      const parent = byId.get(String(result.asset_id));
      card.append(
        head,
        node("p", "result-meta", parent ? assetName(parent) : "Source asset")
      );
      const bar = node("div", "score-bar");
      const fill = document.createElement("i");
      fill.style.width = `${pct.toFixed(1)}%`;
      bar.appendChild(fill);
      card.appendChild(bar);
      card.addEventListener("click", () => {
        if (result.asset_id) selectAssetById(result.asset_id);
        if ($("asset-detail")) $("asset-detail").scrollIntoView({ behavior: "smooth", block: "start" });
      });
      grid.appendChild(card);
    });
    container.appendChild(grid);
  }

  if ($("search-scope")) $("search-scope").addEventListener("change", syncSearchScopeOptions);

  if ($("search-button")) $("search-button").addEventListener("click", async () => {
    const button = $("search-button");
    const payload = currentSearchPayload();
    if (!payload.query) {
      setStatus("search-status", "Enter a search query.", "error");
      return;
    }
    setBusy(button, true, "Searching…");
    setStatus("search-status", "Searching…", "busy");
    state.searchResults = [];
    state.searchPayload = null;
    renderSearchResults();
    if ($("subasset-button")) $("subasset-button").disabled = true;
    try {
      const result = await request("/v1/search", {
        method: "POST",
        headers: tenantHeaders(true),
        body: JSON.stringify(payload),
      });
      state.searchResults = (result && result.results) || [];
      state.searchPayload = payload;
      renderSearchResults();
      if ($("subasset-button")) $("subasset-button").disabled = !state.searchResults.length;
      setStatus("search-status", `${state.searchResults.length} result${state.searchResults.length === 1 ? "" : "s"}.`, "ok");
    } catch (err) {
      setStatus("search-status", cleanMessage(err.message || err), "error");
    } finally {
      setBusy(button, false);
    }
  });

  if ($("subasset-button")) $("subasset-button").addEventListener("click", async () => {
    const button = $("subasset-button");
    const payload = state.searchPayload || currentSearchPayload();
    if (!payload.query || !state.searchResults.length) {
      setStatus("search-status", "Run a search before saving results.", "error");
      return;
    }
    setBusy(button, true, "Saving…");
    setStatus("search-status", "Saving search results as a sub-asset…", "busy");
    try {
      const record = await request("/v1/sub-assets/from-search", {
        method: "POST",
        headers: tenantHeaders(true),
        body: JSON.stringify(payload),
      });
      await loadAssets({ quiet: true });
      if (record.asset_id) selectAssetById(record.asset_id);
      const count = Number.isFinite(record.source_count) ? record.source_count : state.searchResults.length;
      setStatus("search-status", `Saved “${record.asset_name || "Sub-asset"}” with ${count} matched image${count === 1 ? "" : "s"}.`, "ok");
      setStatus("gallery-status", `Sub-asset saved from search: “${record.query || payload.query}”.`, "ok");
    } catch (err) {
      setStatus("search-status", cleanMessage(err.message || err), "error");
    } finally {
      setBusy(button, false);
    }
  });

  // Agent
  const AGENT_SESSION_KEY = "agentic-cv-agent-session";
  const AGENT_ACTIVE = new Set(["calibrating", "labeling", "exporting"]);
  const AGENT_STATUS_LABEL = {
    awaiting_confirmation: "Awaiting confirmation",
    calibrating: "Calibrating",
    needs_attention: "Needs attention",
    labeling: "Labeling",
    exporting: "Packaging dataset",
    complete: "Dataset ready",
    failed: "Failed",
    cancelled: "Cancelled",
  };

  function agentStatusLabel(status) {
    return AGENT_STATUS_LABEL[status] || status || "Planning";
  }

  function agentPillStatus(status) {
    if (status === "complete") return "succeeded";
    if (status === "failed" || status === "cancelled" || status === "needs_attention") return "failed";
    if (AGENT_ACTIVE.has(status)) return "running";
    return status;
  }

  function renderAgentAssets(session) {
    const select = $("agent-asset");
    if (!select) return;
    const previous = select.value
      || (session && session.asset_id)
      || (state.selectedAsset && state.selectedAsset.asset_id)
      || "";
    select.innerHTML = "";
    if (!state.assets.length) {
      const option = document.createElement("option");
      option.value = "";
      option.textContent = "No assets yet — upload media first";
      select.appendChild(option);
      select.disabled = true;
    } else {
      state.assets.forEach((asset) => {
        const option = document.createElement("option");
        option.value = String(asset.asset_id);
        const count = (asset.sources || []).length;
        option.textContent = `${assetName(asset)} · ${count} source${count === 1 ? "" : "s"}`;
        select.appendChild(option);
      });
      if (previous) {
        const match = state.assets.find((asset) => String(asset.asset_id) === String(previous));
        if (match) select.value = String(match.asset_id);
      }
      select.disabled = !!session;
    }
    if ($("agent-message")) $("agent-message").disabled = !!session;
    if ($("agent-plan")) $("agent-plan").disabled = !!session || !state.assets.length;
  }

  function renderAgentMessages(session) {
    const list = $("agent-messages");
    if (!list) return;
    list.innerHTML = "";
    if (!session || !session.messages || !session.messages.length) {
      const empty = node("div", "agent-empty-chat");
      empty.textContent = "Start with a task such as “Create a segmentation dataset of workers wearing yellow vests.”";
      list.appendChild(empty);
      return;
    }
    session.messages.forEach((message) => {
      const row = node("div", `agent-message agent-message-${message.role === "user" ? "user" : "agent"}`);
      const bubble = node("div", "agent-bubble", cleanMessage(message.content));
      row.append(node("div", "agent-message-role", message.role === "user" ? "You" : "Agent"), bubble);
      list.appendChild(row);
    });
    list.scrollTop = list.scrollHeight;
  }

  function agentSpecRow(term, value) {
    const dt = node("dt", null, term);
    const dd = node("dd", null, value);
    return [dt, dd];
  }

  function renderAgentSpec(session) {
    const box = $("agent-spec");
    if (!box) return;
    box.innerHTML = "";
    const spec = session && session.spec;
    if (!spec) return;
    const grid = node("dl", "agent-spec-grid");
    const prompt = session.prompt || (spec.target_objects || []).join(" and ");
    [
      ["Task", spec.task_type],
      ["Prompt", prompt],
      ["Calibration", `${spec.sample_size} sample image${spec.sample_size === 1 ? "" : "s"} · ${spec.max_iterations} attempt${spec.max_iterations === 1 ? "" : "s"}`],
      ["Asset", session.asset_name],
      ["Planner", session.planner_mode === "ollama" ? "Local model" : "Local rules"],
    ].forEach(([term, value]) => {
      const [dt, dd] = agentSpecRow(term, value);
      grid.append(dt, dd);
    });
    box.appendChild(grid);
    if (session.planner_note) {
      box.appendChild(node("p", "agent-note", cleanMessage(session.planner_note)));
    }
  }

  function agentStep(title, detail, state) {
    const row = node("li", "agent-step");
    row.dataset.state = state;
    const mark = node("span", "agent-step-mark", state === "done" ? "✓" : state === "error" ? "!" : state === "active" ? "•" : "");
    const text = node("div", "agent-step-text");
    text.append(node("strong", null, title), node("span", null, detail));
    row.append(mark, text);
    return row;
  }

  function renderAgentProgress(session) {
    const list = $("agent-progress");
    if (!list) return;
    list.innerHTML = "";
    const hasSpec = !!session.spec;
    list.appendChild(agentStep("Task plan", hasSpec ? "Specification ready" : "Waiting for a task", hasSpec ? "done" : "pending"));

    let calibrationState = "pending";
    let calibrationDetail = "Runs after confirmation";
    if (session.status === "needs_attention") {
      calibrationState = "error";
      calibrationDetail = "No threshold met the sample agreement";
    } else if (session.status === "failed" && !session.threshold) {
      calibrationState = "error";
      calibrationDetail = cleanMessage(session.error || "Calibration stopped");
    } else if (session.threshold != null) {
      calibrationState = "done";
      calibrationDetail = `Threshold ${Number(session.threshold).toFixed(2)}`;
    } else if (session.status === "calibrating") {
      calibrationState = "active";
      calibrationDetail = "Testing thresholds on a sample";
    }
    list.appendChild(agentStep("Calibration", calibrationDetail, calibrationState));

    let labelState = "pending";
    let labelDetail = "Starts after confirmation";
    const labelStatus = session.label_status;
    if (labelStatus === "succeeded") {
      labelState = "done";
      labelDetail = "Whole-asset labeling finished";
    } else if (labelStatus === "failed") {
      labelState = "error";
      labelDetail = "Labeling failed";
    } else if (labelStatus === "cancelled") {
      labelState = "error";
      labelDetail = "Labeling cancelled";
    } else if (labelStatus === "queued" || labelStatus === "running") {
      labelState = "active";
      labelDetail = labelStatus === "queued" ? "Queued for the label worker" : "Labeling images in the background";
    } else if (session.status === "calibrating") {
      labelDetail = "Starts after calibration";
    }
    list.appendChild(agentStep("Whole-asset labeling", labelDetail, labelState));

    const datasetState = session.dataset ? "done" : session.status === "exporting" ? "active" : "pending";
    const datasetDetail = session.dataset
      ? `${session.dataset.image_count} image${session.dataset.image_count === 1 ? "" : "s"} · ${session.dataset.annotation_count} annotation${session.dataset.annotation_count === 1 ? "" : "s"}`
      : session.status === "exporting" ? "Building from label revisions" : "Builds from the latest annotations";
    list.appendChild(agentStep("Dataset version", datasetDetail, datasetState));

    let exportState = "pending";
    let exportDetail = "Runs after the dataset is built";
    if (session.export_status === "succeeded") {
      exportState = "done";
      exportDetail = "COCO archive ready to download";
    } else if (session.export_status === "failed") {
      exportState = "error";
      exportDetail = "Export failed";
    } else if (session.export_status === "cancelled") {
      exportState = "error";
      exportDetail = "Export cancelled";
    } else if (session.export_status === "queued" || session.export_status === "running") {
      exportState = "active";
      exportDetail = "Packaging the COCO archive";
    }
    list.appendChild(agentStep("Download package", exportDetail, exportState));
  }

  function applyAgentSession(session) {
    state.agentSession = session;
    try { localStorage.setItem(AGENT_SESSION_KEY, String(session.session_id)); } catch { /* local storage is optional */ }
    const empty = $("agent-run-empty");
    const content = $("agent-run-content");
    if (empty) empty.hidden = true;
    if (content) content.hidden = false;
    renderAgentAssets(session);
    renderAgentMessages(session);
    renderAgentSpec(session);
    renderAgentProgress(session);

    if ($("agent-run-title")) $("agent-run-title").textContent = session.prompt || session.asset_name || "Task plan";
    const pill = $("agent-run-pill");
    if (pill) {
      pill.textContent = agentStatusLabel(session.status);
      pill.dataset.status = agentPillStatus(session.status);
    }

    const confirm = $("agent-confirm");
    if (confirm) {
      confirm.hidden = session.status !== "awaiting_confirmation";
      confirm.disabled = false;
    }
    const retry = $("agent-retry");
    if (retry) {
      retry.hidden = session.status !== "needs_attention";
      retry.disabled = false;
    }
    const cancel = $("agent-cancel");
    if (cancel) {
      const cancellable = !["complete", "failed", "cancelled"].includes(session.status);
      cancel.hidden = !cancellable;
      cancel.disabled = false;
    }
    const download = $("agent-download");
    if (download) {
      const ready = !!session.download_path;
      download.hidden = !ready;
      if (ready) download.href = session.download_path;
      else download.removeAttribute("href");
    }

    const error = $("agent-run-error");
    if (error) {
      error.hidden = !session.error;
      error.textContent = session.error ? cleanMessage(session.error) : "";
    }

    const label = agentStatusLabel(session.status);
    if (session.error) setStatus("agent-status", cleanMessage(session.error), "error");
    else if (session.status === "complete") setStatus("agent-status", "Dataset ready. Download the COCO archive from the run panel.", "ok");
    else if (AGENT_ACTIVE.has(session.status)) setStatus("agent-status", `${label}…`, "busy");
    else if (session.status === "needs_attention") setStatus("agent-status", "Calibration needs attention. Retry at 0.50 or start a new task.", "error");
    else if (session.status === "awaiting_confirmation") setStatus("agent-status", "Task planned. Confirm the specification to start.", "ok");
    else if (session.status === "cancelled") setStatus("agent-status", "Run cancelled.", "error");
  }

  function resetAgent() {
    state.agentPollToken += 1;
    state.agentSession = null;
    try { localStorage.removeItem(AGENT_SESSION_KEY); } catch { /* local storage is optional */ }
    const empty = $("agent-run-empty");
    const content = $("agent-run-content");
    if (empty) empty.hidden = false;
    if (content) content.hidden = true;
    ["agent-spec", "agent-progress", "agent-run-error"].forEach((id) => {
      const el = $(id);
      if (el) { el.innerHTML = ""; if (id === "agent-run-error") el.hidden = true; }
    });
    ["agent-confirm", "agent-retry", "agent-cancel", "agent-download"].forEach((id) => {
      const el = $(id);
      if (el) el.hidden = true;
    });
    renderAgentMessages(null);
    renderAgentAssets(null);
    const input = $("agent-message");
    if (input) input.value = "";
    setStatus("agent-status", "", "");
  }

  async function refreshAgentSession() {
    const session = state.agentSession;
    if (!session) return;
    try {
      const fresh = await request(`/v1/agent/sessions/${encodeURIComponent(session.session_id)}`, {
        headers: tenantHeaders(false),
      });
      applyAgentSession(fresh);
    } catch (err) {
      setStatus("agent-status", `Could not refresh the run: ${cleanMessage(err.message || err)}`, "error");
    }
  }

  function startAgentPolling() {
    const token = ++state.agentPollToken;
    async function tick() {
      if (token !== state.agentPollToken) return;
      const session = state.agentSession;
      if (!session) return;
      try {
        const fresh = await request(`/v1/agent/sessions/${encodeURIComponent(session.session_id)}`, {
          headers: tenantHeaders(false),
        });
        if (token !== state.agentPollToken) return;
        applyAgentSession(fresh);
        if (AGENT_ACTIVE.has(fresh.status)) setTimeout(tick, 700);
      } catch (err) {
        if (token !== state.agentPollToken) return;
        setStatus("agent-status", `Could not refresh the run: ${cleanMessage(err.message || err)}`, "error");
        setTimeout(tick, 2000);
      }
    }
    setTimeout(tick, 200);
  }

  async function restoreAgentSession() {
    let sessionId = null;
    try { sessionId = localStorage.getItem(AGENT_SESSION_KEY); } catch { sessionId = null; }
    if (!sessionId) return;
    try {
      const session = await request(`/v1/agent/sessions/${encodeURIComponent(sessionId)}`, {
        headers: tenantHeaders(false),
      });
      applyAgentSession(session);
      if (AGENT_ACTIVE.has(session.status)) startAgentPolling();
    } catch {
      try { localStorage.removeItem(AGENT_SESSION_KEY); } catch { /* local storage is optional */ }
    }
  }

  async function initAgentView() {
    if (!state.assets.length) await loadAssets({ quiet: true });
    if (state.agentSession) {
      applyAgentSession(state.agentSession);
      if (AGENT_ACTIVE.has(state.agentSession.status)) startAgentPolling();
      return;
    }
    renderAgentAssets(null);
    await restoreAgentSession();
  }

  async function planAgentTask() {
    const button = $("agent-plan");
    const assetId = $("agent-asset") ? $("agent-asset").value : "";
    const message = (($("agent-message") && $("agent-message").value) || "").trim();
    if (!assetId) {
      setStatus("agent-status", "Upload and select an asset before planning a task.", "error");
      return;
    }
    if (!message) {
      setStatus("agent-status", "Describe what the agent should label.", "error");
      return;
    }
    setBusy(button, true, "Planning…");
    setStatus("agent-status", "Planning the task…", "busy");
    try {
      const session = await request("/v1/agent/sessions", {
        method: "POST",
        headers: tenantHeaders(true),
        body: JSON.stringify({ asset_id: assetId, message }),
      });
      applyAgentSession(session);
      setStatus("agent-status", "Task planned. Confirm the specification to start.", "ok");
    } catch (err) {
      setStatus("agent-status", cleanMessage(err.message || err), "error");
    } finally {
      setBusy(button, false);
    }
  }

  async function confirmAgentTask(threshold) {
    const session = state.agentSession;
    if (!session) return;
    const button = threshold == null ? $("agent-confirm") : $("agent-retry");
    const original = button ? button.textContent : "";
    if (button) {
      button.disabled = true;
      button.textContent = threshold == null ? "Calibrating…" : "Starting…";
    }
    setStatus(
      "agent-status",
      threshold == null ? "Calibrating on a sample. This can take a moment…" : "Starting at the fixed threshold…",
      "busy",
    );
    try {
      const body = threshold == null ? {} : { threshold };
      const fresh = await request(`/v1/agent/sessions/${encodeURIComponent(session.session_id)}/confirm`, {
        method: "POST",
        headers: tenantHeaders(true),
        body: JSON.stringify(body),
      });
      applyAgentSession(fresh);
      if (AGENT_ACTIVE.has(fresh.status)) startAgentPolling();
    } catch (err) {
      setStatus("agent-status", cleanMessage(err.message || err), "error");
      await refreshAgentSession();
    } finally {
      if (button) {
        button.disabled = false;
        button.textContent = original;
      }
    }
  }

  async function cancelAgentTask() {
    const session = state.agentSession;
    if (!session) return;
    const button = $("agent-cancel");
    setBusy(button, true, "Cancelling…");
    try {
      const fresh = await request(`/v1/agent/sessions/${encodeURIComponent(session.session_id)}/cancel`, {
        method: "POST",
        headers: tenantHeaders(false),
      });
      state.agentPollToken += 1;
      applyAgentSession(fresh);
      setStatus("agent-status", "Run cancelled.", "error");
    } catch (err) {
      setStatus("agent-status", cleanMessage(err.message || err), "error");
    } finally {
      setBusy(button, false);
    }
  }

  if ($("agent-plan")) $("agent-plan").addEventListener("click", planAgentTask);
  if ($("agent-confirm")) $("agent-confirm").addEventListener("click", () => confirmAgentTask(null));
  if ($("agent-retry")) $("agent-retry").addEventListener("click", () => confirmAgentTask(0.5));
  if ($("agent-cancel")) $("agent-cancel").addEventListener("click", cancelAgentTask);
  if ($("agent-new")) $("agent-new").addEventListener("click", resetAgent);
  if ($("agent-download")) $("agent-download").addEventListener("click", async (event) => {
    event.preventDefault();
    const session = state.agentSession;
    if (!session || !session.download_path) return;
    try {
      const name = ((session.dataset && session.dataset.name) || "dataset")
        .replace(/[^a-z0-9._-]+/gi, "-");
      await downloadWithTenant(session.download_path, `${name}.zip`);
      setStatus("agent-status", "Dataset download started.", "ok");
    } catch (err) {
      setStatus("agent-status", cleanMessage(err.message || err), "error");
    }
  });

  // Datasets
  async function loadDatasets() {
    const list = $("datasets-list");
    const meta = $("datasets-meta");
    if (!list) return;
    if (meta) meta.textContent = "Loading dataset versions…";
    setStatus("datasets-status", "");
    try {
      const datasets = await request("/v1/datasets", { headers: tenantHeaders(false) });
      list.innerHTML = "";
      if (!datasets || !datasets.length) {
        const empty = node("div", "empty-state");
        empty.append(
          node("strong", null, "No dataset versions yet."),
          node("p", null, "Run the Agent on an asset to build a dataset automatically.")
        );
        list.appendChild(empty);
      } else {
        datasets.forEach((dataset) => list.appendChild(renderDatasetCard(dataset)));
      }
      if (meta) meta.textContent = `${(datasets || []).length} dataset version${(datasets || []).length === 1 ? "" : "s"}`;
    } catch (err) {
      setStatus("datasets-status", cleanMessage(err.message || err), "error");
      list.innerHTML = "";
      if (meta) meta.textContent = "Could not load datasets";
    }
  }

  function renderDatasetCard(dataset) {
    const card = node("article", "card dataset-card");
    const head = node("div", "dataset-head");
    const title = node("div");
    title.append(
      node("h2", "card-title", dataset.name || "Untitled dataset"),
      node("p", "dataset-meta", `${dataset.image_count} image${dataset.image_count === 1 ? "" : "s"} · ${dataset.annotation_count} annotation${dataset.annotation_count === 1 ? "" : "s"} · ${formatDate(dataset.created_at)}`)
    );
    const actions = node("div", "actions");
    const exportButton = node("button", "btn btn-secondary btn-small", "Export COCO");
    exportButton.type = "button";
    exportButton.disabled = !dataset.annotation_count;
    exportButton.addEventListener("click", () => exportDataset(dataset, exportButton, card));
    actions.appendChild(exportButton);
    head.append(title, actions);
    card.appendChild(head);

    if ((dataset.categories || []).length) {
      const categories = node("div", "category-row");
      dataset.categories.forEach((category) => {
        categories.appendChild(node("span", "category-chip", category.name));
      });
      card.appendChild(categories);
    }
    const status = node("div", "status");
    status.dataset.datasetStatus = "true";
    card.appendChild(status);
    return card;
  }

  async function exportDataset(dataset, button, card) {
    setBusy(button, true, "Exporting…");
    const status = card.querySelector('[data-dataset-status="true"]');
    if (status) {
      status.textContent = "Queuing COCO export…";
      status.dataset.kind = "busy";
    }
    try {
      let job = await request("/v1/dataset-exports", {
        method: "POST",
        headers: tenantHeaders(true),
        body: JSON.stringify({
          version_id: dataset.version_id,
          format: "coco",
          idempotency_key: `ui-coco-${dataset.version_id}-${Date.now()}`,
        }),
      });
      for (let attempt = 0; attempt < 120; attempt += 1) {
        if (job.status === "succeeded" || job.status === "failed" || job.status === "cancelled") break;
        await sleep(500);
        job = await request(`/v1/dataset-exports/${encodeURIComponent(job.job_id)}`, {
          headers: tenantHeaders(false),
        });
      }
      if (job.status === "succeeded") {
        if (status) {
          status.textContent = "COCO export ready. Download starting…";
          status.dataset.kind = "ok";
        }
        await downloadWithTenant(
          `/v1/dataset-exports/${encodeURIComponent(job.job_id)}/download`,
          `${(dataset.name || "dataset").replace(/[^a-z0-9._-]+/gi, "-")}.zip`,
        );
      } else {
        if (status) {
          status.textContent = `Export ${job.status}${job.error ? `: ${cleanMessage(job.error)}` : ""}`;
          status.dataset.kind = "error";
        }
      }
    } catch (err) {
      if (status) {
        status.textContent = cleanMessage(err.message || err);
        status.dataset.kind = "error";
      }
    } finally {
      setBusy(button, false);
    }
  }

  if ($("datasets-refresh")) $("datasets-refresh").addEventListener("click", loadDatasets);

  // Jobs
  function jobKindLabel(kind) {
    return {
      "batch-label": "Labeling",
      "index-asset": "Search indexing",
      "dataset-export": "Dataset export",
    }[kind] || "Background job";
  }

  async function loadJobs() {
    const list = $("jobs-list");
    const meta = $("jobs-meta");
    if (!list) return;
    if (meta) meta.textContent = "Loading jobs…";
    setStatus("jobs-status", "");
    try {
      const data = await request("/v1/jobs", { headers: tenantHeaders(false) });
      const jobs = (data && data.jobs) || [];
      list.innerHTML = "";
      if (!jobs.length) {
        const empty = node("div", "empty-state");
        empty.append(
          node("strong", null, "No background jobs yet."),
          node("p", null, "Uploading an asset queues a search-indexing job.")
        );
        list.appendChild(empty);
      } else {
        jobs.slice().reverse().forEach((job) => list.appendChild(renderJobCard(job)));
      }
      if (meta) meta.textContent = `${jobs.length} job${jobs.length === 1 ? "" : "s"}`;
    } catch (err) {
      setStatus("jobs-status", cleanMessage(err.message || err), "error");
      list.innerHTML = "";
      if (meta) meta.textContent = "Could not load jobs";
    }
  }

  function renderJobCard(job) {
    const card = node("article", "card job-card");
    const head = node("div", "job-head");
    const text = node("div");
    text.append(
      node("h2", "card-title", jobKindLabel(job.kind)),
      node("p", "job-meta", `Started ${formatDate(job.created_at)} · ${job.attempts} attempt${job.attempts === 1 ? "" : "s"}`)
    );
    const pill = node("span", "status-pill", job.status);
    pill.dataset.status = job.status;
    head.append(text, pill);
    card.appendChild(head);
    if (job.error) card.appendChild(node("p", "job-error", cleanMessage(job.error)));
    return card;
  }

  if ($("jobs-refresh")) $("jobs-refresh").addEventListener("click", loadJobs);

  // Init
  checkRuntime();
  loadAssets({ quiet: true });
})();
