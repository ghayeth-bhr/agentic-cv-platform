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
    assetsLoading: false,
    assetsError: null,
    selectedAsset: null,
    lineage: new Map(),
    searchPayload: null,
    searchResults: [],
    searchSelected: new Set(),
    uploadFiles: [],
    datasets: [],
    datasetThumbUrls: [],
    agentSession: null,
    agentPollToken: 0,
    agentOverlayUrls: [],
    agentCalibrationKey: "",
    review: {
      open: false, versionId: null, dataset: null, panel: null, images: [],
      index: -1, image: null, objectUrl: null, socket: null, snapshot: null,
      mode: "positive", status: null, listEl: null, save: null, canvas: null,
      imageEl: null, labelEl: null,
    },
    gallery: {
      open: false, versionId: null, dataset: null, images: [], visible: [], index: -1,
      total: 0, offset: 0, hasMore: false, filters: {
        categoryId: "all", split: "all", unannotated: false, lowConfidence: false,
      },
      toggles: { boxes: true, masks: true, labels: true },
      objectUrls: [],
    },
    agentLive: {
      sessionId: null, jobId: null, cursor: 0, processed: 0, total: 0,
      images: [], stopped: false, controller: null, retryTimer: null, ready: false,
    },
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

  const CLASS_COLORS = ["#ff4040", "#40a0ff", "#40d278", "#f5b437", "#b07cff", "#ff7ac6"];

  function classColor(categoryId) {
    const index = Math.abs(Number(categoryId) || 0) % CLASS_COLORS.length;
    return CLASS_COLORS[index];
  }

  function polygonArea(coords) {
    if (!coords || coords.length < 6) return 0;
    let sum = 0;
    for (let i = 0; i + 3 < coords.length; i += 2) {
      sum += coords[i] * coords[i + 3] - coords[i + 2] * coords[i + 1];
    }
    return Math.abs(sum) / 2;
  }

  function annotationArea(annotation, image) {
    const polygons = (annotation.segmentation || []).map(polygonArea);
    const maskArea = polygons.reduce((total, value) => total + value, 0);
    if (maskArea > 0) return maskArea;
    const box = annotation.bbox || [];
    return Math.max(0, (Number(box[2]) || 0) * (Number(box[3]) || 0));
  }

  function node(tag, className, text) {
    const el = document.createElement(tag);
    if (className) el.className = className;
    if (text != null) el.textContent = text;
    return el;
  }

  function statePanel(kind, title, detail, actionLabel, action) {
    const panel = node("div", `empty-state state-panel state-panel-${kind}`);
    panel.dataset.state = kind;
    panel.setAttribute("role", kind === "error" ? "alert" : "status");
    panel.appendChild(node("strong", null, title));
    if (detail) panel.appendChild(node("p", null, detail));
    if (actionLabel && action) {
      const button = node("button", "btn btn-secondary btn-small", actionLabel);
      button.type = "button";
      button.addEventListener("click", action);
      panel.appendChild(button);
    }
    return panel;
  }

  function setBusyRegion(id, busy) {
    const el = $(id);
    if (el) el.setAttribute("aria-busy", busy ? "true" : "false");
  }

  function assetSkeletons(count) {
    const fragment = document.createDocumentFragment();
    for (let index = 0; index < count; index += 1) {
      const card = node("div", "asset-card skeleton-card");
      card.setAttribute("aria-hidden", "true");
      const thumb = node("div", "asset-thumb skeleton-block");
      const body = node("div", "asset-body");
      body.append(
        node("div", "skeleton-line skeleton-line-title"),
        node("div", "skeleton-line skeleton-line-meta")
      );
      card.append(thumb, body);
      fragment.appendChild(card);
    }
    return fragment;
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
    if (name !== "datasets") {
      closeReview();
      closeDatasetGallery();
    }
    if (name === "assets") loadAssets();
    if (name === "agent") initAgentView();
    if (name === "datasets") showDatasetsForHash();
    if (name === "jobs") loadJobs();
  }

  document.querySelectorAll(".nav-item").forEach((btn) => {
    btn.addEventListener("click", () => {
      if (btn.dataset.view === "datasets" && galleryVersionFromHash()) {
        window.history.replaceState(null, "", "#/datasets");
      }
      showView(btn.dataset.view);
    });
  });

  function galleryVersionFromHash() {
    const match = /^#\/datasets\/([0-9a-f-]{36})$/i.exec(window.location.hash || "");
    return match ? match[1] : null;
  }

  window.addEventListener("hashchange", () => {
    const versionId = galleryVersionFromHash();
    if (versionId) {
      if (!$("view-datasets") || $("view-datasets").hidden) showView("datasets");
      else openDatasetGallery(versionId);
    } else if (window.location.hash === "#/datasets" && $("view-datasets") && !$("view-datasets").hidden) {
      showDatasetList();
    }
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
    state.assetsLoading = true;
    state.assetsError = null;
    setBusyRegion("asset-gallery", true);
    if (meta && !quiet) meta.textContent = "Loading assets…";
    if (!state.assets.length) {
      const grid = $("asset-gallery");
      if (grid) {
        grid.innerHTML = "";
        grid.appendChild(assetSkeletons(3));
      }
    }
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
      state.assetsLoading = false;
      state.assetsError = null;
      setBusyRegion("asset-gallery", false);
      renderAssets();
      renderAssetDetail();
      syncSearchScopeOptions();
      const n = state.assets.length;
      if (meta) meta.textContent = n ? `${n} asset${n === 1 ? "" : "s"}` : "No assets yet";
      return state.assets;
    } catch (err) {
      const message = cleanMessage(err.message || err);
      state.assetsLoading = false;
      state.assetsError = message;
      state.assets = [];
      state.selectedAsset = null;
      state.lineage = new Map();
      setBusyRegion("asset-gallery", false);
      if (meta) meta.textContent = "Could not load assets";
      if (!quiet) setStatus("gallery-status", message, "error");
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
    if (state.assetsLoading && !state.assets.length) {
      grid.appendChild(assetSkeletons(3));
      return;
    }
    if (state.assetsError) {
      grid.appendChild(
        statePanel(
          "error",
          "Could not load assets.",
          state.assetsError,
          "Try again",
          () => loadAssets()
        )
      );
      return;
    }
    if (!assets.length) {
      if (state.assets.length) {
        grid.appendChild(
          statePanel(
            "empty",
            "No matching assets.",
            "Try a different filter, or clear it to see the whole library.",
            "Clear filter",
            () => {
              const input = $("asset-filter");
              if (input) input.value = "";
              renderAssets();
            }
          )
        );
      } else {
        grid.appendChild(
          statePanel(
            "empty",
            "No assets yet.",
            "Upload photos or videos to start searching and labeling.",
            "Upload media",
            () => {
              showUploadPanel(true);
              if ($("asset-name")) $("asset-name").focus();
            }
          )
        );
      }
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
    const requested = Number($("search-limit") && $("search-limit").value);
    const limit = Number.isFinite(requested) ? Math.max(1, Math.min(500, Math.round(requested))) : 20;
    const payload = { query, limit };
    const scope = $("search-scope") ? $("search-scope").value : "all";
    if (scope === "selected" && state.selectedAsset) {
      payload.asset_ids = [state.selectedAsset.asset_id];
    }
    return payload;
  }

  function assetsById() {
    return new Map(state.assets.map((asset) => [String(asset.asset_id), asset]));
  }

  function searchMinScore() {
    const slider = $("search-min-score");
    return slider ? Number(slider.value) / 100 : 0;
  }

  function updateSearchSelection() {
    const selected = state.searchSelected;
    const total = state.searchResults.length;
    const count = state.searchResults.filter((result) => selected.has(result.record_id)).length;
    if ($("search-selection")) $("search-selection").hidden = !total;
    if ($("search-selected-count")) $("search-selected-count").textContent = `${count} of ${total} selected`;
    const button = $("subasset-button");
    if (button && !button.dataset.busy) {
      button.disabled = !count;
      button.textContent = count ? `Save ${count} selected as asset` : "Save selected as asset";
    }
    const threshold = searchMinScore();
    if ($("search-min-score-value")) $("search-min-score-value").textContent = `${Math.round(threshold * 100)}%`;
    document.querySelectorAll("#search-results .result-card").forEach((card) => {
      const id = card.dataset.recordId;
      const isSelected = selected.has(id);
      card.classList.toggle("is-selected", isSelected);
      card.setAttribute("aria-pressed", String(isSelected));
      const box = card.querySelector(".result-check");
      if (box) box.checked = isSelected;
      card.classList.toggle("below-threshold", Number(card.dataset.score) < threshold);
    });
  }

  function renderSearchResults() {
    const container = $("search-results");
    if (!container) return;
    container.innerHTML = "";
    if (!state.searchResults.length) {
      container.appendChild(node("div", "results-empty", "No results yet. Try a description of the people, objects, or scene."));
      updateSearchSelection();
      return;
    }
    const byId = assetsById();
    const grid = node("div", "result-gallery");
    state.searchResults.forEach((result, index) => {
      const score = Number(result.score);
      const pct = Number.isFinite(score) ? Math.max(0, Math.min(100, score * 100)) : 0;
      const card = node("div", "result-card");
      card.dataset.recordId = result.record_id;
      card.dataset.score = Number.isFinite(score) ? String(score) : "0";
      card.tabIndex = 0;
      card.setAttribute("role", "button");
      card.setAttribute("aria-label", `Match ${index + 1}: include in the new asset`);
      const check = document.createElement("input");
      check.type = "checkbox";
      check.className = "result-check";
      check.tabIndex = -1;
      check.setAttribute("aria-hidden", "true");
      card.appendChild(check);
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
      const open = node("button", "result-open", parent ? `${assetName(parent)} ↗` : "Source asset ↗");
      open.type = "button";
      open.title = "Open the source asset";
      open.addEventListener("click", (event) => {
        event.stopPropagation();
        if (result.asset_id) selectAssetById(result.asset_id);
        if ($("asset-detail")) $("asset-detail").scrollIntoView({ behavior: "smooth", block: "start" });
      });
      card.append(head, open);
      const bar = node("div", "score-bar");
      const fill = document.createElement("i");
      fill.style.width = `${pct.toFixed(1)}%`;
      bar.appendChild(fill);
      card.appendChild(bar);
      const toggle = () => {
        if (state.searchSelected.has(result.record_id)) state.searchSelected.delete(result.record_id);
        else state.searchSelected.add(result.record_id);
        updateSearchSelection();
      };
      card.addEventListener("click", toggle);
      card.addEventListener("keydown", (event) => {
        if (event.key === " " || event.key === "Enter") {
          event.preventDefault();
          toggle();
        }
      });
      grid.appendChild(card);
    });
    container.appendChild(grid);
    updateSearchSelection();
  }

  if ($("search-select-all")) $("search-select-all").addEventListener("click", () => {
    state.searchResults.forEach((result) => state.searchSelected.add(result.record_id));
    updateSearchSelection();
  });
  if ($("search-select-none")) $("search-select-none").addEventListener("click", () => {
    state.searchSelected.clear();
    updateSearchSelection();
  });
  if ($("search-min-score")) $("search-min-score").addEventListener("input", updateSearchSelection);
  if ($("search-select-above")) $("search-select-above").addEventListener("click", () => {
    const threshold = searchMinScore();
    state.searchSelected = new Set(
      state.searchResults.filter((result) => Number(result.score) >= threshold).map((result) => result.record_id)
    );
    updateSearchSelection();
  });

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
    state.searchSelected = new Set();
    state.searchPayload = null;
    renderSearchResults();
    try {
      const result = await request("/v1/search", {
        method: "POST",
        headers: tenantHeaders(true),
        body: JSON.stringify(payload),
      });
      state.searchResults = (result && result.results) || [];
      // Everything starts selected; untick the images that do not belong.
      state.searchSelected = new Set(state.searchResults.map((item) => item.record_id));
      state.searchPayload = payload;
      renderSearchResults();
      setStatus(
        "search-status",
        `${state.searchResults.length} result${state.searchResults.length === 1 ? "" : "s"}. Untick the images that should not go into the new asset.`,
        "ok",
      );
    } catch (err) {
      setStatus("search-status", cleanMessage(err.message || err), "error");
    } finally {
      setBusy(button, false);
    }
  });

  if ($("subasset-button")) $("subasset-button").addEventListener("click", async () => {
    const button = $("subasset-button");
    const payload = state.searchPayload || currentSearchPayload();
    const recordIds = state.searchResults
      .map((result) => result.record_id)
      .filter((id) => state.searchSelected.has(id));
    if (!payload.query || !state.searchResults.length) {
      setStatus("search-status", "Run a search before saving results.", "error");
      return;
    }
    if (!recordIds.length) {
      setStatus("search-status", "Select at least one image to save.", "error");
      return;
    }
    const name = (($("subasset-name") && $("subasset-name").value) || "").trim();
    button.dataset.busy = "true";
    setBusy(button, true, "Saving…");
    setStatus("search-status", `Saving ${recordIds.length} selected image${recordIds.length === 1 ? "" : "s"} as a new asset…`, "busy");
    try {
      const record = await request("/v1/sub-assets/from-search", {
        method: "POST",
        headers: tenantHeaders(true),
        body: JSON.stringify({ query: payload.query, record_ids: recordIds, ...(name ? { name } : {}) }),
      });
      if ($("subasset-name")) $("subasset-name").value = "";
      await loadAssets({ quiet: true });
      if (record.asset_id) selectAssetById(record.asset_id);
      const count = Number.isFinite(record.source_count) ? record.source_count : recordIds.length;
      setStatus("search-status", `Saved “${record.asset_name || "Sub-asset"}” with ${count} matched image${count === 1 ? "" : "s"}.`, "ok");
      setStatus("gallery-status", `Sub-asset saved from search: “${record.query || payload.query}”.`, "ok");
    } catch (err) {
      setStatus("search-status", cleanMessage(err.message || err), "error");
    } finally {
      delete button.dataset.busy;
      setBusy(button, false);
      updateSearchSelection();
    }
  });

  // Agent
  const AGENT_SESSION_KEY = "agentic-cv-agent-session";
  const AGENT_ACTIVE = new Set(["calibrating", "labeling", "exporting"]);
  const AGENT_OPEN = new Set(["gathering", "awaiting_confirmation", "needs_attention"]);
  const AGENT_DONE = new Set(["complete", "failed", "cancelled"]);
  const agentSelection = new Set();
  const AGENT_STATUS_LABEL = {
    gathering: "Needs details",
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

  function agentComposeOpen(session) {
    return !session || AGENT_OPEN.has(session.status) || AGENT_DONE.has(session.status);
  }

  function syncAgentSelection(session) {
    const known = new Set(state.assets.map((asset) => String(asset.asset_id)));
    if (session && AGENT_OPEN.has(session.status) && (session.asset_ids || []).length) {
      agentSelection.clear();
      session.asset_ids.forEach((id) => agentSelection.add(String(id)));
    }
    [...agentSelection].forEach((id) => { if (known.size && !known.has(id)) agentSelection.delete(id); });
    if (!agentSelection.size && state.assets.length) {
      const preferred = state.selectedAsset && known.has(String(state.selectedAsset.asset_id))
        ? String(state.selectedAsset.asset_id)
        : String(state.assets[0].asset_id);
      agentSelection.add(preferred);
    }
  }

  function renderAgentAssets(session) {
    const box = $("agent-assets");
    if (!box) return;
    box.innerHTML = "";
    const locked = !!session && !agentComposeOpen(session);
    let ready = false;
    if (state.assetsLoading && !state.assets.length) {
      box.appendChild(node("p", "agent-assets-state", "Loading assets…"));
    } else if (state.assetsError) {
      box.appendChild(node("p", "agent-assets-state", "Assets unavailable"));
    } else if (!state.assets.length) {
      box.appendChild(node("p", "agent-assets-state", "No assets yet — upload media first"));
    } else {
      ready = true;
      syncAgentSelection(session);
      state.assets.forEach((asset) => {
        const id = String(asset.asset_id);
        const option = node("label", "agent-asset-option");
        const input = document.createElement("input");
        input.type = "checkbox";
        input.value = id;
        input.checked = agentSelection.has(id);
        input.disabled = locked;
        input.addEventListener("change", () => {
          if (input.checked) agentSelection.add(id);
          else agentSelection.delete(id);
          updateAgentAssetCount();
        });
        const count = (asset.sources || []).length;
        option.append(input, node("span", "agent-asset-name", assetName(asset)),
          node("span", "agent-asset-meta", `${count} source${count === 1 ? "" : "s"}`));
        box.appendChild(option);
      });
    }
    box.setAttribute("aria-disabled", String(locked || !ready));
    updateAgentAssetCount();
    const open = agentComposeOpen(session);
    if ($("agent-message")) $("agent-message").disabled = !open;
    if ($("agent-plan")) {
      $("agent-plan").disabled = !open || !ready;
      $("agent-plan").textContent = session && AGENT_OPEN.has(session.status) ? "Send" : "Plan task";
    }
  }

  function updateAgentAssetCount() {
    const count = $("agent-assets-count");
    if (!count) return;
    const n = agentSelection.size;
    count.textContent = n ? `${n} selected` : "none selected";
  }

  function renderAgentQuickReplies(session) {
    const box = $("agent-quick-replies");
    if (!box) return;
    box.innerHTML = "";
    const replies = (session && AGENT_OPEN.has(session.status) && session.quick_replies) || [];
    box.hidden = !replies.length;
    replies.forEach((reply) => {
      const chip = node("button", "chip agent-quick-reply", reply);
      chip.type = "button";
      chip.addEventListener("click", () => sendAgentMessage(reply));
      box.appendChild(chip);
    });
  }

  function renderAgentMessages(session) {
    const list = $("agent-messages");
    if (!list) return;
    list.innerHTML = "";
    if (!session || !session.messages || !session.messages.length) {
      const empty = node("div", "agent-empty-chat");
      empty.textContent = "Describe the dataset you need, for example “Segment hard hats and safety vests, export YOLO.” I'll ask for anything that's missing.";
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
    if (!session) return;
    const classes = session.classes || [];
    const grid = node("dl", "agent-spec-grid");
    const exportLabel = { coco: "COCO", yolo: "YOLO" }[session.export_format] || "—";
    const sample = (session.spec && session.spec.sample_size) || 5;
    [
      ["Task", session.task_type || "—"],
      ["Assets", session.asset_name],
      ["Export", exportLabel],
      ["Calibration", `${sample} sample image${sample === 1 ? "" : "s"} per class`],
      ["Planner", session.planner_mode === "ollama" ? "Local model" : "Local rules"],
    ].forEach(([term, value]) => {
      const [dt, dd] = agentSpecRow(term, value);
      grid.append(dt, dd);
    });
    box.appendChild(grid);
    const list = node("ul", "agent-class-list");
    if (!classes.length) list.appendChild(node("li", "agent-class-empty", "No classes yet"));
    classes.forEach((item) => {
      const row = node("li", "agent-class");
      row.appendChild(node("strong", null, item.name));
      if ((item.attributes || []).length) row.appendChild(node("span", "agent-class-attrs", item.attributes.join(", ")));
      const prompt = (item.prompts || [])[0];
      if (prompt) row.appendChild(node("span", "agent-class-prompt", `SAM 3: “${prompt}”`));
      list.appendChild(row);
    });
    box.appendChild(list);
    const missing = session.missing_slots || [];
    if (missing.length) {
      const names = { assets: "assets", classes: "object classes", task_type: "detection or segmentation" };
      box.appendChild(node("p", "agent-note", `Still needed: ${missing.map((slot) => names[slot] || slot).join(", ")}.`));
    }
    (session.assumptions || []).forEach((text) => box.appendChild(node("p", "agent-note", cleanMessage(text))));
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

  function agentFormatLabel(session) {
    if (session.export_format !== "yolo") return "COCO";
    return session.task_type === "segmentation" ? "YOLO segmentation" : "YOLO detection";
  }

  function renderAgentProgress(session) {
    const list = $("agent-progress");
    if (!list) return;
    list.innerHTML = "";
    const hasSpec = !!session.spec;
    const gathering = session.status === "gathering";
    list.appendChild(agentStep(
      "Task plan",
      hasSpec ? "Specification ready" : gathering ? "Waiting for your answer" : "Waiting for a task",
      hasSpec ? "done" : gathering ? "active" : "pending",
    ));

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
      exportDetail = `${agentFormatLabel(session)} archive ready to download`;
    } else if (session.export_status === "failed") {
      exportState = "error";
      exportDetail = "Export failed";
    } else if (session.export_status === "cancelled") {
      exportState = "error";
      exportDetail = "Export cancelled";
    } else if (session.export_status === "queued" || session.export_status === "running") {
      exportState = "active";
      exportDetail = `Packaging the ${agentFormatLabel(session)} archive`;
    }
    list.appendChild(agentStep("Download package", exportDetail, exportState));
  }

  function revokeAgentOverlays() {
    state.agentOverlayUrls.forEach((url) => URL.revokeObjectURL(url));
    state.agentOverlayUrls = [];
  }

  function calibrationModeLabel(calibration) {
    if (!calibration) return "";
    if (calibration.validator_mode === "vlm") {
      return calibration.validator_model
        ? `Vision validation · ${calibration.validator_model}`
        : "Vision validation";
    }
    if (calibration.validator_mode === "manual") return "Fixed threshold · no calibration";
    if (calibration.validator_mode === "custom") return "Custom validator";
    return "Fallback validator · at least one prediction";
  }

  async function loadAgentOverlay(session, overlayId, img) {
    if (!session || !overlayId || !img || img.dataset.loaded) return;
    img.dataset.loaded = "loading";
    try {
      const path = `/v1/agent/sessions/${encodeURIComponent(session.session_id)}/calibration-overlays/${encodeURIComponent(overlayId)}`;
      const response = await fetch(path, { headers: tenantHeaders(false) });
      if (!response.ok) throw new Error(`Overlay unavailable (${response.status})`);
      const objectUrl = URL.createObjectURL(await response.blob());
      state.agentOverlayUrls.push(objectUrl);
      img.src = objectUrl;
      img.dataset.loaded = "true";
      img.hidden = false;
    } catch (err) {
      img.dataset.loaded = "error";
      if (img.parentElement) {
        img.parentElement.appendChild(node("span", "agent-calibration-overlay-error", cleanMessage(err.message || err)));
      }
    }
  }

  function renderAgentCalibration(session) {
    const box = $("agent-calibration");
    if (!box) return;
    const calibration = session && session.calibration;
    const nextKey = JSON.stringify(calibration || null);
    if (nextKey === state.agentCalibrationKey) return;
    state.agentCalibrationKey = nextKey;
    revokeAgentOverlays();
    box.innerHTML = "";
    if (!calibration) {
      box.hidden = true;
      return;
    }
    box.hidden = false;
    const head = node("div", "agent-calibration-head");
    head.append(
      node("strong", null, calibrationModeLabel(calibration)),
      node("span", "agent-calibration-count", `${(calibration.attempts || []).length} threshold${(calibration.attempts || []).length === 1 ? "" : "s"} tried`)
    );
    box.appendChild(head);
    if (calibration.notice) {
      box.appendChild(node("p", "agent-calibration-notice", cleanMessage(calibration.notice)));
    }
    if (calibration.status === "manual") {
      box.appendChild(node("p", "agent-calibration-empty", `Running at the fixed threshold ${Number(calibration.threshold).toFixed(2)}.`));
      return;
    }
    const perClass = calibration.classes || [];
    if (perClass.length <= 1 && (calibration.prompts_tried || []).length > 1) {
      box.appendChild(node("p", "agent-calibration-notice", `Prompts tried: ${calibration.prompts_tried.map((prompt) => `“${cleanMessage(prompt)}”`).join(", ")}`));
    }
    if (perClass.length > 1) {
      perClass.forEach((item) => {
        const result = item.status === "calibrated" && item.threshold != null
          ? `threshold ${Number(item.threshold).toFixed(2)}`
          : "no threshold passed";
        box.appendChild(node("h4", "agent-calibration-class", `${cleanMessage(item.name)} · “${cleanMessage(item.prompt)}” · ${result}`));
        if ((item.prompts_tried || []).length > 1) {
          box.appendChild(node("p", "agent-calibration-notice", `Prompts tried: ${item.prompts_tried.map((prompt) => `“${cleanMessage(prompt)}”`).join(", ")}`));
        }
        renderCalibrationAttempts(box, item, session);
      });
      return;
    }
    renderCalibrationAttempts(box, calibration, session);
  }

  function renderCalibrationAttempts(box, calibration, session) {
    const attempts = calibration.attempts || [];
    if (!attempts.length) {
      box.appendChild(node("p", "agent-calibration-empty", calibration.reason || "No threshold attempts were recorded."));
      return;
    }
    attempts.forEach((attempt) => {
      const accepted = Number(attempt.accepted_count) || 0;
      const total = Number(attempt.sample_count) || 0;
      const pct = total ? Math.round((Number(attempt.agreement) || 0) * 100) : 0;
      const details = node("details", "agent-calibration-attempt");
      details.dataset.state = attempt.threshold != null && calibration.threshold != null && Number(attempt.threshold) === Number(calibration.threshold) ? "chosen" : "tried";
      const summary = node("summary");
      summary.append(
        node("span", "agent-threshold", `Threshold ${Number(attempt.threshold).toFixed(2)}`),
        node("span", "agent-agreement", `${accepted}/${total} accepted · ${pct}%`)
      );
      details.appendChild(summary);
      const list = node("div", "agent-calibration-samples");
      (attempt.verdicts || []).forEach((verdict) => {
        const row = node("div", "agent-calibration-sample");
        row.dataset.accepted = verdict.accepted ? "true" : "false";
        const copy = node("div", "agent-calibration-sample-copy");
        const emptyResult = verdict.prediction_count === 0;
        const verdictText = verdict.accepted
          ? (emptyResult ? "Correct · no target" : "Accepted")
          : (emptyResult && verdict.target_present ? "Missed" : "Rejected");
        const confidence = verdict.confidence == null ? "" : ` · confidence ${Number(verdict.confidence).toFixed(2)}`;
        copy.append(
          node("strong", null, `${verdictText} · ${cleanMessage(verdict.image_id)}${confidence}`),
          node("span", null, cleanMessage(verdict.reasoning || "No reasoning recorded."))
        );
        row.appendChild(copy);
        if (verdict.overlay_id) {
          const figure = node("figure", "agent-calibration-overlay");
          const img = document.createElement("img");
          img.alt = `Calibration overlay for ${cleanMessage(verdict.image_id)}`;
          img.hidden = true;
          img.loading = "lazy";
          figure.appendChild(img);
          row.appendChild(figure);
          details.addEventListener("toggle", () => {
            if (details.open) loadAgentOverlay(session, verdict.overlay_id, img);
          }, { once: false });
        }
        list.appendChild(row);
      });
      details.appendChild(list);
      box.appendChild(details);
    });
  }

  function resetAgentLive() {
    const live = state.agentLive;
    stopAgentLiveStream(true);
    Object.assign(live, {
      sessionId: null, jobId: null, cursor: 0, processed: 0, total: 0,
      images: [], stopped: false, ready: false,
    });
    const section = $("agent-live");
    if (section) section.hidden = true;
    const badge = $("agent-live-badge");
    if (badge) badge.hidden = true;
    const grid = $("agent-live-grid");
    if (grid) grid.replaceChildren();
    const galleryLink = $("agent-open-gallery");
    if (galleryLink) galleryLink.hidden = true;
    setStatus("agent-live-notice", "");
  }

  function stopAgentLiveStream(clear) {
    const live = state.agentLive;
    if (live.retryTimer) {
      clearTimeout(live.retryTimer);
      live.retryTimer = null;
    }
    if (live.controller) {
      live.controller.abort();
      live.controller = null;
    }
    live.reading = false;
    live.stopped = !!clear;
    if (clear) {
      live.cursor = 0;
      live.processed = 0;
      live.total = 0;
      live.images = [];
    }
  }

  function updateAgentLiveMeta() {
    const live = state.agentLive;
    if ($("agent-live-count")) {
      $("agent-live-count").textContent = `${live.processed} / ${live.total || "?"} images`;
    }
    const bar = $("agent-live-progress-bar");
    if (bar) {
      const pct = live.total ? Math.min(100, Math.round((live.processed / live.total) * 100)) : 0;
      bar.style.width = `${pct}%`;
    }
  }

  function renderAgentLiveGrid() {
    const live = state.agentLive;
    const grid = $("agent-live-grid");
    if (!grid) return;
    grid.replaceChildren();
    live.images.forEach((image, index) => {
      const figure = node("figure");
      figure.dataset.imageId = String(image.imageId);
      const img = node("img");
      img.alt = `Live label preview ${index + 1}`;
      img.loading = "lazy";
      img.src = `/v1/label-jobs/${encodeURIComponent(live.jobId)}/preview/${encodeURIComponent(image.imageId)}/image?tenant_id=${encodeURIComponent(DEFAULT_TENANT)}`;
      const caption = node("figcaption", null, cleanMessage(image.imageId).split(":").pop());
      figure.append(img, caption);
      grid.appendChild(figure);
    });
  }

  function handlePreviewEvent(payload) {
    const live = state.agentLive;
    const eventId = Number(payload && payload.event_id) || 0;
    if (eventId && eventId <= live.cursor) return;
    if (eventId) live.cursor = eventId;
    live.processed = Number(payload && payload.processed) || live.processed;
    live.total = Number(payload && payload.total) || live.total;
    const imageId = payload && payload.image_id;
    if (imageId && !live.images.some((item) => item.imageId === String(imageId))) {
      live.images.push({ imageId: String(imageId) });
      renderAgentLiveGrid();
    }
    updateAgentLiveMeta();
    setStatus("agent-live-notice", "");
  }

  async function readAgentLiveStream() {
    const live = state.agentLive;
    if (live.stopped || !live.jobId || live.reading) return;
    live.reading = true;
    try {
      const response = await fetch(
        `/v1/jobs/${encodeURIComponent(live.jobId)}/preview?after_event_id=${live.cursor}&tenant_id=${encodeURIComponent(DEFAULT_TENANT)}`,
        { headers: tenantHeaders(false), signal: live.controller ? live.controller.signal : undefined }
      );
      if (!response.ok) throw new Error(`Live view unavailable (${response.status})`);
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      while (true) {
        const chunk = await reader.read();
        if (chunk.done) break;
        buffer += decoder.decode(chunk.value, { stream: true });
        const parts = buffer.split("\n\n");
        buffer = parts.pop();
        parts.forEach((part) => {
          const line = part.split("\n").find((item) => item.startsWith("data:"));
          if (!line) return;
          try { handlePreviewEvent(JSON.parse(line.slice(5).trim())); } catch { /* ignore malformed frames */ }
        });
      }
      buffer += decoder.decode();
      const trailing = buffer.split("\n\n").find((part) => part.includes("data:"));
      if (trailing) {
        const line = trailing.split("\n").find((item) => item.startsWith("data:"));
        if (line) {
          try { handlePreviewEvent(JSON.parse(line.slice(5).trim())); } catch { /* ignore malformed frames */ }
        }
      }
    } catch (err) {
      if (!live.stopped && !(err && err.name === "AbortError")) {
        setStatus(
          "agent-live-notice",
          `Live view connection dropped (${cleanMessage(err.message || err)}). Reconnecting without losing displayed images.`,
          "error"
        );
      }
    } finally {
      live.reading = false;
      if (!live.stopped && live.jobId) {
        live.retryTimer = setTimeout(readAgentLiveStream, 300);
      }
    }
  }

  function startAgentLiveStream(session) {
    const live = state.agentLive;
    const jobId = session.label_job_id ? String(session.label_job_id) : null;
    if (!jobId) return;
    if (live.jobId === jobId && (live.reading || live.retryTimer || live.controller)) return;
    stopAgentLiveStream(true);
    Object.assign(live, {
      sessionId: String(session.session_id), jobId, cursor: 0, processed: 0,
      total: live.total || 0, images: [], stopped: false, ready: false,
      controller: new AbortController(), retryTimer: null, reading: false,
    });
    renderAgentLiveGrid();
    updateAgentLiveMeta();
    if ($("agent-live-phase")) $("agent-live-phase").textContent = "Streaming";
    if ($("agent-live-summary")) $("agent-live-summary").textContent = "Images appear here as the label worker finishes each one.";
    if ($("agent-live-badge")) $("agent-live-badge").hidden = false;
    readAgentLiveStream();
  }

  function syncAgentLiveView(session) {
    const section = $("agent-live");
    if (!section) return;
    const jobId = session.label_job_id ? String(session.label_job_id) : null;
    const datasetVersionId = session.dataset_version_id ? String(session.dataset_version_id) : null;
    if (!jobId) {
      resetAgentLive();
      return;
    }
    section.hidden = false;
    const live = state.agentLive;
    const running = !datasetVersionId && !["failed", "cancelled"].includes(session.status)
      && ["queued", "running"].includes(session.label_status)
      || session.status === "labeling";
    if (running) {
      startAgentLiveStream(session);
      return;
    }
    stopAgentLiveStream(false);
    if ($("agent-live-badge")) $("agent-live-badge").hidden = true;
    if ($("agent-live-phase")) {
      if (datasetVersionId) $("agent-live-phase").textContent = "Ready";
      else if (session.label_status === "succeeded") $("agent-live-phase").textContent = "Labeling complete";
      else if (session.status === "cancelled") $("agent-live-phase").textContent = "Cancelled";
      else if (session.status === "failed") $("agent-live-phase").textContent = "Failed";
      else $("agent-live-phase").textContent = "Stopped";
    }
    if ($("agent-live-summary")) {
      $("agent-live-summary").textContent = datasetVersionId
        ? `${live.images.length} images labeled. The dataset version is ready.`
        : `${live.images.length} images labeled so far.`;
    }
    const link = $("agent-open-gallery");
    if (link) {
      link.hidden = !datasetVersionId;
      if (datasetVersionId) {
        link.href = `#/datasets/${encodeURIComponent(datasetVersionId)}`;
        link.dataset.galleryVersion = datasetVersionId;
      } else link.removeAttribute("href");
    }
    if (session.status === "cancelled") {
      setStatus("agent-live-notice", "Run cancelled. Live view stopped.", "error");
    } else if (session.status === "failed" || session.label_status === "failed") {
      setStatus("agent-live-notice", "Labeling failed. Live view stopped.", "error");
    } else if (datasetVersionId) {
      setStatus("agent-live-notice", "Dataset ready. Open the gallery to inspect every image.", "ok");
    }
    if (live.images.length) setStatus("agent-live-notice",
      datasetVersionId ? "Dataset ready. Open the gallery to inspect every image." :
        (session.status === "cancelled" ? "Run cancelled. Live view stopped." : $("agent-live-notice").textContent),
      datasetVersionId ? "ok" : (session.status === "cancelled" ? "error" : null));
  }

  if ($("agent-open-gallery")) {
    $("agent-open-gallery").addEventListener("click", (event) => {
      const versionId = event.currentTarget.dataset.galleryVersion;
      if (!versionId) return;
      event.preventDefault();
      window.location.hash = `#/datasets/${versionId}`;
      showView("datasets");
    });
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
    renderAgentQuickReplies(session);
    renderAgentSpec(session);
    renderAgentProgress(session);
    renderAgentCalibration(session);
    syncAgentLiveView(session);

    const classNames = (session.classes || []).map((item) => item.name).join(", ");
    if ($("agent-run-title")) $("agent-run-title").textContent = classNames || session.asset_name || "Task plan";
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
    else if (session.status === "complete") setStatus("agent-status", `Dataset ready. Download the ${agentFormatLabel(session)} archive from the run panel.`, "ok");
    else if (session.status === "gathering") setStatus("agent-status", "Answer the agent's question to continue.", "ok");
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
    revokeAgentOverlays();
    resetAgentLive();
    state.agentCalibrationKey = "";
    ["agent-spec", "agent-progress", "agent-calibration", "agent-run-error"].forEach((id) => {
      const el = $(id);
      if (el) {
        el.innerHTML = "";
        if (id === "agent-run-error" || id === "agent-calibration") el.hidden = true;
      }
    });
    ["agent-confirm", "agent-retry", "agent-cancel", "agent-download"].forEach((id) => {
      const el = $(id);
      if (el) el.hidden = true;
    });
    renderAgentMessages(null);
    renderAgentQuickReplies(null);
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
    if (!state.assets.length) {
      state.assetsLoading = true;
      renderAgentAssets(null);
      await loadAssets({ quiet: true });
    }
    if (state.agentSession) {
      applyAgentSession(state.agentSession);
      if (AGENT_ACTIVE.has(state.agentSession.status)) startAgentPolling();
      return;
    }
    renderAgentAssets(null);
    await restoreAgentSession();
    if (!state.agentSession && !state.assets.length) {
      setStatus(
        "agent-status",
        state.assetsError
          ? `Could not load assets: ${state.assetsError}`
          : "No assets yet. Upload media in Assets before planning a task.",
        "error"
      );
    }
  }

  async function sendAgentMessage(text) {
    const button = $("agent-plan");
    const input = $("agent-message");
    const message = (text != null ? text : (input && input.value) || "").trim();
    const session = state.agentSession;
    const assetIds = [...agentSelection];
    if (!assetIds.length && !(session && AGENT_OPEN.has(session.status))) {
      setStatus("agent-status", "Select at least one asset before planning a task.", "error");
      return;
    }
    if (!message) {
      setStatus("agent-status", "Describe what the agent should label.", "error");
      return;
    }
    const continuing = session && AGENT_OPEN.has(session.status);
    setBusy(button, true, continuing ? "Sending…" : "Planning…");
    setStatus("agent-status", continuing ? "Updating the plan…" : "Planning the task…", "busy");
    try {
      const fresh = continuing
        ? await request(`/v1/agent/sessions/${encodeURIComponent(session.session_id)}/messages`, {
          method: "POST",
          headers: tenantHeaders(true),
          body: JSON.stringify({ message, asset_ids: assetIds }),
        })
        : await request("/v1/agent/sessions", {
          method: "POST",
          headers: tenantHeaders(true),
          body: JSON.stringify({ asset_ids: assetIds, message }),
        });
      if (!continuing) {
        revokeAgentOverlays();
        resetAgentLive();
        state.agentCalibrationKey = "";
      }
      if (input) input.value = "";
      applyAgentSession(fresh);
    } catch (err) {
      setStatus("agent-status", cleanMessage(err.message || err), "error");
    } finally {
      setBusy(button, false);
      renderAgentAssets(state.agentSession);
    }
  }

  async function planAgentTask() {
    await sendAgentMessage(null);
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
    const confirm = $("agent-confirm");
    const retry = $("agent-retry");
    const cancel = $("agent-cancel");
    if (confirm && threshold == null) confirm.hidden = true;
    if (retry && threshold != null) retry.hidden = true;
    if (cancel) {
      cancel.hidden = false;
      cancel.disabled = false;
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
  if ($("agent-message")) $("agent-message").addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      if (!$("agent-plan").disabled) planAgentTask();
    }
  });
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
  function reviewStatus(message, kind) {
    const el = state.review.status;
    if (!el) return;
    el.textContent = message || "";
    if (kind) el.dataset.kind = kind; else delete el.dataset.kind;
  }

  function closeReview() {
    const review = state.review;
    if (review.socket) {
      review.socket.onclose = null;
      review.socket.close();
    }
    if (review.objectUrl) URL.revokeObjectURL(review.objectUrl);
    if (review.panel) review.panel.hidden = true;
    state.review = {
      open: false, versionId: null, dataset: null, panel: null, images: [],
      index: -1, image: null, objectUrl: null, socket: null, snapshot: null,
      mode: "positive", status: null, listEl: null, save: null, canvas: null,
      imageEl: null, labelEl: null,
    };
  }

  function showDatasetList(opts = {}) {
    if (window.location.hash !== "#/datasets") {
      window.history.replaceState(null, "", "#/datasets");
    }
    closeDatasetGallery();
    if (!opts.keepCards) loadDatasets();
  }

  function showDatasetsForHash() {
    const versionId = galleryVersionFromHash();
    if (versionId) openDatasetGallery(versionId);
    else {
      closeDatasetGallery();
      loadDatasets();
    }
  }

  function closeDatasetGallery() {
    const gallery = state.gallery;
    gallery.objectUrls.forEach((url) => URL.revokeObjectURL(url));
    gallery.objectUrls = [];
    state.gallery.images.forEach((image) => { delete image._url; });
    Object.assign(gallery, {
      open: false, versionId: null, dataset: null, images: [], visible: [], index: -1,
      total: 0, offset: 0, hasMore: false,
    });
    const section = $("dataset-gallery");
    if (section) section.hidden = true;
    const list = $("datasets-list");
    if (list) list.hidden = false;
  }

  function galleryClassName(dataset, categoryId) {
    const categories = (dataset && dataset.categories) || [];
    const match = categories.find(
      (category) => String(category.category_id) === String(categoryId)
    );
    return match ? match.name : `class ${categoryId}`;
  }

  function galleryStatus(message, kind) {
    setStatus("gallery-status", message, kind);
  }

  async function openDatasetGallery(versionId, knownDataset = null) {
    if (state.gallery.open && state.gallery.versionId === String(versionId)) return;
    closeDatasetGallery();
    const section = $("dataset-gallery");
    const list = $("datasets-list");
    if (!section) return;
    if (list) list.hidden = true;
    section.hidden = false;
    Object.assign(state.gallery, {
      open: true,
      versionId: String(versionId),
      dataset: knownDataset,
      index: -1,
      filters: { categoryId: "all", split: "all", unannotated: false, lowConfidence: false },
      toggles: { boxes: true, masks: true, labels: true },
    });
    if ($("gallery-title")) $("gallery-title").textContent = "Dataset gallery";
    if ($("gallery-subtitle")) $("gallery-subtitle").textContent = `Version ${String(versionId).slice(0, 8)}`;
    if ($("gallery-filter-class")) $("gallery-filter-class").value = "all";
    if ($("gallery-filter-split")) $("gallery-filter-split").value = "all";
    if ($("gallery-filter-unannotated")) $("gallery-filter-unannotated").checked = false;
    if ($("gallery-filter-lowconfidence")) $("gallery-filter-lowconfidence").checked = false;
    if ($("gallery-toggle-boxes")) $("gallery-toggle-boxes").checked = true;
    if ($("gallery-toggle-masks")) $("gallery-toggle-masks").checked = true;
    if ($("gallery-toggle-labels")) $("gallery-toggle-labels").checked = true;
    if ($("gallery-counter")) $("gallery-counter").textContent = "image 0 of 0";
    galleryStatus("Loading dataset images…", "busy");
    try {
      if (!state.gallery.dataset) {
        const datasets = await request("/v1/datasets", { headers: tenantHeaders(false) });
        state.gallery.dataset = (datasets || []).find(
          (item) => String(item.version_id) === String(versionId)
        ) || null;
      }
      const dataset = state.gallery.dataset;
      if (!dataset) throw new Error("Dataset version not found");
      if ($("gallery-title")) $("gallery-title").textContent = dataset.name || "Dataset gallery";
      if ($("gallery-subtitle")) {
        $("gallery-subtitle").textContent = `${dataset.source_asset || "Source asset"} · ${formatDate(dataset.created_at)}`;
      }
      renderGalleryClassCounts();
      const classSelect = $("gallery-filter-class");
      if (classSelect) {
        classSelect.innerHTML = "";
        const all = node("option", null, "All classes");
        all.value = "all";
        classSelect.appendChild(all);
        const counts = dataset.class_counts || dataset.categories || [];
        counts.forEach((item) => {
          const option = node("option", null, `${item.name}${item.count != null ? ` (${item.count})` : ""}`);
          option.value = String(item.category_id);
          classSelect.appendChild(option);
        });
      }
      if (window.location.hash !== `#/datasets/${versionId}`) {
        window.history.replaceState(null, "", `#/datasets/${versionId}`);
      }
      await loadGalleryImages(versionId, true);
    } catch (err) {
      galleryStatus(cleanMessage(err.message || err), "error");
    }
  }

  function renderGalleryClassCounts() {
    const row = $("gallery-class-counts");
    if (!row) return;
    row.replaceChildren();
    const dataset = state.gallery.dataset;
    const counts = (dataset && dataset.class_counts) || [];
    if (!counts.length) {
      row.appendChild(node("span", "gallery-empty-note", "No annotations in this version."));
      return;
    }
    counts.forEach((item) => {
      const chip = node("span", "category-chip", `${item.name} · ${item.count}`);
      chip.dataset.classCount = String(item.category_id);
      chip.style.borderColor = item.color || classColor(item.category_id);
      row.appendChild(chip);
    });
  }

  async function galleryImageUrl(image) {
    if (image._url) return image._url;
    const path = `/v1/datasets/${encodeURIComponent(state.gallery.versionId)}/images/${encodeURIComponent(image.image_id)}/file`;
    const response = await fetch(path, { headers: tenantHeaders(false) });
    if (!response.ok) throw new Error(`Image request failed (${response.status})`);
    const url = URL.createObjectURL(await response.blob());
    image._url = url;
    state.gallery.objectUrls.push(url);
    return url;
  }

  function galleryVisibleImages() {
    const filters = state.gallery.filters;
    return state.gallery.images.filter((image) => {
      const annotations = image.annotations || [];
      if (filters.categoryId !== "all" && !annotations.some(
        (item) => String(item.category_id) === String(filters.categoryId)
      )) return false;
      if (filters.split !== "all" && image.split !== filters.split) return false;
      if (filters.unannotated && annotations.length) return false;
      if (filters.lowConfidence && !annotations.some(
        (item) => item.score != null && Number(item.score) < 0.5
      )) return false;
      return true;
    });
  }

  function applyGalleryFilters() {
    const gallery = state.gallery;
    gallery.visible = galleryVisibleImages();
    if (gallery.index >= gallery.visible.length) {
      gallery.index = gallery.visible.length ? gallery.visible.length - 1 : -1;
    }
  }

  async function loadGalleryImages(versionId, reset) {
    const gallery = state.gallery;
    if (reset) {
      gallery.images = [];
      gallery.offset = 0;
      gallery.total = 0;
      gallery.hasMore = true;
      gallery.index = -1;
    }
    if (!gallery.hasMore) return;
    galleryStatus("Loading images…", "busy");
    try {
      const data = await request(
        `/v1/datasets/${encodeURIComponent(versionId)}/images?limit=200&offset=${gallery.offset}`,
        { headers: tenantHeaders(false) }
      );
      const page = (data && data.images) || [];
      gallery.images = gallery.images.concat(page);
      gallery.total = Number(data && data.total) || gallery.images.length;
      gallery.offset = gallery.images.length;
      gallery.hasMore = !!(data && data.has_more);
      await Promise.all(gallery.images.map((image) => galleryImageUrl(image).catch(() => null)));
      applyGalleryFilters();
      if (gallery.visible.length && gallery.index < 0) gallery.index = 0;
      renderGallery();
      galleryStatus("", null);
    } catch (err) {
      galleryStatus(cleanMessage(err.message || err), "error");
    }
  }

  async function selectGalleryImage(index) {
    const gallery = state.gallery;
    if (!gallery.open || index < 0 || index >= gallery.visible.length) return;
    gallery.index = index;
    const image = gallery.visible[index];
    renderGallery();
    try {
      const url = await galleryImageUrl(image);
      if (!gallery.open || gallery.visible[index] !== image) return;
      const img = $("gallery-image");
      if (!img) return;
      await new Promise((resolve, reject) => {
        img.onload = resolve;
        img.onerror = () => reject(new Error("Could not display this image"));
        img.src = url;
      });
      if (!gallery.open || gallery.visible[index] !== image) return;
      drawGalleryOverlay();
    } catch (err) {
      galleryStatus(cleanMessage(err.message || err), "error");
    }
  }

  function renderGallery() {
    const gallery = state.gallery;
    if (!gallery.open) return;
    const visible = gallery.visible;
    const image = visible[gallery.index] || null;
    if ($("gallery-counter")) {
      $("gallery-counter").textContent = `image ${image ? gallery.index + 1 : 0} of ${visible.length}`;
    }
    const filmstrip = $("gallery-filmstrip");
    if (filmstrip) {
      filmstrip.replaceChildren();
      visible.forEach((item, index) => {
        const button = node("button", "gallery-thumb");
        button.type = "button";
        button.dataset.galleryIndex = String(index);
        button.dataset.annotated = String((item.annotations || []).length > 0);
        button.setAttribute("aria-label", `Open ${cleanMessage(item.file_name)}`);
        if (index === gallery.index) button.classList.add("is-active");
        const thumb = node("img");
        thumb.alt = "";
        thumb.loading = "lazy";
        if (item._url) thumb.src = item._url;
        button.appendChild(thumb);
        button.addEventListener("click", () => selectGalleryImage(index));
        filmstrip.appendChild(button);
      });
    }
    const loadMore = $("gallery-load-more");
    if (loadMore) {
      loadMore.hidden = !gallery.hasMore;
      loadMore.textContent = `Load more images (${Math.max(0, gallery.total - gallery.images.length)} remaining)`;
    }
    if ($("gallery-prev")) $("gallery-prev").disabled = gallery.index <= 0;
    if ($("gallery-next")) $("gallery-next").disabled = gallery.index < 0 || gallery.index >= visible.length - 1;
    if ($("gallery-caption")) {
      $("gallery-caption").textContent = image
        ? `${cleanMessage(image.file_name)} · ${image.split || "train"}`
        : "No images match these filters.";
    }
    const details = $("gallery-details");
    if (details) {
      details.replaceChildren();
      const rows = image ? [
        ["File", cleanMessage(image.file_name)],
        ["Size", formatBytes(image.file_size)],
        ["Split", image.split || "train"],
        ["Dimensions", `${image.width}×${image.height}`],
        ["Annotations", String((image.annotations || []).length)],
      ] : [["File", "—"]];
      rows.forEach(([term, value]) => {
        details.appendChild(node("dt", null, term));
        details.appendChild(node("dd", null, value));
      });
    }
    const annotations = $("gallery-annotations");
    if (annotations) {
      annotations.replaceChildren();
      const items = (image && image.annotations) || [];
      if (!items.length) {
        annotations.appendChild(node("p", "gallery-empty-note", "No annotations for this image."));
      }
      items.forEach((annotation) => {
        const row = node("div", "gallery-annotation");
        row.dataset.annotationId = String(annotation.annotation_id);
        const swatch = node("span", "gallery-annotation-swatch");
        swatch.style.background = classColor(annotation.category_id);
        const copy = node("div");
        const score = annotation.score == null ? "" : ` · ${Number(annotation.score).toFixed(2)}`;
        copy.append(
          node("strong", null, `${cleanMessage(galleryClassName(gallery.dataset, annotation.category_id))}${score}`),
          node("span", "gallery-annotation-meta", `${Math.round(annotationArea(annotation, image))} px²`)
        );
        row.append(swatch, copy);
        annotations.appendChild(row);
      });
    }
    const refine = $("gallery-refine");
    if (refine) refine.disabled = !image;
    drawGalleryOverlay();
  }

  function drawGalleryOverlay() {
    const gallery = state.gallery;
    const canvas = $("gallery-overlay");
    const img = $("gallery-image");
    if (!canvas || !img || !img.complete || !img.naturalWidth) return;
    const width = img.naturalWidth;
    const height = img.naturalHeight;
    if (canvas.width !== width) canvas.width = width;
    if (canvas.height !== height) canvas.height = height;
    canvas.dataset.boxes = String(gallery.toggles.boxes);
    canvas.dataset.masks = String(gallery.toggles.masks);
    canvas.dataset.labels = String(gallery.toggles.labels);
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, width, height);
    const image = gallery.visible[gallery.index];
    const annotations = (image && image.annotations) || [];
    annotations.forEach((annotation) => {
      const color = classColor(annotation.category_id);
      if (gallery.toggles.masks) {
        (annotation.segmentation || []).forEach((coords) => {
          if (!coords || coords.length < 6) return;
          ctx.beginPath();
          ctx.moveTo(coords[0], coords[1]);
          for (let i = 2; i + 1 < coords.length; i += 2) ctx.lineTo(coords[i], coords[i + 1]);
          ctx.closePath();
          ctx.fillStyle = `${color}2e`;
          ctx.fill();
          ctx.strokeStyle = color;
          ctx.lineWidth = Math.max(1, width / 500);
          ctx.stroke();
        });
      }
      if (gallery.toggles.boxes) {
        const [x, y, boxWidth, boxHeight] = annotation.bbox || [];
        if (x != null && y != null && boxWidth != null && boxHeight != null) {
          ctx.strokeStyle = color;
          ctx.lineWidth = Math.max(2, width / 300);
          ctx.strokeRect(x, y, boxWidth, boxHeight);
        }
      }
      if (gallery.toggles.labels) {
        const [x, y] = annotation.bbox || [];
        const score = annotation.score == null ? "" : ` ${Number(annotation.score).toFixed(2)}`;
        const text = `${galleryClassName(gallery.dataset, annotation.category_id)}${score}`;
        const size = Math.max(11, Math.round(width / 44));
        ctx.font = `${size}px sans-serif`;
        const textWidth = ctx.measureText(text).width;
        const labelY = Math.max(size, (Number(y) || 0) - 4);
        ctx.fillStyle = color;
        ctx.fillRect(Number(x) || 0, labelY - size, textWidth + 8, size + 3);
        ctx.fillStyle = "#081015";
        ctx.fillText(text, (Number(x) || 0) + 4, labelY);
      }
    });
  }

  async function refineGalleryImage() {
    const gallery = state.gallery;
    const image = gallery.visible[gallery.index];
    if (!gallery.open || !image) return;
    const fullIndex = gallery.images.indexOf(image);
    let dataset = gallery.dataset;
    let card = document.querySelector(`[data-version-id="${CSS.escape(String(gallery.versionId))}"]`);
    if (!dataset || !card) {
      await loadDatasets();
      dataset = (state.datasets || []).find((item) => String(item.version_id) === String(gallery.versionId)) || dataset;
      card = document.querySelector(`[data-version-id="${CSS.escape(String(gallery.versionId))}"]`);
    }
    showDatasetList({ keepCards: true });
    if (dataset && card) openDatasetReview(dataset, card, Math.max(0, fullIndex));
  }

  if ($("dataset-gallery-back")) {
    $("dataset-gallery-back").addEventListener("click", (event) => {
      event.preventDefault();
      showDatasetList();
    });
  }
  if ($("gallery-prev")) $("gallery-prev").addEventListener("click", () => selectGalleryImage(state.gallery.index - 1));
  if ($("gallery-next")) $("gallery-next").addEventListener("click", () => selectGalleryImage(state.gallery.index + 1));
  if ($("gallery-image")) $("gallery-image").addEventListener("load", drawGalleryOverlay);
  if ($("gallery-filter-class")) {
    $("gallery-filter-class").addEventListener("change", (event) => {
      state.gallery.filters.categoryId = event.target.value;
      state.gallery.index = 0;
      applyGalleryFilters();
      renderGallery();
    });
  }
  if ($("gallery-filter-split")) {
    $("gallery-filter-split").addEventListener("change", (event) => {
      state.gallery.filters.split = event.target.value;
      state.gallery.index = 0;
      applyGalleryFilters();
      renderGallery();
    });
  }
  if ($("gallery-filter-unannotated")) {
    $("gallery-filter-unannotated").addEventListener("change", (event) => {
      state.gallery.filters.unannotated = event.target.checked;
      state.gallery.index = 0;
      applyGalleryFilters();
      renderGallery();
    });
  }
  if ($("gallery-filter-lowconfidence")) {
    $("gallery-filter-lowconfidence").addEventListener("change", (event) => {
      state.gallery.filters.lowConfidence = event.target.checked;
      state.gallery.index = 0;
      applyGalleryFilters();
      renderGallery();
    });
  }
  if ($("gallery-reset-filters")) {
    $("gallery-reset-filters").addEventListener("click", () => {
      Object.assign(state.gallery.filters, { categoryId: "all", split: "all", unannotated: false, lowConfidence: false });
      if ($("gallery-filter-class")) $("gallery-filter-class").value = "all";
      if ($("gallery-filter-split")) $("gallery-filter-split").value = "all";
      if ($("gallery-filter-unannotated")) $("gallery-filter-unannotated").checked = false;
      if ($("gallery-filter-lowconfidence")) $("gallery-filter-lowconfidence").checked = false;
      state.gallery.index = 0;
      applyGalleryFilters();
      renderGallery();
    });
  }
  [["gallery-toggle-boxes", "boxes"], ["gallery-toggle-masks", "masks"], ["gallery-toggle-labels", "labels"]].forEach(([id, key]) => {
    const input = $(id);
    if (!input) return;
    input.addEventListener("change", (event) => {
      state.gallery.toggles[key] = event.target.checked;
      drawGalleryOverlay();
    });
  });
  if ($("gallery-load-more")) {
    $("gallery-load-more").addEventListener("click", () => {
      loadGalleryImages(state.gallery.versionId, false);
    });
  }
  if ($("gallery-refine")) $("gallery-refine").addEventListener("click", refineGalleryImage);
  document.addEventListener("keydown", (event) => {
    if (!state.gallery.open || $("view-datasets").hidden) return;
    if (["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement && document.activeElement.tagName)) return;
    if (event.key === "ArrowLeft") {
      event.preventDefault();
      selectGalleryImage(state.gallery.index - 1);
    } else if (event.key === "ArrowRight") {
      event.preventDefault();
      selectGalleryImage(state.gallery.index + 1);
    }
  });

  async function openDatasetReview(dataset, card, initialIndex = 0) {
    const panel = card.querySelector('[data-review-panel="true"]');
    if (!panel) return;
    if (state.review.open && state.review.versionId === String(dataset.version_id)) {
      closeReview();
      return;
    }
    closeReview();
    panel.hidden = false;
    panel.replaceChildren();
    state.review.open = true;
    state.review.versionId = String(dataset.version_id);
    state.review.dataset = dataset;
    state.review.panel = panel;

    const head = node("div", "review-head");
    const copy = node("div");
    copy.append(
      node("h3", "card-title", "Review labels"),
      node(
        "p",
        "card-desc",
        "Pick an image, click the object you want to correct, then accept the mask. The correction becomes a new annotation revision and a fresh dataset version."
      )
    );
    const close = node("button", "btn btn-ghost btn-small", "Close");
    close.type = "button";
    close.addEventListener("click", closeReview);
    head.append(copy, close);

    const status = node("div", "status");
    status.dataset.reviewStatus = "true";
    state.review.status = status;
    panel.append(head, status);

    const body = node("div", "review-body");
    const listCol = node("div", "review-list-col");
    const list = node("div", "review-image-list");
    list.dataset.reviewList = "true";
    state.review.listEl = list;
    listCol.append(node("p", "review-list-title", "Images in this version"), list);

    const workspace = node("div", "review-workspace");
    const stage = node("div", "review-stage");
    const imageEl = node("img", "review-image");
    imageEl.alt = "Selected dataset image";
    const canvas = node("canvas", "review-canvas");
    canvas.setAttribute("aria-label", "Label mask overlay");
    stage.append(imageEl, canvas);
    const labelEl = node("p", "review-image-caption", "Select an image to begin.");
    state.review.imageEl = imageEl;
    state.review.canvas = canvas;
    state.review.labelEl = labelEl;
    canvas.addEventListener("click", handleReviewCanvasClick);

    const controls = node("div", "review-controls");
    const modes = node("div", "review-modes");
    const positive = node("button", "btn btn-small btn-secondary is-active", "Positive point");
    const negative = node("button", "btn btn-small btn-ghost", "Negative point");
    positive.type = "button";
    negative.type = "button";
    const setMode = (mode, active, other) => {
      state.review.mode = mode;
      active.classList.add("is-active");
      active.classList.remove("btn-ghost");
      active.classList.add("btn-secondary");
      other.classList.remove("is-active", "btn-secondary");
      other.classList.add("btn-ghost");
    };
    positive.addEventListener("click", () => setMode("positive", positive, negative));
    negative.addEventListener("click", () => setMode("negative", negative, positive));
    modes.append(positive, negative);

    const save = node("button", "btn btn-primary", "Save corrected mask");
    save.type = "button";
    save.disabled = true;
    state.review.save = save;
    save.addEventListener("click", acceptReview);
    const hint = node(
      "p",
      "review-hint",
      "A click ranks the SAM 3 masks for this image's category. Click on an existing object to correct it, or elsewhere to add a missed one. Negative clicks push a mask away; if no preview appears, try another point."
    );
    controls.append(modes, save, hint);
    workspace.append(stage, labelEl, controls);
    body.append(listCol, workspace);
    panel.appendChild(body);

    reviewStatus("Loading images…", "busy");
    try {
      const data = await request(
        `/v1/datasets/${encodeURIComponent(dataset.version_id)}/images`,
        { headers: tenantHeaders(false) }
      );
      const images = (data && data.images) || [];
      if (!images.length) {
        reviewStatus("This dataset has no images to review.", "error");
        return;
      }
      state.review.images = images;
      list.replaceChildren();
      images.forEach((image, index) => {
        const button = node("button", "review-image-item");
        button.type = "button";
        const name = String(image.file_name || "image").split("/").pop();
        button.append(
          node("span", "review-image-name", cleanMessage(name)),
          node(
            "span",
            "review-image-count",
            `${image.annotations.length} label${image.annotations.length === 1 ? "" : "s"}`
          )
        );
        button.addEventListener("click", () => selectReviewImage(index));
        list.appendChild(button);
      });
      await selectReviewImage(initialIndex);
    } catch (err) {
      reviewStatus(cleanMessage(err.message || err), "error");
    }
  }

  async function selectReviewImage(index) {
    const review = state.review;
    if (!review.open || index < 0 || index >= review.images.length) return;
    review.index = index;
    const image = review.images[index];
    review.image = image;
    if (review.socket) {
      review.socket.onclose = null;
      review.socket.close();
      review.socket = null;
    }
    review.snapshot = null;
    if (review.save) review.save.disabled = true;
    if (review.objectUrl) URL.revokeObjectURL(review.objectUrl);
    review.objectUrl = null;
    Array.from(review.listEl ? review.listEl.children : []).forEach((item, itemIndex) => {
      item.classList.toggle("is-active", itemIndex === index);
    });
    const name = String(image.file_name || "image").split("/").pop();
    if (review.labelEl) {
      review.labelEl.textContent = `${cleanMessage(name)} · ${image.width}×${image.height} · ${image.annotations.length} existing label${image.annotations.length === 1 ? "" : "s"}`;
    }
    reviewStatus("Loading image…", "busy");
    try {
      const path = `/v1/datasets/${encodeURIComponent(review.dataset.version_id)}/images/${encodeURIComponent(image.image_id)}/file`;
      const response = await fetch(path, { headers: tenantHeaders(false) });
      if (!response.ok) throw new Error(`Image request failed (${response.status})`);
      const blob = await response.blob();
      if (!review.open || review.index !== index) return;
      review.objectUrl = URL.createObjectURL(blob);
      await new Promise((resolve, reject) => {
        review.imageEl.onload = resolve;
        review.imageEl.onerror = () => reject(new Error("Could not display this image"));
        review.imageEl.src = review.objectUrl;
      });
      if (!review.open || review.index !== index) return;
      drawReviewOverlay();
      openReviewSocket();
    } catch (err) {
      reviewStatus(cleanMessage(err.message || err), "error");
    }
  }

  function openReviewSocket() {
    const review = state.review;
    if (!review.open || !review.image) return;
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const path = `/v1/refinement/${encodeURIComponent(review.dataset.version_id)}/${encodeURIComponent(review.image.image_id)}?tenant_id=${encodeURIComponent(DEFAULT_TENANT)}`;
    const socket = new WebSocket(`${protocol}//${window.location.host}${path}`);
    review.socket = socket;
    reviewStatus("Connecting to the refinement service…", "busy");
    socket.addEventListener("open", () => {
      if (review.socket !== socket) return;
      reviewStatus("Click the object to preview a corrected mask.", "busy");
    });
    socket.addEventListener("message", (event) => {
      if (review.socket !== socket) return;
      let message = null;
      try { message = JSON.parse(event.data); } catch { return; }
      if (message.type === "error") {
        reviewStatus(cleanMessage(message.detail || "Refinement failed"), "error");
        return;
      }
      if (message.type !== "snapshot") return;
      review.snapshot = message;
      const hasMask = !!(message.predictions && message.predictions.length);
      if (review.save) review.save.disabled = !hasMask;
      drawReviewOverlay();
      if (hasMask) {
        reviewStatus("Mask preview ready. Accept it or add another point.", "ok");
      } else if (!message.revision) {
        reviewStatus("Click the object to preview a corrected mask.", "busy");
      } else {
        reviewStatus("No mask found at that point. Try another point.", "error");
      }
    });
    socket.addEventListener("close", () => {
      if (review.socket !== socket) return;
      review.socket = null;
      if (review.open) {
        reviewStatus("Refinement connection closed. Select the image again to reconnect.", "error");
      }
    });
  }

  function drawReviewOverlay() {
    const review = state.review;
    const canvas = review.canvas;
    const imageEl = review.imageEl;
    if (!canvas || !imageEl || !imageEl.complete || !imageEl.naturalWidth) return;
    const width = imageEl.naturalWidth;
    const height = imageEl.naturalHeight;
    if (canvas.width !== width) canvas.width = width;
    if (canvas.height !== height) canvas.height = height;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, width, height);

    const drawPolygons = (segmentation, fill, stroke) => {
      (segmentation || []).forEach((coords) => {
        if (!coords || coords.length < 6) return;
        ctx.beginPath();
        ctx.moveTo(coords[0], coords[1]);
        for (let i = 2; i + 1 < coords.length; i += 2) ctx.lineTo(coords[i], coords[i + 1]);
        ctx.closePath();
        ctx.fillStyle = fill;
        ctx.fill();
        ctx.strokeStyle = stroke;
        ctx.lineWidth = Math.max(1, width / 500);
        ctx.stroke();
      });
    };

    (review.image && review.image.annotations ? review.image.annotations : []).forEach((annotation) => {
      drawPolygons(annotation.segmentation, "rgba(24,182,196,0.18)", "rgba(95,211,222,0.85)");
    });

    const prediction = review.snapshot && review.snapshot.predictions
      ? review.snapshot.predictions[0]
      : null;
    if (prediction && prediction.mask && prediction.mask.length) {
      const maskCanvas = document.createElement("canvas");
      maskCanvas.width = prediction.mask[0].length;
      maskCanvas.height = prediction.mask.length;
      const maskCtx = maskCanvas.getContext("2d");
      const pixels = maskCtx.createImageData(maskCanvas.width, maskCanvas.height);
      for (let y = 0; y < maskCanvas.height; y += 1) {
        for (let x = 0; x < maskCanvas.width; x += 1) {
          if (!prediction.mask[y][x]) continue;
          const offset = (y * maskCanvas.width + x) * 4;
          pixels.data[offset] = 63;
          pixels.data[offset + 1] = 185;
          pixels.data[offset + 2] = 80;
          pixels.data[offset + 3] = 140;
        }
      }
      maskCtx.putImageData(pixels, 0, 0);
      ctx.drawImage(maskCanvas, 0, 0, width, height);
      if (prediction.box) {
        const [x1, y1, x2, y2] = prediction.box;
        ctx.strokeStyle = "rgba(63,185,80,0.95)";
        ctx.lineWidth = Math.max(2, width / 300);
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      }
    }
  }

  function handleReviewCanvasClick(event) {
    const review = state.review;
    if (!review.open || !review.snapshot || !review.socket || review.socket.readyState !== WebSocket.OPEN) return;
    const rect = review.canvas.getBoundingClientRect();
    if (!rect.width || !rect.height) return;
    const x = (event.clientX - rect.left) * (review.canvas.width / rect.width);
    const y = (event.clientY - rect.top) * (review.canvas.height / rect.height);
    review.socket.send(JSON.stringify({
      type: "click",
      expected_revision: review.snapshot.revision,
      x,
      y,
      label: review.mode,
    }));
    if (review.save) review.save.disabled = true;
    reviewStatus("Finding the best mask for that point…", "busy");
  }

  // Clicking on an existing object corrects that annotation; clicking anywhere
  // else adds a new one. (Replacing the first annotation blindly overwrote
  // correct labels when the user was adding a missed object.)
  function annotationUnderClick(annotations, points) {
    const positive = (points || []).find((point) => point.label === "positive");
    if (!positive || !annotations || !annotations.length) return null;
    let best = null;
    let bestArea = Infinity;
    for (const annotation of annotations) {
      const [x, y, width, height] = annotation.bbox || [];
      if (positive.x >= x && positive.x <= x + width && positive.y >= y && positive.y <= y + height) {
        const area = width * height;
        if (area < bestArea) {
          best = annotation;
          bestArea = area;
        }
      }
    }
    return best;
  }

  async function acceptReview() {
    const review = state.review;
    const snapshot = review.snapshot;
    if (!review.open || !snapshot || !snapshot.predictions || !snapshot.predictions.length) return;
    const dataset = review.dataset;
    const image = review.image;
    const save = review.save;
    setBusy(save, true, "Saving…");
    reviewStatus("Saving the annotation revision…", "busy");
    try {
      const existing = annotationUnderClick(image.annotations, snapshot.points);
      const accepted = await request(
        `/v1/refinements/${encodeURIComponent(snapshot.session_id)}/accept`,
        {
          method: "POST",
          headers: tenantHeaders(true),
          body: JSON.stringify({
            annotation_id: existing ? existing.annotation_id : null,
            prediction_index: 0,
            refinement_revision: snapshot.revision,
          }),
        }
      );
      reviewStatus(`Revision ${accepted.revision} saved. Building an updated dataset…`, "busy");
      const rebuilt = await request(
        `/v1/datasets/${encodeURIComponent(dataset.version_id)}/rebuild`,
        {
          method: "POST",
          headers: tenantHeaders(true),
          body: JSON.stringify({
            name: `${dataset.name || "Dataset"} — refined`,
            idempotency_key: `ui-refine-${dataset.version_id}-${accepted.revision}-${snapshot.session_id}`,
          }),
        }
      );
      closeReview();
      await loadDatasets();
      setStatus(
        "datasets-status",
        `Correction saved as revision ${accepted.revision}. Updated dataset version is ready to export.`,
        "ok"
      );
    } catch (err) {
      reviewStatus(cleanMessage(err.message || err), "error");
    } finally {
      setBusy(save, false);
    }
  }

  async function loadDatasets(opts = {}) {
    if (!opts.keepGallery) closeReview();
    const list = $("datasets-list");
    const meta = $("datasets-meta");
    if (!list) return;
    if (meta) meta.textContent = "Loading dataset versions…";
    setStatus("datasets-status", "");
    setBusyRegion("datasets-list", true);
    list.innerHTML = "";
    list.appendChild(statePanel("loading", "Loading dataset versions…"));
    try {
      const datasets = await request("/v1/datasets", { headers: tenantHeaders(false) });
      state.datasets = datasets || [];
      list.innerHTML = "";
      setBusyRegion("datasets-list", false);
      if (!state.datasets.length) {
        list.appendChild(
          statePanel(
            "empty",
            "No dataset versions yet.",
            "Run the Agent on an asset to build a dataset automatically.",
            "Open Agent",
            () => showView("agent")
          )
        );
      } else {
        state.datasets.forEach((dataset) => list.appendChild(renderDatasetCard(dataset)));
        loadDatasetMosaics(state.datasets);
      }
      if (meta) meta.textContent = `${state.datasets.length} dataset version${state.datasets.length === 1 ? "" : "s"}`;
      if (state.gallery.open) {
        const current = state.datasets.find(
          (item) => String(item.version_id) === String(state.gallery.versionId)
        );
        if (current) {
          state.gallery.dataset = current;
          renderGalleryClassCounts();
        }
      }
    } catch (err) {
      const message = cleanMessage(err.message || err);
      setStatus("datasets-status", message, "error");
      setBusyRegion("datasets-list", false);
      list.innerHTML = "";
      list.appendChild(
        statePanel("error", "Could not load dataset versions.", message, "Try again", loadDatasets)
      );
      if (meta) meta.textContent = "Could not load datasets";
    }
  }

  async function loadDatasetMosaics(datasets) {
    for (const dataset of datasets.slice(0, 12)) {
      const card = document.querySelector(
        `[data-version-id="${CSS.escape(String(dataset.version_id))}"]`
      );
      const mosaic = card && card.querySelector('[data-dataset-mosaic="true"]');
      if (!mosaic) continue;
      try {
        const data = await request(
          `/v1/datasets/${encodeURIComponent(dataset.version_id)}/images?limit=4&offset=0`,
          { headers: tenantHeaders(false) }
        );
        const images = (data && data.images) || [];
        mosaic.replaceChildren();
        for (const image of images) {
          const path = `/v1/datasets/${encodeURIComponent(dataset.version_id)}/images/${encodeURIComponent(image.image_id)}/file`;
          const response = await fetch(path, { headers: tenantHeaders(false) });
          if (!response.ok) continue;
          const url = URL.createObjectURL(await response.blob());
          state.datasetThumbUrls.push(url);
          const thumb = node("img");
          thumb.alt = cleanMessage(image.file_name);
          thumb.loading = "lazy";
          thumb.src = url;
          mosaic.appendChild(thumb);
        }
      } catch {
        /* A card still works without its thumbnail mosaic. */
      }
    }
  }

  function renderDatasetCard(dataset) {
    const card = node("article", "card dataset-card");
    card.dataset.versionId = String(dataset.version_id);
    card.dataset.galleryOpen = "false";
    const head = node("div", "dataset-head");
    const title = node("div");
    const openButton = node("button", "dataset-title-button");
    openButton.type = "button";
    openButton.dataset.openGallery = "true";
    openButton.appendChild(node("h2", "card-title", dataset.name || "Untitled dataset"));
    openButton.addEventListener("click", (event) => {
      event.stopPropagation();
      openDatasetGallery(dataset.version_id, dataset);
    });
    title.append(
      openButton,
      node(
        "p",
        "dataset-meta",
        `${dataset.source_asset || "Source asset"} · ${dataset.image_count} image${dataset.image_count === 1 ? "" : "s"} · ${dataset.annotation_count} annotation${dataset.annotation_count === 1 ? "" : "s"} · ${formatDate(dataset.created_at)}`
      )
    );
    const actions = node("div", "actions");
    const reviewButton = node("button", "btn btn-secondary btn-small", "Review labels");
    reviewButton.type = "button";
    reviewButton.addEventListener("click", (event) => {
      event.stopPropagation();
      openDatasetReview(dataset, card);
    });
    actions.appendChild(reviewButton);
    const format = node("select", "dataset-format");
    format.dataset.exportFormat = "true";
    format.setAttribute("aria-label", "Export format");
    [
      ["coco", "COCO"],
      ["yolo-detection", "YOLO detection"],
      ["yolo-segmentation", "YOLO segmentation"],
    ].forEach(([value, text]) => {
      const option = node("option", null, text);
      option.value = value;
      format.appendChild(option);
    });
    actions.appendChild(format);
    const exportButton = node("button", "btn btn-secondary btn-small", "Export");
    exportButton.type = "button";
    exportButton.disabled = !dataset.annotation_count;
    exportButton.addEventListener("click", (event) => {
      event.stopPropagation();
      exportDataset(dataset, exportButton, card, format.value);
    });
    actions.appendChild(exportButton);
    head.append(title, actions);
    card.appendChild(head);

    const mosaic = node("div", "dataset-mosaic");
    mosaic.dataset.datasetMosaic = "true";
    card.appendChild(mosaic);

    const counts = (dataset.class_counts || []).length
      ? dataset.class_counts
      : (dataset.categories || []).map((category) => ({
          category_id: category.category_id,
          name: category.name,
          count: 0,
          color: classColor(category.category_id),
        }));
    if (counts.length) {
      const categories = node("div", "category-row");
      counts.forEach((item) => {
        const chip = node("span", "category-chip", `${item.name} · ${item.count}`);
        chip.dataset.classCount = String(item.category_id);
        chip.style.borderColor = item.color || classColor(item.category_id);
        categories.appendChild(chip);
      });
      card.appendChild(categories);
    }
    const status = node("div", "status");
    status.dataset.datasetStatus = "true";
    status.textContent = dataset.status || "ready";
    status.dataset.kind = "ok";
    card.appendChild(status);
    const reviewPanel = node("div", "review-panel");
    reviewPanel.dataset.reviewPanel = "true";
    reviewPanel.hidden = true;
    card.appendChild(reviewPanel);
    card.addEventListener("click", (event) => {
      if (event.target.closest("button, select, a, input, .review-panel")) return;
      openDatasetGallery(dataset.version_id, dataset);
    });
    return card;
  }

  function exportFormatLabel(format) {
    return {
      coco: "COCO",
      "yolo-detection": "YOLO detection",
      "yolo-segmentation": "YOLO segmentation",
    }[format] || "COCO";
  }

  async function exportDataset(dataset, button, card, format = "coco") {
    const formatLabel = exportFormatLabel(format);
    setBusy(button, true, "Exporting…");
    const status = card.querySelector('[data-dataset-status="true"]');
    if (status) {
      status.textContent = `Queuing ${formatLabel} export…`;
      status.dataset.kind = "busy";
    }
    try {
      let job = await request("/v1/dataset-exports", {
        method: "POST",
        headers: tenantHeaders(true),
        body: JSON.stringify({
          version_id: dataset.version_id,
          format,
          idempotency_key: `ui-${format}-${dataset.version_id}-${Date.now()}`,
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
          status.textContent = `${formatLabel} export ready. Download starting…`;
          status.dataset.kind = "ok";
        }
        await downloadWithTenant(
          `/v1/dataset-exports/${encodeURIComponent(job.job_id)}/download`,
          `${(dataset.name || "dataset").replace(/[^a-z0-9._-]+/gi, "-")}-${format}.zip`,
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
    setBusyRegion("jobs-list", true);
    list.innerHTML = "";
    list.appendChild(statePanel("loading", "Loading background jobs…"));
    try {
      const data = await request("/v1/jobs", { headers: tenantHeaders(false) });
      const jobs = (data && data.jobs) || [];
      list.innerHTML = "";
      setBusyRegion("jobs-list", false);
      if (!jobs.length) {
        list.appendChild(
          statePanel(
            "empty",
            "No background jobs yet.",
            "Uploading an asset queues a search-indexing job. Labeling and exports appear here too.",
            "Open Assets",
            () => showView("assets")
          )
        );
      } else {
        jobs.slice().reverse().forEach((job) => list.appendChild(renderJobCard(job)));
      }
      if (meta) meta.textContent = `${jobs.length} job${jobs.length === 1 ? "" : "s"}`;
    } catch (err) {
      const message = cleanMessage(err.message || err);
      setStatus("jobs-status", message, "error");
      setBusyRegion("jobs-list", false);
      list.innerHTML = "";
      list.appendChild(
        statePanel("error", "Could not load background jobs.", message, "Try again", loadJobs)
      );
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
  if (window.location.hash === "#/datasets" || galleryVersionFromHash()) {
    showView("datasets");
  }
})();
