const $ = (id) => document.getElementById(id);
const tenantId = () => $("tenant-id").value.trim();
const setStatus = (id, message, error = false) => { const el = $(id); el.textContent = message; el.classList.toggle("error", error); };
const showResult = (id, value, empty = false) => { const el = $(id); el.textContent = typeof value === "string" ? value : JSON.stringify(value, null, 2); el.classList.toggle("empty", empty); };

async function request(path, options = {}) {
  const response = await fetch(path, options);
  const text = await response.text();
  let body; try { body = text ? JSON.parse(text) : {}; } catch { body = { detail: text }; }
  if (!response.ok) throw new Error(body.detail || `Request failed (${response.status})`);
  return body;
}

async function checkRuntime() {
  try { await request("/health/ready"); $("runtime-status").textContent = "Local runtime ready"; $(".status-dot").classList.add("ready"); }
  catch { $("runtime-status").textContent = "Runtime unavailable"; }
}

function bindFile(inputId, labelId) { $(inputId).addEventListener("change", () => { const file = $(inputId).files[0]; $(labelId).textContent = file ? file.name : "Choose an image"; }); }
bindFile("asset-file", "file-label"); bindFile("predict-file", "predict-file-label"); checkRuntime();

$("upload-button").addEventListener("click", async () => {
  const file = $("asset-file").files[0]; if (!file) return setStatus("upload-status", "Choose an image first.", true);
  try {
    setStatus("upload-status", "Hashing and creating upload...");
    const bytes = await file.arrayBuffer(); const digest = [...new Uint8Array(await crypto.subtle.digest("SHA-256", bytes))].map(x => x.toString(16).padStart(2, "0")).join("");
    const headers = { "Content-Type": "application/json", "X-Tenant-ID": tenantId() };
    const asset = await request("/v1/assets", { method: "POST", headers, body: JSON.stringify({ name: $("asset-name").value }) });
    const upload = await request(`/v1/assets/${asset.asset_id}/uploads`, { method: "POST", headers, body: JSON.stringify({ filename: file.name, media_type: file.type || "image/jpeg", expected_sha256: digest }) });
    await request(`/v1/uploads/${upload.upload_id}/parts/1`, { method: "PUT", body: file });
    const source = await request(`/v1/uploads/${upload.upload_id}/complete`, { method: "POST" });
    setStatus("upload-status", `Uploaded ${file.name} · source ${source.source_id}`);
  } catch (error) { setStatus("upload-status", error.message, true); }
});

$("predict-button").addEventListener("click", async () => {
  const file = $("predict-file").files[0]; if (!file) return showResult("predict-result", "Choose an image first.", true);
  const form = new FormData(); form.append("image", file); form.append("prompt", $("predict-prompt").value);
  try { showResult("predict-result", "Running local SAM3..."); const result = await request("/v1/predict", { method: "POST", body: form }); showResult("predict-result", result); }
  catch (error) { showResult("predict-result", `Prediction unavailable: ${error.message}`); }
});

$("search-button").addEventListener("click", async () => {
  try { showResult("search-result", "Searching local index..."); const result = await request("/v1/search", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ query: $("search-query").value }) }); showResult("search-result", result); }
  catch (error) { showResult("search-result", `Search unavailable: ${error.message}`); }
});

$("export-button").addEventListener("click", async () => {
  try {
    const result = await request("/v1/dataset-exports", { method: "POST", headers: { "Content-Type": "application/json", "X-Tenant-ID": tenantId() }, body: JSON.stringify({ version_id: $("version-id").value, format: $("export-format").value, idempotency_key: $("export-key").value }) });
    setStatus("export-status", `Export queued · job ${result.job_id}`);
  } catch (error) { setStatus("export-status", error.message, true); }
});
