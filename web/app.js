const $ = (s) => document.querySelector(s);
const $$ = (s) => [...document.querySelectorAll(s)];

function esc(s) {
  if (s == null) return "";
  const d = document.createElement("div");
  d.textContent = String(s);
  return d.innerHTML;
}
function mb(m) { return m >= 1024 ? (m / 1024).toFixed(1) + " GB" : Math.round(m) + " MB"; }

let connectOpenState = new Map(); // Track open/closed state of connect boxes per engine port
let engines = [];
let localModels = [];
let hardware = null;
let mparams = {};
let mport = null;
let testMsgs = [];
let configMsgs = [];

function toast(msg, isErr = false) {
  const t = $("#toast");
  t.textContent = msg;
  t.className = "toast" + (isErr ? " err" : "");
  clearTimeout(t._h);
  t._h = setTimeout(() => t.classList.add("hidden"), 3400);
}

async function api(path, opts = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...opts,
    body: opts.body ? JSON.stringify(opts.body) : undefined,
  });
  if (res.status >= 400) {
    const t = await res.text();
    throw new Error(t || res.statusText);
  }
  const ct = res.headers.get("content-type") || "";
  return ct.includes("json") ? res.json() : res.text();
}

/* ---------------- TABS / MODALS ---------------- */
function switchTab(name) {
  $$(".tab").forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
  $$(".panel").forEach((p) => p.classList.toggle("active", p.id === "tab-" + name));
}
$$(".tab").forEach((b) => (b.onclick = () => switchTab(b.dataset.tab)));

$("#settingsBtn").onclick = () => $("#settingsModal").classList.remove("hidden");
$("#settingsClose").onclick = () => $("#settingsModal").classList.add("hidden");
$("#settingsSave").onclick = async () => {
  await api("/api/settings", { method: "POST", body: {
    openrouter_key: $("#orKey").value.trim() || undefined,
    hf_token: $("#hfKey").value.trim() || undefined,
    openrouter_model: $("#orModel").value.trim(),
  }});
  $("#settingsModal").classList.add("hidden");
  toast("settings saved");
};

function showModal(title, html) {
  $("#fileModalTitle").textContent = title;
  $("#fileModalBody").innerHTML = html;
  $("#fileModal").classList.remove("hidden");
}
$("#fileModalClose").onclick = () => $("#fileModal").classList.add("hidden");
$("#fileModal").addEventListener("click", (e) => { if (e.target.id === "fileModal") $("#fileModal").classList.add("hidden"); });

/* ---------------- UTILITIES (benchmark / tool test / code edit / cleanup) ---------------- */
$("#cleanupBtn").onclick = async () => {
  if (!confirm("Stop ALL engines and kill leftover llama-server processes? This frees GPU + RAM.")) return;
  try {
    const r = await api("/api/cleanup", { method: "POST", body: {} });
    toast(`cleanup: stopped ${r.stopped.length} engine(s), GPU free ${mb(r.gpu_free_mb)}`);
  } catch (e) { toast("cleanup failed: " + e.message, true); }
  pollEngines();
};
$("#benchRunBtn").onclick = runBench;
$("#toolRunBtn").onclick = runToolTest;
$("#codeRunBtn").onclick = runCodeEditTest;
$("#benchBtn").onclick = () => { switchTab("models"); $("#benchOut").scrollIntoView({ block: "center" }); };
$("#toolBtn").onclick = () => { switchTab("models"); $("#toolOut").scrollIntoView({ block: "center" }); };

async function runBench() {
  const port = $("#testModelSel").value;
  if (!port) return toast("load a model first", true);
  const btn = $("#benchRunBtn"); btn.disabled = true;
  $("#benchOut").textContent = "benchmarking…";
  try {
    const r = await api(`/api/engines/${port}/benchmark`, { method: "POST", body: {
      runs: +$("#benchRuns").value || 3, max_tokens: +$("#benchTokens").value || 200,
    }});
    $("#benchOut").textContent = Object.entries(r.results || []).map(([, x]) =>
      `run ${x.run}: ${x.tps ?? "-"} tok/s · TTFT ${x.ttft_s ?? "-"}s · ${x.tokens ?? "-"} tokens`).join("\n");

    if (r.best_tps != null) $("#benchOut").textContent += `\n\nBEST: ${r.best_tps} tok/s  AVG: ${r.avg_tps} tok/s\nparams: ctx=${r.params?.ctx} ngl=${r.params?.n_gpu_layers} fa=${r.params?.flash_attn} kv=${r.params?.kv_type} batch=${r.params?.batch}`;
  } catch (e) { $("#benchOut").textContent = "error: " + e.message; }
  btn.disabled = false;
}
async function runToolTest() {
  const port = $("#testModelSel").value;
  if (!port) return toast("load a model first", true);
  const btn = $("#toolRunBtn"); btn.disabled = true;
  $("#toolOut").textContent = "testing tool calls…";
  try {
    const r = await api(`/api/engines/${port}/tool-test`, { method: "POST", body: { prompt: $("#toolPrompt").value.trim() } });
    let s = `tool_calls_made: ${r.tool_calls_made ?? 0}\nsupports_tools: ${r.supports_tools ?? false}\ntool_score: ${r.tool_score_pct ?? 0}% ✅ valid JSON\nengine: ${r.engine ?? "-"}\n\n`;
    for (const c of r.calls || []) {
      s += `• ${c.name} → ${c.valid_json ? "valid JSON" : "BROKEN"}\n  ${(c.arguments_raw || "").slice(0, 220)}\n`;
    }
    if (r.text_fallback) s += `\n(fallback text — model may not support tools)\n${r.text_fallback}`;
    if (r.errors && r.errors.length) s += `\nerrors: ${r.errors.join(" | ")}`;
    if (r.elapsed_s != null) s += `\nelapsed: ${r.elapsed_s}s`;
    $("#toolOut").textContent = s;
    if (r.supports_tools === false) toast("model did not emit tool_calls — check chat template", true);
  } catch (e) { $("#toolOut").textContent = "error: " + e.message; toast("tool test failed: " + e.message, true); }
  btn.disabled = false;
}
async function runCodeEditTest() {
  const port = $("#testModelSel").value;
  if (!port) return toast("load a model first", true);
  const btn = $("#codeRunBtn"); btn.disabled = true;
  $("#codeOut").textContent = "running code-edit test…";
  try {
    const r = await api(`/api/engines/${port}/code-edit-test`, { method: "POST", body: { task: $("#codeTask").value.trim() } });
    let s = r.error ? "error: " + r.error
      : `valid_python: ${r.valid_python ? "✅ yes" : "❌ no (not parseable)"}\ntokens: ${r.tokens} · ${r.tps ?? 0} tok/s · ${r.elapsed_s ?? "-"}s\n\n`;
    s += (r.last_block || r.output || "").slice(0, 900);
    $("#codeOut").textContent = s;
  } catch (e) { $("#codeOut").textContent = "error: " + e.message; toast("code-edit test failed: " + e.message, true); }
  btn.disabled = false;
}

/* ---------------- POLLING ---------------- */
setInterval(pollEngines, 2500);
setInterval(pollHardware, 3000);
setInterval(pollDownloads, 1300);

async function pollEngines() {
  try { engines = await api("/api/engines"); } catch (e) { engines = []; }
  renderEngines();
  fillSelects();
}
async function pollHardware() {
  try { hardware = await api("/api/hardware"); } catch (e) {}
  renderHardware();
}
async function pollDownloads() {
  try { const ds = await api("/api/downloads"); if (ds.length) renderDownloads(ds); } catch (e) {}
}

/* ---------------- HARDWARE ---------------- */
function renderHardware() {
  const gpus = hardware?.gpus || [];
  let free = 0, total = 0;
  gpus.forEach((g) => { free += g.vram_free_mb; total += g.vram_total_mb; });
  const used = total - free;
  $("#vramBar").style.width = total ? Math.round((used / total) * 100) + "%" : "0%";
  $("#vramText").textContent = total ? `${mb(used)} / ${mb(total)}` : "no GPU";
  $("#ramBar").style.width = (hardware?.ram?.used_pct || 0) + "%";
  $("#ramText").textContent = hardware?.ram ? `${hardware.ram.used_gb}/${hardware.ram.total_gb} GB` : "—";
  const g = hardware?.active_gpu || gpus[0];
  $("#gpuUtil").textContent = g ? `${g.name.split(" ").slice(0, 3).join(" ")} · ${mb(g.vram_free_mb)} free · ${g.util_pct}%` : "no GPU";
  const running = engines.filter((x) => x.running).length;
  $("#engineState").textContent = running ? `up (${running})` : "idle";
  if (g) $("#hubFitNote").textContent = `(matching ${g.name.split(",")[0]} · ${mb(g.vram_free_mb)} free VRAM)`;
}

/* ---------------- ENGINES ---------------- */
function offloadCls(e) {
  const t = e.layers_total;
  if (!t) return "gpu-none";
  const r = e.layers_offloaded / t;
  return r >= 0.995 ? "gpu-full" : r >= 0.4 ? "gpu-part" : "gpu-none";
}
function engCard(e) {
  const dot = e.running ? (e.ready ? "dot-on" : "dot-loading") : "dot-off";
  const p = e.params || {};
  const m = localModels.find((x) => x.path === e.model_path);
  const size = m ? m.size_gb + " GB" : "—";
  const tps = e.metrics?.predicted_speed ? e.metrics.predicted_speed.toFixed(1) : "—";
  const fit = e.gpu_fit_pct != null ? Math.round(e.gpu_fit_pct) + "%" : "?";
  const vramTxt = e.vram_used_engine_mb ? ` ≈${mb(e.vram_used_engine_mb)} live` : "";
  const c = e.connect || {};
  return `<div class="card">
    <div class="d">PORT <b>${e.port}</b> <span class="status-dot ${dot}"></span> ${e.ready ? "ready" : (e.running ? "loading" : "stopped")}${e.pid ? " · pid " + e.pid : ""}</div>
    <div class="t">${esc(e.model_name)}</div>
    <div class="row">
      <span class="stat-chip ${offloadCls(e)}">GPU offload ${fitLabel(e)}</span>
      <span class="stat-chip">model ${mb(e.model_size_mb)}</span>
    </div>
    <div class="row">
      <span class="stat-chip">VRAM ≈ ${mb(e.est_vram_mb)}${vramTxt}</span>
      <span class="stat-chip">GPU ${mb(e.gpu_weights_mb)} / RAM ${mb(e.ram_weights_mb)} w</span>
    </div>
    <div class="row">
      <span class="stat-chip">TPS ${tps}</span>
      <span class="stat-chip">ctx ${p.ctx ?? "-"}</span>
      <span class="stat-chip">KV ${p.kv_type ?? "f16"}${p.flash_attn ? "+fa" : ""}</span>
      <span class="stat-chip">RAM ${e.ram_used_engine_mb ? Math.round(e.ram_used_engine_mb) + " MB" : "—"}</span>
    </div>
    <div class="connect-box ${e.ready ? "ready" : ""}">
      <div class="connect-head" onclick="toggleConnect(this.parentElement)">
        <span class="connect-title">${e.ready ? "● Ready — click to view API connection details" : "Connecting info"}</span>
        <span class="connect-caret">▾</span>
      </div>
      <div class="connect-body">
        <div class="muted">OpenAI-compatible endpoint — use from opencode, Claude Code, Cline, Roo, VS Code AI plugins.</div>
        <div class="kv"><label>Base URL</label><code id="cu-$base-${e.port}">${esc(c?.api_url || e.base_url)}</code><button class="small" onclick="copyText('cu-$base-${e.port}')">Copy</button></div>
        <div class="kv"><label>Models</label><code>${esc(c?.models_url || e.base_url + "/v1/models")}</code></div>
        <div class="kv"><label>Health</label><code>${esc(c?.health_url || e.base_url + "/health")}</code></div>
        <div class="kv"><label>Headers</label><code>Content-Type: application/json  ·  Authorization: Bearer local</code></div>
        <div class="kv full">
          <label>OpenCode / Claude Code provider</label>
          <pre>// ~/.opencode/opencode.json  or  CLAUDE_CODE provider
{
  "provider": "opencode-local",
  "base_url": "${esc(c?.base_url || e.base_url)}/v1",
  "api_key": "local"
}</pre>
        </div>
        <div class="kv full">
          <label>Cline / Roo</label>
          <pre>Settings → API Provider: OpenAI Compatible
API Base URL: ${esc(c?.base_url || e.base_url)}/v1
API Key: any value (e.g. "local")</pre>
        </div>
        <div class="kv"><label>curl</label><code>curl ${esc(c?.api_url || e.base_url + "/v1/chat/completions")} -H "Content-Type: application/json" -d '{"model":"local","messages":[{"role":"user","content":"hello"}]}'</code></div>
      </div>
    </div>
    <div class="btn-row" style="margin-top:8px">
      <button class="small primary" onclick="testModel(${e.port})">Test</button>
      <button class="small" onclick="editEngine(${e.port})">Edit</button>
      <button class="small" onclick="showLogs(${e.port})">Logs</button>
      <button class="small warn" onclick="stopEngine(${e.port})">Stop</button>
    </div></div>`;
}
function toggleConnect(el) { if (el) el.classList.toggle("open"); }
window.toggleConnect = toggleConnect;
function copyText(id) {
  const el = document.getElementById(id);
  if (navigator.clipboard && el) navigator.clipboard.writeText(el.textContent).then(() => toast("copied"));
}
function fitLabel(e) {
  const o = e.layers_offloaded, t = e.layers_total;
  if (!o || !t) return "fit unknown";
  const r = o / t;
  return r >= 0.995 ? "full (100%)" : r >= 0.5 ? `split (${Math.round(r * 100)}%)` : `partial (${Math.round(r * 100)}%)`;
}
function renderEngines() {
  $("#engineList").innerHTML = engines.length
    ? engines.map(engCard).join("")
    : '<div class="muted">No engines running. Click “Load Model”.</div>';
  $("#engineCount").textContent = engines.length;
}
function fillSelects() {
  const opts = engines.filter((x) => x.running && x.ready);
  for (const sel of ["#testModelSel", "#configEngineSel"]) {
    const el = $(sel);
    const prev = el.value;
    el.innerHTML = opts.map((x) => `<option value="${x.port}">${esc(x.model_name)} (${x.port})</option>`).join("");
    if (opts.some((x) => x.port == prev)) el.value = prev;
    if (!el.value && opts.length) el.value = opts[0].port;
  }
}
function testModel(port) { $("#testModelSel").value = port; switchTab("test"); }
function editEngine(port) { openLoadModalFor(port); }
async function showLogs(port) {
  const r = await api(`/api/engines/${port}/logs?n=160`);
  showModal("llama-server logs", `<pre style="max-height:60vh;overflow:auto;margin:0">${esc((r.logs || []).join("\n"))}</pre>`);
}
function stopEngine(port) {
  if (!confirm(`Stop engine on port ${port}?`)) return;
  api(`/api/engines/${port}/stop`, { method: "POST", body: {} }).then(pollEngines);
}

/* ---------------- LOCAL MODELS ---------------- */
async function refreshLocal() {
  localModels = await api("/api/local");
  const sel = $("#loadFileSel");
  sel.innerHTML = '<option value="">— select a local GGUF —</option>' +
    localModels.map((m) => `<option value="${esc(m.path)}">${esc(m.file)} (${m.size_gb} GB · ${esc(m.repo)})</option>`).join("");
  renderLocal();
}
function renderLocal() {
  $("#localList").innerHTML = localModels.map((m) => `
    <div class="card"><div class="t">${esc(m.file)}</div>
    <div class="d">${esc(m.repo)} · ${m.size_gb} GB</div>
    <button class="small primary" onclick="openLoadByPath('${esc(m.path)}')">Load</button></div>`).join("");
}
function openLoadByPath(p) { $("#loadFileSel").value = p; openLoadModalFor(null); }

/* ---------------- HUB (VRAM-filtered) ---------------- */
$("#searchBtn").onclick = searchNow;
$("#searchInput").onkeydown = (e) => { if (e.key === "Enter") searchNow(); };
async function loadHub() {
  try {
    const r = await api("/api/hub");
    renderHub(r.models || [], "Compatible with your GPU — fully offloadable:");
    if (r.gpu) $("#hubFitNote").textContent = `(matching ${r.gpu.name.split(",")[0]} · ${mb(r.vram_free_mb)} free VRAM)`;
  } catch (e) { $("#hubResults").innerHTML = '<div class="muted">hub unavailable</div>'; }
}
function renderHub(rows, label) {
  $("#hubResults").innerHTML = rows.length
    ? `<div class="muted" style="margin-bottom:6px">${esc(label)}</div>` + rows.map((m) => `
      <div class="card">
        <div class="t">${esc(m.id)}</div>
        <div class="d">fits VRAM: <b class="gpu-full">✅ ${m.best_size_gb ?? "?"} GB</b> · files ${m.file_count ?? "?"}
          ${m.downloads != null ? ` · dl ${m.downloads}` : ""}</div>
        <div class="btn-row" style="margin-top:6px">
          <button class="small primary" onclick="inspectRepoDownload('${esc(m.id)}')">Files</button>
        </div>
      </div>`).join("")
    : '<div class="muted">No models on HuggingFace fit your current GPU VRAM. Lower VRAM usage or clear GPU with ⚡ Cleanup.</div>';
}
async function searchNow() {
  const q = $("#searchInput").value.trim() || "instruct";
  try {
    const rows = await api(`/api/search?q=${encodeURIComponent(q)}&limit=40`);
    renderHub(rows, q + " — only models fitting VRAM");
    if (!rows.length) toast("No search results fit your GPU — try a smaller quant", true);
  } catch (e) { toast(e.message, true); }
}

async function inspectRepoDownload(repo) {
  try {
    const d = await api("/api/model/" + encodeURIComponent(repo));
    const files = (d.files || []).filter((f) => f.size > 0);
    if (!files.length) return toast("no GGUF files in repo", true);
    const freeVram = hardware?.active_gpu?.vram_free_mb ?? 0;
    const maxBytes = Math.max(0, freeVram * 1024 * 1024 - 400 * 1024 * 1024);
    const list = files.map((f) => {
      const fits = f.size <= maxBytes;
      return `<div class="card"><div class="t">${esc(f.name)}</div>
        <div class="d">${(f.size / 2 ** 30).toFixed(2)} GB · <span class="${fits ? "m-fit" : "m-nofit"}">${fits ? "✅ fits GPU" : "⚠ exceeds VRAM"}</span></div>
        <button class="small primary" ${fits ? "" : "disabled"} onclick="downloadModel('${esc(repo)}','${esc(f.name)}')">${fits ? "Download" : "Too big"}</button></div>`;
    }).join("");
    showModal("Download · " + repo, list);
  } catch (e) { toast(e.message, true); }
}
async function downloadModel(repo, file) {
  await api("/api/download", { method: "POST", body: { repo, file } });
  toast("downloading " + file);
}
function renderDownloads(list) {
  $("#dlList").innerHTML = list.map((d) => {
    const pct = d.total ? Math.round((d.done / d.total) * 100) : 0;
    const spd = d.speed ? (d.speed / 2 ** 20).toFixed(1) + " MB/s" : "";
    const err = d.error ? ` <span style="color:var(--red)">${esc(d.error)}</span>` : "";
    return `<div class="card"><div class="d">${esc(d.file)} — <b>${d.status}</b> ${spd}${err}</div>
      <div class="progress"><div style="width:${pct}%"></div></div>
      <div class="d">${gb2(d.done)} / ${gb2(d.total)} GB</div></div>`;
  }).join("");
}
function gb2(b) { return (b / 2 ** 30).toFixed(2); }

/* ---------------- LOAD MODAL ---------------- */
$("#loadBtn").onclick = () => openLoadModalFor(null);
$("#modalClose").onclick = () => $("#modal").classList.add("hidden");
$("#modal").addEventListener("click", (e) => { if (e.target.id === "modal") $("#modal").classList.add("hidden"); });
$("#autoTuneBtn").onclick = async () => {
  const path = $("#loadFileSel").value;
  if (!path) return toast("select a model file first", true);
  try {
    const est = await api("/api/estimate?path=" + encodeURIComponent(path));
    applyTune(est.tuned || {});
    toast("auto-tuned for current hardware");
  } catch (e) { toast("auto-tune failed: " + e.message, true); }
};
$("#fitFullBtn").onclick = () => { mparams.n_gpu_layers = specs.find((s) => s.key === "n_gpu_layers")?.max ?? 999; renderModalParams(); };
$("#fitRamBtn").onclick = () => {
  const p = $("#loadFileSel").value;
  if (!p) return;
  api("/api/estimate?path=" + encodeURIComponent(p))
    .then((e) => { mparams.n_gpu_layers = e.suggested_ngl; renderModalParams(); })
    .catch(() => {});
};
$("#param-n_gpu_layers").addEventListener("input", () => {
  mparams.n_gpu_layers = +$("#param-n_gpu_layers").value;
  $("#nglVal").textContent = mparams.n_gpu_layers;
  updateEstVram();
});
$("#loadFileSel").addEventListener("change", (e) => updateFileInfo(e.target.value));
$("#loadConfirm").onclick = doLoad;

function applyTune(tuned) {
  Object.assign(mparams, tuned || {});
  specs.forEach((s) => { if (mparams[s.key] == null) mparams[s.key] = s.default; });
  renderModalParams();
}

function openLoadModalFor(port = null) {
  const eng = port ? engines.find((e) => e.port === port) : null;
  mport = port;
  specs.forEach((s) => { mparams[s.key] = eng ? (eng.params[s.key] ?? s.default) : s.default; });
  if (eng && eng.model_path) {
    $("#loadFileSel").value = eng.model_path;
  }
  $("#loadConfirm").textContent = port ? "Save & restart" : "Load & start server";
  renderModalParams();
  updateFileInfo($("#loadFileSel").value);
  $("#modal").classList.remove("hidden");
}

function renderModalParams() {
  const ngl = $("#param-n_gpu_layers");
  ngl.value = mparams.n_gpu_layers;
  $("#nglVal").textContent = mparams.n_gpu_layers;
  const box = $("#paramList");
  box.innerHTML = specs.filter((s) => s.key !== "n_gpu_layers").map((s) => {
    const v = mparams[s.key];
    const ccat = s.cat_color || "green";
    const tag = `<span class="cat-tag ${ccat}">${esc(s.cat_label || "")}</span>`;
    if (s.type === "bool")
      return `<div class="param-row"><div class="param-head"><span>${esc(s.label)}</span>${tag}</div>
        <label class="chk"><input type="checkbox" ${v ? "checked" : ""} onchange="setBool('${s.key}',this.checked)">${esc(s.help)}</label></div>`;
    if (s.type === "enum")
      return `<div class="param-row"><div class="param-head"><span>${esc(s.label)}</span>${tag}</div>
        <select onchange="setVal('${s.key}',this.value)">${s.options.map((o) => `<option ${o === v ? "selected" : ""}>${o}</option>`).join("")}</select></div>`;
    return `<div class="param-row"><div class="param-head"><span>${esc(s.label)}</span><span class="val" id="pv-${s.key}">${v}</span></div>
      ${tag}
      <input type="range" id="pr-${s.key}" min="${s.min}" max="${s.max}" step="${s.step}" value="${v}" data-k="${s.key}">
      <div class="muted">${esc(s.help)}</div></div>`;
  }).join("");
  box.querySelectorAll("input[type=range]").forEach((r) => r.oninput = () => {
    mparams[r.dataset.k] = +r.value;
    $("#pv-" + r.dataset.k).textContent = r.value;
    updateEstVram();
  });
}
window.setBool = (k, v) => { mparams[k] = v; };
window.setVal = (k, v) => { mparams[k] = v; };

async function updateFileInfo(path) {
  if (!path) {
    $("#stLayers").textContent = "-"; $("#stSize").textContent = "-";
    $("#stVram").textContent = "-"; $("#stFree").textContent = "-";
    const h = $("#fitHint"); h.textContent = ""; h.className = "fit-hint";
    return;
  }
  try {
    const g = await api("/api/gguf-meta?path=" + encodeURIComponent(path));
    const est = await api("/api/estimate?path=" + encodeURIComponent(path));
    const layers = g.block_count ?? "?";
    if (layers !== "?") $("#param-n_gpu_layers").max = layers;
    $("#stLayers").textContent = layers;
    $("#stSize").textContent = (g.size_gb ?? "?") + " GB";
    $("#stVram").textContent = mb(est.est_vram_mb);
    $("#stFree").textContent = mb(est.free_vram_mb) + " free";
    const hint = $("#fitHint");
    if (est.fits_fully) { hint.textContent = "✅ Fits fully on GPU — maximum speed (auto-tuned)"; hint.className = "fit-hint ok"; }
    else { hint.textContent = "⚠ Does NOT fully fit — remainder runs from system RAM"; hint.className = "fit-hint warn"; }
    if (mport == null && !autoApplied) {
      autoApplied = true;
      if (est.tuned) { Object.assign(mparams, est.tuned); renderModalParams(); }
      else mparams.n_gpu_layers = est.suggested_ngl;
    }
  } catch (e) {}
}
let autoApplied = false;
function updateEstVram() {
  const size = parseFloat(($("#stSize").textContent || "0"));
  const layers = parseInt($("#stLayers").textContent || "1", 10);
  const n = mparams.n_gpu_layers || 0;
  const frac = layers > 1 ? Math.min(1, n / layers) : 1;
  $("#stVram").textContent = mb(Math.round((size * 1024) * frac));
}
async function doLoad() {
  const path = $("#loadFileSel").value;
  if (!path) return toast("select a model file", true);
  const btn = $("#loadConfirm");
  btn.disabled = true;
  try {
    if (mport) {
      await api(`/api/engines/${mport}/restart`, { method: "POST", body: { params: mparams } });
    } else {
      await api("/api/engines/load", { method: "POST", body: { path, params: mparams, auto_tune: true, name: path.split(/[\\/]/).pop() } });
    }
    $("#modal").classList.add("hidden");
    toast(mport ? "engine restarted with new settings" : "model loaded");
  } catch (e) { toast("failed: " + e.message, true); }
  btn.disabled = false;
  pollEngines();
}

/* ---------------- TEST CHAT ---------------- */
$("#testSend").onclick = sendTest;
$("#testInput").onkeydown = (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendTest(); } };
$("#clearChatBtn").onclick = () => { $("#testChat").innerHTML = ""; testMsgs = []; };

function renderTestParams() {
  const keys = ["temperature", "top_k", "top_p", "min_p", "repeat_penalty"];
  $("#testParams").innerHTML = specs.filter((s) => keys.includes(s.key)).map((s) => `
    <div class="param-row"><div class="param-head"><span>${esc(s.label)}</span><span class="val">${s.default}</span></div>
    <input type="range" id="slider-${s.key}" min="${s.min}" max="${s.max}" step="${s.step}" value="${s.default}" data-k="${s.key}"></div>`).join("");
  $("#testParams").querySelectorAll("input").forEach((r) =>
    r.oninput = () => { r.parentElement.querySelector(".val").textContent = r.value; });
}

function addMsg(chatSel, role, text) {
  const el = document.createElement("div");
  el.className = "msg " + role;
  if (role === "assistant") el.innerHTML = '<div class="msg-role">assistant</div>';
  const body = document.createElement("div");
  body.className = "msg-body";
  body.textContent = text;
  el.appendChild(body);
  $(chatSel).appendChild(el);
  $(chatSel).scrollTop = $(chatSel).scrollHeight;
  return { el, body };
}

async function readSSE(res, h) {
  if (!res.ok) throw new Error(await res.text());
  const ct = res.headers.get("content-type") || "";
  if (ct.includes("json")) {
    const o = await res.json();
    if (o.error) throw new Error(o.error);
    if (h.done) h.done(o);
    return;
  }
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = "", ev = "message";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let i;
    while ((i = buf.indexOf("\n\n")) >= 0) {
      const blk = buf.slice(0, i);
      buf = buf.slice(i + 2);
      let data = "";
      for (const l of blk.split("\n")) {
        if (l.startsWith("event:")) ev = l.slice(6).trim();
        else if (l.startsWith("data:")) data += l.slice(5).trim();
      }
      if (!data) continue;
      let o; try { o = JSON.parse(data); } catch (e) { continue; }
      if (h.event) h.event(ev, o);
      if (ev === "meta") h.meta && h.meta(o);
      else if (ev === "error") h.error && h.error(o);
      else if (ev === "done") h.done && h.done(o);
      else h.data && h.data(o);
      ev = "message";
    }
  }
}

const TEST_TOOLS = [
  { type: "function", function: { name: "get_weather", description: "Get weather for a city", parameters: { type: "object", properties: { city: { type: "string" } }, required: ["city"] } } },
  { type: "function", function: { name: "search_web", description: "Search the web", parameters: { type: "object", properties: { query: { type: "string" } }, required: ["query"] } } },
  { type: "function", function: { name: "run_python", description: "Run python", parameters: { type: "object", properties: { code: { type: "string" } }, required: ["code"] } } },
  { type: "function", function: { name: "list_files", description: "List files in a directory", parameters: { type: "object", properties: { path: { type: "string" } } } } },
];

async function sendTest() {
  const port = $("#testModelSel").value;
  if (!port) return toast("load a model first", true);
  const input = $("#testInput").value.trim();
  if (!input) return;
  $("#testInput").value = "";
  const body = {
    messages: [...testMsgs, { role: "user", content: input }],
    max_tokens: +$("#testMaxTokens").value || 512,
  };
  for (const k of ["temperature", "top_k", "top_p", "min_p", "repeat_penalty"]) {
    const el = $("#slider-" + k);
    if (el) body[k] = parseFloat(el.value);
  }
  if ($("#useToolsChk")?.checked) { body.tools = TEST_TOOLS; body.parallel_tool_calls = true; }
  testMsgs.push({ role: "user", content: input });
  addMsg("#testChat", "user", input);
  const as = addMsg("#testChat", "assistant", "");
  let text = "";
  try {
    const res = await fetch(`/api/engines/${port}/chat`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    await readSSE(res, {
      data(o) {
        const delta = o.choices?.[0]?.delta;
        const c = delta?.content;
        if (c) { text += c; as.body.textContent = text; as.el.scrollIntoView({ block: "end" }); }
        const tcs = delta?.tool_calls || [];
        for (const tc of tcs) {
          const fn = tc.function || {};
          const line = `[tool_call] ${fn.name}(${fn.arguments || ""})`;
          text += (text.endsWith("\n") ? "" : "\n") + line;
          as.body.className = "msg-body tool-body";
          as.body.textContent = text;
        }
      },
      meta(m) {
        const d = document.createElement("div");
        d.className = "meta-line" + (m.tps >= 15 ? " good" : "");
        d.textContent = `${m.tokens} tok · ${m.tps} tok/s · first ${m.first_token_s}s · ${m.elapsed_s}s`;
        as.el.appendChild(d);
      },
      error(e) { toast("chat error: " + (e.error || "?"), true); },
    });
    testMsgs.push({ role: "assistant", content: text });
  } catch (e) { toast("chat error: " + e.message, true); }
  pollEngines();
}

/* ---------------- CONFIG CHAT ---------------- */
$("#configSend").onclick = sendConfig;
$("#configInput").onkeydown = (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendConfig(); } };
$("#orRefresh").onclick = refreshFreeModels;

async function initConfig() {
  const s = await api("/api/settings");
  if (s.openrouter_model) $("#orModel").value = s.openrouter_model;
  refreshFreeModels();
}
async function refreshFreeModels() {
  try {
    const list = await api("/api/free-models");
    $("#orModelList").innerHTML = list.map((m) => `<option value="${esc(m.id)}">${esc(m.id)}</option>`).join("");
  } catch (e) {}
}

async function sendConfig() {
  const q = $("#configInput").value.trim();
  if (!q) return;
  $("#configInput").value = "";
  configMsgs.push({ role: "user", content: q });
  addMsg("#configChat", "user", q);
  const port = $("#configEngineSel").value;
  const msg = addMsg("#configChat", "assistant", "");
  let text = "";
  try {
    const res = await fetch("/api/config-chat", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages: configMsgs, port: port ? +port : null, auto_apply: $("#autoApply").checked }),
    });
    await readSSE(res, {
      data(o) { if (o.delta) { text += o.delta; msg.body.textContent = text; msg.el.scrollIntoView({ block: "end" }); } },
      done(o) {
        if (o.applied && o.applied.params_after) {
          const d = document.createElement("div");
          d.className = "meta-line good";
          d.textContent = "Applied live → engine restarted. Reason: " + (o.applied.reason || "—");
          msg.el.appendChild(d);
          pollEngines();
        } else if (o.changes && Object.keys(o.changes).length) {
          const d = document.createElement("div");
          d.className = "meta-line";
          d.textContent = "Suggested changes (auto-apply off): " + JSON.stringify(o.changes);
          msg.el.appendChild(d);
        }
      },
      error(e) {
        const d = document.createElement("div");
        d.className = "meta-line";
        d.textContent = "error: " + (e.error || "?");
        msg.el.appendChild(d);
      },
    });
    configMsgs.push({ role: "assistant", content: text });
  } catch (e) { toast("config error: " + e.message, true); }
}

/* ---------------- INIT ---------------- */
(async function init() {
  try {
    specs = await api("/api/spec");
    renderTestParams();
    await refreshLocal();
    await initConfig();
    await loadHub();
    pollEngines();
    pollHardware();
    renderModalParams();
  } catch (e) { toast("init error: " + e.message, true); }
})();