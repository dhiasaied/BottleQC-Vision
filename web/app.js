/* BottleQC Vision dashboard */
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];

const COLOR_HEX = {
  red: "#ef4444", orange: "#f97316", yellow: "#facc15", green: "#22c55e", blue: "#3b82f6",
  purple: "#a855f7", pink: "#e879b9", white: "#f1f5f9", black: "#1f2937", unknown: "#64748b", none: "#475569", any: "conic-gradient(#ef4444,#facc15,#22c55e,#3b82f6,#a855f7,#ef4444)",
};
const STATUS_LABEL = { OK: "OK", NO_CAP: "Missing cap", WRONG_COLOR: "Wrong colour" };

const state = {
  stats: null, filter: "", offset: 0, pageSize: 50, sessionId: null, config: null,
  charts: {}, ws: null, streaming: false, lastDefectToast: 0,
};

/* ------------------------------------------------------------ api */
async function api(path, opts = {}) {
  const res = await fetch(path, {
    headers: opts.body && !(opts.body instanceof FormData) ? { "Content-Type": "application/json" } : {},
    ...opts,
    body: opts.body && !(opts.body instanceof FormData) ? JSON.stringify(opts.body) : opts.body,
  });
  if (!res.ok) {
    let msg = res.statusText;
    try { const j = await res.json(); msg = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail); } catch {}
    throw new Error(msg);
  }
  return res.headers.get("content-type")?.includes("json") ? res.json() : res;
}

/* ------------------------------------------------------------ helpers */
const fmtTime = (ts) => new Date(ts * 1000).toLocaleTimeString([], { hour12: false });
const fmtDate = (ts) => new Date(ts * 1000).toLocaleString([], { hour12: false });
const fmtDur = (s) => { s = Math.floor(s || 0); return [s / 3600, (s % 3600) / 60, s % 60].map((v) => String(Math.floor(v)).padStart(2, "0")).join(":"); };
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
function toast(msg, kind = "", img = null, ms = 3500) {
  const t = document.createElement("div");
  t.className = `toast ${kind}`;
  t.innerHTML = (img ? `<img src="${img}" alt="">` : "") + `<div>${msg}</div>`;
  $("#toasts").appendChild(t);
  while ($("#toasts").children.length > 4) $("#toasts").firstChild.remove();
  setTimeout(() => t.remove(), ms);
}
function animateNumber(el, value) {
  const cur = Number(el.dataset.v || 0);
  if (cur === value) return;
  el.dataset.v = value;
  const start = performance.now(), d = 350;
  const step = (now) => {
    const p = Math.min(1, (now - start) / d);
    el.textContent = Math.round(cur + (value - cur) * (1 - Math.pow(1 - p, 3))).toLocaleString();
    if (p < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

/* ------------------------------------------------------------ charts */
function cssVar(n) { return getComputedStyle(document.documentElement).getPropertyValue(n).trim(); }
function initCharts() {
  if (!window.Chart) return;
  Chart.defaults.color = cssVar("--muted");
  Chart.defaults.borderColor = cssVar("--border");
  Chart.defaults.font.family = "Inter, system-ui, sans-serif";
  state.charts.timeline = new Chart($("#c-timeline"), {
    type: "line",
    data: { labels: [], datasets: [
      { label: "Counted", data: [], borderColor: cssVar("--accent"), backgroundColor: "rgba(56,189,248,.12)", fill: true, tension: .3, pointRadius: 0, borderWidth: 2 },
      { label: "Defects", data: [], borderColor: cssVar("--bad"), backgroundColor: "rgba(239,68,68,.10)", fill: true, tension: .3, pointRadius: 0, borderWidth: 2 },
    ] },
    options: { animation: false, maintainAspectRatio: false, interaction: { mode: "index", intersect: false },
      plugins: { legend: { position: "top", align: "end", labels: { boxWidth: 10, boxHeight: 10 } } },
      scales: { x: { ticks: { maxTicksLimit: 8 }, grid: { display: false } }, y: { beginAtZero: true, ticks: { precision: 0 } } } },
  });
  state.charts.results = new Chart($("#c-results"), {
    type: "doughnut",
    data: { labels: ["OK", "Missing cap", "Wrong colour"], datasets: [{ data: [0, 0, 0], backgroundColor: [cssVar("--ok"), cssVar("--bad"), cssVar("--warn")], borderWidth: 0 }] },
    options: { maintainAspectRatio: false, cutout: "68%", plugins: { legend: { position: "bottom", labels: { boxWidth: 10, boxHeight: 10 } } } },
  });
  state.charts.colors = new Chart($("#c-colors"), {
    type: "bar",
    data: { labels: [], datasets: [{ data: [], backgroundColor: [], borderRadius: 6 }] },
    options: { maintainAspectRatio: false, plugins: { legend: { display: false } },
      scales: { x: { grid: { display: false } }, y: { beginAtZero: true, ticks: { precision: 0 } } } },
  });
}
function updateCharts(s) {
  const c = state.charts;
  if (!c.timeline) return;
  c.timeline.data.labels = s.timeline.map((p) => fmtTime(p[0]));
  c.timeline.data.datasets[0].data = s.timeline.map((p) => p[1]);
  c.timeline.data.datasets[1].data = s.timeline.map((p) => p[2]);
  c.timeline.update("none");
  c.results.data.datasets[0].data = [s.ok, s.no_cap, s.wrong_color];
  c.results.update();
  const entries = Object.entries(s.by_color).sort((a, b) => b[1] - a[1]);
  c.colors.data.labels = entries.map((e) => e[0]);
  c.colors.data.datasets[0].data = entries.map((e) => e[1]);
  c.colors.data.datasets[0].backgroundColor = entries.map((e) => COLOR_HEX[e[0]] || "#64748b");
  c.colors.update();
}

/* ------------------------------------------------------------ stats render */
function renderStats(s) {
  const prev = state.stats;
  state.stats = s;
  const pill = $("#state-pill");
  pill.className = `pill ${s.state}`;
  $("b", pill).textContent = s.state[0].toUpperCase() + s.state.slice(1);
  if (s.error && (!prev || prev.error !== s.error)) toast(`⚠ ${esc(s.error)}`, "bad", null, 6000);

  if (s.model) $("#model-line").textContent = `${s.model.path} · ${s.model.task} · caps via ${s.model.cap_mode}`;
  $("#session-label").textContent = s.session_id ? `Session #${s.session_id}` : "No session";
  $("#elapsed").textContent = fmtDur(s.elapsed);
  $("#b-fps").textContent = `${s.fps.toFixed(1)} FPS`;
  $("#b-infer").textContent = `${s.infer_ms.toFixed(0)} ms`;
  $("#b-tracks").textContent = `${s.active_tracks} tracked`;

  animateNumber($("#k-total"), s.total);
  animateNumber($("#k-ok"), s.ok);
  animateNumber($("#k-def"), s.defects);
  $("#k-nocap").textContent = s.no_cap;
  $("#k-wrong").textContent = s.wrong_color;
  $("#k-avg").textContent = `avg ${s.avg_throughput_min ?? "—"} /min`;
  $("#k-ok-pct").textContent = s.total ? `${((s.ok / s.total) * 100).toFixed(1)}% of output` : "—";
  $("#k-tp").textContent = `${s.throughput_min} bottles / last min`;
  $("#k-yield").textContent = s.yield == null ? "—" : `${s.yield.toFixed(1)}%`;
  const ring = $("#yield-ring");
  ring.style.strokeDasharray = `${s.yield ?? 0} 100`;
  ring.style.stroke = s.yield == null ? cssVar("--muted") : s.yield >= 95 ? cssVar("--ok") : s.yield >= 85 ? cssVar("--warn") : cssVar("--bad");
  if (prev && s.defects > prev.defects) { $(".kpi.bad").classList.remove("flash"); void $(".kpi.bad").offsetWidth; $(".kpi.bad").classList.add("flash"); }

  const active = s.state === "running" || s.state === "paused" || s.state === "loading";
  $("#btn-start").disabled = s.state === "running" || s.state === "loading";
  $("#btn-pause").disabled = !(s.state === "running" || s.state === "paused");
  $("#btn-pause span").textContent = s.state === "paused" ? "Resume" : "Pause";
  $("#btn-stop").disabled = !active;
  setStreaming(active);
  if (!state.config) applyConfigToForm(s.config);
  setExpectedChip(s.config.expected_color);

  if (s.session_id !== state.sessionId) { state.sessionId = s.session_id; loadLog(true); }
  updateCharts(s);
}
function setStreaming(on) {
  const img = $("#video");
  if (on && !state.streaming) { img.src = `/api/stream.mjpg?t=${Date.now()}`; state.streaming = true; }
  if (!on && state.streaming) {
    state.streaming = false;
    img.src = `/api/frame.jpg?t=${Date.now()}`;  // freeze last frame
  }
}
$("#video").addEventListener("load", () => $(".video-wrap").classList.add("live"));
$("#video").addEventListener("error", () => { if (!state.streaming) $(".video-wrap").classList.remove("live"); });

/* ------------------------------------------------------------ inspection log */
function rowHtml(r, isNew = false) {
  const snap = r.snapshot ? `/api/snapshots/${r.snapshot}` : "";
  const color = r.cap_color ? `<span class="swatch-dot" style="background:${COLOR_HEX[r.cap_color] || "#64748b"}"></span>${esc(r.cap_color)}` : `<span style="color:var(--muted)">—</span>`;
  const pct = Math.round((r.color_conf || 0) * 100);
  return `<tr class="${isNew ? "new" : ""}">
    <td>${snap ? `<img class="thumb" loading="lazy" src="${snap}" alt="bottle ${r.id}">` : ""}</td>
    <td class="mono">${r.id}</td><td class="mono">${fmtTime(r.ts)}</td><td class="mono">#${r.track_id}</td>
    <td><span class="status ${r.status}">${STATUS_LABEL[r.status] || r.status}</span></td>
    <td>${color}</td>
    <td>${r.cap_color ? `<span class="bar-mini"><b style="width:${pct}%"></b></span><span class="mono">${pct}%</span>` : "—"}</td>
    <td class="mono">${(r.det_conf || 0).toFixed(2)}</td><td class="mono">${r.observations}</td></tr>`;
}
async function loadLog(reset = false) {
  if (reset) state.offset = 0;
  const q = new URLSearchParams({ limit: state.pageSize, offset: state.offset });
  if (state.filter) q.set("status", state.filter);
  if (state.sessionId) q.set("session_id", state.sessionId);
  try {
    const rows = state.sessionId ? await api(`/api/inspections?${q}`) : [];
    const body = $("#log-body");
    if (reset) body.innerHTML = "";
    body.insertAdjacentHTML("beforeend", rows.map((r) => rowHtml(r)).join(""));
    if (!body.children.length) body.innerHTML = `<tr class="empty"><td colspan="9">No inspections yet</td></tr>`;
    state.offset += rows.length;
    $("#log-more").hidden = rows.length < state.pageSize;
  } catch (e) { toast(`Could not load log: ${esc(e.message)}`, "bad"); }
}
function matchesFilter(r) {
  if (!state.filter) return true;
  if (state.filter === "DEFECT") return r.status !== "OK";
  return r.status === state.filter;
}
function onEvents(events) {
  const body = $("#log-body");
  for (const r of events) {
    if (r.session_id !== state.sessionId) continue;
    if (matchesFilter(r)) {
      body.querySelector("tr.empty")?.remove();
      body.insertAdjacentHTML("afterbegin", rowHtml(r, true));
      state.offset++;
      while (body.children.length > 400) body.lastElementChild.remove();
    }
    if (r.status !== "OK" && Date.now() - state.lastDefectToast > 700) {
      state.lastDefectToast = Date.now();
      toast(`<b>${STATUS_LABEL[r.status]}</b> · bottle #${r.id}${r.cap_color ? ` (${esc(r.cap_color)})` : ""}`,
        r.status === "NO_CAP" ? "bad" : "warn", r.snapshot ? `/api/snapshots/${r.snapshot}` : null, 2600);
    }
  }
}
$("#log-tabs").addEventListener("click", (e) => {
  const b = e.target.closest("button"); if (!b) return;
  $$("#log-tabs button").forEach((x) => x.classList.toggle("active", x === b));
  state.filter = b.dataset.f; loadLog(true);
});
$("#log-more").addEventListener("click", () => loadLog());
$("#log-body").addEventListener("click", (e) => {
  const img = e.target.closest(".thumb"); if (!img) return;
  const lb = document.createElement("div");
  lb.className = "lightbox"; lb.innerHTML = `<img src="${img.src}" alt="">`;
  lb.onclick = () => lb.remove(); document.body.appendChild(lb);
});

/* ------------------------------------------------------------ live connection */
function connectWS() {
  const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
  state.ws = ws;
  ws.onmessage = (m) => { const d = JSON.parse(m.data); if (d.events?.length) onEvents(d.events); renderStats(d.stats); };
  ws.onclose = () => { setTimeout(connectWS, 1500); };
  ws.onerror = () => ws.close();
}

/* ------------------------------------------------------------ source + controls */
function showSourceFields() {
  const t = $("#src-type").value;
  $$(".src-opt").forEach((el) => el.classList.toggle("show", el.dataset.for === t));
}
$("#src-type").addEventListener("change", showSourceFields);
async function loadVideos(select) {
  const vids = await api("/api/videos");
  $("#src-file").innerHTML = vids.length ? vids.map((v) => `<option value="${esc(v.path)}">${esc(v.name)}</option>`).join("") : `<option value="">No videos — upload one</option>`;
  if (select) $("#src-file").value = select;
}
$("#upload").addEventListener("change", async (e) => {
  const f = e.target.files[0]; if (!f) return;
  const fd = new FormData(); fd.append("file", f);
  toast(`Uploading ${esc(f.name)}…`);
  try { const r = await api("/api/upload", { method: "POST", body: fd }); await loadVideos(r.path); toast("Upload complete", "ok"); }
  catch (err) { toast(`Upload failed: ${esc(err.message)}`, "bad"); }
  e.target.value = "";
});
function sourceReq() {
  const type = $("#src-type").value;
  const value = type === "file" ? $("#src-file").value : type === "camera" ? $("#src-camera").value : type === "stream" ? $("#src-stream").value.trim() : "";
  return { type, value };
}
$("#btn-start").addEventListener("click", async () => {
  const src = sourceReq();
  if ((src.type === "file" || src.type === "stream") && !src.value) return toast("Select a video or enter a stream URL", "warn");
  $("#btn-start").disabled = true;
  try { await api("/api/control", { method: "POST", body: { action: "start", source: src } }); toast("Inspection started", "ok"); }
  catch (e) { toast(`Start failed: ${esc(e.message)}`, "bad", null, 6000); $("#btn-start").disabled = false; }
});
$("#btn-pause").addEventListener("click", () => api("/api/control", { method: "POST", body: { action: state.stats?.state === "paused" ? "resume" : "pause" } }).catch((e) => toast(esc(e.message), "bad")));
$("#btn-stop").addEventListener("click", async () => {
  try { await api("/api/control", { method: "POST", body: { action: "stop" } }); toast("Session stopped — report ready in Reports", "ok"); }
  catch (e) { toast(esc(e.message), "bad"); }
});

/* ------------------------------------------------------------ settings */
let cfgTimer;
function pushConfig(patch) {
  clearTimeout(cfgTimer);
  Object.assign(state.pending ||= {}, patch);
  cfgTimer = setTimeout(async () => {
    const body = state.pending; state.pending = null;
    try { const r = await api("/api/config", { method: "POST", body }); state.config = r.config; }
    catch (e) { toast(`Settings: ${esc(e.message)}`, "bad"); }
  }, 250);
}
function applyConfigToForm(c) {
  state.config = c;
  $("#s-conf").value = c.conf; $("#o-conf").textContent = c.conf.toFixed(2);
  $("#s-iou").value = c.iou; $("#o-iou").textContent = c.iou.toFixed(2);
  $("#s-line").value = c.line_pos; $("#o-line").textContent = `${Math.round(c.line_pos * 100)}%`;
  $("#s-presence").value = c.min_presence; $("#o-presence").textContent = `${Math.round(c.min_presence * 100)}%`;
  $("#s-dir").value = c.direction; $("#s-tracker").value = c.tracker; $("#s-imgsz").value = String(c.imgsz);
  $("#s-loop").checked = c.loop_video; $("#s-realtime").checked = c.realtime;
  $$("#swatches button").forEach((b) => b.classList.toggle("active", b.dataset.c === c.expected_color));
}
function setExpectedChip(c) {
  $("#expected-chip span").textContent = c;
  $("#expected-chip i").style.background = COLOR_HEX[c] || "#64748b";
}
const bindRange = (id, out, key, fmt) => $(id).addEventListener("input", (e) => { const v = parseFloat(e.target.value); $(out).textContent = fmt(v); pushConfig({ [key]: v }); });
bindRange("#s-conf", "#o-conf", "conf", (v) => v.toFixed(2));
bindRange("#s-iou", "#o-iou", "iou", (v) => v.toFixed(2));
bindRange("#s-line", "#o-line", "line_pos", (v) => `${Math.round(v * 100)}%`);
bindRange("#s-presence", "#o-presence", "min_presence", (v) => `${Math.round(v * 100)}%`);
$("#s-dir").addEventListener("change", (e) => pushConfig({ direction: e.target.value }));
$("#s-tracker").addEventListener("change", (e) => pushConfig({ tracker: e.target.value }));
$("#s-imgsz").addEventListener("change", (e) => pushConfig({ imgsz: parseInt(e.target.value, 10) }));
$("#s-loop").addEventListener("change", (e) => pushConfig({ loop_video: e.target.checked }));
$("#s-realtime").addEventListener("change", (e) => pushConfig({ realtime: e.target.checked }));
$("#swatches").addEventListener("click", (e) => {
  const b = e.target.closest("button"); if (!b) return;
  $$("#swatches button").forEach((x) => x.classList.toggle("active", x === b));
  setExpectedChip(b.dataset.c);
  pushConfig({ expected_color: b.dataset.c });
});
$("#s-model").addEventListener("change", async (e) => {
  toast(`Loading ${esc(e.target.value)}…`);
  try { const m = await api(`/api/model?name=${encodeURIComponent(e.target.value)}`, { method: "POST" }); toast(`Model ready · caps via ${m.cap_mode}`, "ok"); }
  catch (err) { toast(`Model: ${esc(err.message)}`, "bad", null, 6000); }
});
async function loadSettingsMeta() {
  const meta = await api("/api/config");
  $("#swatches").innerHTML = meta.colors.map((c) => `<button data-c="${c}"><span class="swatch-dot" style="background:${COLOR_HEX[c]}"></span>${c}</button>`).join("");
  const current = meta.config.model_path.split(/[\\/]/).pop();
  const models = [...new Set([current, ...meta.models])];
  $("#s-model").innerHTML = models.map((m) => `<option ${m === current ? "selected" : ""}>${esc(m)}</option>`).join("");
  $("#s-model-info").textContent = "Custom models with a 'cap' class detect caps directly; COCO models (yolo11n.pt) fall back to colour heuristics.";
  applyConfigToForm(meta.config);
}

/* ------------------------------------------------------------ drawers & reports */
function openDrawer(id) { $(`#${id}`).classList.add("open"); $("#scrim").classList.add("show"); if (id === "reports") loadSessions(); }
function closeDrawers() { $$(".drawer").forEach((d) => d.classList.remove("open")); $("#scrim").classList.remove("show"); }
$("#open-settings").addEventListener("click", () => openDrawer("settings"));
$("#expected-chip").addEventListener("click", () => openDrawer("settings"));
$("#open-reports").addEventListener("click", () => openDrawer("reports"));
$("#scrim").addEventListener("click", closeDrawers);
$$("[data-close]").forEach((b) => b.addEventListener("click", closeDrawers));
document.addEventListener("keydown", (e) => { if (e.key === "Escape") { closeDrawers(); $(".lightbox")?.remove(); } });

async function loadSessions() {
  const list = await api("/api/sessions");
  $("#r-scope").innerHTML = `<option value="current">Current session${state.sessionId ? ` (#${state.sessionId})` : ""}</option><option value="all">All sessions</option>` +
    list.map((s) => `<option value="${s.id}">Session #${s.id} · ${fmtDate(s.started_at)}</option>`).join("");
  $("#sessions").innerHTML = list.length ? list.map((s) => {
    const y = s.total ? ((s.ok / s.total) * 100).toFixed(1) + "%" : "—";
    const dur = s.ended_at ? fmtDur(s.ended_at - s.started_at) : "running";
    return `<li><b>#${s.id} · ${esc(s.source)}</b><span class="y" style="color:${!s.total ? "var(--muted)" : s.ok / s.total >= .95 ? "var(--ok)" : "var(--warn)"}">${y}</span>
      <small>${fmtDate(s.started_at)} · ${dur} · ${s.total} bottles · expected ${esc(s.expected_color)}</small></li>`;
  }).join("") : `<li><small>No sessions yet</small></li>`;
}
$$(".export-btns .btn").forEach((b) => b.addEventListener("click", () => {
  const scope = $("#r-scope").value;
  const q = new URLSearchParams({ fmt: b.dataset.fmt });
  if (scope === "all") q.set("scope", "all");
  else if (scope !== "current") q.set("session_id", scope);
  else if (!state.sessionId) return toast("No current session — pick one from the list", "warn");
  const a = document.createElement("a");
  a.href = `/api/reports/export?${q}`; a.download = ""; document.body.appendChild(a); a.click(); a.remove();
  toast(`Exporting ${b.dataset.fmt.toUpperCase()}…`, "ok");
}));

/* ------------------------------------------------------------ boot */
window.addEventListener("DOMContentLoaded", async () => {
  initCharts();
  showSourceFields();
  try { await Promise.all([loadVideos(), loadSettingsMeta()]); } catch (e) { toast(`Init: ${esc(e.message)}`, "bad"); }
  try { renderStats(await api("/api/stats")); } catch {}
  connectWS();
});
