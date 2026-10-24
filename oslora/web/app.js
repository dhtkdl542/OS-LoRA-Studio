"use strict";
// OS LoRA Studio front end. Plain JS, no build step.

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const S = {
  view: "dataset", defaults: null, ds: null, page: 0, perPage: 24, sel: -1,
  run: null, runs: [], detail: null, preset: "character", capTimer: null,
};
const SEC_PER_STEP = { 512: 2.3, 768: 4.5, 1024: 8.8, 1280: 14.5 };

// ------------------------------------------------------------------ utils
async function api(path, body) {
  const opt = body === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
  const r = await fetch(path, opt);
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.detail || `요청 실패 (${r.status})`);
  return data;
}
let toastTimer;
function toast(msg, err = false) {
  const t = $("#toast");
  t.textContent = msg; t.className = "toast" + (err ? " err" : ""); t.hidden = false;
  clearTimeout(toastTimer); toastTimer = setTimeout(() => (t.hidden = true), err ? 6000 : 3000);
}
const thumb = (p, size = 384) => `/api/thumb?size=${size}&path=${encodeURIComponent(p)}`;
const full = p => `/api/image?path=${encodeURIComponent(p)}`;
const fmtLR = v => (v == null ? "-" : Number(v).toExponential(2).replace("e-", "e-"));
function fmtTime(sec) {
  if (!sec || sec < 0) return "-";
  const h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60);
  return h ? `${h}시간 ${m}분` : `${m}분`;
}
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
function segValue(id) { return $(`#${id} button.on`)?.dataset.v; }
function bindSeg(id, onChange) {
  $$(`#${id} button`).forEach(b => b.addEventListener("click", () => {
    $$(`#${id} button`).forEach(x => x.classList.toggle("on", x === b));
    onChange && onChange(b.dataset.v);
  }));
}
function setSeg(id, v) { $$(`#${id} button`).forEach(x => x.classList.toggle("on", x.dataset.v == String(v))); }

// ------------------------------------------------------------------ navigation
function show(view) {
  if (!$(`#view-${view}`)) view = "dataset";
  S.view = view;
  history.replaceState(null, "", `#${view}`);
  $$(".nav-item").forEach(b => b.classList.toggle("active", b.dataset.view === view));
  $$(".view").forEach(v => v.classList.toggle("active", v.id === `view-${view}`));
  if (view === "dataset") layoutGrid();
  if (view === "monitor" || view === "results") refreshRun();
}
$$(".nav-item").forEach(b => b.addEventListener("click", () => show(b.dataset.view)));
$("#job").addEventListener("click", () => show("monitor"));

// ------------------------------------------------------------------ runs (rail + top bar)
const PHASE = { starting: "준비 중", caching: "캐싱", training: "학습 중", sampling: "샘플 생성", done: "완료", error: "오류", paused: "⏸ 일시정지" };
async function refreshRuns() {
  try {
    const { runs } = await api("/api/runs");
    S.runs = runs;
    const live = runs.find(r => r.alive) || runs.find(r => ["training", "sampling", "caching", "starting", "paused"].includes(r.phase) && Date.now() / 1000 - r.updated < 120);
    if (!S.run && runs.length) S.run = (live || runs[0]).name;
    $("#run-list").innerHTML = runs.map(r => {
      const st = r.max_steps && r.phase !== "done" ? `${PHASE[r.phase] || r.phase} ${r.step}/${r.max_steps}` : (PHASE[r.phase] || r.phase);
      return `<li class="${r === live ? "live" : ""}"><button class="${r.name === S.run ? "sel" : ""}" data-run="${esc(r.name)}">
        <span class="rn">${esc(r.name)}</span><span class="rs">${esc(st)}${r.test ? " (테스트)" : ""}</span></button></li>`;
    }).join("") || `<li class="hint" style="padding:0 10px">아직 없습니다</li>`;
    $$("#run-list button").forEach(b => b.addEventListener("click", () => {
      S.run = b.dataset.run; refreshRuns();
      if (S.view !== "results") show("monitor"); else refreshRun();
    }));
    const job = $("#job");
    if (live && live.max_steps) {
      job.hidden = false;
      $("#job-text").textContent = `${live.name} ${Math.round((100 * live.step) / live.max_steps)}%${live.phase === "paused" ? " ⏸ 일시정지" : ""}`;
      $("#job-fill").style.width = `${(100 * live.step) / live.max_steps}%`;
    } else if (live) {
      job.hidden = false; $("#job-text").textContent = `${live.name} ${PHASE[live.phase] || ""}`; $("#job-fill").style.width = "0";
    } else job.hidden = true;
    // while a run is going, "start" queues the next one
    S.busyTrain = !!live;
    // a paused run can take changed settings (int8, LR range, max steps, samples, photo texture)
    S.pausedRun = live && live.phase === "paused" ? live.name : null;
    $("#tr-start").textContent = S.pausedRun ? "바뀐 설정 반영해서 재개" : (live ? "대기열에 추가" : "학습 시작");
    refreshQueue();
  } catch (e) { /* server restarting */ }
}

// ------------------------------------------------------------------ dataset
async function loadDataset() {
  const folder = $("#ds-folder").value.trim().replace(/^"|"$/g, "");
  if (!folder) return toast("폴더 경로를 입력하세요.", true);
  try {
    S.ds = await api("/api/dataset", { folder });
    localStorage.setItem("oslora.folder", folder);
  } catch (e) { return toast(e.message, true); }
  S.page = 0; S.sel = -1;
  const d = S.ds, n = d.items.length;
  $("#ds-stats").innerHTML = `
    <div><b>${n}</b><span>이미지</span></div>
    <div class="${d.captioned < n ? "warn" : ""}"><b>${d.captioned}</b><span>캡션 있음</span></div>
    <div><b>${d.holdout || "-"}</b><span>${d.holdout ? "검증용 홀드아웃" : "홀드아웃 없음 (16장 미만)"}</span></div>
    ${d.small ? `<div class="warn"><b>${d.small}</b><span>저해상도</span></div>` : ""}`;
  selectImage(-1);
  layoutGrid();
  updateSummary();
  startDiagnosis();
}
function layoutGrid() {
  const grid = $("#ds-grid");
  if (!S.ds || !S.ds.items.length) return;
  const wrap = $(".ds-gridwrap");
  const w = grid.clientWidth, h = wrap.clientHeight - 40;
  const gap = 8, target = 132;
  const cols = Math.max(2, Math.floor((w + gap) / (target + gap)));
  const tileW = (w - gap * (cols - 1)) / cols, tileH = (tileW * 4) / 3;
  const rows = Math.max(1, Math.floor((h + gap) / (tileH + gap)));
  S.perPage = cols * rows;
  grid.style.gridTemplateColumns = `repeat(${cols}, minmax(0, 1fr))`;
  const pages = Math.ceil(S.ds.items.length / S.perPage);
  S.page = Math.min(S.page, pages - 1);
  renderGrid(pages);
}
function renderGrid(pages) {
  const items = S.ds.items, start = S.page * S.perPage;
  $("#ds-grid").innerHTML = items.slice(start, start + S.perPage).map((it, k) => {
    const i = start + k;
    const flag = S.flags && S.flags[it.name];
    const tip = (flag ? `[${flag.titles.join(", ")}]\n` : "") + (it.caption || "캡션 없음");
    return `<button class="tile ${i === S.sel ? "sel" : ""}" data-i="${i}" title="${esc(tip)}">
      <img loading="lazy" src="${thumb(it.path, 320)}" alt="${esc(it.name)}">${it.caption ? "" : `<span class="nocap">캡션 없음</span>`}${flag ? `<span class="flag ${flag.level}"></span>` : ""}</button>`;
  }).join("");
  $$("#ds-grid .tile").forEach(t => t.addEventListener("click", () => selectImage(+t.dataset.i)));
  $("#ds-pager").hidden = pages <= 1;
  $("#pg-label").textContent = `${S.page + 1} / ${pages}`;
}
function selectImage(i) {
  S.sel = i;
  $$("#ds-grid .tile").forEach(t => t.classList.toggle("sel", +t.dataset.i === i));
  const it = S.ds && S.ds.items[i];
  // right panel: check-up results until an image is picked, then the caption editor
  $("#ds-diag").hidden = !!it;
  $("#ds-edit").hidden = !it;
  $("#ed-img").classList.toggle("on", !!it);
  $("#ed-empty").hidden = !!it;
  $("#ed-cap").disabled = $("#ed-save").disabled = !it;
  $("#ed-msg").textContent = "";
  if (!it) { $("#ed-cap").value = ""; $("#ed-meta").textContent = ""; return; }
  $("#ed-img").src = thumb(it.path, 900);
  $("#ed-meta").textContent = `${it.name}  ${it.w}×${it.h}`;
  $("#ed-cap").value = it.caption;
}
// ------------------------------------------------------------------ dataset check-up
const LEVEL_RANK = { bad: 3, warn: 2, info: 1 };
async function startDiagnosis() {
  if (!S.ds) return;
  clearTimeout(S.diagTimer);
  try { await api("/api/diagnose", { folder: S.ds.folder, resolution: +segValue("tr-res") || 1024 }); }
  catch (e) { $("#diag-body").innerHTML = `<div class="empty small">${esc(e.message)}</div>`; return; }
  pollDiagnosis();
}
async function pollDiagnosis() {
  let st;
  try { st = await api("/api/diagnose/status"); } catch { return; }
  const body = $("#diag-body");
  if (st.running) {
    const pct = st.total ? (100 * st.done) / st.total : 0;
    body.innerHTML = `<div class="diag-progress"><span>진단 중… ${st.done}/${st.total || "?"} ${esc(st.current || "")}</span>
      <div class="bar"><span style="width:${pct}%"></span></div>
      <span>얼굴 인식·흐림·중복·캡션을 확인하고 있어요 (CPU만 써서 학습 중에도 괜찮아요).</span></div>`;
    S.diagTimer = setTimeout(pollDiagnosis, 800);
    return;
  }
  if (st.error) { body.innerHTML = `<div class="empty small">진단 실패: ${esc(st.error)}</div>`; return; }
  if (!st.findings) return;
  renderFindings(st.findings);
}
function renderFindings(findings) {
  S.findings = findings;
  S.flags = {};
  findings.forEach(f => {
    if (!LEVEL_RANK[f.level]) return;
    f.files.forEach(n => {
      const cur = S.flags[n] || { level: f.level, titles: [] };
      if (LEVEL_RANK[f.level] > LEVEL_RANK[cur.level]) cur.level = f.level;
      cur.titles.push(f.title);
      S.flags[n] = cur;
    });
  });
  $("#diag-body").innerHTML = findings.map((f, k) => `
    <div class="finding ${f.level}">
      <div class="ft"><i></i><span>${esc(f.title)}</span></div>
      <div class="fd">${esc(f.detail)}</div>
      ${f.files.length ? `<div class="chips">${f.files.slice(0, 8).map(n => `<button data-name="${esc(n)}" title="${esc(n)}">${esc(n)}</button>`).join("")}${f.files.length > 8 ? `<button disabled>+${f.files.length - 8}</button>` : ""}</div>` : ""}
      ${f.fix ? `<button class="btn primary fix" data-k="${k}">${esc(f.fix_label)}</button>` : ""}
    </div>`).join("");
  $$("#diag-body .chips button[data-name]").forEach(b => b.addEventListener("click", () => jumpTo(b.dataset.name)));
  $$("#diag-body .fix").forEach(b => b.addEventListener("click", () => applyFix(findings[+b.dataset.k])));
  if (S.ds) layoutGrid();
}
function jumpTo(name) {
  const i = S.ds.items.findIndex(it => it.name === name);
  if (i < 0) return toast("이미 옮겨졌거나 없는 파일이에요.", true);
  S.page = Math.floor(i / S.perPage);
  layoutGrid();
  selectImage(i);
}
async function applyFix(f) {
  if (f.fix === "caption") {
    try {
      await api("/api/caption/start", { folder: S.ds.folder, ...capSettings(), overwrite: false });
      $("#cap-start").disabled = true;
      toast("캡션 없는 이미지만 자동 캡션을 시작했어요 (아래 자동 캡션 막대에서 진행 상황이 보여요)");
      pollCaption();
    } catch (e) { toast(e.message, true); }
    return;
  }
  const what = f.fix === "facecrop" ? `${f.files.length}장에서 얼굴 클로즈업 사본을 만들까요? (원본은 그대로)`
    : f.fix === "augment" ? `${f.files.length}장에 화질 변형(노이즈·흔들림·저해상도·초점 흐림·필름·폰 셀카·밤 셀카, 절반은 색 필터도)을 건 사본을 만들까요? (원본은 그대로, 캡션 맨 앞에 효과를 적어요)`
    : `${f.files.length}장을 excluded 폴더로 옮길까요? (지우지 않아요, 나중에 다시 꺼낼 수 있어요)`;
  if (!confirm(what)) return;
  try {
    const r = await api("/api/diagnose/fix", { folder: S.ds.folder, action: f.fix, files: f.files });
    toast(r.message);
    await loadDataset();  // reloads the grid and re-runs the check-up
  } catch (e) { toast(e.message, true); }
}
$("#diag-rerun").addEventListener("click", startDiagnosis);
$("#ed-back").addEventListener("click", () => selectImage(-1));

async function saveCaption() {
  const it = S.ds.items[S.sel];
  try {
    await api("/api/caption/save", { path: it.path, caption: $("#ed-cap").value });
    it.caption = $("#ed-cap").value.trim();
    $("#ed-msg").textContent = "저장했어요";
    S.ds.captioned = S.ds.items.filter(x => x.caption).length;
    layoutGrid();
  } catch (e) { toast(e.message, true); }
}
// native Windows picker (opened by the local server, so the full path comes back)
async function pickPath(kind, initial, title) {
  try {
    const r = await api("/api/pick", { kind, initial: initial || "", title: title || "" });
    return r.path || "";
  } catch (e) { toast(e.message, true); return ""; }
}
$("#ds-load").addEventListener("click", async () => {
  const btn = $("#ds-load");
  btn.disabled = true; btn.textContent = "폴더 선택 창 열림…";
  const path = await pickPath("folder", $("#ds-folder").value, "데이터셋 폴더 선택");
  btn.disabled = false; btn.textContent = "폴더 선택";
  if (path) { $("#ds-folder").value = path; loadDataset(); }
});
$$(".pick-file").forEach(b => b.addEventListener("click", async () => {
  const input = $(`#${b.dataset.for}`);
  const path = await pickPath("file", input.value, b.dataset.title || "모델 파일 선택");
  if (path) { input.value = path; saveTrainSettings(); saveModelPaths(); }
}));
// model files: paths live in settings.json on the server (shared by training, captions, comparison)
const MODEL_INPUTS = { "tr-dit": "dit", "tr-idit": "inference_dit", "tr-te": "text_encoder", "tr-vae": "vae" };
function renderModelStatus(models) {
  Object.entries(MODEL_INPUTS).forEach(([id, key]) => {
    const m = models[key], input = $(`#${id}`);
    if (!m) return;
    const optional = key === "inference_dit" && !input.value.trim();
    input.classList.toggle("missing", !m.ok && !optional);
    input.placeholder = `경로 지정 (예: ComfyUI/${m.folder}/${m.file})`;
    input.title = m.ok ? m.path : (optional ? "비우면 학습용 DiT로 샘플을 그려요" : `${m.label} 파일이 없어요. 📁로 지정하거나 ⬇로 받으세요`);
    const a = $(`.dl[data-model="${key}"]`);
    a.href = m.url;
    a.title = `${m.label} 받기: ${m.file} (${m.size}) · Hugging Face. 받은 뒤 📁로 지정하세요`;
  });
}
async function saveModelPaths() {
  const body = { model: S.model || "moss_image" };
  Object.entries(MODEL_INPUTS).forEach(([id, key]) => (body[key] = $(`#${id}`).value));
  try {
    const r = await api("/api/settings/models", body);
    S.defaults.model_files[body.model] = r.models;
    renderModelStatus(r.models);
  } catch (e) { toast(e.message, true); }
}
// model choice: swaps the model-file fields (each model keeps its own paths) and the LoRA name prefix
const NAME_PREFIX = { moss_image: "OS_Moss_", krea2: "OS_Krea2_", qwen21: "OS_Qwen_" };
function selectModel(key, keepName = false) {
  const d = S.defaults;
  if (!d.model_files[key]) key = "moss_image";
  const prev = S.model;
  S.model = key;
  setSeg("tr-model", key);
  const files = d.model_files[key];
  Object.entries(MODEL_INPUTS).forEach(([id, k]) => ($(`#${id}`).value = files[k] ? files[k].path : ""));
  renderModelStatus(files);
  $("#tr-idit-label").textContent = `샘플용 DiT (${d.inference_labels[key] || "추론용"})`;
  $("#model-chip").textContent = d.model_list[key];
  const name = $("#tr-name");
  if (!keepName && prev && prev !== key && name.value.startsWith(NAME_PREFIX[prev] || "\u0000"))
    name.value = NAME_PREFIX[key] + name.value.slice(NAME_PREFIX[prev].length);
  try { localStorage.setItem("oslora.model", key); } catch { }
}
Object.keys(MODEL_INPUTS).forEach(id => $(`#${id}`).addEventListener("change", saveModelPaths));
$("#ds-folder").addEventListener("keydown", e => e.key === "Enter" && loadDataset());
$("#pg-prev").addEventListener("click", () => { if (S.page > 0) { S.page--; layoutGrid(); } });
$("#pg-next").addEventListener("click", () => { S.page++; layoutGrid(); });
$("#ed-save").addEventListener("click", saveCaption);
$("#ed-cap").addEventListener("keydown", e => { if (e.key === "s" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); saveCaption(); } });
document.addEventListener("keydown", e => {
  if (S.view !== "dataset" || !S.ds || document.activeElement.tagName === "TEXTAREA" || document.activeElement.tagName === "INPUT") return;
  if (e.key === "ArrowRight" && S.sel < S.ds.items.length - 1) { selectImage(S.sel + 1); }
  if (e.key === "ArrowLeft" && S.sel > 0) { selectImage(S.sel - 1); }
});
window.addEventListener("resize", () => { if (S.view === "dataset") layoutGrid(); if (S.view === "monitor") renderMonitor(); });
// the grid area also changes size without a window resize (e.g. the caption status line
// appearing under the auto-caption bar): re-fit the page whenever it does
let lastGridBox = "";
new ResizeObserver(() => {
  const w = $(".ds-gridwrap");
  const box = `${w.clientWidth}x${w.clientHeight}`;
  if (box !== lastGridBox && S.view === "dataset") { lastGridBox = box; layoutGrid(); }
}).observe($(".ds-gridwrap"));

// ------------------------------------------------------------------ captioning
bindSeg("cap-style"); bindSeg("cap-subject");
// auto-caption choices survive a page refresh (per-browser convenience only)
function capSettings() {
  return { model: $("#cap-model").value, style: segValue("cap-style"), subject: segValue("cap-subject"),
           trigger: $("#cap-trigger").value, overwrite: $("#cap-over").checked };
}
function saveCapSettings() { try { localStorage.setItem("oslora.caption", JSON.stringify(capSettings())); } catch { } }
function restoreCapSettings() {
  let s = null;
  try { s = JSON.parse(localStorage.getItem("oslora.caption") || "null"); } catch { }
  if (!s) return;
  if (s.model && [...$("#cap-model").options].some(o => o.value === s.model)) $("#cap-model").value = s.model;
  if (s.style) setSeg("cap-style", s.style);
  if (s.subject) setSeg("cap-subject", s.subject);
  if (typeof s.trigger === "string") $("#cap-trigger").value = s.trigger;
  $("#cap-over").checked = !!s.overwrite;
}
["#cap-model", "#cap-trigger", "#cap-over"].forEach(sel => $(sel).addEventListener("change", saveCapSettings));
$$("#cap-style button, #cap-subject button").forEach(b => b.addEventListener("click", saveCapSettings));

async function startCaption() {
  if (!S.ds) return toast("먼저 데이터셋 폴더를 불러오세요.", true);
  try {
    saveCapSettings();
    await api("/api/caption/start", { folder: S.ds.folder, ...capSettings() });
    $("#cap-start").disabled = true;
    pollCaption();
  } catch (e) { toast(e.message, true); }
}
async function pollCaption() {
  clearTimeout(S.capTimer);
  let st;
  try { st = await api("/api/caption/status"); } catch { return; }
  const box = $("#cap-status");
  if (st.phase === "error") {
    box.innerHTML = `<span style="color:var(--red)">${esc(st.message)}</span> ${esc((st.log || "").split("\n").filter(Boolean).slice(-1)[0] || "")}`;
    $("#cap-start").disabled = false; return;
  }
  if (st.total) {
    const pct = (100 * (st.done || 0)) / st.total;
    box.innerHTML = `<span class="bar"><span style="width:${pct}%"></span></span>${esc(st.message || "")}${st.caption ? ` — <i>${esc(st.caption)}</i>` : ""}`;
  } else box.textContent = st.message || "";
  if (st.phase === "done") {
    $("#cap-start").disabled = false;
    if (S.ds) loadDataset();
    toast(st.message || "캡션 완료");
    return;
  }
  if (st.running || st.phase === "loading" || st.phase === "captioning") S.capTimer = setTimeout(pollCaption, 1500);
  else $("#cap-start").disabled = false;
}
$("#cap-start").addEventListener("click", startCaption);
$("#cap-trigger").addEventListener("input", e => { if (!$("#tr-trigger").dataset.touched) $("#tr-trigger").value = e.target.value; });

// ------------------------------------------------------------------ training form
function applyPreset(key) {
  S.preset = key;
  S.trainKind = key === "style" ? "style" : "character";
  const p = S.defaults.presets[key];
  $$(".preset").forEach(b => b.classList.toggle("on", b.dataset.k === key));
  setSeg("tr-res", p.resolution);
  $("#tr-rank").value = p.rank; $("#tr-alpha").value = p.alpha;
  $("#tr-minlr").value = p.min_lr; $("#tr-maxlr").value = p.max_lr; $("#tr-startlr").value = p.start_lr;
  $("#tr-cdrop").value = p.caption_dropout; $("#tr-steps").value = p.max_steps;
  updateSummary();
}
function updateSummary() {
  const res = +segValue("tr-res");
  const n = S.ds ? S.ds.items.length - (S.ds.holdout || 0) : null;
  let steps = $("#tr-steps").value.trim();
  if (steps === "auto" && n) steps = String(Math.min(5000, Math.max(1500, Math.round(n * (n <= 40 ? 80 : 60)))));
  const sps = SEC_PER_STEP[res] || 8.8;
  const total = /^\d+$/.test(steps) ? +steps * sps * 1.12 : null;
  $("#tr-summary").innerHTML = `
    <div><span>학습 이미지</span><b>${n ?? "-"}</b></div>
    <div><span>최대 step</span><b>${steps}</b></div>
    <div><span>예상 속도</span><b>${sps}초/step</b></div>
    <div><span>최대 소요 (조기 종료 전)</span><b>${total ? fmtTime(total) : "-"}</b></div>`;
  $(".autopilot").classList.toggle("off", !$("#tr-ap").checked);
  $("#tr-fixed").hidden = $("#tr-ap").checked;
}
bindSeg("tr-res", updateSummary);
["#tr-steps", "#tr-ap"].forEach(s => $(s).addEventListener("input", updateSummary));
$("#tr-trigger").addEventListener("input", e => (e.target.dataset.touched = "1"));
async function resumeWithSettings() {
  try {
    await api("/api/train/resume_with", {
      run: S.pausedRun, int8_compute: false, photo_texture: $("#tr-photo").checked,
      min_lr: +$("#tr-minlr").value, max_lr: +$("#tr-maxlr").value, max_steps: $("#tr-steps").value.trim(),
      samples: $("#tr-samples").value.split(/^\s*---\s*$/m).map(s => s.trim()).filter(Boolean),
      sample_every: +$("#tr-severy").value || 500, sample_w: +$("#tr-sw").value, sample_h: +$("#tr-sh").value,
    });
    S.run = S.pausedRun;
    toast("바뀐 설정으로 이어서 학습해요. 해상도·rank·데이터셋은 새 학습에서만 바뀌어요");
    await refreshRuns();
    show("monitor");
  } catch (e) { $("#tr-msg").textContent = e.message; }
}
async function startTraining() {
  $("#tr-msg").textContent = "";
  if (S.pausedRun) return resumeWithSettings();
  if (!S.ds) { $("#tr-msg").textContent = "1번 데이터셋에서 폴더를 먼저 불러오세요."; return; }
  const body = {
    name: $("#tr-name").value.trim(), folder: S.ds.folder, trigger: $("#tr-trigger").value,
    model: S.model || "moss_image",
    resolution: +segValue("tr-res"), rank: +$("#tr-rank").value, alpha: +$("#tr-alpha").value,
    max_steps: $("#tr-steps").value.trim(), autopilot: $("#tr-ap").checked,
    min_lr: +$("#tr-minlr").value, max_lr: +$("#tr-maxlr").value, start_lr: +$("#tr-startlr").value,
    early_stop: $("#tr-early").checked,
    holdout: $("#tr-holdout").value, caption_dropout: +$("#tr-cdrop").value, photo_texture: $("#tr-photo").checked, int8_compute: false, multires: $("#tr-multires").checked, ema: +$("#tr-ema").value,
    optimizer: $("#tr-opt").value, lr: +$("#tr-lr").value,
    // several prompts are separated by a line containing only ---; line breaks inside a prompt stay
    samples: $("#tr-samples").value.split(/^\s*---\s*$/m).map(s => s.trim()).filter(Boolean),
    sample_every: +$("#tr-severy").value || 500,
    sample_w: +$("#tr-sw").value, sample_h: +$("#tr-sh").value,
    dit: $("#tr-dit").value, inference_dit: $("#tr-idit").value, text_encoder: $("#tr-te").value, vae: $("#tr-vae").value,
  };
  try {
    body.queue = !!S.busyTrain;  // something is running: queue it instead of refusing
    body.kind = S.trainKind || (S.preset === "style" ? "style" : "character");
    const r = await api("/api/train/start", body);
    if (r.queued) {
      toast(`${r.run}을(를) 대기열 ${r.position}번째에 넣었어요. 지금 학습이 끝나면 자동으로 시작해요`);
      refreshQueue();
      return;
    }
    S.run = r.run;
    toast(`${r.run} 학습을 시작했어요`);
    await refreshRuns();
    show("monitor");
  } catch (e) { $("#tr-msg").textContent = e.message; }
}
$("#tr-start").addEventListener("click", () => { saveTrainSettings(); startTraining(); });

// ------------------------------------------------------------------ training queue
async function refreshQueue(data) {
  let q = data;
  if (!q) { try { q = await api("/api/queue"); } catch { return; } }
  S.queue = q;
  const btn = $("#tr-queue");
  btn.hidden = !q.items.length && !q.paused;
  btn.textContent = `대기열 ${q.items.length}개${q.paused ? " (멈춤)" : ""}`;
  btn.classList.toggle("warn", q.paused);
  if (!$("#queue-modal").hidden) renderQueue();
}
function renderQueue() {
  const q = S.queue || { items: [], paused: false };
  $("#queue-state").textContent = q.paused ? "⏸ 멈춤 - 이전 학습이 오류나 중지로 끝났어요" : (q.items.length ? "대기 중" : "비어 있어요");
  $("#queue-resume").hidden = !q.paused;
  $("#queue-list").innerHTML = q.items.map((it, i) => `
    <li><span class="qn">${i + 1}</span>
      <div class="qm"><b>${esc(it.name)}</b><small>${esc(it.folder)} · ${it.resolution}px · 최대 ${esc(it.max_steps)} step</small></div>
      <div class="qa">
        <button class="btn ghost" data-op="up" data-i="${i}" ${i === 0 ? "disabled" : ""} aria-label="위로">↑</button>
        <button class="btn ghost" data-op="down" data-i="${i}" ${i === q.items.length - 1 ? "disabled" : ""} aria-label="아래로">↓</button>
        <button class="btn ghost" data-op="remove" data-i="${i}" aria-label="빼기">✕</button>
      </div></li>`).join("") || `<li class="hint">대기 중인 학습이 없어요. 학습 중에 [대기열에 추가]를 누르면 여기에 쌓여요.</li>`;
  $$("#queue-list [data-op]").forEach(b => b.addEventListener("click", async () => {
    refreshQueue(await api("/api/queue", { op: b.dataset.op, index: +b.dataset.i }));
  }));
}
$("#tr-queue").addEventListener("click", () => { $("#queue-modal").hidden = false; renderQueue(); });
$("#queue-close").addEventListener("click", () => ($("#queue-modal").hidden = true));
$("#queue-resume").addEventListener("click", async () => refreshQueue(await api("/api/queue", { op: "resume" })));
$("#queue-modal").addEventListener("click", e => { if (e.target.id === "queue-modal") $("#queue-modal").hidden = true; });

// ------------------------------------------------------------------ auto settings
async function openAuto(kind) {
  if (!S.ds) { $("#tr-msg").textContent = "1번 데이터셋에서 폴더를 먼저 불러오세요."; return; }
  kind = kind || (S.preset === "style" ? "style" : segValue("auto-kind") || "character");
  setSeg("auto-kind", kind);
  $("#auto-modal").hidden = false;
  $("#auto-apply").disabled = true;
  $("#auto-reasons").innerHTML = `<li class="hint">데이터셋을 살펴보는 중… (처음이면 10초쯤 걸려요)</li>`;
  try {
    S.auto = await api("/api/autoconfig", { folder: S.ds.folder, kind, trigger: $("#tr-trigger").value, model: S.model || "moss_image" });
  } catch (e) { $("#auto-reasons").innerHTML = `<li class="warn">${esc(e.message)}</li>`; return; }
  $("#auto-reasons").innerHTML = S.auto.reasons.map(r => {
    const warn = r.startsWith("⚠");
    const i = r.indexOf(": ");
    const body = !warn && i > 0 ? `<b>${esc(r.slice(0, i))}</b> ${esc(r.slice(i + 2))}` : esc(r);
    return `<li class="${warn ? "warn" : ""}">${body}</li>`;
  }).join("");
  $("#auto-apply").disabled = false;
}
function applyAuto() {
  const a = S.auto && S.auto.settings;
  if (!a) return;
  const preset = S.auto.kind === "style" ? "style" : "character";
  S.trainKind = S.auto.kind;
  S.preset = preset;
  $$(".preset").forEach(b => b.classList.toggle("on", b.dataset.k === preset));
  setSeg("tr-res", a.resolution);
  const set = (id, v) => { if (v !== undefined && v !== null) $(`#${id}`).value = v; };
  set("tr-name", a.name); set("tr-trigger", a.trigger); set("tr-rank", a.rank); set("tr-alpha", a.alpha);
  set("tr-startlr", a.start_lr); set("tr-minlr", a.min_lr); set("tr-maxlr", a.max_lr);
  set("tr-steps", a.max_steps); set("tr-holdout", a.holdout); set("tr-cdrop", a.caption_dropout);
  set("tr-sw", a.sample_w); set("tr-sh", a.sample_h); set("tr-severy", a.sample_every);
  set("tr-ema", a.ema); set("tr-opt", a.optimizer);
  $("#tr-ap").checked = !!a.autopilot; $("#tr-early").checked = !!a.early_stop;
  updateSummary();
  saveTrainSettings();
  $("#auto-modal").hidden = true;
  toast("자동 설정을 적용했어요. 확인하고 학습 시작을 누르세요");
}
$("#tr-auto").addEventListener("click", () => openAuto());
bindSeg("auto-kind", k => openAuto(k));
$("#auto-cancel").addEventListener("click", () => ($("#auto-modal").hidden = true));
$("#auto-apply").addEventListener("click", applyAuto);
$("#auto-modal").addEventListener("click", e => { if (e.target.id === "auto-modal") $("#auto-modal").hidden = true; });

// training form survives a page refresh / server restart (per-browser convenience only)
const TRAIN_FIELDS = ["tr-name", "tr-trigger", "tr-steps", "tr-rank", "tr-alpha", "tr-minlr", "tr-startlr", "tr-maxlr",
  "tr-holdout", "tr-ema", "tr-cdrop", "tr-opt", "tr-lr", "tr-samples", "tr-severy", "tr-sw", "tr-sh"];
const TRAIN_CHECKS = ["tr-ap", "tr-early", "tr-photo", "tr-multires"];
function saveTrainSettings() {
  const s = { preset: S.preset, res: segValue("tr-res"), samplesSig: S.defaults && S.defaults.samples_sig };
  TRAIN_FIELDS.forEach(id => (s[id] = $(`#${id}`).value));
  TRAIN_CHECKS.forEach(id => (s[id] = $(`#${id}`).checked));
  try { localStorage.setItem("oslora.train", JSON.stringify(s)); } catch { }
}
function restoreTrainSettings() {
  let s = null;
  try { s = JSON.parse(localStorage.getItem("oslora.train") || "null"); } catch { }
  if (!s) return;
  if (s.preset && S.defaults.presets[s.preset]) {
    S.preset = s.preset;
    $$(".preset").forEach(b => b.classList.toggle("on", b.dataset.k === s.preset));
  }
  if (s.res) setSeg("tr-res", s.res);
  // new default sample prompts/size (server update) replace the remembered ones once
  const fresh = s.samplesSig !== S.defaults.samples_sig ? ["tr-samples", "tr-sw", "tr-sh"] : [];
  if (["1.5e-4", "0.00015"].includes(s["tr-maxlr"])) fresh.push("tr-maxlr");  // old default: the cap is now 1e-4
  TRAIN_FIELDS.forEach(id => {
    if (!fresh.includes(id) && typeof s[id] === "string" && s[id] !== "") $(`#${id}`).value = s[id];
  });
  TRAIN_CHECKS.forEach(id => { if (typeof s[id] === "boolean") $(`#${id}`).checked = s[id]; });
  updateSummary();
}
$("#view-train").addEventListener("input", saveTrainSettings);
$("#view-train").addEventListener("change", saveTrainSettings);
$("#view-train").addEventListener("click", e => { if (e.target.closest(".seg button, .preset")) setTimeout(saveTrainSettings, 0); });

// ------------------------------------------------------------------ charts
function drawChart(canvas, series, opt = {}) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (!w || !h) return;
  canvas.width = w * dpr; canvas.height = h * dpr;
  const g = canvas.getContext("2d");
  g.scale(dpr, dpr);
  g.clearRect(0, 0, w, h);
  const pts = series.flatMap(s => s.data);
  if (pts.length < 2) {
    g.fillStyle = "#5f5873"; g.font = "12px Pretendard Variable, sans-serif"; g.textAlign = "center";
    g.fillText("데이터를 기다리는 중", w / 2, h / 2); return;
  }
  const pad = { l: 44, r: 8, t: 6, b: 16 };
  let x0 = Math.min(...pts.map(p => p[0])), x1 = Math.max(...pts.map(p => p[0]));
  if (opt.xmax) x1 = Math.max(x1, opt.xmax);
  let ys = pts.map(p => p[1]).filter(Number.isFinite);
  let y0 = Math.min(...ys), y1 = Math.max(...ys);
  if (opt.clip) { const sorted = [...ys].sort((a, b) => a - b); y1 = sorted[Math.floor(sorted.length * 0.98)]; y0 = sorted[Math.floor(sorted.length * 0.01)]; }
  if (opt.zero) y0 = Math.min(0, y0);
  if (y1 - y0 < 1e-12) { y1 += Math.abs(y1) * 0.05 + 1e-9; y0 -= Math.abs(y0) * 0.05 + 1e-9; }
  const span = y1 - y0; y0 -= span * 0.06; y1 += span * 0.06;
  const X = x => pad.l + ((x - x0) / Math.max(1, x1 - x0)) * (w - pad.l - pad.r);
  const Y = y => pad.t + (1 - (y - y0) / (y1 - y0)) * (h - pad.t - pad.b);
  g.strokeStyle = "#2f2840"; g.lineWidth = 1; g.fillStyle = "#5f5873"; g.font = "10px Pretendard Variable, sans-serif";
  for (let k = 0; k <= 2; k++) {
    const v = y0 + ((y1 - y0) * k) / 2, yy = Math.round(Y(v)) + 0.5;
    g.beginPath(); g.moveTo(pad.l, yy); g.lineTo(w - pad.r, yy); g.stroke();
    g.textAlign = "right"; g.fillText((opt.fmt || (v => v.toFixed(3)))(v), pad.l - 6, yy + 3);
  }
  g.textAlign = "left"; g.fillText(String(Math.round(x0)), pad.l, h - 3);
  g.textAlign = "right"; g.fillText(String(Math.round(x1)), w - pad.r, h - 3);
  for (const s of series) {
    if (s.data.length < 1) continue;
    g.strokeStyle = s.color; g.lineWidth = s.width || 1.6; g.globalAlpha = s.alpha ?? 1;
    g.beginPath();
    s.data.forEach((p, i) => { const fx = X(p[0]), fy = Y(Math.min(Math.max(p[1], y0), y1)); i ? g.lineTo(fx, fy) : g.moveTo(fx, fy); });
    g.stroke(); g.globalAlpha = 1;
    if (s.dots) {
      g.fillStyle = s.color;
      s.data.forEach(p => { g.beginPath(); g.arc(X(p[0]), Y(p[1]), 2.2, 0, 7); g.fill(); });
    }
    if (s.marks) {
      g.fillStyle = "#b69cff";
      s.marks.forEach(p => { g.beginPath(); g.arc(X(p[0]), Y(p[1]), 4, 0, 7); g.fill(); });
    }
  }
}
function ema(arr, a = 0.08) { let m = null; return arr.map(([x, y]) => [x, (m = m == null ? y : m + a * (y - m))]); }

// ------------------------------------------------------------------ monitor + results
async function refreshRun() {
  if (!S.run) return;
  try { S.detail = await api(`/api/runs/${encodeURIComponent(S.run)}`); } catch (e) { return; }
  if (S.view === "monitor") renderMonitor();
  if (S.view === "results") renderResults();
}
function renderMonitor() {
  const d = S.detail;
  if (!d) return;
  const st = d.status || {};
  $("#mo-name").textContent = d.name;
  const ph = $("#mo-phase");
  ph.textContent = st.message || PHASE[st.phase] || "";
  ph.classList.toggle("warn", st.phase === "error" || st.last_action === "overfit");
  $("#mo-step").textContent = st.step ?? 0;
  $("#mo-max").textContent = `/ ${st.max_steps ?? "-"}`;
  $("#mo-lr").textContent = fmtLR(st.lr);
  $("#mo-loss").textContent = st.loss != null ? st.loss.toFixed(4) : "-";
  $("#mo-eta").textContent = st.phase === "done" ? "완료" : fmtTime(st.eta_sec);
  $("#mo-best").textContent = st.best_step ? `step ${st.best_step}` : "-";
  const sk = $("#mo-strikes"); sk.textContent = `${st.strikes ?? 0} / 3`; sk.classList.toggle("warn", (st.strikes || 0) > 0);
  $("#mo-fill").style.width = st.max_steps ? `${(100 * (st.step || 0)) / st.max_steps}%` : "0";
  const running = (d.alive || ["training", "sampling", "caching", "starting", "paused"].includes(st.phase)) && st.phase !== "done" && st.phase !== "error";
  $("#mo-stop").hidden = !running;
  // pause takes effect after the current step: show "requested" until the trainer confirms
  const paused = st.phase === "paused";
  $("#mo-pause").hidden = !running || d.pause_requested || paused;
  $("#mo-resume").hidden = !running || !(d.pause_requested || paused);
  if (d.pause_requested && !paused && running) ph.textContent = "일시정지 요청됨 - 지금 하던 step(또는 평가/샘플)을 마치면 GPU를 비워요";
  if (paused && !d.pause_requested) ph.textContent = "재개하는 중 - 모델을 GPU로 다시 올리고 있어요";

  const steps = d.steps, evals = d.evals;
  // live runs show the whole planned range (progress at a glance); finished runs only what ran
  const live = d.alive || ["training", "sampling", "paused"].includes(st.phase);
  const xmax = live ? st.max_steps : undefined;
  const loss = steps.map(s => [s[0], s[1]]);
  drawChart($("#ch-loss"), [
    { data: loss, color: "#6f5bb3", alpha: 0.45, width: 1 },
    { data: ema(loss), color: "#b69cff", width: 2 },
  ], { clip: true, xmax });
  drawChart($("#ch-lr"), [{ data: steps.map(s => [s[0], s[2]]), color: "#7cc4ff", width: 2 }], { fmt: v => v.toExponential(1), xmax, zero: true });
  if (evals.length) {
    const f0 = evals[0].fit, v0 = evals[0].val;
    const fit = evals.map(e => [e.step, e.fit / f0]);
    const ser = [{ data: fit, color: "#b69cff", width: 2, dots: true, marks: evals.filter(e => e.best).map(e => [e.step, e.fit / f0]) }];
    if (v0) ser.push({ data: evals.map(e => [e.step, e.val / v0]), color: "#ffb86b", width: 2, dots: true });
    drawChart($("#ch-probe"), ser, { xmax });
    drawChart($("#ch-leak"), [{ data: evals.map(e => [e.step, e.leak]), color: "#ff6b81", width: 2, dots: true }], { fmt: v => v.toExponential(1), xmax, zero: true });
  } else {
    drawChart($("#ch-probe"), []); drawChart($("#ch-leak"), []);
  }
  $("#ch-probe").previousElementSibling.innerHTML = `프로브 loss <em>보라 = 학습 이미지${evals[0]?.val ? ", 주황 = 홀드아웃" : ""} (시작 = 1)</em>`;

  const entries = [
    ...evals.map(e => ({ step: e.step, order: 0, html: `
      <li class="${esc(e.action || "")} ${e.best ? "best" : ""}">
        <span class="js">${e.step}</span>
        <div><div class="jr">${esc(e.reason || (e.action === "baseline" ? "시작 기준점 측정" : e.action || ""))}</div>
        <div class="jv">fit ${e.fit?.toFixed(4)}${e.val != null ? `  val ${e.val.toFixed(4)}` : ""}  leak ${Number(e.leak).toExponential(1)}  LR ${fmtLR(e.lr)}</div></div>
      </li>` })),
    ...(d.quality || []).filter(q => !q.final).map(q => ({ step: q.step, order: 1, html: `
      <li class="quality ${/과적합/.test(q.note || "") ? "overfit" : ""}">
        <span class="js">${q.step}</span>
        <div><div class="jr">${esc(q.note || "샘플 품질 측정")}</div>
        <div class="jv">${q.likeness != null ? `닮음 ${q.likeness.toFixed(2)}${q.likeness_iqr != null ? `±${(q.likeness_iqr / 2).toFixed(2)}` : ""}  ` : ""}준수 ${q.adherence.toFixed(1)}  유사도 ${q.copy.toFixed(2)}${q.diversity != null ? `  다양성 ${q.diversity.toFixed(2)}` : ""}${q.aesthetic != null ? `  미적 ${q.aesthetic.toFixed(2)}` : ""}</div>
        ${flexLine(q)}</div>
      </li>` })),
  ].sort((a, b) => b.step - a.step || b.order - a.order);
  $("#journal").innerHTML = entries.map(e => e.html).join("") || `<li class="hint">첫 평가를 기다리는 중</li>`;

  renderSheet(d);
}
// flexibility checks: clean-look override (x1 = the base model's own look), attribute edit, identity bleed
function flexLine(q) {
  const parts = [];
  if (q.style_ratio != null) parts.push(`<span class="${q.style_ratio > 2 ? "warn" : ""}" title="흔들림·노이즈 프롬프트의 전체 선명도 ÷ LoRA 없을 때. ×1이 이상적, 높을수록 LoRA가 효과를 지우는 중">화질 덮어쓰기 ×${q.style_ratio.toFixed(2)}</span>`);
  if (q.edit_delta != null) parts.push(`<span class="${q.edit_delta < -2 ? "warn" : ""}" title="'짧은 금발 단발 + 안경' 요청을 얼마나 따르는지 (LoRA 없을 때 대비). 내려가면 LoRA가 요청을 덮어쓰는 중">유연성 ${q.edit_delta >= 0 ? "+" : ""}${q.edit_delta.toFixed(1)}</span>`);
  if (q.bleed != null) parts.push(`<span class="${q.bleed > 0.45 ? "warn" : ""}" title="'노년 남성'을 그렸을 때 그 얼굴이 캐릭터와 닮은 정도. 낮아야 정상">번짐 ${q.bleed.toFixed(2)}</span>`);
  return parts.length ? `<div class="jv flex">${parts.join("  ")}</div>` : "";
}
function renderSheet(d) {
  const byStep = new Map();
  d.samples.forEach(s => { if (!byStep.has(s.step)) byStep.set(s.step, []); byStep.get(s.step)[s.prompt] = s.path; });
  const steps = [...byStep.keys()].sort((a, b) => a - b);
  const best = d.status?.best_step;
  // once a run is done its last samples are rendered with the selected (recommended) weights
  const finalStep = d.status?.phase === "done" ? steps[steps.length - 1] : null;
  const label = step => step === 0 ? "베이스" : step === finalStep ? `최종 (추천, step ${best})` : `step ${step}`;
  $("#sheet-hint").textContent = steps.length ? `열 = step, 행 = 샘플 프롬프트. 클릭하면 크게 봐요.` : "";
  // quality per column: the final render of the selected weights is logged with final=true
  const qByStep = new Map();
  (d.quality || []).forEach(q => qByStep.set(q.step, q));
  const q0 = qByStep.get(0);
  // quality checks run on their own schedule: label each column with the latest check at or before it
  const qSteps = [...qByStep.keys()].sort((a, b) => a - b);
  const qAt = step => { let hit; for (const s of qSteps) { if (s <= step) hit = s; else break; } return hit == null ? null : qByStep.get(hit); };
  const qLine = step => {
    const q = qAt(step);
    if (!q) return "";
    const bad = /과적합|저하/.test(q.note || "");
    return `<span class="sc-q ${bad ? "bad" : ""}">${q.likeness != null ? `닮음 <b>${q.likeness.toFixed(2)}</b> ` : ""}준수 <b>${q.adherence.toFixed(1)}</b></span>`;
  };
  const qs = [...qByStep.values()].filter(q => q.step > 0);
  const ql = qs[qs.length - 1];
  // adherence always moves a little once the face changes, so its delta is shown neutral
  const delta = (a, b, digits, goodUp = true, neutral = false) => {
    const dv = a - b, cls = neutral ? "" : (dv >= 0) === goodUp ? "up" : "down";
    return `<span class="${cls}">${dv >= 0 ? "+" : ""}${dv.toFixed(digits)}</span>`;
  };
  $("#qsum").innerHTML = ql && q0 ? `
    ${ql.likeness != null ? `<span>닮음 <b>${ql.likeness.toFixed(2)}</b> ${delta(ql.likeness, q0.likeness, 2)}</span>` : ""}
    <span>프롬프트 준수 <b>${ql.adherence.toFixed(1)}</b> ${delta(ql.adherence, q0.adherence, 1, true, true)}</span>
    <span>학습 이미지 유사도 <b>${ql.copy.toFixed(2)}</b></span>
    ${ql.diversity != null ? `<span>다양성 <b>${ql.diversity.toFixed(2)}</b></span>` : ""}
    ${ql.style_ratio != null ? `<span title="×1이 이상적">화질 덮어쓰기 <b>×${ql.style_ratio.toFixed(2)}</b></span>` : ""}
    ${ql.edit_delta != null ? `<span>유연성 <b>${ql.edit_delta >= 0 ? "+" : ""}${ql.edit_delta.toFixed(1)}</b></span>` : ""}
    ${ql.bleed != null ? `<span>번짐 <b>${ql.bleed.toFixed(2)}</b></span>` : ""}` : "";
  $("#sheet").innerHTML = steps.map(step => `
    <div class="sheet-col ${step === finalStep ? "best" : ""}">
      <div class="sc-step">${label(step)}${qLine(step)}</div>
      ${(byStep.get(step) || []).map((p, i) => p ? `<img src="${thumb(p, 480)}" data-full="${esc(p)}" data-cap="${esc(`${label(step)} · ${d.sample_prompts[i] || ""}`)}" alt="">` : "").join("")}
    </div>`).join("") || `<div class="empty small">샘플 프롬프트가 있으면 학습 중 주기적으로 여기에 쌓여요.</div>`;
  const body = $("#sheet");
  body.scrollLeft = body.scrollWidth;
  $$("#sheet img").forEach(img => img.addEventListener("click", () => openLightbox(img.dataset.full, img.dataset.cap)));
}
function renderContinue(d, st) {
  const box = $("#re-continue");
  const pts = d.resume_points || [];
  const busy = d.alive || ["training", "sampling", "caching", "starting", "paused"].includes(st.phase);
  box.hidden = !pts.length || busy;
  if (box.hidden) return;
  // the results view re-renders every few seconds: only rebuild the list when it changes
  const key = d.name + "|" + pts.map(p => `${p.step}${p.exact ? "e" : ""}`).join(",");
  if (S.resumeKey !== key) {
    S.resumeKey = key;
    $("#re-step").innerHTML = pts.map(p =>
      `<option value="${p.step}">step ${p.step}${p.exact ? " (멈춘 지점, 정확히 이어짐)" : " (체크포인트)"}</option>`).join("");
    $("#re-max").value = Math.max(st.max_steps || 0, pts[0].step + 500);
  }
  const p = pts.find(x => x.step === +$("#re-step").value) || pts[0];
  $("#re-cont-hint").textContent = p.exact
    ? "옵티마이저와 Autopilot 상태까지 그대로 이어서 학습해요."
    : "이 체크포인트의 LoRA에서 다시 시작해요. 옵티마이저는 새로 만들어서 처음 몇십 step은 천천히 다시 올라가요. 이 step 이후에 만들어진 파일은 별도 폴더로 옮겨져요(삭제 안 함).";
}
$("#re-step").addEventListener("change", () => S.detail && renderContinue(S.detail, S.detail.status || {}));
$("#re-go").addEventListener("click", async () => {
  if (!S.run) return;
  const step = +$("#re-step").value, max = +$("#re-max").value || null;
  if (!confirm(`${S.run}을(를) step ${step}부터 최대 ${max} step까지 이어서 학습할까요?`)) return;
  try {
    await api("/api/train/continue", { run: S.run, step, max_steps: max, int8_compute: false });
    toast(`step ${step}부터 이어서 학습을 시작했어요`);
    S.resumeKey = null;
    show("monitor");
  } catch (e) { toast(e.message, true); }
});

// ------------------------------------------------------------------ checkpoint comparison
bindSeg("re-mode", () => renderResults());
function renderCompare(d) {
  // checkpoint chips: rebuilt only when the run's list changes (keeps the user's ticks)
  const cks = [{ key: "base", label: "베이스" }, ...(d.checkpoints || [])];
  const key = d.name + "|" + cks.map(c => c.key).join(",");
  if (S.cmpKey !== key) {
    S.cmpKey = key;
    $("#cmp-cks").innerHTML = cks.map(c =>
      `<label><input type="checkbox" value="${esc(c.key)}" checked>${esc(c.label)}</label>`).join("");
    $("#cmp-prompts").value = (d.sample_prompts || []).join("\n---\n");
  }
  // the run's model: what its inference model is called, and Turbo's fixed step count
  const model = (d.config || {}).model || "moss_image", label = S.defaults.inference_labels[model] || "추론용";
  $("#cmp-model").options[0].textContent = `${label} (실사용)`;
  $$(".idit-name").forEach(e => (e.textContent = label));
  const fixed = (S.defaults.fixed_steps || {})[model], onInf = $("#cmp-model").value === "sft";
  $("#cmp-steps").disabled = !!(fixed && onInf);
  if (fixed && onInf) $("#cmp-steps").value = fixed;
  const busy = S.busyTrain;
  $("#cmp-start").disabled = !!busy || S.cmpRunning;
  $("#cmp-msg").textContent = busy ? "학습 중에는 GPU가 비지 않아서 비교할 수 없어요. 끝난 뒤 눌러주세요." : "";
  const now = Date.now();
  if (!S.cmpFetched || now - S.cmpFetched > 2500) { S.cmpFetched = now; loadCompare(d.name); }
}
async function loadCompare(run) {
  let r;
  try { r = await api(`/api/compare/${encodeURIComponent(run)}`); } catch { return; }
  const res = r.result;
  S.cmpRunning = !!res && (res.status === "running" || res.status === "preparing");
  if (!res) return;
  const done = res.done || 0, total = res.total || 0;
  const eta = res.eta ? ` · 남은 시간 약 ${fmtTime(res.eta)}` : "";
  $("#re-cap").textContent = S.cmpRunning ? `${res.message} (${done}/${total || "?"})${eta}` : (res.message || "");
  const best = res.columns.reduce((b, c, i) => (c.likeness != null && (b < 0 || c.likeness > res.columns[b].likeness)) ? i : b, -1);
  const head = res.columns.map((c, i) => `<th class="${i === best ? "top" : ""}"><b>${esc(c.label)}${i === best ? " ★" : ""}</b>
      ${c.likeness != null ? `닮음 ${c.likeness.toFixed(2)} · 준수 ${c.adherence.toFixed(1)}` : ""}
      ${c.file && res.status === "done" && c.label !== "추천" ? `<br><button class="btn" data-promote="${esc(c.file)}">추천 파일로</button>` : ""}</th>`).join("");
  const rows = res.prompts.map((p, pi) => `<tr><td class="rowh" title="${esc(p)}">프롬프트 ${pi + 1}</td>${res.columns.map((c, ci) => {
    const cell = res.cells[pi][ci];
    if (!cell) return `<td class="pending">${S.cmpRunning ? "그리는 중…" : "-"}</td>`;
    const path = `${res.dir}\\${cell.img}`;
    return `<td class="${ci === best ? "top" : ""}"><img src="${thumb(path, 520)}" data-full="${esc(path)}" data-cap="${esc(c.label + " · " + p)}" alt="">
      ${cell.likeness != null ? `<div class="sc">닮음 <b>${cell.likeness.toFixed(2)}</b> · 준수 <b>${cell.adherence.toFixed(1)}</b></div>` : ""}</td>`;
  }).join("")}</tr>`).join("");
  $("#cmp-table").innerHTML = `<table><thead><tr><th></th>${head}</tr></thead><tbody>${rows}</tbody></table>`;
  $$("#cmp-table img").forEach(img => img.addEventListener("click", () => openLightbox(img.dataset.full, img.dataset.cap)));
  $$("#cmp-table [data-promote]").forEach(b => b.addEventListener("click", async () => {
    if (!confirm(`${b.dataset.promote}을(를) 추천 파일로 바꿀까요? 지금 추천 파일은 백업해 둬요.`)) return;
    const col = res.columns.find(c => c.file === b.dataset.promote) || {};
    const top = best >= 0 ? res.columns[best].likeness : null;
    try { toast((await api("/api/promote", { run, file: b.dataset.promote, picked_likeness: col.likeness ?? null, top_likeness: top })).message); refreshRun(); }
    catch (e) { toast(e.message, true); }
  }));
  if (S.cmpRunning) $("#cmp-start").disabled = true;
}
$("#cmp-start").addEventListener("click", async () => {
  if (!S.run) return;
  const checkpoints = $$("#cmp-cks input:checked").map(i => i.value);
  if (!checkpoints.length) return toast("비교할 체크포인트를 하나 이상 고르세요.", true);
  const prompts = $("#cmp-prompts").value.split(/^\s*---\s*$/m).map(s => s.trim()).filter(Boolean);
  const est = checkpoints.length * prompts.length;
  if (!confirm(`${checkpoints.length}개 체크포인트 × 프롬프트 ${prompts.length}개 = ${est}장을 그려요 (장당 약 1분). 시작할까요?`)) return;
  try {
    await api("/api/compare/start", {
      run: S.run, prompts, checkpoints, seed: +$("#cmp-seed").value || 42,
      strength: +$("#cmp-strength").value || 1, steps: +$("#cmp-steps").value || 24, model: $("#cmp-model").value,
      width: +$("#tr-sw").value || 768, height: +$("#tr-sh").value || 1024,
    });
    S.cmpRunning = true; S.cmpFetched = 0;
    toast("비교 이미지를 그리기 시작했어요. 다 그린 칸부터 차례로 채워져요.");
    refreshRun();
  } catch (e) { toast(e.message, true); }
});

function renderResults() {
  const d = S.detail;
  if (!d) return;
  const st = d.status || {};
  $("#re-title").textContent = d.name;
  $("#re-reason").textContent = st.phase === "done" ? `${st.message || "완료"}. 추천 파일은 step ${st.best_step} 시점이에요.` : `아직 진행 중이에요 (${PHASE[st.phase] || st.phase}).`;
  const kind = { best: "★ 추천 (가장 점수가 좋았던 시점)", last: "마지막 step", checkpoint: "중간 저장" };
  $("#re-files").innerHTML = d.files.sort((a, b) => (a.kind === "best" ? -1 : b.kind === "best" ? 1 : a.name.localeCompare(b.name)))
    .map(f => `<li class="${f.kind}"><span class="fn">${esc(f.name)}${f.tag ? ` <em class="tag">★ ${esc(f.tag)}</em>` : ""}</span><span class="fs">${f.mb} MB</span><span class="fk">${kind[f.kind]}</span>${st.phase === "done" ? `<button class="btn ghost used" data-used="${esc(f.name)}" title="ComfyUI 등에서 실제로 이 파일을 쓰셨다면 눌러주세요. 추천 파일로 바꾸고, 다음 추천이 취향에 맞춰져요">이걸 썼어요</button>` : ""}</li>`).join("")
    || `<li class="hint">학습이 끝나면 파일이 생겨요.</li>`;
  $$("#re-files [data-used]").forEach(b => b.addEventListener("click", async () => {
    try { toast((await api("/api/promote", { run: d.name, file: b.dataset.used })).message); refreshRun(); }
    catch (e) { toast(e.message, true); }
  }));
  renderContinue(d, st);
  const mode = segValue("re-mode") || "samples";
  $("#re-grid").hidden = mode !== "samples";
  $("#re-cmp").hidden = mode !== "compare";
  if (mode === "compare") { renderCompare(d); return; }
  const steps = [...new Set(d.samples.map(s => s.step))].sort((a, b) => a - b);
  const last = steps[steps.length - 1];
  const loraLabel = st.phase === "done" ? `LoRA 추천 파일 (step ${st.best_step})` : `LoRA (step ${last})`;
  const cells = [];
  if (steps.length > 1) {
    d.sample_prompts.forEach((p, i) => {
      const b = d.samples.find(s => s.step === 0 && s.prompt === i), f = d.samples.find(s => s.step === last && s.prompt === i);
      if (b) cells.push(`<figure><img src="${thumb(b.path, 900)}" data-full="${esc(b.path)}" data-cap="베이스 모델 · ${esc(p)}"><figcaption>베이스 모델</figcaption></figure>`);
      if (f) cells.push(`<figure class="lora"><img src="${thumb(f.path, 900)}" data-full="${esc(f.path)}" data-cap="${esc(loraLabel)} · ${esc(p)}"><figcaption>${esc(loraLabel)}</figcaption></figure>`);
    });
  }
  $("#re-cap").textContent = cells.length ? "같은 시드에서 베이스 모델과 LoRA 적용 결과를 나란히 비교해요." : "";
  $("#re-grid").innerHTML = cells.join("") || `<div class="empty small">샘플이 생기면 베이스 모델과 비교해서 보여줘요.</div>`;
  $$("#re-grid img").forEach(img => img.addEventListener("click", () => openLightbox(img.dataset.full, img.dataset.cap)));
}
$("#mo-stop").addEventListener("click", async () => {
  if (!S.run || !confirm(`${S.run} 학습을 중지할까요? 지금까지 가장 좋았던 시점은 저장됩니다.`)) return;
  try { await api("/api/train/stop", { run: S.run }); toast("중지 요청을 보냈어요. 현재 step을 마치고 저장한 뒤 멈춥니다."); } catch (e) { toast(e.message, true); }
});
$("#mo-pause").addEventListener("click", async () => {
  if (!S.run) return;
  try {
    await api("/api/train/pause", { run: S.run });
    toast("일시정지를 요청했어요. 지금 하던 step을 마치면 GPU를 비우고 기다려요.");
    refreshRun();
  } catch (e) { toast(e.message, true); }
});
$("#mo-resume").addEventListener("click", async () => {
  if (!S.run) return;
  try {
    await api("/api/train/resume", { run: S.run });
    toast("다시 시작해요. 모델을 GPU로 올린 뒤 같은 지점부터 이어서 학습해요.");
    refreshRun();
  } catch (e) { toast(e.message, true); }
});
$("#re-open").addEventListener("click", () => S.run && api("/api/open", { run: S.run }).catch(e => toast(e.message, true)));

// ------------------------------------------------------------------ system meters
function setMeter(id, frac, text, hot) {
  const m = $(id);
  m.querySelector("i").style.width = `${Math.max(0, Math.min(1, frac)) * 100}%`;
  m.querySelector(".mv").textContent = text;
  m.classList.toggle("hot", !!hot);
}
async function refreshSystem() {
  let s;
  try { s = await api("/api/system"); } catch { return; }
  setMeter("#m-cpu", s.cpu / 100, `${Math.round(s.cpu)}%`, s.cpu > 90);
  setMeter("#m-ram", s.ram_used / s.ram_total, `${s.ram_used.toFixed(1)}/${Math.round(s.ram_total)}GB`, s.ram_used / s.ram_total > 0.9);
  if (s.vram_total) {
    setMeter("#m-vram", s.vram_used / s.vram_total, `${s.vram_used.toFixed(1)}/${Math.round(s.vram_total)}GB`, s.vram_used / s.vram_total > 0.92);
    setMeter("#m-gpu", s.gpu / 100, `${s.gpu}%  ${s.gpu_temp}°C`, s.gpu_temp >= 83);
    $("#m-gpu").title = s.gpu_name || "";
    if (s.power_limit_w) {
      $("#m-pwr").hidden = false;
      setMeter("#m-pwr", s.power_w / s.power_limit_w, `${Math.round(s.power_w)}/${Math.round(s.power_limit_w)}W`, false);
    }
  } else {
    $("#m-vram").hidden = $("#m-gpu").hidden = true;
  }
}

// ------------------------------------------------------------------ lightbox
function openLightbox(path, cap) {
  // captions are "label · prompt": keep the label and the prompt's opening words on one line,
  // the whole prompt on hover (sample prompts can run to hundreds of words)
  const text = cap || "";
  const [label, ...rest] = text.split(" · ");
  const prompt = rest.join(" · ");
  const head = prompt.length > 90 ? prompt.slice(0, 90).replace(/\s+\S*$/, "") + "…" : prompt;
  $("#lb-img").src = full(path);
  $("#lb-cap").textContent = prompt ? `${label} · ${head}` : label;
  $("#lb-cap").title = text;
  $("#lightbox").hidden = false;
}
$("#lightbox").addEventListener("click", () => ($("#lightbox").hidden = true));
document.addEventListener("keydown", e => { if (e.key === "Escape") { $("#lightbox").hidden = true; $("#auto-modal").hidden = true; $("#queue-modal").hidden = true; } });

// ------------------------------------------------------------------ boot
async function boot() {
  S.defaults = await api("/api/defaults");
  const d = S.defaults;
  $("#cap-model").innerHTML = Object.entries(d.caption_models).map(([k, v]) => `<option value="${k}">${esc(v)}</option>`).join("");
  restoreCapSettings();
  $("#tr-presets").innerHTML = Object.entries(d.presets).map(([k, p]) =>
    `<button class="preset" data-k="${k}"><b>${esc(p.label)}</b><small>${esc(p.hint)}</small></button>`).join("");
  $$(".preset").forEach(b => b.addEventListener("click", () => applyPreset(b.dataset.k)));
  $("#tr-samples").value = d.samples.join("\n---\n");
  $("#tr-sw").value = d.sample_size[0]; $("#tr-sh").value = d.sample_size[1];
  $("#tr-dit").value = d.paths.dit; $("#tr-idit").value = d.paths.inference_dit || ""; $("#tr-te").value = d.paths.text_encoder; $("#tr-vae").value = d.paths.vae;
  $("#tr-model").innerHTML = Object.entries(d.model_list).map(([k, v]) => `<button data-v="${k}">${esc(v)}</button>`).join("");
  $$("#tr-model button").forEach(b => b.addEventListener("click", () => { selectModel(b.dataset.v); saveTrainSettings(); }));
  let savedModel = "moss_image";
  try { savedModel = localStorage.getItem("oslora.model") || "moss_image"; } catch { }
  selectModel(savedModel, true);
  applyPreset("character");
  restoreTrainSettings();
  const needed = Object.values(d.models).filter(m => !m.ok && m !== d.models.inference_dit);
  if (needed.length) toast(`모델 파일 ${needed.length}개가 아직 지정되지 않았어요. 학습 설정 > 모델 파일에서 📁로 지정하거나 ⬇로 받으세요.`, true);
  let saved = null;
  try { saved = new URLSearchParams(location.search).get("folder") || localStorage.getItem("oslora.folder"); } catch { }
  if (saved) { $("#ds-folder").value = saved; await loadDataset(); }
  await refreshRuns();
  const params = new URLSearchParams(location.search);
  if (params.get("run")) S.run = params.get("run");
  if (params.get("re") === "compare") setSeg("re-mode", "compare");
  if (location.hash.length > 1) show(location.hash.slice(1));
  if (d.busy.caption) pollCaption();
  setInterval(refreshRuns, 5000);
  refreshSystem();
  setInterval(refreshSystem, 2000);
  setInterval(() => { if (S.view === "monitor" || S.view === "results") refreshRun(); }, 4000);
}
boot().catch(e => toast("서버에 연결할 수 없어요: " + e.message, true));
