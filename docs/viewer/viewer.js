/* MazeRunner trace viewer.
 *
 * Serverless: index.json is the eager manifest; each attempt lazy-loads its
 * own JSON. The replayer walks the submitted polyline by arclength — the same
 * logic the promo renderer uses — so playback speed reflects drag distance,
 * not point density. All drawing is done in task-pixel space on an offscreen
 * transform so zoom cannot desynchronize overlay and image (the promo's
 * aspect-ratio drift bug, fixed once, stays fixed here by construction).
 */

"use strict";

const $ = (id) => document.getElementById(id);
const DATA = "data";

const state = {
  manifest: null,
  filtered: [],
  selected: null,     // attempt summary
  attempt: null,      // full payload
  task: null,
  images: {},         // input, mask
  frac: 1.0,          // replay position 0..1
  playing: false,
  speed: 1,
  zoomed: false,
  lengths: null,      // cumulative arclengths of submission
};

/* ---------------- data ---------------- */

async function boot() {
  state.manifest = await (await fetch(`${DATA}/index.json`)).json();
  const families = new Set(), modes = new Set();
  for (const t of Object.values(state.manifest.tasks)) families.add(t.family);
  for (const a of state.manifest.attempts) if (a.fm && !a.ok) modes.add(a.fm);
  fill($("f-model"), state.manifest.models.map((m) => [m.id, m.name]));
  fill($("f-family"), [...families].sort().map((f) => [f, f]));
  fill($("f-mode"), [...modes].sort().map((m) => [m, m.replaceAll("_", " ")]));
  for (const el of document.querySelectorAll("#filters select"))
    el.addEventListener("change", applyFilters);
  applyFilters();
  const hash = location.hash.slice(1);
  if (hash) selectByKey(hash.replaceAll("/", "--"));
}

function fill(sel, pairs) {
  for (const [value, label] of pairs) {
    const o = document.createElement("option");
    o.value = value; o.textContent = label;
    sel.appendChild(o);
  }
}

function applyFilters() {
  const model = $("f-model").value, outcome = $("f-outcome").value;
  const family = $("f-family").value, tier = $("f-tier").value;
  const mode = $("f-mode").value, sort = $("f-sort").value;
  const tasks = state.manifest.tasks;
  let rows = state.manifest.attempts.filter((a) => {
    const t = tasks[a.m];
    return (!model || a.p === model)
        && (!outcome || String(a.ok) === outcome)
        && (!family || t.family === family)
        && (!tier || t.tier === tier)
        && (!mode || a.fm === mode);
  });
  const key = { "rp-desc": (a) => -a.rp, "rp-asc": (a) => a.rp, "lat-desc": (a) => -(a.lat || 0) }[sort];
  rows.sort((x, y) => key(x) - key(y));
  state.filtered = rows.slice(0, 400);
  $("count").textContent =
    `${rows.length} attempts` + (rows.length > 400 ? " · showing 400" : "");
  renderList();
}

function renderList() {
  const list = $("list");
  list.textContent = "";
  const tasks = state.manifest.tasks;
  for (const a of state.filtered) {
    const row = document.createElement("div");
    row.className = "row" + (state.selected?.k === a.k ? " sel" : "");
    const t = tasks[a.m];
    row.innerHTML =
      `<span class="badge ${a.ok ? "win" : "fail"}">${a.ok ? "WIN" : "FAIL"}</span>` +
      `<span class="who">${modelName(a.p)}<small>${a.m} · ${t.tier} · trial ${a.t + 1}` +
      `${a.fm && !a.ok ? " · " + a.fm.replaceAll("_", " ") : ""}</small></span>` +
      `<span class="rp">${(a.rp * 100).toFixed(0)}%</span>`;
    row.onclick = () => select(a);
    list.appendChild(row);
  }
}

const modelName = (p) =>
  state.manifest.models.find((m) => m.id === p)?.name ?? p;

function selectByKey(key) {
  const a = state.manifest.attempts.find((x) => x.k === key);
  if (a) select(a);
}

async function select(a) {
  state.selected = a;
  location.hash = a.k.replaceAll("--", "/");
  renderList();
  const [attempt, task] = await Promise.all([
    fetch(`${DATA}/attempts/${a.k}.json`).then((r) => r.json()),
    fetch(`${DATA}/tasks/${a.m}/task.json`).then((r) => r.json()),
  ]);
  state.attempt = attempt;
  state.task = task;
  state.images.input = await loadImage(`${DATA}/tasks/${a.m}/input.png`);
  state.images.mask = await loadImage(`${DATA}/tasks/${a.m}/mask.png`);
  state.lengths = cumulative(pointsPx());
  state.frac = 1.0; state.playing = false; state.zoomed = false;
  $("scrub").value = 1000;
  $("empty").style.display = "none";
  $("zoom").disabled = !attempt.evaluation?.first_collision;
  renderSide();
  draw();
}

const loadImage = (src) => new Promise((ok, err) => {
  const im = new Image(); im.onload = () => ok(im); im.onerror = err; im.src = src;
});

/* ---------------- geometry ---------------- */

function pointsPx() {
  const { task, attempt } = state;
  const pts = attempt?.submission?.points ?? [];
  return pts.map((p) => [p.x * (task.width - 1), p.y * (task.height - 1)]);
}

function cumulative(pts) {
  const out = [0];
  for (let i = 1; i < pts.length; i++)
    out.push(out[i - 1] + Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]));
  return out;
}

/* Points along the polyline up to `frac` of total arclength. */
function walk(pts, lens, frac) {
  if (pts.length < 2) return pts;
  const target = lens[lens.length - 1] * frac;
  const out = [pts[0]];
  for (let i = 1; i < pts.length; i++) {
    if (lens[i] >= target) {
      const seg = lens[i] - lens[i - 1];
      const t = seg > 0 ? (target - lens[i - 1]) / seg : 0;
      out.push([
        pts[i - 1][0] + (pts[i][0] - pts[i - 1][0]) * t,
        pts[i - 1][1] + (pts[i][1] - pts[i - 1][1]) * t,
      ]);
      return out;
    }
    out.push(pts[i]);
  }
  return out;
}

/* ---------------- drawing ---------------- */

function draw() {
  const cv = $("cv"), { task } = state;
  if (!task) return;
  cv.width = task.width; cv.height = task.height;
  const ctx = cv.getContext("2d");

  // Zoom is a pure canvas transform: image and overlay share it, so they
  // cannot drift apart. Aspect ratio is preserved by using one scale factor.
  ctx.save();
  if (state.zoomed && state.attempt.evaluation?.first_collision) {
    const c = state.attempt.evaluation.first_collision;
    const half = 150 * Math.min(1, task.width / task.height);
    const scale = Math.min(task.width, task.height) / (2 * half);
    const cx = clamp(c.x_px, half, task.width - half);
    const cy = clamp(c.y_px, half, task.height - half);
    ctx.setTransform(scale, 0, 0, scale, task.width / 2 - cx * scale, task.height / 2 - cy * scale);
  }

  ctx.drawImage(state.images.input, 0, 0);

  if ($("t-mask").checked && state.images.mask) {
    ctx.save();
    ctx.globalAlpha = 0.35;
    ctx.globalCompositeOperation = "screen";
    ctx.drawImage(state.images.mask, 0, 0);
    ctx.restore();
  }

  if ($("t-ref").checked) {
    const ref = task.reference_path.map((p) => [p.x * (task.width - 1), p.y * (task.height - 1)]);
    stroke(ctx, ref, "rgba(52,211,153,0.9)", 2, [6, 5]);
  }

  if ($("t-radii").checked) {
    circle(ctx, task.start.x * (task.width - 1), task.start.y * (task.height - 1),
           task.start_radius_px, "rgba(34,211,238,0.8)");
    circle(ctx, task.goal.x * (task.width - 1), task.goal.y * (task.height - 1),
           task.goal_radius_px, "rgba(245,158,11,0.8)");
  }

  const pts = pointsPx();
  const drawn = walk(pts, state.lengths, state.frac);
  if (drawn.length >= 2) {
    const win = state.attempt.evaluation?.success;
    stroke(ctx, drawn, "rgba(255,255,255,0.95)", 7);
    stroke(ctx, drawn, state.frac < 1 || win ? "#22D3EE" : "#F84454", 3.5);
    const head = drawn[drawn.length - 1];
    dot(ctx, head[0], head[1], 5, "#FFFFFF");
  }

  const col = state.attempt.evaluation?.first_collision;
  if (col && state.frac >= 1) {
    circle(ctx, col.x_px, col.y_px, 12, "#F84454", 3.5);
    ctx.strokeStyle = "#F84454"; ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(col.x_px - 8, col.y_px - 8); ctx.lineTo(col.x_px + 8, col.y_px + 8);
    ctx.moveTo(col.x_px - 8, col.y_px + 8); ctx.lineTo(col.x_px + 8, col.y_px - 8);
    ctx.stroke();
  }
  ctx.restore();
}

const clamp = (v, lo, hi) => Math.min(Math.max(v, lo), hi);

function stroke(ctx, pts, style, width, dash = []) {
  ctx.save();
  ctx.strokeStyle = style; ctx.lineWidth = width;
  ctx.lineJoin = "round"; ctx.lineCap = "round";
  ctx.setLineDash(dash);
  ctx.beginPath();
  ctx.moveTo(pts[0][0], pts[0][1]);
  for (const [x, y] of pts.slice(1)) ctx.lineTo(x, y);
  ctx.stroke();
  ctx.restore();
}

function circle(ctx, x, y, r, style, width = 2.5) {
  ctx.save();
  ctx.strokeStyle = style; ctx.lineWidth = width;
  ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.stroke();
  ctx.restore();
}

function dot(ctx, x, y, r, style) {
  ctx.save();
  ctx.fillStyle = style;
  ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.fill();
  ctx.restore();
}

/* ---------------- side panel ---------------- */

function renderSide() {
  const a = state.attempt, ev = a.evaluation || {}, d = a.derived || {};
  $("v-title").textContent = `${a.model_name} · ${a.maze}`;
  $("v-meta").textContent =
    `trial ${a.trial + 1} · ${state.task.family}/${state.task.tier}/${state.task.archetype}` +
    ` · route progress ${(d.route_progress * 100 || 0).toFixed(0)}%` +
    (a.latency_s ? ` · ${a.latency_s}s` : "");
  const chips = $("v-chips");
  chips.textContent = "";
  const spec = [
    ["starts on badge", ev.starts_correctly],
    ["reaches goal", ev.ends_correctly],
    ["collision-free", ev.collision_free],
    [`efficiency ${ev.efficiency ? ev.efficiency.toFixed(2) : "—"}`, ev.success || null],
  ];
  for (const [label, good] of spec) {
    const c = document.createElement("span");
    c.className = "chip" + (good === true ? " good" : good === false ? " bad" : "");
    c.textContent = label;
    chips.appendChild(c);
  }
  const mode = $("mode");
  if (a.failure_mode && !ev.success) {
    mode.hidden = false;
    $("m-name").textContent = a.failure_mode.primary.replaceAll("_", " ");
    $("m-why").textContent = a.failure_mode.why || "";
    const q = $("m-quote");
    if (a.failure_mode.quote) { q.hidden = false; q.textContent = `“…${a.failure_mode.quote}…”`; }
    else q.hidden = true;
  } else mode.hidden = true;

  const tr = $("trace");
  if (a.reasoning) { tr.classList.remove("empty"); tr.textContent = a.reasoning; }
  else { tr.classList.add("empty"); tr.textContent = "provider returned no reasoning for this attempt"; }
}

/* ---------------- controls ---------------- */

function tick(ts) {
  if (!state.playing) return;
  const total = state.lengths?.[state.lengths.length - 1] || 1;
  const pxPerSec = 260 * state.speed;               // drag speed
  state.frac = Math.min(1, state.frac + (pxPerSec / 60) / total);
  $("scrub").value = Math.round(state.frac * 1000);
  draw();
  if (state.frac < 1) requestAnimationFrame(tick);
  else { state.playing = false; $("play").textContent = "▶ replay"; }
}

$("play").onclick = () => {
  if (!state.attempt) return;
  if (state.playing) { state.playing = false; $("play").textContent = "▶ replay"; return; }
  if (state.frac >= 1) state.frac = 0;
  state.playing = true;
  $("play").textContent = "❚❚ pause";
  requestAnimationFrame(tick);
};
$("scrub").oninput = (e) => {
  state.playing = false; $("play").textContent = "▶ replay";
  state.frac = e.target.value / 1000; draw();
};
$("speed").onclick = () => {
  state.speed = { 1: 2, 2: 4, 4: 0.5, 0.5: 1 }[state.speed];
  $("speed").textContent = `${state.speed}×`;
};
$("zoom").onclick = () => {
  state.zoomed = !state.zoomed;
  $("zoom").classList.toggle("active", state.zoomed);
  draw();
};
for (const id of ["t-ref", "t-mask", "t-radii"]) $(id).onchange = draw;

boot();
