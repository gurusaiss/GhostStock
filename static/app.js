// GhostStock dashboard — vanilla JS, no build step, no CDN dependency.
// Talks to the FastAPI backend in server.py; all chart rendering is hand-built SVG.

const API = "";
const SEVERITY_HIGH = 85;   // confidence >= this -> "High"
const SEVERITY_MEDIUM = 70; // confidence >= this -> "Medium", else "Low"

const state = {
  source: "synthetic",
  meta: null,
  store: null,
  dates: [],
  dateIdx: 0,
  alerts: [],
  selectedSku: null,
  tab: "alerts",

  // alerts-table view state
  search: "",
  severity: "all",
  sortKey: "confidence_score",
  sortDir: "desc",
  page: 0,
  pageSize: 20,
};

const el = (sel) => document.querySelector(sel);
const fmtINR = (n) => {
  if (n == null || Number.isNaN(n)) return "—";
  const abs = Math.abs(n);
  if (abs >= 1e7) return "₹" + (n / 1e7).toFixed(2) + "Cr";
  if (abs >= 1e5) return "₹" + (n / 1e5).toFixed(2) + "L";
  if (abs >= 1e3) return "₹" + (n / 1e3).toFixed(1) + "K";
  return "₹" + Math.round(n).toLocaleString("en-IN");
};
const fmtPct = (f) => (f == null ? "—" : (f * 100).toFixed(1) + "%");

function severityOf(score) {
  if (score >= SEVERITY_HIGH) return "high";
  if (score >= SEVERITY_MEDIUM) return "medium";
  return "low";
}
const SEVERITY_LABEL = { high: "High", medium: "Medium", low: "Low" };

function toast(msg, isErr) {
  const t = el("#toast");
  t.textContent = msg;
  t.classList.remove("hidden");
  t.classList.toggle("err", !!isErr);
  clearTimeout(toast._t);
  toast._t = setTimeout(() => t.classList.add("hidden"), 4000);
}

async function api(path, opts) {
  const res = await fetch(API + path, opts);
  if (res.status === 401) {
    window.location.href = "/login.html";
    throw new Error("Not authenticated");
  }
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || res.statusText);
  }
  return res.json();
}

// -------------------------------------------------------------------- auth

let currentUser = null;

async function bootstrapAuth() {
  const res = await fetch("/api/auth/me");
  if (!res.ok) {
    window.location.href = "/login.html";
    return false;
  }
  currentUser = await res.json();
  el("#user-email").textContent = currentUser.email;
  el("#user-role-badge").textContent = currentUser.role === "admin" ? "Admin" : "Store manager";
  el("#user-avatar").textContent = currentUser.email.slice(0, 1);
  el("#nav-users").classList.toggle("hidden", currentUser.role !== "admin");
  el("#logout-btn").addEventListener("click", async () => {
    await fetch("/api/auth/logout", { method: "POST" });
    window.location.href = "/login.html";
  });
  return true;
}

// ------------------------------------------------------------------ init

async function init() {
  const ok = await bootstrapAuth();
  if (!ok) return;
  setupUsersPanel();

  el("#latest-btn").addEventListener("click", () => {
    state.dateIdx = state.dates.length - 1;
    el("#date-slider").value = state.dateIdx;
    onDateOrStoreChange();
  });
  el("#date-slider").addEventListener("input", (e) => {
    state.dateIdx = parseInt(e.target.value, 10);
    updateDateReadout();
  });
  el("#date-slider").addEventListener("change", onDateOrStoreChange);
  el("#store-select").addEventListener("change", (e) => {
    state.store = e.target.value;
    onDateOrStoreChange();
  });
  el("#recompute-btn").addEventListener("click", onRecompute);
  el("#back-to-alerts-btn").addEventListener("click", () => switchTab("alerts"));
  el("#sku-select").addEventListener("change", (e) => selectSku(e.target.value));

  document.querySelectorAll(".seg-btn").forEach((btn) => {
    btn.addEventListener("click", () => switchSource(btn.dataset.source));
  });
  document.querySelectorAll(".nav-item").forEach((btn) => {
    btn.addEventListener("click", () => switchTab(btn.dataset.tab));
  });

  // toolbar: search / severity filter / page size / export
  el("#sku-search").addEventListener("input", (e) => {
    state.search = e.target.value.trim().toUpperCase();
    state.page = 0;
    renderAlertsTable();
  });
  document.querySelectorAll("#severity-filter .chip").forEach((btn) => {
    btn.addEventListener("click", () => {
      state.severity = btn.dataset.severity;
      state.page = 0;
      document.querySelectorAll("#severity-filter .chip").forEach((b) => b.classList.toggle("active", b === btn));
      renderAlertsTable();
    });
  });
  el("#page-size-select").addEventListener("change", (e) => {
    state.pageSize = parseInt(e.target.value, 10);
    state.page = 0;
    renderAlertsTable();
  });
  el("#export-csv-btn").addEventListener("click", exportCsv);

  setupUpload();
  await loadSourceData();
}

function switchTab(tab) {
  if (tab === "insights" && !state.selectedSku) {
    el("#drilldown-section").classList.add("hidden");
    el("#insights-empty").classList.remove("hidden");
  }
  if (tab === "users") loadUsers();
  state.tab = tab;
  document.querySelectorAll(".nav-item").forEach((b) => {
    const active = b.dataset.tab === tab;
    b.classList.toggle("active", active);
    b.setAttribute("aria-selected", active);
  });
  document.querySelectorAll(".tab-panel").forEach((p) => {
    p.classList.toggle("hidden", p.dataset.panel !== tab);
  });
  // "Users" and "Impact" aren't scoped by store/date — hide the filter there
  // so it doesn't imply it affects numbers that are actually chain-wide.
  const hasData = !!state.meta;
  el("#filter-row").classList.toggle("hidden", tab === "users" || tab === "impact" || !hasData);
}

async function switchSource(source) {
  if (source === state.source) return;
  state.source = source;
  document.querySelectorAll(".seg-btn").forEach((b) =>
    b.classList.toggle("active", b.dataset.source === source)
  );
  el("#upload-panel").classList.toggle("hidden", source !== "upload");
  state.selectedSku = null;
  await loadSourceData();
}

async function loadSourceData() {
  if (state.source === "upload" && !window._uploadedOnce) {
    showEmptyState(true);
    return;
  }
  try {
    const meta = await api(`/api/meta?source=${state.source}`);
    state.meta = meta;
    state.dates = meta.allDates;
    state.dateIdx = state.dates.length - 1;
    state.store = meta.stores[0] || null;

    showEmptyState(false);
    populateStoreSelect(meta.stores);
    const slider = el("#date-slider");
    slider.min = 0;
    slider.max = Math.max(0, state.dates.length - 1);
    slider.value = state.dateIdx;
    updateDateReadout();

    el("#filter-row").classList.remove("hidden");
    el("#alerts-section").classList.remove("hidden");
    switchTab(state.tab === "users" ? "users" : "alerts");

    if (state.source === "synthetic") {
      el("#rupee-section").classList.remove("hidden");
      el("#upload-rough-section").classList.add("hidden");
      loadSummary();
    } else {
      el("#rupee-section").classList.add("hidden");
      el("#upload-rough-section").classList.remove("hidden");
    }

    await refreshAlerts();
  } catch (e) {
    toast(e.message, true);
  }
}

function showEmptyState(show) {
  el("#empty-state").classList.toggle("hidden", !show);
  el("#filter-row").classList.toggle("hidden", show);
  el("#kpi-row").classList.toggle("hidden", show);
  el("#alerts-section").classList.toggle("hidden", show);
  el("#drilldown-section").classList.add("hidden");
  el("#rupee-section").classList.add("hidden");
  el("#upload-rough-section").classList.add("hidden");
}

function populateStoreSelect(stores) {
  const sel = el("#store-select");
  sel.innerHTML = "";
  stores.forEach((s) => {
    const opt = document.createElement("option");
    opt.value = s;
    opt.textContent = s;
    sel.appendChild(opt);
  });
  sel.value = state.store;
}

function updateDateReadout() {
  el("#date-readout").textContent = state.dates[state.dateIdx] || "—";
}

function onDateOrStoreChange() {
  updateDateReadout();
  state.selectedSku = null;
  state.page = 0;
  el("#drilldown-section").classList.add("hidden");
  refreshAlerts();
}

// ------------------------------------------------------------------ alerts

async function refreshAlerts() {
  const date = state.dates[state.dateIdx];
  if (!state.store || !date) return;
  try {
    const data = await api(
      `/api/alerts?store=${encodeURIComponent(state.store)}&date=${date}&source=${state.source}`
    );
    state.alerts = data.alerts;
    state.page = 0;
    renderAlertsSubtitle();
    renderKpiRow();
    renderAlertsTable();
  } catch (e) {
    toast(e.message, true);
  }
}

function renderAlertsSubtitle() {
  el("#alerts-subtitle").textContent = `Store ${state.store} · as of ${state.dates[state.dateIdx]}`;
}

function renderKpiRow() {
  const row = el("#kpi-row");
  if (!state.alerts.length) {
    row.innerHTML = "";
    row.classList.add("hidden");
    return;
  }
  row.classList.remove("hidden");
  const top = state.alerts[0];
  const highCount = state.alerts.filter((a) => severityOf(a.confidence_score) === "high").length;
  const tiles = [
    { label: "Alerts found today", value: state.alerts.length },
    { label: "High severity", value: highCount, cls: highCount ? "accent" : "" },
    { label: "Top confidence", value: top.confidence_score.toFixed(0) + "%" },
    { label: "Longest streak", value: Math.max(...state.alerts.map((a) => a.streak_len)) + " days" },
  ];
  row.innerHTML = tiles
    .map(
      (t) => `<div class="stat-tile">
        <div class="stat-label">${t.label}</div>
        <div class="stat-value ${t.cls || ""}">${t.value}</div>
      </div>`
    )
    .join("");
}

function getFilteredSortedAlerts() {
  let rows = state.alerts;
  if (state.search) rows = rows.filter((a) => a.sku_id.toUpperCase().includes(state.search));
  if (state.severity !== "all") rows = rows.filter((a) => severityOf(a.confidence_score) === state.severity);
  rows = [...rows].sort((a, b) => {
    const va = a[state.sortKey], vb = b[state.sortKey];
    return state.sortDir === "asc" ? va - vb : vb - va;
  });
  return rows;
}

const SORTABLE_COLS = [
  { key: "confidence_score", label: "Confidence" },
  { key: "streak_len", label: "Streak" },
  { key: "current_stock", label: "Stock" },
];

function renderAlertsTable() {
  const wrap = el("#alerts-table-wrap");
  const pager = el("#pagination");

  if (!state.alerts.length) {
    wrap.innerHTML = `<p class="muted-text">No phantom stock-out candidates found for this store/date
      (try a later date — the detector needs ~4 months of history to build a baseline).</p>`;
    pager.innerHTML = "";
    return;
  }

  const filtered = getFilteredSortedAlerts();
  if (!filtered.length) {
    wrap.innerHTML = `<div class="no-results">No alerts match this search/filter. Try clearing it.</div>`;
    pager.innerHTML = "";
    return;
  }

  const totalPages = Math.max(1, Math.ceil(filtered.length / state.pageSize));
  state.page = Math.min(state.page, totalPages - 1);
  const start = state.page * state.pageSize;
  const pageRows = filtered.slice(start, start + state.pageSize);

  const table = document.createElement("table");
  table.className = "alerts-table";
  const thead = document.createElement("thead");
  const headRow = document.createElement("tr");
  headRow.innerHTML = `<th>Rank</th><th>SKU</th>`;
  SORTABLE_COLS.forEach((c) => {
    const th = document.createElement("th");
    th.className = "sortable num-cell";
    th.dataset.key = c.key;
    const arrow = state.sortKey === c.key ? (state.sortDir === "desc" ? "↓" : "↑") : "";
    th.innerHTML = `${c.label}<span class="sort-arrow">${arrow}</span>`;
    th.addEventListener("click", () => {
      if (state.sortKey === c.key) state.sortDir = state.sortDir === "desc" ? "asc" : "desc";
      else { state.sortKey = c.key; state.sortDir = "desc"; }
      renderAlertsTable();
    });
    headRow.appendChild(th);
  });
  // appendChild, not innerHTML +=: the latter re-parses the whole row from a
  // string and would silently drop the click listeners just attached above.
  const paceTh = document.createElement("th");
  paceTh.className = "num-cell";
  paceTh.textContent = "Normal pace";
  headRow.appendChild(paceTh);
  headRow.appendChild(document.createElement("th"));
  thead.appendChild(headRow);
  table.appendChild(thead);

  const tbody = document.createElement("tbody");
  pageRows.forEach((a) => {
    const rank = state.alerts.indexOf(a) + 1;
    const sev = severityOf(a.confidence_score);
    const tr = document.createElement("tr");
    tr.dataset.sku = a.sku_id;
    if (a.sku_id === state.selectedSku) tr.classList.add("selected");
    const pct = Math.max(0, Math.min(100, a.confidence_score));
    const promoBadge = a.promo_fraction_in_streak > 0.05
      ? `<span class="promo-badge">promo ${Math.round(a.promo_fraction_in_streak * 100)}%</span>` : "";
    tr.innerHTML = `
      <td class="rank-cell">${rank}</td>
      <td class="sku-cell"></td>
      <td>
        <div class="conf-meter-wrap">
          <span class="severity-badge severity-${sev}">${SEVERITY_LABEL[sev]}</span>
          <div class="conf-meter-track"><div class="conf-meter-fill" style="width:${pct}%"></div></div>
          <span class="conf-meter-value">${pct.toFixed(0)}%</span>
        </div>
      </td>
      <td class="num-cell">${a.streak_len}d</td>
      <td class="num-cell">${a.current_stock}</td>
      <td class="num-cell">${a.baseline_mean.toFixed(1)}/day</td>
      <td><button class="view-btn">View →</button></td>
    `;
    tr.querySelector(".sku-cell").textContent = a.sku_id;
    tr.querySelector(".sku-cell").insertAdjacentHTML("beforeend", promoBadge);
    const openInsights = () => { switchTab("insights"); selectSku(a.sku_id); };
    tr.addEventListener("click", openInsights);
    tr.querySelector(".view-btn").addEventListener("click", (e) => { e.stopPropagation(); openInsights(); });
    tbody.appendChild(tr);
  });
  table.appendChild(tbody);
  wrap.innerHTML = "";
  wrap.appendChild(table);

  renderPagination(totalPages);
}

function renderPagination(totalPages) {
  const pager = el("#pagination");
  if (totalPages <= 1) { pager.innerHTML = ""; return; }
  const parts = [];
  parts.push(`<button class="page-btn" id="pg-prev" ${state.page === 0 ? "disabled" : ""}>‹</button>`);
  for (let i = 0; i < totalPages; i++) {
    if (totalPages > 7 && Math.abs(i - state.page) > 2 && i !== 0 && i !== totalPages - 1) {
      if (i === 1 || i === totalPages - 2) parts.push(`<span class="page-info">…</span>`);
      continue;
    }
    parts.push(`<button class="page-btn ${i === state.page ? "active" : ""}" data-page="${i}">${i + 1}</button>`);
  }
  parts.push(`<button class="page-btn" id="pg-next" ${state.page === totalPages - 1 ? "disabled" : ""}>›</button>`);
  pager.innerHTML = parts.join("");
  pager.querySelectorAll("[data-page]").forEach((b) =>
    b.addEventListener("click", () => { state.page = parseInt(b.dataset.page, 10); renderAlertsTable(); })
  );
  const prev = el("#pg-prev"), next = el("#pg-next");
  if (prev) prev.addEventListener("click", () => { state.page--; renderAlertsTable(); });
  if (next) next.addEventListener("click", () => { state.page++; renderAlertsTable(); });
}

function exportCsv() {
  const rows = getFilteredSortedAlerts();
  if (!rows.length) { toast("Nothing to export.", true); return; }
  const header = ["rank", "sku", "confidence_pct", "severity", "streak_days", "normal_pace_per_day", "stock_shown", "promo_overlap_pct", "reason"];
  const lines = [header.join(",")];
  rows.forEach((a) => {
    const rank = state.alerts.indexOf(a) + 1;
    const vals = [
      rank, a.sku_id, a.confidence_score, severityOf(a.confidence_score),
      a.streak_len, a.baseline_mean.toFixed(2), a.current_stock,
      Math.round(a.promo_fraction_in_streak * 100),
      `"${a.reason.replace(/"/g, '""')}"`,
    ];
    lines.push(vals.join(","));
  });
  const blob = new Blob([lines.join("\n")], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `ghoststock_alerts_${state.store}_${state.dates[state.dateIdx]}.csv`;
  link.click();
  URL.revokeObjectURL(url);
  toast("CSV exported.");
}

// ------------------------------------------------------------------ drilldown

async function selectSku(sku) {
  state.selectedSku = sku;
  document.querySelectorAll("tr[data-sku]").forEach((tr) =>
    tr.classList.toggle("selected", tr.dataset.sku === sku)
  );
  const row = state.alerts.find((a) => a.sku_id === sku);
  if (!row) return;

  el("#insights-empty").classList.add("hidden");
  el("#drilldown-section").classList.remove("hidden");
  el("#drill-sku").textContent = `${sku} @ ${state.store}`;
  el("#drill-reason").textContent = row.reason;

  // "jump to SKU" dropdown mirrors the current alert list
  const skuSel = el("#sku-select");
  skuSel.innerHTML = state.alerts
    .map((a) => `<option value="${a.sku_id}" ${a.sku_id === sku ? "selected" : ""}>${a.sku_id} (${a.confidence_score.toFixed(0)}%)</option>`)
    .join("");

  const sev = severityOf(row.confidence_score);
  const stats = [
    { label: "Confidence", value: row.confidence_score.toFixed(0) + "%", cls: "accent" },
    { label: "Severity", value: SEVERITY_LABEL[sev] },
    { label: "Zero-sales streak", value: row.streak_len + " days" },
    { label: "System stock shown", value: row.current_stock + " units" },
    { label: "Normal pace", value: row.baseline_mean.toFixed(2) + "/day" },
  ];
  el("#drill-stats").innerHTML = stats
    .map(
      (s) => `<div class="stat-tile">
        <div class="stat-label">${s.label}</div>
        <div class="stat-value ${s.cls || ""}">${s.value}</div>
      </div>`
    )
    .join("");

  try {
    const date = state.dates[state.dateIdx];
    const days = Math.min(180, Math.max(60, row.streak_len + 25));
    const chartData = await api(
      `/api/chart?store=${encodeURIComponent(state.store)}&sku=${encodeURIComponent(sku)}&date=${date}&source=${state.source}&days=${days}`
    );
    renderDualChart(el("#chart-block"), chartData);
  } catch (e) {
    toast(e.message, true);
  }
}

// ------------------------------------------------------------------ charts
// Two stacked SVGs sharing an x-axis (bars = daily units sold, line = stock
// on hand) — never a dual-axis chart. A single synced crosshair + tooltip
// covers both, since they're one story at one date.

const SVGNS = "http://www.w3.org/2000/svg";
function svgEl(tag, attrs) {
  const n = document.createElementNS(SVGNS, tag);
  for (const k in attrs) n.setAttribute(k, attrs[k]);
  return n;
}

function renderDualChart(container, data) {
  container.innerHTML = "";
  const { dates, sold, stock, promoActive } = data;
  const n = dates.length;
  if (!n) {
    container.innerHTML = `<p class="muted-text">No history in this window.</p>`;
    return;
  }

  const width = Math.max(container.clientWidth || 760, 320);
  const marginL = 34, marginR = 10;
  const plotW = width - marginL - marginR;
  const slotW = plotW / n;
  const barW = Math.min(20, Math.max(2, slotW - 3));

  const legend = document.createElement("div");
  legend.className = "chart-legend";
  legend.innerHTML = `
    <span class="legend-item"><span class="legend-key bar" style="background:var(--series-2)"></span>Daily units sold</span>
    <span class="legend-item"><span class="legend-key" style="background:var(--series-1)"></span>System stock on hand</span>`;
  container.appendChild(legend);

  const barH = 110;
  const maxSold = Math.max(1, ...sold) * 1.15;
  const barSvg = svgEl("svg", { width, height: barH + 8, viewBox: `0 0 ${width} ${barH + 8}`, class: "chart-svg" });
  const baselineY = barH;

  barSvg.appendChild(svgEl("line", {
    x1: marginL, x2: width - marginR, y1: baselineY, y2: baselineY,
    stroke: "var(--baseline)", "stroke-width": 1,
  }));

  sold.forEach((v, i) => {
    if (v <= 0) return;
    const h = (v / maxSold) * (barH - 10);
    const x = marginL + i * slotW + (slotW - barW) / 2;
    const yTop = baselineY - h;
    const r = Math.min(4, h / 2);
    const path = `M${x + r},${yTop} L${x + barW - r},${yTop} Q${x + barW},${yTop} ${x + barW},${yTop + r}
      L${x + barW},${baselineY} L${x},${baselineY} L${x},${yTop + r} Q${x},${yTop} ${x + r},${yTop} Z`;
    barSvg.appendChild(svgEl("path", { d: path, fill: "var(--series-2)" }));
  });

  promoActive.forEach((p, i) => {
    if (!p) return;
    const x = marginL + i * slotW + slotW / 2;
    barSvg.appendChild(svgEl("rect", {
      x: x - 1.5, y: 2, width: 3, height: 4, rx: 1.5,
      fill: "var(--status-warning)",
    }));
  });

  const barCrosshair = svgEl("line", {
    x1: 0, x2: 0, y1: 0, y2: barH, stroke: "var(--muted-ink)", "stroke-width": 1, opacity: 0,
  });
  barSvg.appendChild(barCrosshair);

  const lineH = 130;
  const minStock = Math.min(...stock);
  const maxStock = Math.max(...stock);
  const pad = Math.max(1, (maxStock - minStock) * 0.15);
  const domainMin = Math.max(0, minStock - pad);
  const domainMax = maxStock + pad || 1;
  const yScale = (v) => lineH - 18 - ((v - domainMin) / (domainMax - domainMin || 1)) * (lineH - 28);

  const lineSvg = svgEl("svg", { width, height: lineH + 20, viewBox: `0 0 ${width} ${lineH + 20}`, class: "chart-svg" });

  [domainMin, (domainMin + domainMax) / 2, domainMax].forEach((tickVal) => {
    const y = yScale(tickVal);
    lineSvg.appendChild(svgEl("line", {
      x1: marginL, x2: width - marginR, y1: y, y2: y, stroke: "var(--gridline)", "stroke-width": 1,
    }));
    const t = svgEl("text", {
      x: marginL - 6, y: y + 3, "text-anchor": "end", fill: "var(--muted-ink)", "font-size": 10,
    });
    t.textContent = Math.round(tickVal);
    lineSvg.appendChild(t);
  });

  let points = stock.map((v, i) => `${marginL + i * slotW + slotW / 2},${yScale(v)}`).join(" ");
  lineSvg.appendChild(svgEl("polyline", {
    points, fill: "none", stroke: "var(--series-1)", "stroke-width": 2,
    "stroke-linejoin": "round", "stroke-linecap": "round",
  }));

  const lastX = marginL + (n - 1) * slotW + slotW / 2;
  const lastY = yScale(stock[n - 1]);
  lineSvg.appendChild(svgEl("circle", { cx: lastX, cy: lastY, r: 4, fill: "var(--series-1)", stroke: "var(--surface-1)", "stroke-width": 2 }));
  const endLabel = svgEl("text", {
    x: Math.min(lastX + 6, width - marginR - 2), y: lastY - 8, "text-anchor": "end",
    fill: "var(--text-primary)", "font-size": 11, "font-weight": 650,
  });
  endLabel.textContent = stock[n - 1];
  lineSvg.appendChild(endLabel);

  const tickEvery = Math.max(1, Math.round(n / 7));
  for (let i = 0; i < n; i += tickEvery) {
    const x = marginL + i * slotW + slotW / 2;
    const t = svgEl("text", {
      x, y: lineH + 14, "text-anchor": "middle", fill: "var(--muted-ink)", "font-size": 10,
    });
    t.textContent = dates[i].slice(5);
    lineSvg.appendChild(t);
  }

  const lineCrosshair = svgEl("line", {
    x1: 0, x2: 0, y1: 0, y2: lineH, stroke: "var(--muted-ink)", "stroke-width": 1, opacity: 0,
  });
  lineSvg.appendChild(lineCrosshair);

  const wrap = document.createElement("div");
  wrap.style.position = "relative";
  const tooltip = document.createElement("div");
  tooltip.className = "chart-tooltip";
  wrap.appendChild(barSvg);
  wrap.appendChild(lineSvg);
  wrap.appendChild(tooltip);
  container.appendChild(wrap);

  function handleMove(evt, svgNode) {
    const rect = svgNode.getBoundingClientRect();
    const scale = width / rect.width;
    const xPix = (evt.clientX - rect.left) * scale;
    let idx = Math.floor((xPix - marginL) / slotW);
    idx = Math.max(0, Math.min(n - 1, idx));
    const cx = marginL + idx * slotW + slotW / 2;
    barCrosshair.setAttribute("x1", cx); barCrosshair.setAttribute("x2", cx); barCrosshair.setAttribute("opacity", 1);
    lineCrosshair.setAttribute("x1", cx); lineCrosshair.setAttribute("x2", cx); lineCrosshair.setAttribute("opacity", 1);

    tooltip.innerHTML = `
      <div class="tt-date">${dates[idx]}${promoActive[idx] ? " · promo active" : ""}</div>
      <div class="tt-row"><span class="tt-key" style="background:var(--series-2)"></span>Units sold<span class="tt-value">${sold[idx]}</span></div>
      <div class="tt-row"><span class="tt-key" style="background:var(--series-1)"></span>Stock on hand<span class="tt-value">${stock[idx]}</span></div>
    `;
    tooltip.style.opacity = 1;
    const tipX = Math.min(cx / scale + 12, rect.width - 170);
    tooltip.style.left = Math.max(4, tipX) + "px";
    tooltip.style.top = "4px";
  }
  function handleLeave() {
    barCrosshair.setAttribute("opacity", 0);
    lineCrosshair.setAttribute("opacity", 0);
    tooltip.style.opacity = 0;
  }
  [barSvg, lineSvg].forEach((svgNode) => {
    svgNode.addEventListener("pointermove", (e) => handleMove(e, svgNode));
    svgNode.addEventListener("pointerleave", handleLeave);
  });
}

// ------------------------------------------------------------------ rupee summary

async function loadSummary() {
  try {
    const s = await api(`/api/summary?source=${state.source}`);
    renderSummary(s);
  } catch (e) {
    toast(e.message, true);
  }
}

function renderSummary(s) {
  if (!s.available) {
    el("#rupee-section").classList.add("hidden");
    return;
  }
  el("#rupee-note").textContent = s.precomputed
    ? "Precomputed by evaluate.py — click the button above to re-run live."
    : "Freshly recomputed live.";

  const kpis = [
    { label: "Confirmed phantom cases caught", value: `${s.n_confirmed} / ${s.n_ground_truth_cases}` },
    { label: "₹ lost in observed window (~18mo)", value: fmtINR(s.total_rupees_window) },
    { label: "Est. ₹ recovered / year", value: fmtINR(s.annualized_rupees), cls: "accent" },
  ];
  if (s.precision_at_20 != null) {
    kpis.push({ label: "Precision@20", value: fmtPct(s.precision_at_20) });
    kpis.push({ label: "Recall", value: fmtPct(s.recall), cls: "good" });
  }
  el("#rupee-kpis").innerHTML = kpis
    .map(
      (k) => `<div class="stat-tile">
        <div class="stat-label">${k.label}</div>
        <div class="stat-value ${k.cls || ""}">${k.value}</div>
      </div>`
    )
    .join("");

  el("#arithmetic-body").innerHTML = `<div class="arithmetic-body-inner">For each confirmed phantom case:

  rupees_lost = avg_daily_sales_when_in_stock &times; unit_price &times; days_lost

Summed across <span class="hl">${s.n_confirmed}</span> confirmed/likely cases (caught by the
detector in that store's top-20 by the time the streak had played out):

  &Sigma; rupees_lost  =  <span class="hl">${fmtINR(s.total_rupees_window)}</span>   (value lost over the
  ${s.n_days_in_data}-day observed window, ~${s.years_in_data.toFixed(2)} years)

Annualized:

  ${fmtINR(s.total_rupees_window)}  /  ${s.years_in_data.toFixed(2)} years  =  <span class="hl">${fmtINR(s.annualized_rupees)}</span> / year

across all ${s.n_stores} stores in the chain.</div>`;
}

async function onRecompute() {
  const btn = el("#recompute-btn");
  btn.disabled = true;
  btn.textContent = "Running full scan across all stores — this takes ~3-4 minutes...";
  try {
    const s = await api("/api/summary/recompute", { method: "POST" });
    renderSummary(s);
    toast("Recomputed live.");
  } catch (e) {
    toast(e.message, true);
  } finally {
    btn.disabled = false;
    btn.textContent = "Recompute live across all stores (~3-4 min)";
  }
}

// ------------------------------------------------------------------ upload

function setupUpload() {
  const dz = el("#dropzone");
  const input = el("#file-input");
  dz.addEventListener("click", () => input.click());
  dz.addEventListener("dragover", (e) => { e.preventDefault(); dz.classList.add("dragover"); });
  dz.addEventListener("dragleave", () => dz.classList.remove("dragover"));
  dz.addEventListener("drop", (e) => {
    e.preventDefault();
    dz.classList.remove("dragover");
    if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
  });
  input.addEventListener("change", () => {
    if (input.files[0]) handleFile(input.files[0]);
  });
}

async function handleFile(file) {
  const statusEl = el("#upload-status");
  statusEl.textContent = "Uploading & cleaning...";
  statusEl.className = "upload-status";
  const form = new FormData();
  form.append("file", file);
  try {
    const res = await fetch("/api/upload", { method: "POST", body: form });
    const body = await res.json();
    if (!res.ok) throw new Error(body.detail || "Upload failed");
    statusEl.textContent = `Loaded ${body.nRows.toLocaleString()} rows from ${body.filename}: ${body.nStores} stores, ${body.nSkus} SKUs.`;
    statusEl.className = "upload-status ok";
    window._uploadedOnce = true;
    await loadSourceData();
  } catch (e) {
    statusEl.textContent = e.message;
    statusEl.className = "upload-status err";
  }
}

// -------------------------------------------------------------- admin: users

function setupUsersPanel() {
  if (currentUser.role !== "admin") return;

  el("#add-user-btn").addEventListener("click", openUserModal);
  el("#modal-close-btn").addEventListener("click", closeUserModal);
  el("#user-modal-overlay").addEventListener("click", (e) => {
    if (e.target.id === "user-modal-overlay") closeUserModal();
  });
  el("#new-role").addEventListener("change", (e) => {
    el("#store-scope-field").classList.toggle("hidden", e.target.value === "admin");
  });
  el("#gen-password-btn").addEventListener("click", async () => {
    try {
      const { password } = await api("/api/admin/generate_password", { method: "POST" });
      el("#new-password").value = password;
    } catch (e) {
      toast(e.message, true);
    }
  });
  el("#user-form").addEventListener("submit", submitNewUser);
}

function openUserModal() {
  el("#user-form").reset();
  el("#modal-error").classList.remove("show");
  el("#store-scope-field").classList.remove("hidden");
  // unchecked by default: scoping a manager to specific stores only works if
  // the admin has to deliberately opt stores in, not uncheck the ones to exclude
  const stores = (state.meta && state.meta.stores) || [];
  el("#store-checklist").innerHTML = stores
    .map((s) => `<label class="store-check-item"><input type="checkbox" value="${s}"> ${s}</label>`)
    .join("");
  el("#user-modal-overlay").classList.remove("hidden");
}

function closeUserModal() {
  el("#user-modal-overlay").classList.add("hidden");
}

async function submitNewUser(e) {
  e.preventDefault();
  const email = el("#new-email").value.trim();
  const password = el("#new-password").value;
  const role = el("#new-role").value;
  let storeIds = "ALL";
  if (role !== "admin") {
    const checked = Array.from(document.querySelectorAll("#store-checklist input:checked")).map((i) => i.value);
    if (!checked.length) {
      showModalError("Pick at least one store for a manager account (or make them an admin instead).");
      return;
    }
    storeIds = checked.join(",");
  }
  const btn = el("#user-form-submit");
  btn.disabled = true;
  btn.textContent = "Creating…";
  try {
    await api("/api/admin/users", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password, role, store_ids: storeIds }),
    });
    closeUserModal();
    toast(`User ${email} created.`);
    loadUsers();
  } catch (e) {
    showModalError(e.message);
  } finally {
    btn.disabled = false;
    btn.textContent = "Create user";
  }
}

function showModalError(msg) {
  const box = el("#modal-error");
  box.textContent = msg;
  box.classList.add("show");
}

async function loadUsers() {
  if (currentUser.role !== "admin") return;
  try {
    const users = await api("/api/admin/users");
    renderUsersTable(users);
  } catch (e) {
    toast(e.message, true);
  }
}

function renderUsersTable(users) {
  const wrap = el("#users-table-wrap");
  if (!users.length) {
    wrap.innerHTML = `<p class="muted-text">No users yet.</p>`;
    return;
  }
  const table = document.createElement("table");
  table.className = "users-table";
  table.innerHTML = `<thead><tr><th>Email</th><th>Role</th><th>Store access</th><th>Added</th><th></th></tr></thead>`;
  const tbody = document.createElement("tbody");
  users.forEach((u) => {
    const tr = document.createElement("tr");
    const scope = u.store_ids === "ALL" ? "All stores" : u.store_ids.split(",").join(", ");
    const added = new Date(u.created_at * 1000).toLocaleDateString();
    const emailCell = document.createElement("td");
    emailCell.textContent = u.email;
    const roleCell = document.createElement("td");
    roleCell.innerHTML = `<span class="role-pill ${u.role}">${u.role === "admin" ? "Admin" : "Manager"}</span>`;
    const scopeCell = document.createElement("td");
    scopeCell.className = "store-scope-text";
    scopeCell.textContent = scope;
    const addedCell = document.createElement("td");
    addedCell.className = "store-scope-text";
    addedCell.textContent = added;
    const actionsCell = document.createElement("td");
    const resetBtn = document.createElement("button");
    resetBtn.className = "view-btn";
    resetBtn.textContent = "Reset password";
    resetBtn.style.marginRight = "6px";
    resetBtn.addEventListener("click", () => resetUserPassword(u));
    const delBtn = document.createElement("button");
    delBtn.className = "btn-danger-text";
    delBtn.textContent = "Remove";
    delBtn.addEventListener("click", () => removeUser(u));
    actionsCell.appendChild(resetBtn);
    actionsCell.appendChild(delBtn);

    tr.append(emailCell, roleCell, scopeCell, addedCell, actionsCell);
    tbody.appendChild(tr);
  });
  table.appendChild(tbody);
  wrap.innerHTML = "";
  wrap.appendChild(table);
}

async function resetUserPassword(u) {
  const pwd = window.prompt(`New password for ${u.email} (min 8 characters):`);
  if (!pwd) return;
  try {
    await api(`/api/admin/users/${u.id}/reset_password`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ password: pwd }),
    });
    toast(`Password reset for ${u.email}. Share it with them directly.`);
  } catch (e) {
    toast(e.message, true);
  }
}

async function removeUser(u) {
  if (!window.confirm(`Remove ${u.email}? They won't be able to log in again (an already-open session stays valid until it expires).`)) return;
  try {
    await api(`/api/admin/users/${u.id}`, { method: "DELETE" });
    toast(`${u.email} removed.`);
    loadUsers();
  } catch (e) {
    toast(e.message, true);
  }
}

init();
