/* ── Operator Debugger Inspector ────────────────────────────── */

const initialParams = new URLSearchParams(window.location.search);

const state = {
  traceDir: initialParams.get("trace_dir") || "",
  traces: [],
  run: null,
  filteredEpisodes: [],
  episode: null,
  selectedEpisodeId: initialParams.get("episode") || "",
  selectedStepIndex: Math.max(0, Number(initialParams.get("step") || 0) || 0),
  replayStatus: null,
  replayPollHandle: null,
  loading: new Set(),
};

const sectionOrder = [
  "secReasoning",
  "secActions",
  "secAct",
  "secDiffs",
  "secHtml",
  "secShots",
  "secReplay",
];

const sectionLabels = [
  "Reasoning",
  "Actions",
  "/act",
  "Diffs",
  "HTML",
  "Shots",
  "Replay",
];

/* ── Utilities ─────────────────────────────────────────────── */

function q(id) {
  return document.getElementById(id);
}

function pretty(value) {
  return JSON.stringify(value ?? {}, null, 2);
}

function clipMiddle(value, max = 72) {
  const text = String(value || "");
  if (text.length <= max) return text;
  const left = Math.max(20, Math.floor((max - 3) * 0.65));
  const right = Math.max(10, max - 3 - left);
  return `${text.slice(0, left)}...${text.slice(-right)}`;
}

function copyToClipboard(text, successMsg = "Copied") {
  return navigator.clipboard.writeText(String(text ?? "")).then(() => {
    toast(successMsg, "success");
  });
}

function setText(id, value) {
  const el = q(id);
  if (el) el.textContent = String(value ?? "");
}

function setPre(id, value) {
  const el = q(id);
  if (el) el.textContent = typeof value === "string" ? value : pretty(value);
}

function syncUrlState() {
  const next = new URL(window.location.href);
  if (state.traceDir) next.searchParams.set("trace_dir", state.traceDir);
  else next.searchParams.delete("trace_dir");
  if (state.selectedEpisodeId) next.searchParams.set("episode", state.selectedEpisodeId);
  else next.searchParams.delete("episode");
  next.searchParams.set("step", String(state.selectedStepIndex));
  window.history.replaceState({}, "", next.toString());
}

/* ── Toast notifications ───────────────────────────────────── */

function toast(message, type = "info") {
  const container = q("toastContainer");
  if (!container) return;
  const el = document.createElement("div");
  el.className = `toast ${type}`;
  el.textContent = message;
  container.appendChild(el);
  setTimeout(() => {
    el.style.opacity = "0";
    el.style.transform = "translateY(10px)";
    el.style.transition = "all 200ms ease";
    setTimeout(() => el.remove(), 220);
  }, 2800);
}

/* ── Loading indicators ────────────────────────────────────── */

function showLoading(paneId) {
  state.loading.add(paneId);
  const pane = q(paneId);
  if (!pane || pane.querySelector(".loading-overlay")) return;
  const overlay = document.createElement("div");
  overlay.className = "loading-overlay";
  overlay.innerHTML = '<div class="spinner"></div>';
  pane.appendChild(overlay);
}

function hideLoading(paneId) {
  state.loading.delete(paneId);
  const pane = q(paneId);
  if (!pane) return;
  const overlay = pane.querySelector(".loading-overlay");
  if (overlay) overlay.remove();
}

/* ── API ───────────────────────────────────────────────────── */

async function api(path, options = {}) {
  const url = new URL(path, window.location.origin);
  if (state.traceDir) url.searchParams.set("trace_dir", state.traceDir);
  const res = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const text = await res.text();
    toast(`API error: ${text.slice(0, 120)}`, "error");
    throw new Error(text);
  }
  return await res.json();
}

/* ── Chip helper ───────────────────────────────────────────── */

function createChip(text, klass = "") {
  const span = document.createElement("span");
  span.className = `chip ${klass}`.trim();
  span.textContent = String(text ?? "");
  return span;
}

/* ── Render: run summary metrics ───────────────────────────── */

function renderRunSummary() {
  const box = q("runSummary");
  if (!box) return;
  const traceIndex = state.run?.trace_index || {};
  const summary = traceIndex.summary || {};
  const episodes = state.run?.episodes || [];
  const traceDir = String(state.run?.trace_dir || "");
  const traceEl = q("traceDirText");
  if (traceEl) {
    traceEl.textContent = traceDir ? clipMiddle(traceDir, 90) : "No trace loaded";
    traceEl.title = traceDir || "";
  }

  const successRate = summary.success_rate;
  const avgScore = summary.avg_score;

  const metrics = [
    ["Project", traceIndex.web_project_id || "-", null],
    ["Provider", traceIndex.provider || "-", null],
    ["Model", traceIndex.model || "-", null],
    ["Episodes", episodes.length, null],
    ["Max Steps", traceIndex.max_steps ?? "-", null],
    ["Duration", summary.timing?.total_seconds != null ? `${summary.timing.total_seconds}s` : "-", null],
    ["Avg Score", avgScore ?? "-", avgScore != null ? (avgScore >= 0.7 ? "score-ok" : avgScore >= 0.4 ? "score-warn" : "score-bad") : null],
    ["Success", successRate ?? "-", successRate != null ? (successRate >= 0.7 ? "score-ok" : successRate >= 0.4 ? "score-warn" : "score-bad") : null],
  ];

  box.innerHTML = "";
  for (const [k, v, scoreClass] of metrics) {
    const card = document.createElement("div");
    card.className = "metric";
    const title = document.createElement("div");
    title.className = "k";
    title.textContent = String(k);
    const value = document.createElement("div");
    value.className = `v ${scoreClass || ""}`.trim();
    value.textContent = String(v);
    card.appendChild(title);
    card.appendChild(value);
    box.appendChild(card);
  }
}

/* ── Render: trace picker ──────────────────────────────────── */

function renderTracePicker() {
  const picker = q("tracePicker");
  if (!picker) return;
  picker.innerHTML = "";
  for (const item of state.traces || []) {
    const opt = document.createElement("option");
    opt.value = item.trace_dir;
    const parts = [item.name || item.trace_dir];
    if (item.project) parts.push(item.project);
    if (item.model) parts.push(item.model);
    parts.push(`${item.episodes || 0} ep`);
    opt.textContent = parts.join(" \u00b7 ");
    if (item.trace_dir === state.traceDir) opt.selected = true;
    picker.appendChild(opt);
  }
}

/* ── Render: episode filters ───────────────────────────────── */

function renderEpisodeFilters() {
  const episodes = state.run?.episodes || [];
  const useCases = [...new Set(episodes.map((ep) => String(ep.use_case || "").trim()).filter(Boolean))].sort();
  const failureCategories = [...new Set(episodes.map((ep) => String(ep.failure_category || "").trim()).filter(Boolean))].sort();

  const useCaseSel = q("filterUseCase");
  const failureSel = q("filterFailureCategory");

  if (useCaseSel) {
    useCaseSel.innerHTML = '<option value="">All</option>';
    for (const v of useCases) {
      const opt = document.createElement("option");
      opt.value = v;
      opt.textContent = v;
      useCaseSel.appendChild(opt);
    }
  }

  if (failureSel) {
    failureSel.innerHTML = '<option value="">All</option>';
    for (const v of failureCategories) {
      const opt = document.createElement("option");
      opt.value = v;
      opt.textContent = v;
      failureSel.appendChild(opt);
    }
  }
}

/* ── Filter episodes ───────────────────────────────────────── */

function applyEpisodeFilters() {
  const useCase = q("filterUseCase")?.value || "";
  const status = q("filterStatus")?.value || "";
  const failureCategory = q("filterFailureCategory")?.value || "";
  const episodes = state.run?.episodes || [];

  state.filteredEpisodes = episodes.filter((ep) => {
    if (useCase && String(ep.use_case || "") !== useCase) return false;
    if (status === "success" && !ep.success) return false;
    if (status === "failure" && ep.success) return false;
    if (failureCategory && String(ep.failure_category || "") !== failureCategory) return false;
    return true;
  });

  if (!state.filteredEpisodes.some((ep) => ep.episode_task_id === state.selectedEpisodeId)) {
    state.selectedEpisodeId = state.filteredEpisodes[0]?.episode_task_id || "";
    state.selectedStepIndex = 0;
    state.episode = null;
    if (state.selectedEpisodeId) {
      void loadEpisode(state.selectedEpisodeId);
      return;
    }
  }
  renderEpisodes();
  renderSteps();
}

/* ── Render: episode list ──────────────────────────────────── */

function renderEpisodes() {
  const root = q("episodes");
  if (!root) return;
  root.innerHTML = "";

  const episodes = state.filteredEpisodes || [];
  if (!episodes.length) {
    const empty = document.createElement("div");
    empty.className = "empty";
    empty.textContent = "No episodes match filters.";
    root.appendChild(empty);
    return;
  }

  for (const ep of episodes) {
    const item = document.createElement("article");
    item.className = `item ${ep.episode_task_id === state.selectedEpisodeId ? "active" : ""}`.trim();

    const title = document.createElement("strong");
    title.textContent = ep.use_case || ep.task_id || "episode";

    const meta = document.createElement("div");
    meta.className = "meta";
    const scoreStr = (ep.score ?? 0).toFixed(2);
    meta.textContent = `score ${scoreStr}  \u00b7  ${ep.steps ?? 0} steps  \u00b7  ${ep.llm_calls ?? 0} llm`;

    const chips = document.createElement("div");
    chips.className = "chips";
    chips.appendChild(createChip(ep.success ? "PASS" : "FAIL", ep.success ? "ok" : "bad"));
    if (ep.failure_category) chips.appendChild(createChip(ep.failure_category));
    if (ep.estimated_cost_usd) chips.appendChild(createChip(`$${Number(ep.estimated_cost_usd).toFixed(3)}`));

    item.appendChild(title);
    item.appendChild(meta);
    item.appendChild(chips);
    item.onclick = () => void loadEpisode(ep.episode_task_id);
    root.appendChild(item);
  }
}

/* ── Step helpers ──────────────────────────────────────────── */

function stepCount() {
  return (state.episode?.steps || []).length;
}

function currentStep() {
  return (state.episode?.steps || [])[state.selectedStepIndex] || null;
}

function normalizeActions(step) {
  if (!step || typeof step !== "object") return [];
  const many = step.actions;
  if (Array.isArray(many) && many.length) return many;
  const single = step.action;
  if (single && typeof single === "object") return [single];
  return [];
}

function normalizeToolCalls(step) {
  const tc = step?.act_response?.tool_calls;
  return Array.isArray(tc) ? tc : [];
}

/* ── Render: step timeline ─────────────────────────────────── */

function renderSteps() {
  const root = q("steps");
  if (!root) return;
  root.innerHTML = "";

  const summaries = state.episode?.step_summaries || [];
  if (!summaries.length) {
    const empty = document.createElement("div");
    empty.className = "empty";
    empty.textContent = "Select an episode to view steps.";
    root.appendChild(empty);
    return;
  }

  for (let idx = 0; idx < summaries.length; idx += 1) {
    const s = summaries[idx];
    const item = document.createElement("article");
    item.className = `item ${idx === state.selectedStepIndex ? "active" : ""}`.trim();

    const title = document.createElement("strong");
    const normalized = normalizeActions(state.episode?.steps?.[idx] || {});
    const actionTypes = (s.action_types || []).filter(Boolean);
    if (!actionTypes.length && normalized.length) {
      for (const a of normalized) {
        const t = String(a?.type || "").trim();
        if (t && !actionTypes.includes(t)) actionTypes.push(t);
      }
    }
    const actionLabel = actionTypes.length ? actionTypes.join(", ") : (s.done ? "done" : "-");
    title.textContent = `Step ${idx}  \u2014  ${actionLabel}`;

    const meta = document.createElement("div");
    meta.className = "meta";
    const scoreDelta = ((s.after_score ?? 0) - (s.before_score ?? 0));
    const deltaStr = scoreDelta >= 0 ? `+${scoreDelta.toFixed(2)}` : scoreDelta.toFixed(2);
    meta.textContent = `${(s.before_score ?? 0).toFixed(2)} \u2192 ${(s.after_score ?? 0).toFixed(2)} (${deltaStr})${s.exec_ok === false ? "  \u00b7  EXEC FAIL" : ""}`;

    // Score bar
    const scoreBar = document.createElement("div");
    scoreBar.className = "score-bar";
    const fill = document.createElement("div");
    fill.className = "score-bar-fill";
    fill.style.width = `${Math.min(100, Math.max(0, (s.after_score ?? 0) * 100))}%`;
    scoreBar.appendChild(fill);

    item.appendChild(title);
    item.appendChild(meta);
    item.appendChild(scoreBar);
    item.onclick = () => setStep(idx);
    root.appendChild(item);
  }
}

/* ── Render: episode summary ───────────────────────────────── */

function renderEpisodeSummary() {
  const box = q("episodeSummary");
  if (!box) return;
  box.innerHTML = "";

  if (!state.episode) {
    const empty = document.createElement("div");
    empty.className = "empty";
    empty.textContent = "Select an episode.";
    box.appendChild(empty);
    return;
  }

  const ep = state.episode.episode || {};

  const title = document.createElement("div");
  const titleStrong = document.createElement("strong");
  titleStrong.textContent = ep.use_case || ep.task_id || "episode";
  title.appendChild(titleStrong);

  const meta = document.createElement("div");
  meta.className = "meta";
  const traceFile = String(state.episode.trace_file || "");
  meta.textContent = traceFile ? clipMiddle(traceFile, 90) : "-";
  meta.title = traceFile;

  const metaActions = document.createElement("div");
  metaActions.className = "chips";
  if (traceFile) {
    const copyBtn = document.createElement("button");
    copyBtn.className = "btn tiny";
    copyBtn.textContent = "Copy trace file";
    copyBtn.onclick = () => {
      void copyToClipboard(traceFile, "Trace file copied");
    };
    metaActions.appendChild(copyBtn);
  }

  const chips = document.createElement("div");
  chips.className = "chips";
  chips.appendChild(createChip(ep.success ? "PASS" : "FAIL", ep.success ? "ok" : "bad"));
  chips.appendChild(createChip(`score ${(ep.score ?? 0).toFixed(2)}`));
  chips.appendChild(createChip(`${ep.steps ?? 0} steps`));
  chips.appendChild(createChip(`${ep.task_seconds ?? 0}s`));
  chips.appendChild(createChip(`${ep.operator_duration_ms ?? 0}ms op`));
  chips.appendChild(createChip(`${ep.llm_calls ?? 0} llm`));
  if (ep.estimated_cost_usd) chips.appendChild(createChip(`$${Number(ep.estimated_cost_usd).toFixed(3)}`));
  if (ep.failure_category) chips.appendChild(createChip(ep.failure_category, "bad"));

  box.appendChild(title);
  box.appendChild(meta);
  box.appendChild(metaActions);
  box.appendChild(chips);
}

/* ── Render: step headline ─────────────────────────────────── */

function renderStepHeadline() {
  const box = q("stepHeadline");
  if (!box) return;

  const step = currentStep();
  if (!step) {
    box.innerHTML = '<div class="empty">Select a step to inspect.</div>';
    setText("stepPosition", "Step \u2014 / \u2014");
    return;
  }

  const before = step.before || {};
  const after = step.after || {};
  const execution = step.execution || {};
  const actions = normalizeActions(step).map((a) => a?.type).filter(Boolean).join(", ");
  const total = stepCount();
  setText("stepPosition", `${state.selectedStepIndex + 1} / ${total}`);

  box.innerHTML = "";

  const title = document.createElement("div");
  const titleStrong = document.createElement("strong");
  titleStrong.textContent = `Step ${state.selectedStepIndex}`;
  title.appendChild(titleStrong);
  if (actions) {
    const actionSpan = document.createElement("span");
    actionSpan.style.cssText = "color: var(--ink-soft); font-weight: 500; margin-left: 8px;";
    actionSpan.textContent = actions;
    title.appendChild(actionSpan);
  }

  const meta = document.createElement("div");
  meta.className = "meta";
  const execLabel = execution.exec_ok === false ? "EXEC FAIL" : "exec ok";
  meta.textContent = `${(before.score ?? 0).toFixed(2)} \u2192 ${(after.score ?? 0).toFixed(2)}  \u00b7  ${execLabel}`;

  const urls = document.createElement("div");
  urls.className = "meta";
  urls.textContent = `${before.url || "-"}\n\u2192 ${after.url || "-"}`;

  box.appendChild(title);
  box.appendChild(meta);
  box.appendChild(urls);
}

/* ── Render: step inspector ────────────────────────────────── */

function renderStepInspector() {
  const step = currentStep();
  if (!step) {
    setPre("reasoningPre", "");
    setPre("metricsPre", {});
    setPre("stateInPre", {});
    setPre("stateOutPre", {});
    setPre("actionsPre", []);
    setPre("executionPre", {});
    setPre("actRequestPre", {});
    setPre("actResponsePre", {});
    setPre("stateDiffPre", "");
    setPre("htmlDiffPre", "");
    setPre("replayStatusPre", "Select an episode/step first.");
    ["htmlBeforeText", "htmlAfterText"].forEach((id) => {
      const el = q(id);
      if (el) el.value = "";
    });
    ["htmlBeforeFrame", "htmlAfterFrame"].forEach((id) => {
      const el = q(id);
      if (el) el.srcdoc = "";
    });
    toggleImage("shotBefore", "shotBeforeEmpty", "");
    toggleImage("shotAfter", "shotAfterEmpty", "");
    return;
  }

  const agent = step.agent || {};
  const before = step.before || {};
  const after = step.after || {};
  const execution = step.execution || {};
  const actRequest = step.act_request || {};
  const actResponse = step.act_response || {};
  const diffs = step.diffs || {};

  setPre("reasoningPre", {
    reasoning: agent.reasoning || "",
    content: agent.content || "",
    done: Boolean(agent.done),
  });
  setPre("metricsPre", agent.metrics || {});
  setPre("stateInPre", agent.state_in || {});
  setPre("stateOutPre", agent.state_out || {});
  const normalizedActions = normalizeActions(step);
  const toolCalls = normalizeToolCalls(step);
  setPre("actionsPre", {
    actions: normalizedActions,
    tool_calls: toolCalls,
  });
  setPre("executionPre", execution);
  setPre("actRequestPre", actRequest);
  setPre("actResponsePre", actResponse);
  setPre("stateDiffPre", diffs.state || "");
  setPre("htmlDiffPre", diffs.html || "");

  const beforeHtml = String(actRequest.snapshot_html || "");
  const afterHtml = String(after.html || "");
  const htmlBeforeText = q("htmlBeforeText");
  const htmlAfterText = q("htmlAfterText");
  const htmlBeforeFrame = q("htmlBeforeFrame");
  const htmlAfterFrame = q("htmlAfterFrame");
  if (htmlBeforeText) htmlBeforeText.value = beforeHtml;
  if (htmlAfterText) htmlAfterText.value = afterHtml;
  if (htmlBeforeFrame) htmlBeforeFrame.srcdoc = beforeHtml;
  if (htmlAfterFrame) htmlAfterFrame.srcdoc = afterHtml;

  toggleImage("shotBefore", "shotBeforeEmpty", before.screenshot || "");
  toggleImage("shotAfter", "shotAfterEmpty", after.screenshot || "");

  const prevBtn = q("prevStepBtn");
  const nextBtn = q("nextStepBtn");
  if (prevBtn) prevBtn.disabled = state.selectedStepIndex <= 0;
  if (nextBtn) nextBtn.disabled = state.selectedStepIndex >= stepCount() - 1;
}

function toggleImage(imgId, emptyId, src) {
  const img = q(imgId);
  const empty = q(emptyId);
  if (!img || !empty) return;
  if (!src) {
    img.classList.add("hidden");
    empty.classList.remove("hidden");
    img.removeAttribute("src");
    return;
  }
  img.src = src;
  img.classList.remove("hidden");
  empty.classList.add("hidden");
}

/* ── Quick-section nav ─────────────────────────────────────── */

function renderQuickSections() {
  const nav = q("quickSections");
  if (!nav) return;
  nav.innerHTML = "";
  for (let i = 0; i < sectionOrder.length; i += 1) {
    const id = sectionOrder[i];
    const btn = document.createElement("button");
    btn.className = "btn";
    btn.textContent = `${i + 1} ${sectionLabels[i]}`;
    btn.onclick = () => openSection(id);
    nav.appendChild(btn);
  }
}

function openSection(id) {
  const el = q(id);
  if (!el) return;
  el.open = true;
  el.scrollIntoView({ behavior: "smooth", block: "start" });
  if (id === "secReplay") void refreshReplayStatus();
}

/* ── Step navigation ───────────────────────────────────────── */

function setStep(index) {
  const total = stepCount();
  state.selectedStepIndex = Math.min(Math.max(0, Number(index) || 0), Math.max(0, total - 1));
  syncUrlState();
  renderSteps();
  renderStepHeadline();
  renderStepInspector();

  // Scroll active step into view in the timeline
  const stepsRoot = q("steps");
  if (stepsRoot) {
    const active = stepsRoot.querySelector(".item.active");
    if (active) active.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }
}

function selectNeighborEpisode(direction) {
  const eps = state.filteredEpisodes || [];
  if (!eps.length) return;
  const idx = eps.findIndex((ep) => ep.episode_task_id === state.selectedEpisodeId);
  if (idx < 0) return;
  const next = Math.min(Math.max(idx + direction, 0), eps.length - 1);
  if (next !== idx) void loadEpisode(eps[next].episode_task_id);
}

/* ── Data loading ──────────────────────────────────────────── */

async function loadTraceCatalog() {
  const catalog = await fetch("/api/traces").then((r) => r.json());
  const all = Array.isArray(catalog.items) ? [...catalog.items] : [];
  const preferred = all
    .sort((a, b) => {
      const epA = Number(a?.episodes || 0);
      const epB = Number(b?.episodes || 0);
      if (epB !== epA) return epB - epA;
      const ca = String(a?.created_at_utc || "");
      const cb = String(b?.created_at_utc || "");
      return cb.localeCompare(ca);
    })[0] || null;
  if (!preferred) {
    state.traces = [];
    renderTracePicker();
    return;
  }
  state.traces = [
    {
      ...preferred,
      name: "eval run",
    },
  ];
  state.traceDir = String(preferred.trace_dir || "");
  renderTracePicker();
}

async function loadRun() {
  showLoading("episodesPane");
  try {
    state.run = await api("/api/run");
    if (!state.traceDir) state.traceDir = state.run.trace_dir || "";
    state.filteredEpisodes = [...(state.run.episodes || [])];
    renderRunSummary();
    renderEpisodeFilters();
    applyEpisodeFilters();

    const episodes = state.filteredEpisodes || [];
    if (episodes.length && !episodes.some((ep) => ep.episode_task_id === state.selectedEpisodeId)) {
      state.selectedEpisodeId = episodes[0].episode_task_id;
      state.selectedStepIndex = 0;
    }

    if (state.selectedEpisodeId) {
      await loadEpisode(state.selectedEpisodeId, { preserveStep: true });
    } else {
      renderEpisodes();
      renderSteps();
      renderEpisodeSummary();
      renderStepHeadline();
      renderStepInspector();
    }
  } finally {
    hideLoading("episodesPane");
  }
}

async function loadEpisode(episodeTaskId, options = {}) {
  state.selectedEpisodeId = episodeTaskId;
  if (!options.preserveStep) state.selectedStepIndex = 0;

  showLoading("stepsPane");
  try {
    state.episode = await api(`/api/episode/${encodeURIComponent(episodeTaskId)}`);
    const total = stepCount();
    if (state.selectedStepIndex >= total) state.selectedStepIndex = Math.max(0, total - 1);
    syncUrlState();
    renderEpisodes();
    renderSteps();
    renderEpisodeSummary();
    renderStepHeadline();
    renderStepInspector();
  } finally {
    hideLoading("stepsPane");
  }
}

async function applyTraceSelection() {
  const picker = q("tracePicker");
  if (!picker || !picker.value) return;
  state.traceDir = picker.value;
  state.run = null;
  state.episode = null;
  state.selectedEpisodeId = "";
  state.selectedStepIndex = 0;
  syncUrlState();
  await loadRun();
  toast("Trace loaded", "success");
}

/* ── Replay ────────────────────────────────────────────────── */

async function postReplay(path) {
  if (!state.selectedEpisodeId) return;
  const payload = {
    episode_task_id: state.selectedEpisodeId,
    step_index: state.selectedStepIndex,
  };
  state.replayStatus = await api(path, { method: "POST", body: JSON.stringify(payload) });
  setPre("replayStatusPre", state.replayStatus || {});
}

async function refreshReplayStatus() {
  try {
    state.replayStatus = await api("/api/replay/status");
    setPre("replayStatusPre", state.replayStatus || {});
  } catch (err) {
    setPre("replayStatusPre", String(err));
  }
}

function shouldPollReplay() {
  const replayDetails = q("secReplay");
  return Boolean(replayDetails?.open || state.replayStatus?.running);
}

function startReplayPolling() {
  if (state.replayPollHandle) return;
  state.replayPollHandle = window.setInterval(() => {
    if (shouldPollReplay()) void refreshReplayStatus();
  }, 1500);
}

/* ── Copy button handler ───────────────────────────────────── */

function handleCopyClick(ev) {
  const target = ev.target;
  if (!(target instanceof HTMLElement)) return;
  const copyId = target.getAttribute("data-copy-target");
  if (!copyId) return;
  const src = q(copyId);
  if (!src) return;
  const text = src.textContent || src.value || "";
  navigator.clipboard.writeText(String(text)).then(() => {
    const prev = target.textContent;
    target.textContent = "Copied!";
    setTimeout(() => { target.textContent = prev; }, 1200);
  });
}

/* ── Event listeners ───────────────────────────────────────── */

function setupListeners() {
  q("copyTraceDirBtn")?.addEventListener("click", () => {
    const full = String(state.run?.trace_dir || "");
    if (!full) return;
    void copyToClipboard(full, "Trace dir copied");
  });

  q("loadTraceBtn")?.addEventListener("click", () => void applyTraceSelection());
  q("refreshBtn")?.addEventListener("click", () => {
    void loadRun();
    toast("Refreshing...");
  });

  q("filterUseCase")?.addEventListener("change", applyEpisodeFilters);
  q("filterStatus")?.addEventListener("change", applyEpisodeFilters);
  q("filterFailureCategory")?.addEventListener("change", applyEpisodeFilters);

  q("prevStepBtn")?.addEventListener("click", () => setStep(state.selectedStepIndex - 1));
  q("nextStepBtn")?.addEventListener("click", () => setStep(state.selectedStepIndex + 1));

  q("replayStartBtn")?.addEventListener("click", () => void postReplay("/api/replay/start"));
  q("replayPauseBtn")?.addEventListener("click", () => void postReplay("/api/replay/pause"));
  q("replayResumeBtn")?.addEventListener("click", () => void postReplay("/api/replay/resume"));
  q("replayStepBtn")?.addEventListener("click", () => void postReplay("/api/replay/step"));
  q("replayResetBtn")?.addEventListener("click", () => void postReplay("/api/replay/reset"));

  for (const sectionId of sectionOrder) {
    q(sectionId)?.addEventListener("toggle", () => {
      if (sectionId === "secReplay" && q("secReplay")?.open) void refreshReplayStatus();
    });
  }

  document.addEventListener("click", handleCopyClick);

  document.addEventListener("keydown", (ev) => {
    const tag = String(ev.target?.tagName || "").toLowerCase();
    if (["input", "textarea", "select"].includes(tag)) return;

    if (ev.key === "j" && ev.shiftKey) {
      ev.preventDefault();
      selectNeighborEpisode(1);
      return;
    }
    if (ev.key === "k" && ev.shiftKey) {
      ev.preventDefault();
      selectNeighborEpisode(-1);
      return;
    }
    if (ev.key === "j") {
      ev.preventDefault();
      setStep(state.selectedStepIndex + 1);
      return;
    }
    if (ev.key === "k") {
      ev.preventDefault();
      setStep(state.selectedStepIndex - 1);
      return;
    }
    const n = Number(ev.key);
    if (Number.isInteger(n) && n >= 1 && n <= sectionOrder.length) {
      ev.preventDefault();
      openSection(sectionOrder[n - 1]);
    }
  });
}

/* ── Boot ──────────────────────────────────────────────────── */

async function boot() {
  renderQuickSections();
  setupListeners();
  startReplayPolling();
  await loadTraceCatalog();
  await loadRun();
  await refreshReplayStatus();
}

void boot().catch((err) => {
  const root = q("stepHeadline");
  if (root) root.innerHTML = `<div class="empty">Boot failed: ${String(err)}</div>`;
});
