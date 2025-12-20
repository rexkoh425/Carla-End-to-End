const runLog = document.getElementById("train-log");
const runList = document.getElementById("run-list");
const trackingState = {
  provider: "none",
  endpoint: "http://localhost:5000",
  experiment: "default",
  run: "lane_train",
};

function appendTrainLog(text) {
  const time = new Date().toLocaleTimeString();
  runLog.textContent = `[${time}] ${text}\n${runLog.textContent}`;
}

function readPlanFromForm() {
  return {
    model: document.getElementById("train-model").value,
    experiment: document.getElementById("experiment-name").value || "experiment",
    dataset: document.getElementById("train-dataset").value,
    epochs: parseInt(document.getElementById("epochs").value, 10) || 1,
    batch: parseInt(document.getElementById("batch-size").value, 10) || 1,
    lr: parseFloat(document.getElementById("lr").value) || 0.0001,
    device: document.getElementById("device").value,
    freeze: document.getElementById("freeze").value,
    augment: document.getElementById("augment-notes").value,
    checkpoint: document.getElementById("checkpoint").value,
    loadedConfig: window._loadedConfigText || "",
    configPath: document.getElementById("config-path").value || "",
    tracking: trackingState,
  };
}

function savePlan() {
  const plan = readPlanFromForm();
  const all = JSON.parse(localStorage.getItem("train-plans") || "[]");
  all.push(plan);
  localStorage.setItem("train-plans", JSON.stringify(all));
  appendTrainLog(`Saved plan ${plan.experiment}`);
  renderPlans();
}

function renderPlans() {
  const all = JSON.parse(localStorage.getItem("train-plans") || "[]");
  runList.innerHTML = "";
  if (!all.length) {
    runList.innerHTML = `<p class="subtext">No saved runs yet.</p>`;
    return;
  }
  all.forEach((plan, idx) => {
    const card = document.createElement("div");
    card.className = "card";
    card.style.padding = "12px";
    card.innerHTML = `
      <strong>${plan.experiment}</strong>
      <p class="subtext">${plan.model} · ${plan.dataset || "dataset?"}</p>
      <div class="badge">epochs ${plan.epochs}</div>
      <div class="badge">batch ${plan.batch}</div>
      <div class="toolbar" style="margin-top:8px;">
        <button class="secondary" data-action="load" data-idx="${idx}">Load</button>
        <button class="secondary" data-action="delete" data-idx="${idx}">Delete</button>
      </div>
    `;
    runList.appendChild(card);
  });

  runList.querySelectorAll("button").forEach((btn) => {
    btn.addEventListener("click", () => {
      const idx = parseInt(btn.dataset.idx, 10);
      const action = btn.dataset.action;
      const allPlans = JSON.parse(localStorage.getItem("train-plans") || "[]");
      if (action === "delete") {
        allPlans.splice(idx, 1);
        localStorage.setItem("train-plans", JSON.stringify(allPlans));
        renderPlans();
        return;
      }
      if (action === "load") {
        const plan = allPlans[idx];
        document.getElementById("train-model").value = plan.model;
        document.getElementById("experiment-name").value = plan.experiment;
        document.getElementById("train-dataset").value = plan.dataset;
        document.getElementById("epochs").value = plan.epochs;
        document.getElementById("batch-size").value = plan.batch;
        document.getElementById("lr").value = plan.lr;
        document.getElementById("device").value = plan.device;
        document.getElementById("freeze").value = plan.freeze;
        document.getElementById("augment-notes").value = plan.augment;
        document.getElementById("checkpoint").value = plan.checkpoint;
        window._loadedConfigText = plan.loadedConfig;
        document.getElementById("config-path").value = plan.configPath || "";
        if (window._loadedConfigText) {
          document.getElementById("config-preview").value = window._loadedConfigText;
        }
        appendTrainLog(`Loaded plan ${plan.experiment}`);
      }
    });
  });
}

function startTraining(simulated) {
  const plan = readPlanFromForm();
  const action = simulated ? "dry run" : "training";
  appendTrainLog(`Started ${action} for ${plan.experiment} (${plan.model})`);
  appendTrainLog(
    ` dataset=${plan.dataset || "?"} epochs=${plan.epochs} batch=${plan.batch} lr=${plan.lr} device=${plan.device}`
  );
  postTrainingEvent(simulated ? "train_dryrun" : "train_start", plan);
}

function exportPlan() {
  const plan = readPlanFromForm();
  const blob = new Blob([JSON.stringify(plan, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `${plan.experiment || "training-plan"}.json`;
  a.click();
  URL.revokeObjectURL(url);
  appendTrainLog("Exported plan JSON");
  postTrainingEvent("train_export", plan);
}

function handleConfigFile(event) {
  const file = event.target.files?.[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = (e) => {
    window._loadedConfigText = e.target.result;
    document.getElementById("config-preview").value = window._loadedConfigText;
    appendTrainLog(`Loaded config file (${file.name}, ${file.size} bytes)`);
  };
  reader.readAsText(file);
}

async function loadConfigFromPath() {
  const path = document.getElementById("config-path").value;
  if (!path) {
    appendTrainLog("Config path is empty.");
    return;
  }
  try {
    const res = await fetch(path);
    if (!res.ok) {
      appendTrainLog(`Failed to fetch config (${res.status})`);
      return;
    }
    const text = await res.text();
    window._loadedConfigText = text;
    document.getElementById("config-preview").value = text;
    appendTrainLog(`Loaded config from ${path}`);
  } catch (err) {
    appendTrainLog(`Error loading config: ${err.message}`);
  }
}

function loadTrainTracking() {
  const stored = localStorage.getItem("tracking-config");
  if (stored) {
    const parsed = JSON.parse(stored);
    trackingState.provider = parsed.provider || "none";
    trackingState.endpoint = parsed.endpoint || trackingState.endpoint;
    trackingState.experiment = parsed.experiment || trackingState.experiment;
    trackingState.run = parsed.run || trackingState.run;
  }
  document.getElementById("train-tracker-provider").value = trackingState.provider;
  document.getElementById("train-tracker-endpoint").value = trackingState.endpoint;
  document.getElementById("train-tracker-experiment").value = trackingState.experiment;
  document.getElementById("train-tracker-run").value = trackingState.run;
}

function saveTrainTracking() {
  trackingState.provider = document.getElementById("train-tracker-provider").value;
  trackingState.endpoint = document.getElementById("train-tracker-endpoint").value;
  trackingState.experiment = document.getElementById("train-tracker-experiment").value || "default";
  trackingState.run = document.getElementById("train-tracker-run").value || "run";
  localStorage.setItem("tracking-config", JSON.stringify(trackingState));
  appendTrainLog(`Tracking saved (${trackingState.provider})`);
}

async function pingTrainTracking() {
  if (trackingState.provider === "none") {
    appendTrainLog("Tracking disabled.");
    return;
  }
  try {
    const res = await fetch(trackingState.endpoint, { method: "GET" });
    appendTrainLog(`Ping ${trackingState.endpoint} -> ${res.status}`);
  } catch (err) {
    appendTrainLog(`Ping failed: ${err.message}`);
  }
}

async function postTrainingEvent(event, payload) {
  if (trackingState.provider === "none" || !trackingState.endpoint) return;
  const body = {
    provider: trackingState.provider,
    experiment: trackingState.experiment,
    run: trackingState.run,
    event,
    payload,
  };
  try {
    const res = await fetch(`${trackingState.endpoint}/api/track`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    appendTrainLog(`Tracking POST ${res.status}`);
  } catch (err) {
    appendTrainLog(`Tracking failed: ${err.message}`);
  }
}

function initTrainingPage() {
  renderPlans();
  document.getElementById("train-btn").onclick = () => startTraining(false);
  document.getElementById("dryrun-btn").onclick = () => startTraining(true);
  document.getElementById("save-train-btn").onclick = savePlan;
  document.getElementById("export-train-json").onclick = exportPlan;
  document.getElementById("config-file").addEventListener("change", handleConfigFile);
  document.getElementById("train-save-tracker").onclick = saveTrainTracking;
  document.getElementById("train-ping-tracker").onclick = pingTrainTracking;
  document.getElementById("load-config-path").onclick = loadConfigFromPath;
  loadTrainTracking();
  appendTrainLog("Training planner ready.");

  const resultsFile = document.getElementById("results-file");
  if (resultsFile) {
    resultsFile.addEventListener("change", handleResultsFile);
  }
}

document.addEventListener("DOMContentLoaded", initTrainingPage);

// Results visualization
const resultsState = {
  metrics: [],
};

function handleResultsFile(event) {
  const file = event.target.files?.[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = (e) => {
    try {
      const text = e.target.result;
      const json = JSON.parse(text);
      const metrics = normalizeMetrics(json);
      if (!metrics.length) {
        appendTrainLog("No metrics found in file.");
        return;
      }
      resultsState.metrics = metrics;
      renderResultsSummary(metrics);
      renderResultsChart(metrics);
      appendTrainLog(`Loaded metrics (${file.name})`);
    } catch (err) {
      appendTrainLog(`Metrics parse error: ${err.message}`);
    }
  };
  reader.readAsText(file);
}

function normalizeMetrics(raw) {
  if (!raw) return [];
  if (Array.isArray(raw)) return sanitizeMetrics(raw);
  if (Array.isArray(raw.metrics)) return sanitizeMetrics(raw.metrics);
  return [];
}

function sanitizeMetrics(arr) {
  return arr
    .map((m) => ({
      epoch: Number(m.epoch ?? m.step ?? m.iter ?? 0),
      train: m.train_loss ?? m.loss ?? null,
      val: m.val_loss ?? m.validation_loss ?? null,
      acc: m.acc ?? m.accuracy ?? m.val_acc ?? null,
    }))
    .filter((m) => !Number.isNaN(m.epoch))
    .sort((a, b) => a.epoch - b.epoch);
}

function renderResultsSummary(metrics) {
  const summary = document.getElementById("results-summary");
  if (!summary) return;
  const last = metrics[metrics.length - 1];
  summary.innerHTML = `
    <div>Epochs: <strong>${metrics.length}</strong></div>
    <div>Final train loss: <strong>${last.train ?? "n/a"}</strong></div>
    <div>Final val loss: <strong>${last.val ?? "n/a"}</strong></div>
    <div>Final acc: <strong>${last.acc ?? "n/a"}</strong></div>
  `;
}

function renderResultsChart(metrics) {
  const canvas = document.getElementById("results-chart");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const padding = { left: 40, right: 10, top: 10, bottom: 30 };
  const width = canvas.width - padding.left - padding.right;
  const height = canvas.height - padding.top - padding.bottom;

  const epochs = metrics.map((m) => m.epoch);
  const losses = metrics.map((m) => m.train ?? m.val).filter((v) => v !== null && v !== undefined);
  if (!epochs.length || !losses.length) return;

  const minEpoch = Math.min(...epochs);
  const maxEpoch = Math.max(...epochs);
  const lossVals = metrics.flatMap((m) => [m.train, m.val]).filter((v) => v !== null && v !== undefined);
  const minLoss = Math.min(...lossVals);
  const maxLoss = Math.max(...lossVals);

  const scaleX = (e) => padding.left + ((e - minEpoch) / Math.max(1, maxEpoch - minEpoch)) * width;
  const scaleY = (v) => padding.top + (1 - (v - minLoss) / Math.max(1e-6, maxLoss - minLoss)) * height;

  ctx.strokeStyle = "rgba(255,255,255,0.08)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(padding.left, padding.top);
  ctx.lineTo(padding.left, padding.top + height);
  ctx.lineTo(padding.left + width, padding.top + height);
  ctx.stroke();

  function plotLine(key, color) {
    const pts = metrics.filter((m) => m[key] !== null && m[key] !== undefined);
    if (!pts.length) return;
    ctx.beginPath();
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    pts.forEach((m, idx) => {
      const x = scaleX(m.epoch);
      const y = scaleY(m[key]);
      if (idx === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }

  plotLine("train", "#7af2c4");
  plotLine("val", "#9cf8ff");

  ctx.fillStyle = "rgba(255,255,255,0.7)";
  ctx.font = "12px Manrope, sans-serif";
  ctx.fillText(`Epoch ${minEpoch}-${maxEpoch}`, padding.left, canvas.height - 10);
}
