const models = [
  {
    id: "mask2former",
    name: "Mask2Former",
    tags: ["segmentation", "lanes"],
    desc: "Panoptic segmentation for road and lane structure.",
    template: {
      model: { id: "facebook/mask2former-swin-large-cityscapes-panoptic", config: "Mask2Former/mask2former_finetune_config.yaml", device: "auto" },
      data: { source: "Storage/Pooled_75205/val", image_root: "Storage/Pooled_75205" },
      runtime: { batch: 4, imgsz: 640, overlay_alpha: 0.6, min_score: 0.4 },
    },
  },
  {
    id: "laneatt",
    name: "LaneATT",
    tags: ["detection", "lanes"],
    desc: "Anchor-based lane detector for fast previews.",
    template: {
      model: { config: "LaneATT/laneatt_model_config.yaml", weights: "LaneATT/checkpoints/laneatt_100.pt" },
      data: { source: "Storage/lanes/val", resize: [640, 360] },
      runtime: { positive_threshold: 0.5, nms: true, nms_threshold: 40 },
    },
  },
  {
    id: "yolopv2",
    name: "YOLOPv2",
    tags: ["multi-task", "lanes"],
    desc: "Drivable area + lane + detection (torchscript, 1280x720).",
    template: {
      model: { weights: "YOLOPv2/data/weights/yolopv2.pt", device: "cuda:0" },
      data: { source: "Storage/lanes/val" },
      runtime: { conf_threshold: 0.3, iou_threshold: 0.45, img_size: 640 },
    },
  },
  {
    id: "yolop",
    name: "YOLOP",
    tags: ["multi-task"],
    desc: "Joint lane + drivable area detector.",
    template: {
      model: { id: "yolop-base", config: "YOLOP/configs/yolop.yaml", device: "auto" },
      data: { source: "Storage/yolop/val" },
      runtime: { conf: 0.25, iou: 0.45 },
    },
  },
  {
    id: "yoloe",
    name: "YOLOE",
    tags: ["detection"],
    desc: "Generic detection backbone.",
    template: { model: { config: "YOLOE/config.yaml", weights: "YOLOE/weights/best.pt" }, data: {}, runtime: { conf: 0.25 } },
  },
  {
    id: "blip",
    name: "BLIP",
    tags: ["vision-language"],
    desc: "Image captioning / VQA.",
    template: { model: { id: "Salesforce/blip-image-captioning-base" }, runtime: { max_length: 64 } },
  },
  {
    id: "cyclegan",
    name: "CycleGAN",
    tags: ["image2image"],
    desc: "Unpaired image-to-image translation.",
    template: { model: { config: "CycleGAN/config.yaml" }, runtime: { direction: "AtoB" } },
  },
  {
    id: "depthanything",
    name: "DepthAnything",
    tags: ["depth"],
    desc: "Monocular depth estimation.",
    template: { model: { id: "depthanything-medium" }, runtime: { imgsz: 640 } },
  },
  {
    id: "dinox",
    name: "DINO-X",
    tags: ["vision"],
    desc: "Self-supervised vision transformer.",
    template: { model: { config: "DINO-X/config.yaml" }, runtime: {} },
  },
  {
    id: "internvl",
    name: "InternVL",
    tags: ["vision-language"],
    desc: "Multimodal large model.",
    template: { model: { id: "InternVL/InternVL2-Llama3-76B" }, runtime: { max_tokens: 128 } },
  },
  {
    id: "moe",
    name: "MoE",
    tags: ["expert"],
    desc: "Mixture-of-experts experiments.",
    template: { model: { config: "MoE/config.yaml" }, runtime: { experts: 8 } },
  },
  {
    id: "persformer",
    name: "PersFormer",
    tags: ["geometry"],
    desc: "Perspective-aware transformer.",
    template: { model: { config: "PersFormer/config.yaml" }, runtime: {} },
  },
  {
    id: "qwen3",
    name: "Qwen3",
    tags: ["llm"],
    desc: "Qwen 3 series LLM.",
    template: { model: { id: "Qwen/Qwen2.5-14B" }, runtime: { max_tokens: 256 } },
  },
  {
    id: "groundingdino",
    name: "GroundingDINO",
    tags: ["detection"],
    desc: "Open-vocabulary detection with text prompts.",
    template: {
      model: {
        config: "models/detector/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        weights: "models/weights/groundingdino_swint_ogc.pth",
        prompt: "car . person . road",
        box_threshold: 0.3,
        text_threshold: 0.25,
        device: "cuda:0",
      },
    },
  },
  {
    id: "deepseek",
    name: "DeepSeek 7B Chat",
    tags: ["llm"],
    desc: "deepseek-ai/deepseek-llm-7b-chat.",
    template: { model: { id: "deepseek-ai/deepseek-llm-7b-chat" }, runtime: { max_tokens: 256 } },
  },
  {
    id: "sdxl",
    name: "SDXL",
    tags: ["diffusion"],
    desc: "Image generation.",
    template: { model: { id: "stabilityai/stable-diffusion-xl-base-1.0" }, runtime: { steps: 25, guidance: 7.5 } },
  },
  {
    id: "tcntrader",
    name: "TCNTransformerMoETrader",
    tags: ["timeseries"],
    desc: "Trading MoE time-series model.",
    template: { model: { config: "TCNTransformerMoETrader/config.yaml" }, runtime: {} },
  },
  {
    id: "videoharvester",
    name: "VideoHarvester",
    tags: ["video"],
    desc: "Video ingestion/harvesting pipelines.",
    template: { model: { config: "VideoHarvester/config.yaml" }, runtime: {} },
  },
  {
    id: "videoproclab",
    name: "VideoProcessingLab",
    tags: ["video"],
    desc: "Video post-processing lab.",
    template: { model: { config: "VideoProcessingLab/config.yaml" }, runtime: {} },
  },
  {
    id: "webharvesting",
    name: "WebHarvesting",
    tags: ["scrape"],
    desc: "Web scraping/harvesting utilities.",
    template: { model: { config: "WebHarvesting/config.yaml" }, runtime: {} },
  },
  {
    id: "custom",
    name: "Custom Node",
    tags: ["script"],
    desc: "Wrap a custom script or API endpoint.",
    template: { model: { id: "local-script.py", args: ["--foo", "bar"] }, data: {}, runtime: {} },
  },
];

const state = {
  selectedModel: models[0].id,
  config: {},
  graph: { nodes: [], links: [] },
  activeLinkStart: null,
  activeLinkRole: null,
  previewStart: null,
  selectedLink: null,
  tracking: {
    provider: "none",
    endpoint: "http://localhost:5000",
    experiment: "default",
    run: "lane_preview",
  },
  filter: "",
  inventory: {},
  validationOk: false,
};

const INPUT_BASE = "Storage"; // base folder for relative picks
const OUTPUT_BASE = "Storage/Output"; // default output base when single-file
const OUTPUT_BASES = [OUTPUT_BASE, OUTPUT_BASE.toLowerCase(), "Storage/output"];

const htmlEscape = (str = "") =>
  str.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c] || c));

function defaultMetaForType(type) {
  if (type === "Input" || type === "Data")
    return { source: "Storage/sample_video.mp4", folder: "Storage/input_folder", mode: "single", previewLocal: "" };
  if (type === "Connector") return { middleware: "" };
  if (type === "Prompt") return { prompt: "car . person . road" };
  if (type === "Debug") return {};
  if (type === "Post") return { output: "Storage/output/", outputPreviewPath: "", outputPreviewLocal: "" };
  if (type === "Model") return { modelId: state.selectedModel || models[0].id };
  return {};
}

function ensureNodeMeta(node) {
  if (!node.meta) node.meta = {};
  const defaults = defaultMetaForType(node.type);
  Object.keys(defaults).forEach((k) => {
    if (node.meta[k] === undefined) node.meta[k] = defaults[k];
  });
  if (node.type === "Model") {
    const m = getModelById(node.meta.modelId) || models[0];
    node.title = m?.name || node.title;
    // Auto-fill LaneATT defaults if selected and missing
    if (node.meta.modelId === "laneatt") {
      // Prefer the pip-compatible config (avoids !!python/tuple tags)
      const badExperimentConfig = "LaneATT/experiments/laneatt_r34_tusimple/config.yaml";
      if (!node.meta.config || node.meta.config === badExperimentConfig) {
        node.meta.config = "LaneATT/laneatt_model_config.yaml";
      }
      if (!node.meta.weights || node.meta.weights.includes("experiments/")) {
        node.meta.weights = "LaneATT/checkpoints/laneatt_100.pt";
      }
    }
    if (node.meta.modelId === "yolopv2") {
      if (!node.meta.weights) {
        node.meta.weights = "YOLOPv2/data/weights/yolopv2.pt";
      }
      if (!node.meta.device) {
        node.meta.device = "cuda:0";
      }
    }
  }
  return node.meta;
}

function getModelById(id) {
  return models.find((m) => m.id === id);
}

function modelLabel(model) {
  if (model?.name) return model.name;
  const id = model?.id || "";
  return id
    .replace(/[-_]+/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase())
    .trim();
}

const schemas = {
  mask2former: { required: ["model"], props: { model: ["config"] } },
  yolopv2: { required: ["model"], props: { model: ["weights"] } },
  laneatt: { required: ["model"], props: { model: ["config", "weights"] } },
  yolop: { required: ["model"], props: { model: ["config"] } },
  yoloe: { required: ["model"], props: { model: ["config"] } },
  sdxl: { required: ["model"], props: { model: ["id"] } },
  blip: { required: ["model"], props: { model: ["id"] } },
};

const el = (id) => document.getElementById(id);
const hasModelUI = () => !!el("model-grid") || !!el("model-select");
const hasConfigUI = () => !!el("config-editor");
const hasTrackingUI = () => !!el("tracker-provider");
const hasGraphUI = () => !!el("graph-canvas");

function bindIf(id, event, handler) {
  const node = el(id);
  if (node) node.addEventListener(event, handler);
  return node;
}

function getLayerRect() {
  const layer = el("link-layer");
  return layer ? layer.getBoundingClientRect() : null;
}

function getConnectorPoint(nodeId, role = "out") {
  const layerRect = getLayerRect();
  if (!layerRect) return null;
  const connector = document.querySelector(`.node[data-id="${nodeId}"] .connector[data-role="${role}"]`);
  if (!connector) return null;
  const rect = connector.getBoundingClientRect();
  return {
    x: rect.left - layerRect.left + rect.width / 2,
    y: rect.top - layerRect.top + rect.height / 2,
  };
}

function connectorRoles(type) {
  switch (type) {
    case "Input":
    case "Data":
      return ["out"];
    case "Post":
      return ["in"];
    case "Debug":
      return ["in", "out"];
    case "Prompt":
      return ["out"];
    case "Model":
    case "Connector":
      // left = input, right = output
      return ["in", "out"];
    default:
      return ["out", "in"];
  }
}

function loadModelConfig(modelId) {
  const model = models.find((m) => m.id === modelId);
  const stored = localStorage.getItem(`model-config-${modelId}`);
  const config = stored ? JSON.parse(stored) : model.template;
  state.config = config;
  if (hasConfigUI()) {
    if (el("config-name")) el("config-name").value = modelId;
    if (el("runtime-device")) el("runtime-device").value = config.model?.device ?? "auto";
    if (el("config-editor")) el("config-editor").value = JSON.stringify(config, null, 2);
  }
  const dropdown = el("model-select");
  if (dropdown && dropdown.value !== modelId) dropdown.value = modelId;
  updateStatusBar();
}

function renderModelCards() {
  if (!hasModelUI()) return;
  const grid = el("model-grid");
  grid.innerHTML = "";
  const filtered = models.filter((m) => {
    if (!state.filter) return true;
    const needle = state.filter.toLowerCase();
    return (
      m.name.toLowerCase().includes(needle) ||
      m.id.toLowerCase().includes(needle) ||
      m.tags.some((t) => t.toLowerCase().includes(needle))
    );
  });
  el("model-count").textContent = `${filtered.length} of ${models.length}`;
  const dropdown = el("model-select");
  if (dropdown) {
    dropdown.innerHTML = "";
    filtered.forEach((m) => {
      const opt = document.createElement("option");
      opt.value = m.id;
      opt.textContent = m.name;
      dropdown.appendChild(opt);
    });
    dropdown.value = state.selectedModel;
  }

  filtered.forEach((model) => {
    const card = document.createElement("div");
    card.className = `model-card ${state.selectedModel === model.id ? "active" : ""}`;
    card.innerHTML = `
      <div style="display:flex;justify-content:space-between;align-items:center;">
        <strong>${model.name}</strong>
        <span class="badge">${model.tags.join(" · ")}</span>
      </div>
      <p class="subtext" style="margin-top:6px;">${model.desc}</p>
    `;
    card.onclick = () => {
      state.selectedModel = model.id;
      renderModelCards();
      loadModelConfig(model.id);
      appendLog(`Selected ${model.name}`);
    };
    grid.appendChild(card);
  });
}

function beautifyConfig() {
  const editor = el("config-editor");
  if (!editor) return;
  try {
    const parsed = JSON.parse(editor.value);
    editor.value = JSON.stringify(parsed, null, 2);
    state.config = parsed;
    validateConfig();
  } catch (err) {
    appendLog(`Format error: ${err.message}`);
  }
}

function resetConfig() {
  const model = models.find((m) => m.id === state.selectedModel);
  state.config = model.template;
  if (hasConfigUI()) {
    if (el("config-editor")) el("config-editor").value = JSON.stringify(model.template, null, 2);
    if (el("config-name")) el("config-name").value = model.id;
    if (el("runtime-device")) el("runtime-device").value = model.template.model?.device ?? "auto";
  }
}

function saveConfig() {
  try {
    const editor = el("config-editor");
    if (!editor) return;
    const parsed = JSON.parse(editor.value);
    const modelId = state.selectedModel;
    localStorage.setItem(`model-config-${modelId}`, JSON.stringify(parsed));
    appendLog(`Saved config for ${modelId}`);
    validateConfig();
  } catch (err) {
    appendLog(`Save failed: ${err.message}`);
  }
}

function appendLog(text) {
  const box = el("log-box");
  if (!box) return;
  const time = new Date().toLocaleTimeString();
  box.textContent = `[${time}] ${text}\n${box.textContent}`;
}

function logAction(text) {
  appendLog(text);
  try {
    console.debug(text);
  } catch (_) {}
}

function initGraph() {
  const stored = localStorage.getItem("graph-state");
  if (stored) {
    try {
      const parsed = JSON.parse(stored);
      if (Array.isArray(parsed?.nodes) && parsed.nodes.length) {
        // Preserve nodes but start with no connections so the user draws links explicitly.
        state.graph = { nodes: parsed.nodes, links: [] };
        state.graph.nodes.forEach(ensureNodeMeta);
        return;
      }
    } catch (err) {
      console.warn("Graph restore failed, using defaults", err);
    }
  }
  // Fallback default graph
  state.graph = {
    nodes: [
      { id: "input-1", title: "Camera feed", type: "Input", x: 60, y: 180, meta: defaultMetaForType("Input") },
      { id: "model-1", title: "Mask2Former", type: "Model", x: 320, y: 180, meta: defaultMetaForType("Model") },
      { id: "post-1", title: "Overlay/Save", type: "Post", x: 580, y: 180, meta: defaultMetaForType("Post") },
    ],
    links: [],
  };
}

function persistGraph() {
  localStorage.setItem("graph-state", JSON.stringify(state.graph));
}

function loadTracking() {
  const stored = localStorage.getItem("tracking-config");
  if (stored) {
    state.tracking = JSON.parse(stored);
  }
  if (hasTrackingUI()) {
    el("tracker-provider").value = state.tracking.provider;
    el("tracker-endpoint").value = state.tracking.endpoint;
    el("tracker-experiment").value = state.tracking.experiment;
    el("tracker-run").value = state.tracking.run;
  }
}

function saveTracking() {
  state.tracking = {
    provider: el("tracker-provider").value,
    endpoint: el("tracker-endpoint").value,
    experiment: el("tracker-experiment").value || "default",
    run: el("tracker-run").value || "run",
  };
  localStorage.setItem("tracking-config", JSON.stringify(state.tracking));
  appendLog(`Tracking saved (${state.tracking.provider})`);
  updateStatusBar();
}

async function pingTracking() {
  if (state.tracking.provider === "none") {
    appendLog("Tracking disabled.");
    return;
  }
  try {
    const url = state.tracking.endpoint;
    const res = await fetch(url, { method: "GET" });
    appendLog(`Ping ${url} -> ${res.status}`);
  } catch (err) {
    appendLog(`Ping failed: ${err.message}`);
  }
}

function addNode(type, pos) {
  const node = {
    id: `${type}-${Date.now()}`,
    title:
      type === "Model"
        ? models.find((m) => m.id === state.selectedModel).name
        : type === "Data" || type === "Input"
        ? "Data Input"
        : type === "Connector"
        ? "Data Middleware"
        : type === "Post"
        ? "Output"
        : `${type} node`,
    type,
    x: pos?.x ?? 140 + Math.random() * 280,
    y: pos?.y ?? 120 + Math.random() * 200,
    meta: defaultMetaForType(type),
  };
  state.graph.nodes.push(node);
  renderGraph();
  persistGraph();
}

function clearGraph() {
  state.graph = { nodes: [], links: [] };
  renderGraph();
  persistGraph();
  state.activeLinkStart = null;
  state.previewStart = null;
  removePreviewLine();
  clearConnectorHighlights();
}

function drawLinks() {
  const layer = el("link-layer");
  const wrapper = el("graph-wrapper");
  const rect = wrapper.getBoundingClientRect();
  layer.setAttribute("width", rect.width);
  layer.setAttribute("height", rect.height);
  ensureLinkLayerDefs(layer);
  layer.querySelectorAll(".link-path, .link-dot, .link-delete, .link-delete-label, .link-delete-hit, #preview-line").forEach((n) => n.remove());
  const validNodeIds = new Set(state.graph.nodes.map((n) => n.id));
  state.graph.links = state.graph.links.filter((l) => validNodeIds.has(l.from) && validNodeIds.has(l.to));

  state.graph.links.forEach((link) => {
    const fromCenter = getConnectorPoint(link.from, "out");
    const toCenter = getConnectorPoint(link.to, "in");
    if (!fromCenter || !toCenter) return;

    // Slight bezier bend to make crossings readable
    const dx = toCenter.x - fromCenter.x;
    const curvature = Math.min(140, Math.max(32, Math.abs(dx) * 0.35));
    const controlX = fromCenter.x + dx * 0.5;
    const control1 = `${fromCenter.x + Math.sign(dx || 1) * curvature} ${fromCenter.y}`;
    const control2 = `${controlX} ${toCenter.y}`;
    const d = `M ${fromCenter.x} ${fromCenter.y} C ${control1}, ${control2}, ${toCenter.x} ${toCenter.y}`;

    // Invisible hit area for easier clicking/removal
    const hit = document.createElementNS("http://www.w3.org/2000/svg", "path");
    hit.classList.add("link-hit");
    hit.setAttribute("d", d);
    hit.dataset.from = link.from;
    hit.dataset.to = link.to;
    hit.addEventListener("click", (e) => {
      e.stopPropagation();
      selectLink(link.from, link.to, null);
    });
    hit.addEventListener("dblclick", (e) => {
      e.stopPropagation();
      removeLink(link.from, link.to);
    });
    layer.appendChild(hit);

    const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
    path.classList.add("link-path");
    path.setAttribute("d", d);
    path.dataset.from = link.from;
    path.dataset.to = link.to;
    path.addEventListener("click", (e) => {
      e.stopPropagation();
      selectLink(link.from, link.to, path);
    });
    path.addEventListener("dblclick", (e) => {
      e.stopPropagation();
      removeLink(link.from, link.to);
    });
    layer.appendChild(path);

    const dot = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    dot.classList.add("link-dot");
    dot.setAttribute("cx", controlX);
    dot.setAttribute("cy", (fromCenter.y + toCenter.y) / 2);
    dot.setAttribute("r", 4);
    layer.appendChild(dot);

    // Inline delete control at the midpoint for easy removal
    const midX = (fromCenter.x + toCenter.x) / 2;
    const midY = (fromCenter.y + toCenter.y) / 2;
    const deleteBtn = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    deleteBtn.classList.add("link-delete");
    deleteBtn.setAttribute("cx", midX);
    deleteBtn.setAttribute("cy", midY);
    deleteBtn.setAttribute("r", 9);
    deleteBtn.dataset.from = link.from;
    deleteBtn.dataset.to = link.to;
    deleteBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      logAction(`Delete link (dot) ${link.from} -> ${link.to}`);
      removeLink(link.from, link.to);
    });
    layer.appendChild(deleteBtn);

    const deleteHit = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    deleteHit.setAttribute("x", midX - 14);
    deleteHit.setAttribute("y", midY - 14);
    deleteHit.setAttribute("width", 28);
    deleteHit.setAttribute("height", 28);
    deleteHit.setAttribute("fill", "transparent");
    deleteHit.dataset.from = link.from;
    deleteHit.dataset.to = link.to;
    deleteHit.classList.add("link-delete-hit");
    deleteHit.addEventListener("click", (e) => {
      e.stopPropagation();
      logAction(`Delete link (hit) ${link.from} -> ${link.to}`);
      removeLink(link.from, link.to);
    });
    layer.appendChild(deleteHit);

    const deleteLabel = document.createElementNS("http://www.w3.org/2000/svg", "text");
    deleteLabel.classList.add("link-delete-label");
    deleteLabel.setAttribute("x", midX);
    deleteLabel.setAttribute("y", midY);
    deleteLabel.textContent = "x";
    deleteLabel.dataset.from = link.from;
    deleteLabel.dataset.to = link.to;
    deleteLabel.addEventListener("click", (e) => {
      e.stopPropagation();
      logAction(`Delete link (label) ${link.from} -> ${link.to}`);
      removeLink(link.from, link.to);
    });
    layer.appendChild(deleteLabel);
  });

  // keep selection only if it still exists
  if (state.selectedLink) {
    const stillExists = state.graph.links.some(
      (l) => l.from === state.selectedLink.from && l.to === state.selectedLink.to
    );
    if (stillExists) {
      highlightSelectedLink();
    } else {
      state.selectedLink = null;
    }
  }
}

function renderGraph() {
  const canvas = el("graph-canvas");
  canvas.innerHTML = "";
  state.graph.nodes.forEach((node) => {
    ensureNodeMeta(node);
    const div = document.createElement("div");
    div.className = `node node-${node.type.toLowerCase()}`;
    div.style.left = `${node.x}px`;
    div.style.top = `${node.y}px`;
    div.dataset.id = node.id;
    div.dataset.type = node.type;
    const roles = connectorRoles(node.type);
    const connectorClass =
      roles.length === 1 && roles[0] === "out"
        ? "connectors connectors-out-only"
        : roles.length === 1 && roles[0] === "in"
        ? "connectors connectors-in-only"
        : "connectors connectors-dual";
    const connectors = roles
      .map(
        (role) => `<div class="connector" data-role="${role}" title="${role === "out" ? "Output" : "Input"}"></div>`
      )
      .join("");
    const controlHtml = renderNodeControl(node);
    div.innerHTML = `
      <h4>${htmlEscape(node.title)} <button type="button" class="node-remove" data-node="${node.id}" title="Delete block">x</button></h4>
      <span class="type">${htmlEscape(node.type)}</span>
      ${controlHtml}
      <div class="${connectorClass}">
        ${connectors}
      </div>
    `;
    setupDrag(div, node);
    div.querySelectorAll(".connector").forEach((c) => {
      c.addEventListener("pointerdown", (e) => {
        e.stopPropagation();
        startLinking(node.id, c.dataset.role);
      });
      c.addEventListener("pointerenter", (e) => {
        e.stopPropagation();
        if (state.activeLinkStart) {
          tryFinalizeLink(node.id, c.dataset.role);
        }
      });
      c.addEventListener("click", (e) => {
        // click fallback
        e.stopPropagation();
        handleLinkClick(node.id, c.dataset.role);
      });
    });
    div.querySelectorAll(".node-remove").forEach((btn) => {
      btn.addEventListener("click", (e) => {
        e.stopPropagation();
        logAction(`Delete node ${node.id}`);
        deleteNode(node.id);
      });
    });
    bindNodeControls(div, node);
    canvas.appendChild(div);
  });
  drawLinks();
}

function renderNodeControl(node) {
  const meta = ensureNodeMeta(node);
  const base = (key, placeholder, actionLabel = "Pick", kind = "file", showView = true) => `
    <div class="node-control" data-control-key="${key}">
      <div class="node-control-row">
        <input class="node-path-input" data-key="${key}" data-node="${node.id}" placeholder="${placeholder}" value="${htmlEscape(
          meta[key] || ""
        )}" />
        <button type="button" class="secondary tiny node-pick" data-node="${node.id}" data-kind="${kind}" data-key="${key}">${actionLabel}</button>
        <button type="button" class="secondary tiny node-clear" title="Clear" data-node="${node.id}" data-key="${key}">x</button>
        ${showView ? `<button type="button" class="secondary tiny node-view" title="Preview" data-node="${node.id}" data-key="${key}">View</button>` : ""}
      </div>
    </div>
  `;

  if (node.type === "Input" || node.type === "Data") {
    const mode = meta.mode || "single";
    const previewUrl =
      mode === "single"
        ? meta.previewLocal || (meta.source ? buildFileUrl(meta.source) : "")
        : "";
    const previewClass = previewUrl ? "node-inline-preview visible" : "node-inline-preview";
    return `
      <div class="node-control">
        <div class="node-control-row">
          <label style="font-size:12px;color:var(--muted);">Mode</label>
          <select class="node-input-mode" data-node="${node.id}">
            <option value="single" ${mode === "single" ? "selected" : ""}>Single file</option>
            <option value="folder" ${mode === "folder" ? "selected" : ""}>Folder</option>
          </select>
        </div>
      </div>
      <div class="node-mode-single ${mode === "single" ? "" : "hidden"}">
        ${base("source", "Select image/video file...", "Pick", "file", true)}
        <div class="${previewClass}">
          <img data-node-preview="${node.id}" src="${previewUrl}" alt="Input preview" />
        </div>
      </div>
      <div class="node-mode-folder ${mode === "folder" ? "" : "hidden"}">
        ${base("folder", "Select folder...", "Folder", "dir", false)}
      </div>
    `;
  }
  if (node.type === "Connector") {
    return base("middleware", "Select middleware file...");
  }
  if (node.type === "Prompt") {
    return `
      <div class="node-control">
        <div class="node-control-row">
          <input class="node-gdino-prompt" data-node="${node.id}" placeholder="Enter prompt e.g. car . person . road" value="${htmlEscape(
            meta.prompt || ""
          )}" />
        </div>
      </div>
    `;
  }
  if (node.type === "Post") {
    const isSingle = isGraphSingleMode();
    const previewUrl =
      isSingle && meta.outputPreviewLocal
        ? meta.outputPreviewLocal
        : isSingle && meta.outputPreviewPath
        ? buildFileUrl(meta.outputPreviewPath)
        : isSingle && meta.output && !isLikelyFolder(meta.output)
        ? buildFileUrl(meta.output)
        : "";
    const previewClass = previewUrl ? "node-output-preview visible" : "node-output-preview";
    return `
      ${base("output", isSingle ? "Select output file..." : "Select output folder...", isSingle ? "Pick" : "Folder", isSingle ? "file" : "dir")}
      <div class="${previewClass}">
        ${previewUrl ? `<img src="${previewUrl}" alt="Output preview" />` : `<div class="preview-placeholder">No output yet</div>`}
        <div class="node-control-row" style="margin-top:6px;">
          <button type="button" class="secondary tiny node-output-load" data-node="${node.id}">Refresh preview</button>
        </div>
      </div>
    `;
  }
  if (node.type === "Debug") {
    const previewUrl = meta.debugPreviewLocal || (meta.debugPreviewPath ? buildFileUrl(meta.debugPreviewPath) : "");
    const previewClass = previewUrl ? "node-output-preview visible" : "node-output-preview";
    return `
      <div class="${previewClass}">
        ${previewUrl ? `<img src="${previewUrl}" alt="Debug preview" />` : `<div class="preview-placeholder">No debug output yet</div>`}
        <div class="node-control-row" style="margin-top:6px;">
          <button type="button" class="secondary tiny node-debug-load" data-node="${node.id}">Refresh preview</button>
        </div>
      </div>
    `;
  }
  if (node.type === "Model") {
    const options = models
      .map(
        (m) =>
          `<option value="${m.id}" ${m.id === meta.modelId ? "selected" : ""}>${htmlEscape(
            modelLabel(m)
          )}</option>`
      )
      .join("");
    const isGDino = meta.modelId === "groundingdino";
    const gdPrompt = meta.prompt || "car . person . road";
    const gdBox = meta.box_threshold !== undefined ? meta.box_threshold : 0.3;
    const gdText = meta.text_threshold !== undefined ? meta.text_threshold : 0.25;
    const gdinoControls = isGDino
      ? `
      <div class="node-control">
        <div class="node-control-row">
          <input class="node-gdino-prompt" data-node="${node.id}" placeholder="Prompt e.g. car . person . road" value="${htmlEscape(
            gdPrompt
          )}" />
        </div>
        <div class="node-control-row">
          <input class="node-gdino-box" type="number" step="0.01" data-node="${node.id}" value="${gdBox}" title="Box threshold" />
          <input class="node-gdino-text" type="number" step="0.01" data-node="${node.id}" value="${gdText}" title="Text threshold" />
        </div>
      </div>`
      : "";
    return `
      <div class="node-control">
        <div class="node-control-row">
          <select class="node-model-select" data-node="${node.id}">
            ${options}
          </select>
        </div>
      </div>
      ${gdinoControls}
    `;
  }
  return "";
}

function bindNodeControls(div, node) {
  div.querySelectorAll(".node-path-input").forEach((input) => {
    input.addEventListener("change", async (e) => {
      const key = e.target.dataset.key;
      let val = e.target.value.trim();
      if (node.type === "Input" || node.type === "Data") {
        val = maybePrefixStorage(val);
        const resolved = await resolveStoragePath(val);
        if (resolved && resolved !== val) {
          val = resolved;
        }
      }
      const updates = { [key]: val };
      if (node.type === "Post" && key === "output") {
        updates.outputPreview = "";
      }
      updateNodeMeta(node.id, updates);
      // Keep the displayed value in sync with the resolved path
      if (val && e.target.value !== val) {
        e.target.value = val;
      }
      let needsRender = false;
      if (node.type === "Input" || node.type === "Data") {
        updatePreview(val, null);
        const changed = applyDefaultOutputsFromInput(val);
        needsRender = needsRender || changed;
        needsRender = true; // input node needs rerender for its own value
      }
      if (node.type === "Post") {
        needsRender = true;
      }
      if (needsRender) renderGraph();
    });
  });
  div.querySelectorAll(".node-clear").forEach((btn) => {
    btn.addEventListener("click", (e) => {
      e.stopPropagation();
      const key = btn.dataset.key;
      const updates = { [key]: "" };
      if ((node.type === "Input" || node.type === "Data") && key === "source") {
        updates.previewLocal = "";
      }
      if (node.type === "Post" && key === "output") {
        updates.outputPreviewPath = "";
        updates.outputPreviewLocal = "";
      }
      updateNodeMeta(node.id, updates);
      renderGraph();
      if (node.type === "Input" || node.type === "Data") {
        updatePreview("", null);
      }
    });
  });
  div.querySelectorAll(".node-pick").forEach((btn) => {
    btn.addEventListener("click", (e) => {
      e.stopPropagation();
      handleNodePick(node, btn.dataset.kind, btn.dataset.key);
    });
  });
  div.querySelectorAll(".node-input-mode").forEach((sel) => {
    sel.addEventListener("change", (e) => {
      const mode = e.target.value || "single";
      updateNodeMeta(node.id, { mode });
      renderGraph();
    });
  });
  div.querySelectorAll(".node-output-load").forEach((btn) => {
    btn.addEventListener("click", async (e) => {
      e.stopPropagation();
      const path = node.meta?.outputPreviewPath || node.meta?.output || node.meta?.debugPreviewPath || "";
      if (!path || isLikelyFolder(path)) {
        appendLog("Set an output file path to refresh preview.");
        return;
      }
      const url = await fetchPreviewUrl(path);
      if (url) {
        const updates = {};
        if (node.type === "Debug") {
          updates.debugPreviewLocal = url;
          updates.debugPreviewPath = path;
        } else {
          updates.outputPreviewLocal = url;
          updates.outputPreviewPath = path;
        }
        updateNodeMeta(node.id, updates);
        renderGraph();
      } else {
        appendLog("Preview refresh failed.");
      }
    });
  });
  div.querySelectorAll(".node-view").forEach((btn) => {
    btn.addEventListener("click", (e) => {
      e.stopPropagation();
      const key = btn.dataset.key;
      const path = node.meta?.[key];
      if (!path) {
        appendLog("No file path set for preview.");
        return;
      }
      const url = buildFileUrl(path);
      updatePreview(path, null);
      window.open(url, "_blank");
    });
  });
  div.querySelectorAll(".node-model-select").forEach((sel) => {
    sel.addEventListener("change", (e) => {
      const modelId = e.target.value;
      const model = getModelById(modelId);
      updateNodeMeta(node.id, { modelId });
      if (model) {
        node.title = model.name;
      }
      // Clear cached previews so downstream Debug/Post nodes don't show stale outputs after a model swap.
      clearCachedPreviews();
      renderGraph();
    });
  });
  div.querySelectorAll(".node-gdino-prompt").forEach((input) => {
    input.addEventListener("change", (e) => {
      updateNodeMeta(node.id, { prompt: e.target.value || "" });
    });
  });
  div.querySelectorAll(".node-gdino-box").forEach((input) => {
    input.addEventListener("change", (e) => {
      const v = parseFloat(e.target.value);
      updateNodeMeta(node.id, { box_threshold: Number.isFinite(v) ? v : 0.3 });
    });
  });
  div.querySelectorAll(".node-gdino-text").forEach((input) => {
    input.addEventListener("change", (e) => {
      const v = parseFloat(e.target.value);
      updateNodeMeta(node.id, { text_threshold: Number.isFinite(v) ? v : 0.25 });
    });
  });
}

function clearCachedPreviews() {
  state.graph.nodes.forEach((n) => {
    if (n.type === "Debug") {
      n.meta.debugPreviewPath = "";
      n.meta.debugPreviewLocal = "";
    }
    if (n.type === "Post") {
      n.meta.outputPreviewPath = "";
      n.meta.outputPreviewLocal = "";
    }
  });
  persistGraph();
}

function loadPipelineFromFile(file) {
  const reader = new FileReader();
  reader.onload = (e) => {
    try {
      const text = e.target.result;
      const payload = JSON.parse(text);
      applyPipeline(payload);
    } catch (err) {
      appendLog(`Load pipeline parse error: ${err.message}`);
    }
  };
  reader.readAsText(file);
}

function updateNodeMeta(nodeId, updates) {
  const node = state.graph.nodes.find((n) => n.id === nodeId);
  if (!node) return;
  ensureNodeMeta(node);
  Object.assign(node.meta, updates);
  persistGraph();
}

async function handleNodePick(node, kind = "file", keyOverride) {
  const key =
    keyOverride || (kind === "dir" ? "output" : node.type === "Connector" ? "middleware" : "source");

  const picked = await pickFileOrDir(kind);
  if (!picked) return;
  const pickedPath = typeof picked === "string" ? picked : picked.path;
  const pickedFile = typeof picked === "object" ? picked.file : null;
  let finalPath = pickedPath;
  const looksLikePath = /^(?:[A-Za-z]:[\\/]|[\\/]|\.{1,2}[\\/])/;
  if (!looksLikePath.test(pickedPath || "")) {
    if (kind === "file" && !hasPathSeparator(finalPath)) {
      const resolved = await searchStorageForName(finalPath);
      if (resolved && resolved.length) {
        finalPath = resolved[0];
        appendLog(
          resolved.length === 1
            ? `Resolved ${finalPath} in Storage/`
            : `Resolved ${finalPath} (first of ${resolved.length} matches in Storage/)`
        );
      } else {
        finalPath = maybePrefixStorage(finalPath);
      }
    } else {
      finalPath = maybePrefixStorage(finalPath);
    }
  }
  if (kind === "file") {
    const resolved = await resolveStoragePath(finalPath);
    if (resolved && resolved !== finalPath) {
      finalPath = resolved;
    }
  }

  const updates = { [key]: finalPath };
  if (kind === "file" && pickedFile) {
    try {
      updates.previewLocal = URL.createObjectURL(pickedFile);
    } catch (_) {
      // ignore
    }
  }
  updateNodeMeta(node.id, updates);
  let needsRender = true; // we picked a file, so rerender to reflect new path
  if (node.type === "Input" || node.type === "Data") {
    updatePreview(finalPath, null);
    const changed = applyDefaultOutputsFromInput(finalPath);
    needsRender = needsRender || changed;
  }
  if (needsRender) renderGraph();
}

async function pickFileOrDir(kind = "file") {
  if (kind === "dir" && window.showDirectoryPicker) {
    try {
      const dir = await window.showDirectoryPicker();
      return { path: (dir.name || "selected-folder") + "/" };
    } catch (err) {
      if (err?.name === "AbortError") return null;
    }
  }
  return new Promise((resolve) => {
    const input = document.createElement("input");
    input.type = "file";
    if (kind === "dir") input.webkitdirectory = true;
    input.style.position = "fixed";
    input.style.opacity = "0";
    input.style.pointerEvents = "none";
    document.body.appendChild(input);
    input.addEventListener("change", () => {
      let val = "";
      let fileObj = null;
      if (kind === "dir" && input.files?.length) {
        const first = input.files[0];
        const rel = first.webkitRelativePath || "";
        const dirName = rel.includes("/") ? rel.split("/")[0] : first.name || "folder";
        val = `${dirName}/`;
      } else if (input.files?.length) {
        const f = input.files[0];
        val = f.webkitRelativePath || f.name || "";
        fileObj = f;
      }
      document.body.removeChild(input);
      if (!val) return resolve(null);
      resolve({ path: val, file: fileObj });
    });
    input.addEventListener("cancel", () => {
      document.body.removeChild(input);
      resolve(null);
    });
    input.click();
  });
}

function setupDrag(element, node) {
  let startX = 0;
  let startY = 0;
  let dragging = false;

  const onMove = (ev) => {
    if (!dragging) return;
    node.x = startX + ev.clientX - startXPointer;
    node.y = startY + ev.clientY - startYPointer;
    element.style.left = `${node.x}px`;
    element.style.top = `${node.y}px`;
    drawLinks();
  };

  const onUp = () => {
    dragging = false;
    document.removeEventListener("pointermove", onMove);
    document.removeEventListener("pointerup", onUp);
    persistGraph();
  };

  let startXPointer = 0;
  let startYPointer = 0;

  element.addEventListener("pointerdown", (ev) => {
    // Ignore pointer-down on controls so clicks can delete/update without starting drag
    if (
      ev.target.classList.contains("connector") ||
      ev.target.closest(".node-remove") ||
      ev.target.closest(".node-control")
    ) {
      return;
    }
    dragging = true;
    startX = node.x;
    startY = node.y;
    startXPointer = ev.clientX;
    startYPointer = ev.clientY;
    element.setPointerCapture(ev.pointerId);
    document.addEventListener("pointermove", onMove);
    document.addEventListener("pointerup", onUp);
  });
}

function handleLinkClick(nodeId, role) {
  clearConnectorHighlights();
  if (!state.activeLinkStart) {
    startLinking(nodeId, role);
    return;
  }
  tryFinalizeLink(nodeId, role);
}

function highlightNode(nodeId) {
  document.querySelectorAll(`.node[data-id="${nodeId}"] .connector`).forEach((c) => c.classList.add("active"));
}

function clearConnectorHighlights() {
  document.querySelectorAll(".connector.active").forEach((c) => c.classList.remove("active"));
}

function startLinking(nodeId, role) {
  // Only allow starting from an "out" connector to keep arrows directional.
  if (role !== "out") {
    appendLog("Start from an output (left) connector.");
    return;
  }
  state.activeLinkStart = nodeId;
  state.activeLinkRole = role;
  highlightNode(nodeId);
  state.previewStart = getConnectorPoint(nodeId, role) || getNodeCenter(nodeId);
  // show a tiny preview line from the start point
  const layerRect = getLayerRect();
  if (state.previewStart && layerRect) {
    updatePreviewLine(state.previewStart.x + layerRect.left, state.previewStart.y + layerRect.top);
  }
  appendLog(`Linking from ${nodeId}...`);
}

function tryFinalizeLink(nodeId, role) {
  if (!state.activeLinkStart || state.activeLinkStart === nodeId) {
    state.activeLinkStart = null;
    state.activeLinkRole = null;
    state.previewStart = null;
    removePreviewLine();
    clearConnectorHighlights();
    return;
  }
  // Enforce direction: out -> in
  if (role !== "in") {
    appendLog("Finish on an input (right) connector.");
    return;
  }
  const exists = state.graph.links.some((l) => l.from === state.activeLinkStart && l.to === nodeId);
  if (exists) {
    appendLog(`Link ${state.activeLinkStart} -> ${nodeId} already exists`);
  } else {
    state.graph.links.push({ from: state.activeLinkStart, to: nodeId });
    appendLog(`Linked ${state.graph.links[state.graph.links.length - 1].from} -> ${nodeId}`);
    drawLinks();
    persistGraph();
  }
  state.activeLinkStart = null;
  state.activeLinkRole = null;
  state.previewStart = null;
  removePreviewLine();
  clearConnectorHighlights();
}

function getNodeCenter(nodeId) {
  const nodeEl = document.querySelector(`.node[data-id="${nodeId}"]`);
  const layerRect = getLayerRect();
  if (!nodeEl || !layerRect) return null;
  const rect = nodeEl.getBoundingClientRect();
  return {
    x: rect.left - layerRect.left + rect.width / 2,
    y: rect.top - layerRect.top + rect.height / 2,
  };
}

function updatePreviewLine(clientX, clientY) {
  if (!state.previewStart) return;
  const layer = el("link-layer");
  if (!layer) return;
  ensureLinkLayerDefs(layer);
  const rect = layer.getBoundingClientRect();
  const x2 = clientX - rect.left;
  const y2 = clientY - rect.top;
  let line = layer.querySelector("#preview-line");
  if (!line) {
    line = document.createElementNS("http://www.w3.org/2000/svg", "line");
    line.id = "preview-line";
    line.setAttribute("stroke", "rgba(156,248,255,0.7)");
    line.setAttribute("stroke-width", "2");
    line.setAttribute("stroke-linecap", "round");
    layer.appendChild(line);
  }
  line.setAttribute("x1", state.previewStart.x);
  line.setAttribute("y1", state.previewStart.y);
  line.setAttribute("x2", x2);
  line.setAttribute("y2", y2);
}

function removePreviewLine() {
  const layer = el("link-layer");
  if (!layer) return;
  const line = layer.querySelector("#preview-line");
  if (line) line.remove();
}

function deleteSelectedLink() {
  if (!state.selectedLink) return;
  removeLink(state.selectedLink.from, state.selectedLink.to);
  state.selectedLink = null;
  document.querySelectorAll(".link-path.selected").forEach((p) => p.classList.remove("selected"));
}

function deleteNode(nodeId) {
  const nextNodes = state.graph.nodes.filter((n) => n.id !== nodeId);
  const nextLinks = state.graph.links.filter((l) => l.from !== nodeId && l.to !== nodeId);
  if (nextNodes.length === state.graph.nodes.length) return;

  state.graph = { ...state.graph, nodes: nextNodes, links: nextLinks };
  if (state.selectedLink && (state.selectedLink.from === nodeId || state.selectedLink.to === nodeId)) {
    state.selectedLink = null;
  }
  state.activeLinkStart = null;
  state.previewStart = null;
  removePreviewLine();
  clearConnectorHighlights();
  logAction(`Removed node ${nodeId}`);
  renderGraph();
  persistGraph();
}

function removeLink(from, to) {
  const nextLinks = state.graph.links.filter((l) => !(l.from === from && l.to === to));
  if (nextLinks.length === state.graph.links.length) return;

  state.graph = { ...state.graph, links: nextLinks };
  if (state.selectedLink && state.selectedLink.from === from && state.selectedLink.to === to) {
    state.selectedLink = null;
  }
  state.activeLinkStart = null;
  state.previewStart = null;
  removePreviewLine();
  clearConnectorHighlights();
  logAction(`Removed link ${from} -> ${to}`);
  drawLinks();
  persistGraph();
}

function selectLink(from, to, pathEl) {
  state.selectedLink = { from, to };
  highlightSelectedLink(pathEl);
}

function highlightSelectedLink(targetPath) {
  document.querySelectorAll(".link-path.selected").forEach((p) => p.classList.remove("selected"));
  if (targetPath) {
    targetPath.classList.add("selected");
    return;
  }
  if (!state.selectedLink) return;
  const layer = el("link-layer");
  if (!layer) return;
  const found = layer.querySelector(
    `.link-path[data-from="${state.selectedLink.from}"][data-to="${state.selectedLink.to}"]`
  );
  if (found) found.classList.add("selected");
}

function simulateRun() {
  beautifyConfig();
  if (!validateConfig()) return;
  const payload = {
    model: state.selectedModel,
    config: state.config,
    graph: state.graph,
    input: el("input-path").value || "Storage/sample.mp4",
    output: el("output-path").value || "Storage/output_preview.png",
    notes: el("preview-notes").value || "",
    tracking: state.tracking,
  };
  appendLog(`Simulated run: ${payload.model} on ${payload.input} -> ${payload.output}`);
  appendLog(`Pipeline nodes: ${payload.graph.nodes.length}, links: ${payload.graph.links.length}`);
  postTrackingEvent("pipeline_simulate", payload);
  updatePreview(payload.input, payload.output);
}

async function runPipeline() {
  beautifyConfig();
  if (!validateConfig()) return;
  const isSingle = isGraphSingleMode();
  const inputNodes = state.graph.nodes.filter((n) => n.type === "Input" || n.type === "Data");
  const inputs = [];
  for (const node of inputNodes) {
    const resolved = await resolveStoragePath(getInputPath(node.meta || {}));
    if (resolved && resolved !== node.meta?.source) {
      node.meta.source = resolved;
    }
    inputs.push(resolved);
  }
  const rawOutputs = state.graph.nodes
    .filter((n) => n.type === "Post")
    .map((n) => normalizeOutputPath(n.meta?.output || "", isSingle));
  const outputs = isSingle ? rawOutputs.map((o) => ensureFileOutput(o, inputs[0])) : rawOutputs;

  // push resolved outputs back into node meta so UI shows the file path
  if (isSingle) {
    state.graph.nodes
      .filter((n) => n.type === "Post")
      .forEach((n, idx) => {
        n.meta.output = outputs[idx] || n.meta.output;
      });
    renderGraph();
  }

  const graphModel = state.graph.nodes.find((n) => n.type === "Model");
  const modelId = graphModel?.meta?.modelId || state.selectedModel;
  const payload = {
    model: modelId,
    config: state.config,
    graph: state.graph,
    inputs,
    middleware: state.graph.nodes.filter((n) => n.type === "Connector").map((n) => n.meta?.middleware || ""),
    outputs,
    tracking: state.tracking,
    name: el("config-name")?.value || modelId,
  };
  if (payload.middleware.length) {
    appendLog("Data middleware nodes are not implemented; dropping from payload.");
    payload.middleware = [];
    payload.graph.links = payload.graph.links.filter(
      (l) =>
        state.graph.nodes.find((n) => n.id === l.from && n.type !== "Connector") &&
        state.graph.nodes.find((n) => n.id === l.to && n.type !== "Connector")
    );
    payload.graph.nodes = payload.graph.nodes.filter((n) => n.type !== "Connector");
  }
  appendLog(
    `Pipeline run: inputs=${payload.inputs.join(", ") || "n/a"}, middleware=${payload.middleware.join(", ") ||
      "n/a"}, outputs=${payload.outputs.join(", ") || "n/a"}`
  );
  postTrackingEvent("pipeline_run", payload);
  const result = await submitPipeline(payload);
  const resolvedOutput = result?.output || payload.outputs[0] || payload.output;
  const resolvedInput =
    (payload.inputs && payload.inputs[0]) ||
    (payload.config?.data && payload.config.data.source) ||
    payload.input;
  // Update previews for Post and Debug nodes based on returned steps.
  if (result?.steps && Array.isArray(result.steps)) {
    for (const step of result.steps) {
      if (!step?.node || !step?.output) continue;
      const node = state.graph.nodes.find((n) => n.id === step.node);
      if (!node) continue;
      if (step.type === "post") {
        const url = await fetchPreviewUrl(step.output);
        updateNodeMeta(node.id, { outputPreviewPath: step.output, outputPreviewLocal: url || "" });
      } else if (step.type === "debug") {
        const url = await fetchPreviewUrl(step.output);
        updateNodeMeta(node.id, { debugPreviewPath: step.output, debugPreviewLocal: url || "" });
      }
    }
    renderGraph();
  }
  if (resolvedOutput && isSingle) {
    const url = await fetchPreviewUrl(resolvedOutput);
    state.graph.nodes
      .filter((n) => n.type === "Post")
      .forEach((n) => {
        n.meta.outputPreviewPath = resolvedOutput;
        if (url) n.meta.outputPreviewLocal = url;
      });
    renderGraph();
  }
  updatePreview(resolvedInput, resolvedOutput);
}

function applyPipeline(payload) {
  try {
    state.selectedModel = payload.model || state.selectedModel;
    state.config = payload.config || state.config;
    state.graph = payload.graph || state.graph;
    state.tracking = payload.tracking || state.tracking;
    state.graph.nodes.forEach(ensureNodeMeta);
    if (hasConfigUI()) {
      const cfgName = el("config-name");
      const runtime = el("runtime-device");
      const editor = el("config-editor");
      if (cfgName && payload.name) cfgName.value = payload.name;
      if (runtime && payload.config?.model?.device) runtime.value = payload.config.model.device;
      if (editor && payload.config) editor.value = JSON.stringify(payload.config, null, 2);
    }
    if (hasGraphUI()) {
      persistGraph();
      renderGraph();
    }
    appendLog(`Loaded pipeline ${payload.name || ""}`.trim());
  } catch (err) {
    appendLog(`Load pipeline failed: ${err.message}`);
  }
}

async function submitPipeline(payload) {
  try {
    const res = await fetch("/api/pipeline/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      appendLog(`Pipeline submit failed (${res.status})`);
      return null;
    }
    const data = await res.json();
    appendLog(`Pipeline submitted: id=${data.id} status=${data.status || "queued"}`);
    if (data.message) appendLog(`Pipeline message: ${data.message}`);
    if (data.id) {
      appendLog(`Saved at ${data.path || "/api/pipeline/run/" + data.id}`);
    }
    if (data.output) {
      appendLog(`Output written to ${data.output}`);
    }
    return data;
  } catch (err) {
    appendLog(`Pipeline submit error: ${err.message}`);
    return null;
  }
}

async function startCvat() {
  try {
    appendLog("Starting CVAT containers...");
    const res = await fetch("/api/cvat/start", { method: "POST" });
    const data = await res.json();
    if (data.status === "ok") {
      appendLog("CVAT start requested. Opening UI...");
      window.open("http://localhost:8080", "_blank");
    } else {
      appendLog(`CVAT start failed: ${data.message || res.status}`);
    }
  } catch (err) {
    appendLog(`CVAT start error: ${err.message}`);
  }
}

function exportJson() {
  beautifyConfig();
  if (!validateConfig()) return;
  const bundle = {
    model: state.selectedModel,
    config: state.config,
    graph: state.graph,
    name: el("config-name").value || state.selectedModel,
    device: el("runtime-device").value,
    tracking: state.tracking,
  };
  const blob = new Blob([JSON.stringify(bundle, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `${bundle.name || "pipeline"}.json`;
  a.click();
  URL.revokeObjectURL(url);
  appendLog("Exported pipeline JSON");
}

async function postTrackingEvent(event, payload) {
  if (state.tracking.provider === "none" || !state.tracking.endpoint) {
    return;
  }
  const body = {
    provider: state.tracking.provider,
    experiment: state.tracking.experiment,
    run: state.tracking.run,
    event,
    payload,
  };
  try {
    const res = await fetch(`${state.tracking.endpoint}/api/track`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    appendLog(`Tracking POST ${res.status}`);
  } catch (err) {
    appendLog(`Tracking failed: ${err.message}`);
  }
}

function validateConfig() {
  const schema = schemas[state.selectedModel];
  if (!schema) return true;
  const errors = [];
  const cfg = state.config;
  if (schema.required) {
    schema.required.forEach((key) => {
      if (!(key in cfg)) errors.push(`Missing required section: ${key}`);
    });
  }
  if (schema.props && cfg.model) {
    schema.props.model.forEach((k) => {
      if (!cfg.model[k]) errors.push(`model.${k} is required`);
    });
  }
  if (errors.length) {
    appendLog(`Validation: ${errors.join("; ")}`);
    state.validationOk = false;
    updateStatusBar(errors.join("; "));
    return false;
  }
  appendLog("Config validated.");
  state.validationOk = true;
  updateStatusBar("ok");
  return true;
}

function buildFileUrl(path) {
  if (!path) return "";
  if (/^https?:\/\//i.test(path)) return path;
  const ts = Date.now(); // bust cache for fresh renders
  const base =
    window.API_FILE_BASE ||
    (window.location.port === "7000" || /:7000$/.test(window.location.host)
      ? "/api/file"
      : "http://localhost:7000/api/file");
  return `${base}?path=${encodeURIComponent(path)}&t=${ts}`;
}

async function searchStorageForName(name) {
  try {
    const res = await fetch(`/api/storage/search?name=${encodeURIComponent(name)}`);
    if (!res.ok) return [];
    return await res.json();
  } catch (_) {
    return [];
  }
}

async function fetchPreviewUrl(path) {
  try {
    const res = await fetch(buildFileUrl(path));
    if (!res.ok) return null;
    const blob = await res.blob();
    return URL.createObjectURL(blob);
  } catch (_) {
    return null;
  }
}

function isShallowStoragePath(path = "") {
  if (!path) return false;
  if (path.startsWith("Storage/")) {
    const rest = path.slice("Storage/".length);
    return !rest.includes("/");
  }
  return false;
}

function ensureStoragePromptStyles() {
  if (document.getElementById("storage-choice-style")) return;
  const style = document.createElement("style");
  style.id = "storage-choice-style";
  style.textContent = `
  .storage-choice-overlay {
    position: fixed;
    inset: 0;
    background: rgba(0,0,0,0.55);
    display: grid;
    place-items: center;
    z-index: 9999;
  }
  .storage-choice-card {
    background: #0f162f;
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    box-shadow: 0 20px 48px rgba(0,0,0,0.4);
    padding: 16px;
    width: min(520px, 90vw);
    color: #e9edf7;
    font-family: "Manrope", "Segoe UI", system-ui, sans-serif;
  }
  .storage-choice-card h4 { margin: 0 0 8px 0; }
  .storage-choice-list { display: grid; gap: 8px; margin: 12px 0; }
  .storage-choice-list button {
    width: 100%;
    text-align: left;
    padding: 10px 12px;
    border-radius: 10px;
    border: 1px solid rgba(255,255,255,0.08);
    background: rgba(255,255,255,0.04);
    color: #e9edf7;
    cursor: pointer;
  }
  .storage-choice-actions { display: flex; justify-content: flex-end; gap: 8px; }
  .storage-choice-cancel {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.12);
    color: #e9edf7;
  }
  `;
  document.head.appendChild(style);
}

function pickStorageMatch(name, matches) {
  if (typeof window === "undefined" || typeof document === "undefined") return matches[0];
  ensureStoragePromptStyles();
  return new Promise((resolve) => {
    const overlay = document.createElement("div");
    overlay.className = "storage-choice-overlay";
    const card = document.createElement("div");
    card.className = "storage-choice-card";
    card.innerHTML = `
      <h4>Choose file for "${name}"</h4>
      <div style="color:#9da7be;font-size:0.95rem;">Multiple matches found under Storage. Pick one to use.</div>
      <div class="storage-choice-list"></div>
      <div class="storage-choice-actions">
        <button class="storage-choice-cancel">Cancel</button>
      </div>
    `;
    const list = card.querySelector(".storage-choice-list");
    matches.slice(0, 5).forEach((m, idx) => {
      const btn = document.createElement("button");
      btn.textContent = `${idx + 1}) ${m}`;
      btn.addEventListener("click", () => {
        cleanup();
        resolve(m);
      });
      list.appendChild(btn);
    });
    const cancelBtn = card.querySelector(".storage-choice-cancel");
    cancelBtn.addEventListener("click", () => {
      cleanup();
      resolve(null);
    });
    const cleanup = () => overlay.remove();
    overlay.appendChild(card);
    document.body.appendChild(overlay);
  });
}

async function resolveStoragePath(path = "") {
  if (!path) return path;
  if (hasPathSeparator(path) && !isShallowStoragePath(path)) return path;
  const name = path.split(/[\\/]/).pop();
  if (!name) return path;
  const matches = await searchStorageForName(name);
  if (!matches || matches.length === 0) return path;
  if (matches.length === 1) return matches[0];
  const choice = await pickStorageMatch(name, matches);
  if (choice) return choice;
  appendLog(
    `Multiple files named "${name}" found under Storage. Candidates: ${matches
      .slice(0, 3)
      .join(", ")}${matches.length > 3 ? " ..." : ""}. Kept original value.`
  );
  return path;
}

function getInputPath(meta = {}) {
  const mode = meta.mode || "single";
  if (mode === "folder") return meta.folder || "";
  return meta.source || "";
}

function isGraphSingleMode() {
  // if any input/data node is folder mode, treat as multi
  return !state.graph.nodes.some((n) => (n.type === "Input" || n.type === "Data") && (n.meta?.mode || "single") === "folder");
}

function applyDefaultOutputsFromInput(inputPath) {
  if (!inputPath || !isGraphSingleMode()) return false;
  let changed = false;
  state.graph.nodes
    .filter((n) => n.type === "Post")
    .forEach((n) => {
      ensureNodeMeta(n);
      const current = n.meta.output || "";
      const trimmed = (current || "").replace(/[\\/]$/, "");
      const trimmedLower = trimmed.toLowerCase();
      const underOutput = OUTPUT_BASES.some((b) => trimmedLower.startsWith(b.toLowerCase()));
      const isMissing = !trimmed;
      // Only override if blank or already under our output base.
      if (!isMissing && !underOutput) return;
      const next = ensureFileOutput("", inputPath);
      if (next !== current) {
        n.meta.output = next;
        persistGraph();
        changed = true;
      }
    });
  return changed;
}

function normalizeOutputPath(path, isSingle) {
  if (!path) return "";
  if (isSingle) return path; // file output is expected
  // ensure folder ends with a slash-like suffix so backend treats it as a directory
  if (/[\\/]$/.test(path)) return path;
  // if it looks like a file (has an extension), keep as-is; else append '/'
  if (/\.[a-z0-9]+$/i.test(path)) return path;
  return `${path}/`;
}

function deriveOutputBaseFromInput(inputPath) {
  if (!inputPath) return OUTPUT_BASE;
  const match = inputPath.match(/^Storage[\\/](.+)$/i);
  if (!match) return OUTPUT_BASE;
  const rest = match[1];
  const restDir = rest.replace(/[^\\/]+$/, "").replace(/[\\/]$/, "");
  const base = restDir ? `${OUTPUT_BASE}/${restDir}` : OUTPUT_BASE;
  return base.replace(/[\\]+/g, "/").replace(/\/+/g, "/").replace(/\/$/, "");
}

function ensureFileOutput(path, inputPath) {
  // If already a file path with extension, keep it.
  const trimmed = (path || "").replace(/[\\/]$/, "");
  const hasExt = /\.[a-z0-9]+$/i.test(trimmed);
  const trimmedLower = trimmed.toLowerCase();
  const baseHit = OUTPUT_BASES.find((b) => trimmedLower.startsWith(b.toLowerCase()));
  const underOutput = !!baseHit;
  const relToOutput = underOutput ? trimmed.slice(baseHit.length).replace(/^[/\\]/, "") : "";
  const flatUnderOutput = underOutput && (!relToOutput || !relToOutput.includes("/"));
  if (hasExt && underOutput) {
    // normalize to canonical base to avoid case/volume mismatches
    const normalized = `${OUTPUT_BASE}/${relToOutput}`.replace(/[\\]+/g, "/");
    return normalized.replace(/\/+/g, "/");
  }
  if (hasExt && !flatUnderOutput) return trimmed; // keep user-specified concrete file path outside our default base
  const baseDir =
    !trimmed || trimmed === OUTPUT_BASE || flatUnderOutput || !hasExt
      ? deriveOutputBaseFromInput(inputPath)
      : trimmed || deriveOutputBaseFromInput(inputPath);
  const name = deriveOutputName(inputPath);
  return `${baseDir.replace(/[\\/]$/, "")}/${name}.png`;
}

function deriveOutputName(inputPath) {
  if (!inputPath) return "output";
  const filename = inputPath.split(/[\\/]/).pop() || "output";
  const stem = filename.replace(/\.[^.]+$/, "");
  return `${stem}_mask`;
}

function hasPathSeparator(p = "") {
  return /[\\/]/.test(p);
}

function maybePrefixStorage(path = "") {
  if (!path) return path;
  if (hasPathSeparator(path)) return path;
  return `${INPUT_BASE}/${path}`;
}

function updatePreview(inputPath, outputPath) {
  const inputEl = el("preview-input");
  const outputEl = el("preview-output");
  if (inputEl) {
    const inUrl = inputPath && !isLikelyFolder(inputPath) ? buildFileUrl(inputPath) : "";
    inputEl.src = inUrl;
    inputEl.alt = inputPath || "Input preview";
  }
  if (outputEl) {
    outputEl.src = outputPath ? buildFileUrl(outputPath) : "";
    outputEl.alt = outputPath || "Output preview";
  }
}

function isLikelyFolder(path) {
  return path && !/\.[a-z0-9]+$/i.test(path);
}

async function fetchInventory() {
  try {
    let res = await fetch("/api/models");
    if (!res.ok) {
      res = await fetch("http://localhost:7000/api/models");
    }
    if (res.ok) {
      const items = await res.json();
      state.inventory = Object.fromEntries(items.map((i) => [i.id, i]));
      mergeInventory();
    }
  } catch (err) {
    appendLog(`Inventory fetch failed: ${err.message}`);
  }
}

function mergeInventory() {
  models.forEach((m) => {
    const inv = state.inventory[m.id];
    if (!inv) return;
    if (inv.config) {
      m.template.model = m.template.model || {};
      if (!m.template.model.config) m.template.model.config = inv.config;
    }
    if (inv.weights) {
      m.template.model = m.template.model || {};
      if (!m.template.model.weights) m.template.model.weights = inv.weights;
    }
  });
  loadModelConfig(state.selectedModel);
  renderModelCards();
  updateStatusBar();
}

function registerEvents() {
  bindIf("beautify-btn", "click", beautifyConfig);
  bindIf("reset-btn", "click", resetConfig);
  bindIf("save-btn", "click", saveConfig);
  bindIf("run-btn", "click", simulateRun);
  bindIf("run-pipeline", "click", runPipeline);
  bindIf("export-btn", "click", exportJson);
  bindIf("add-input", "click", () => addNode("Input"));
  bindIf("add-model", "click", () => addNode("Model"));
  bindIf("add-post", "click", () => addNode("Post"));
  bindIf("add-debug", "click", () => addNode("Debug"));
  const loadPipelineBtn = el("load-pipeline-btn");
  const loadPipelineInput = el("load-pipeline-input");
  if (loadPipelineBtn && loadPipelineInput) {
    loadPipelineBtn.addEventListener("click", () => loadPipelineInput.click());
    loadPipelineInput.addEventListener("change", (e) => {
      const file = e.target.files?.[0];
      if (file) loadPipelineFromFile(file);
    });
  }
  bindIf("clear-graph", "click", clearGraph);
  bindIf("save-tracker", "click", saveTracking);
  bindIf("ping-tracker", "click", pingTracking);
  const modelSearch = el("model-search");
  if (modelSearch) {
    modelSearch.addEventListener("input", (e) => {
      state.filter = e.target.value;
      renderModelCards();
    });
  }
  const modelSelect = el("model-select");
  if (modelSelect) {
    modelSelect.addEventListener("change", (e) => {
      state.selectedModel = e.target.value;
      renderModelCards();
      loadModelConfig(state.selectedModel);
      appendLog(`Selected ${state.selectedModel}`);
    });
  }
  bindIf("refresh-inventory", "click", () => fetchInventory());
  bindIf("start-cvat", "click", startCvat);
  if (hasGraphUI()) {
    window.addEventListener("pointermove", (e) => updatePreviewLine(e.clientX, e.clientY));
    window.addEventListener("resize", drawLinks);
    window.addEventListener("keydown", (e) => {
      if (e.key === "Delete" || e.key === "Backspace") {
        deleteSelectedLink();
      }
    });
  }
  const canvas = el("graph-canvas");
  if (canvas) {
    canvas.addEventListener("click", (e) => {
      const btn = e.target.closest?.(".node-remove");
      if (btn) {
        const id = btn.dataset.node;
        if (id) {
          e.stopPropagation();
          logAction(`Delete node (delegate) ${id}`);
          deleteNode(id);
        }
      }
    });
  }
  // Global safety net to ensure delete clicks are caught even if overlaying elements intercept bubbling.
  document.addEventListener(
    "click",
    (e) => {
      const btn = e.target.closest?.(".node-remove");
      if (!btn) return;
      const id = btn.dataset.node;
      if (!id) return;
      e.stopPropagation();
      logAction(`Delete node (global) ${id}`);
      deleteNode(id);
    },
    true
  );

  // Load pipeline from uploaded file
  document.addEventListener(
    "change",
    (e) => {
      const input = e.target;
      if (input && input.id === "load-pipeline-input" && input.files?.[0]) {
        loadPipelineFromFile(input.files[0]);
      }
    },
    true
  );

  // Delegate link delete clicks from SVG layer
  const layer = el("link-layer");
  if (layer) {
    layer.addEventListener("click", (e) => {
      const target = e.target;
      if (
        target.classList.contains("link-delete") ||
        target.classList.contains("link-delete-label") ||
        target.dataset?.from
      ) {
        const from = target.dataset.from;
        const to = target.dataset.to;
        if (from && to) {
          e.stopPropagation();
          logAction(`Delete link (delegate) ${from} -> ${to}`);
          removeLink(from, to);
        }
      }
    });
  }
}

function initPaletteDrag() {
  const palette = el("block-palette");
  const canvas = el("graph-canvas");
  if (!palette || !canvas) return;

  const setHover = (on) => {
    if (on) {
      canvas.classList.add("drop-hover");
    } else {
      canvas.classList.remove("drop-hover");
    }
  };

  palette.querySelectorAll(".palette-item").forEach((item) => {
    item.addEventListener("dragstart", (e) => {
      e.dataTransfer.effectAllowed = "copy";
      e.dataTransfer.setData("application/x-node-type", item.dataset.type || "");
      e.dataTransfer.setData("text/plain", item.dataset.type || "");
      setHover(true);
    });
    item.addEventListener("dragend", () => setHover(false));
  });

  canvas.addEventListener("dragover", (e) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = "copy";
    setHover(true);
  });
  canvas.addEventListener("dragleave", () => setHover(false));
  canvas.addEventListener("drop", (e) => {
    e.preventDefault();
    setHover(false);
    const type = e.dataTransfer.getData("application/x-node-type") || e.dataTransfer.getData("text/plain");
    if (!type) return;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    addNode(type, { x, y });
  });
}

function ensureLinkLayerDefs(layer) {
  if (layer.querySelector("#link-defs")) {
    layer.classList.add("gradient-ready");
    return;
  }
  const defs = document.createElementNS("http://www.w3.org/2000/svg", "defs");
  defs.id = "link-defs";

  const grad = document.createElementNS("http://www.w3.org/2000/svg", "linearGradient");
  grad.id = "grad";
  grad.innerHTML = `
    <stop offset="0%" stop-color="#9cf8ff" stop-opacity="0.9"/>
    <stop offset="100%" stop-color="#7af2c4" stop-opacity="0.9"/>
  `;
  defs.appendChild(grad);

  layer.prepend(defs);
  layer.classList.add("gradient-ready");
}

function init() {
  initGraph();
  state.graph.nodes.forEach(ensureNodeMeta);
  if (hasModelUI() || hasConfigUI()) {
    renderModelCards();
    loadModelConfig(state.selectedModel);
  } else {
    // ensure config state exists even if UI is absent
    loadModelConfig(state.selectedModel);
  }
  if (hasTrackingUI()) loadTracking();
  if (hasGraphUI()) renderGraph();
  registerEvents();
  initPaletteDrag();
  fetchInventory();
  watchGraphResize();
  appendLog("UI ready.");
  updateStatusBar();
}

function updateStatusBar(message = "") {
  const modelStatus = el("status-model");
  const validationStatus = el("status-validation");
  const trackingStatus = el("status-tracking");

  if (modelStatus) {
    modelStatus.textContent = `Model: ${state.selectedModel}`;
  }
  if (validationStatus) {
    validationStatus.textContent = state.validationOk ? "Validation: ok" : `Validation: ${message || "pending"}`;
    validationStatus.classList.remove("good", "warn", "bad");
    validationStatus.classList.add(state.validationOk ? "good" : "warn");
  }
  if (trackingStatus) {
    const t = state.tracking;
    const label = t.provider === "none" ? "Tracking: none" : `Tracking: ${t.provider} @ ${t.endpoint}`;
    trackingStatus.textContent = label;
  }
}

function watchGraphResize() {
  const wrapper = el("graph-wrapper");
  if (!wrapper || typeof ResizeObserver === "undefined") return;
  const resizeObserver = new ResizeObserver(() => drawLinks());
  resizeObserver.observe(wrapper);
}

document.addEventListener("DOMContentLoaded", init);
