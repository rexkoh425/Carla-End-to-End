// Config Editor: YAML-driven, collapsible tree with inline edits.

// Prefer repo files served under /repo inside the nginx container (dockerized),
// but keep local dev fallback so browsing from the repo still works.
const repoBases = ["/repo", ".", ".."]; // tried in order

// Start empty; populate from yaml_index.json and always append a Custom option.
let configSources = [];

function pathCandidates(raw) {
  if (!raw) return [];
  if (raw.startsWith("http")) return [raw];
  if (raw.startsWith("/")) return [raw];
  const stripped = raw.replace(/^(\.?\.\/)+/, ""); // drop leading ./ or ../
  const paths = repoBases.map((base) => `${base.replace(/\/$/, "")}/${stripped}`);
  paths.push(raw); // original as last-resort
  return Array.from(new Set(paths));
}

function normalizePath(p) {
  return pathCandidates(p)[0] || "";
}

const cfgState = {
  selected: null,
  path: "",
  data: {},
  rawText: "",
  folded: new Set(),
  selectedPath: null,
};

const $ = (id) => document.getElementById(id);

function typeOf(val) {
  if (Array.isArray(val)) return "array";
  if (val === null) return "null";
  return typeof val;
}

function pathToString(pathArr) {
  return pathArr
    .map((p) => (typeof p === "number" ? `[${p}]` : p))
    .join(".");
}

async function loadYaml(path) {
  const candidates = pathCandidates(path);
  if (!candidates.length) {
    setStatus("Load failed: No path provided", true);
    return;
  }

  let lastError = "404 Not Found";
  for (const url of candidates) {
    try {
      const res = await fetch(url);
      if (!res.ok) {
        lastError = `${res.status} ${res.statusText}`;
        continue;
      }
      const text = await res.text();
      const parsed = jsyaml.load(text);
      cfgState.data = parsed || {};
      cfgState.rawText = text;
      cfgState.path = url;
      renderTree();
      renderYAML();
      setStatus(`Loaded ${url}`);
      renderSourcePicker();
      return;
    } catch (err) {
      lastError = err.message;
    }
  }
  setStatus(`Load failed: ${lastError}`, true);
}

function setStatus(msg, bad = false) {
  const el = $("status-line");
  if (!el) return;
  el.textContent = msg;
  el.className = bad ? "status-pill bad" : "status-pill good";
}

function renderSourcePicker() {
  const sel = $("config-source");
  sel.innerHTML = "";
  configSources.forEach((src, idx) => {
    const opt = document.createElement("option");
    opt.value = src.id;
    opt.textContent = `${src.name} - ${src.path}`;
    if ((cfgState.selected && cfgState.selected === src.id) || (!cfgState.selected && idx === 0)) opt.selected = true;
    sel.appendChild(opt);
  });
  if (!cfgState.path && configSources.length) {
    cfgState.path = configSources[0].path;
    cfgState.selected = configSources[0].id;
  }
  $("config-path").value = cfgState.path || "";
}

function toggleFold(pathStr) {
  if (cfgState.folded.has(pathStr)) cfgState.folded.delete(pathStr);
  else cfgState.folded.add(pathStr);
  renderTree();
  setStatus(cfgState.folded.has(pathStr) ? `Collapsed ${pathStr || "<root>"}` : `Expanded ${pathStr || "<root>"}`);
}

function updateValue(pathArr, newVal) {
  let cursor = cfgState.data;
  for (let i = 0; i < pathArr.length - 1; i++) {
    const key = pathArr[i];
    if (!(key in cursor)) return;
    cursor = cursor[key];
  }
  const lastKey = pathArr[pathArr.length - 1];
  cursor[lastKey] = newVal;
  renderTree();
  renderYAML();
  setStatus(`Updated ${pathToString(pathArr)}`);
}

function parseInput(raw, original) {
  const trimmed = raw.trim();
  if (typeOf(original) === "number") {
    const num = Number(trimmed);
    return Number.isNaN(num) ? original : num;
  }
  if (typeOf(original) === "boolean") {
    return trimmed === "true" || trimmed === "1" || trimmed === "yes";
  }
  if (trimmed === "" && original === null) return null;
  return raw;
}

function renderLeaf(pathArr, key, val) {
  const row = document.createElement("div");
  row.className = "tree-row";
  const label = document.createElement("div");
  label.className = "tree-label";
  label.textContent = key;
  const controls = document.createElement("div");
  controls.className = "tree-controls";

  const type = typeOf(val);
  const badge = document.createElement("span");
  badge.className = "pill micro";
  badge.textContent = type;
  controls.appendChild(badge);

  let input;
  if (type === "boolean") {
    input = document.createElement("input");
    input.type = "checkbox";
    input.checked = val === true;
    input.onchange = (e) => updateValue(pathArr, e.target.checked);
  } else {
    input = document.createElement("input");
    input.type = type === "number" ? "number" : "text";
    input.value = val ?? "";
    input.onchange = (e) => updateValue(pathArr, parseInput(e.target.value, val));
  }

  const quickBtn = document.createElement("button");
  quickBtn.className = "secondary tiny";
  quickBtn.textContent = "Select";
  quickBtn.onclick = () => selectPath(pathArr, val);

  controls.appendChild(input);
  controls.appendChild(quickBtn);

  row.appendChild(label);
  row.appendChild(controls);
  return row;
}

function renderNode(obj, pathArr = []) {
  const frag = document.createDocumentFragment();
  if (typeOf(obj) === "array") {
    obj.forEach((item, idx) => {
      const itemPath = [...pathArr, idx];
      const pathStr = pathToString(itemPath);
      const itemType = typeOf(item);
      if (itemType === "object" || itemType === "array") {
        const container = document.createElement("div");
        container.className = "tree-node";
        const header = document.createElement("div");
        header.className = "tree-row";
        const label = document.createElement("div");
        label.className = "tree-label";
        label.innerHTML = `<span class="caret ${cfgState.folded.has(pathStr) ? "" : "open"}"></span> [${idx}]`;
        label.onclick = (e) => {
          e.stopPropagation();
          toggleFold(pathStr);
        };
        header.appendChild(label);
        const controls = document.createElement("div");
        controls.className = "tree-controls";
        const badge = document.createElement("span");
        badge.className = "pill micro";
        badge.textContent = itemType;
        controls.appendChild(badge);
        const pick = document.createElement("button");
        pick.className = "secondary tiny";
        pick.textContent = "Select";
        pick.onclick = (e) => {
          e.stopPropagation();
          selectPath(itemPath, item);
        };
        controls.appendChild(pick);
        header.appendChild(controls);
        container.appendChild(header);
        if (!cfgState.folded.has(pathStr)) {
          container.appendChild(renderNode(item, itemPath));
        }
        frag.appendChild(container);
      } else {
        frag.appendChild(renderLeaf(itemPath, `[${idx}]`, item));
      }
    });
    return frag;
  }

  Object.keys(obj || {}).forEach((key) => {
    const val = obj[key];
    const nodePath = [...pathArr, key];
    const pathStr = pathToString(nodePath);
    const valType = typeOf(val);
    const container = document.createElement("div");
    container.className = "tree-node";
    const header = document.createElement("div");
    header.className = "tree-row";
    const label = document.createElement("div");
    label.className = "tree-label";
    label.innerHTML =
      valType === "object" || valType === "array"
        ? `<span class="caret ${cfgState.folded.has(pathStr) ? "" : "open"}"></span> ${key}`
        : `${key}`;
    label.onclick = (e) => {
      e.stopPropagation();
      selectPath(nodePath, val);
      if (valType === "object" || valType === "array") toggleFold(pathStr);
    };
    header.appendChild(label);
    const controls = document.createElement("div");
    controls.className = "tree-controls";
    const badge = document.createElement("span");
    badge.className = "pill micro";
    badge.textContent = valType;
    controls.appendChild(badge);
    if (valType !== "object" && valType !== "array") {
      const input = document.createElement("input");
      input.type = valType === "number" ? "number" : valType === "boolean" ? "text" : "text";
      if (valType === "boolean") input.value = val ? "true" : "false";
      else input.value = val ?? "";
      input.onchange = (e) => updateValue(nodePath, parseInput(e.target.value, val));
      controls.appendChild(input);
    } else {
      const foldBtn = document.createElement("button");
      foldBtn.className = "secondary tiny";
      foldBtn.textContent = cfgState.folded.has(pathStr) ? "Expand" : "Collapse";
      foldBtn.onclick = (e) => {
        e.stopPropagation();
        toggleFold(pathStr);
      };
      controls.appendChild(foldBtn);
    }
    const pick = document.createElement("button");
    pick.className = "secondary tiny";
    pick.textContent = "Select";
    pick.onclick = (e) => {
      e.stopPropagation();
      selectPath(nodePath, val);
    };
    controls.appendChild(pick);

    header.appendChild(controls);
    container.appendChild(header);

    if ((valType === "object" || valType === "array") && !cfgState.folded.has(pathStr)) {
      container.appendChild(renderNode(val, nodePath));
    }
    frag.appendChild(container);
  });
  return frag;
}

function renderTree() {
  const root = $("tree-root");
  root.innerHTML = "";
  const tree = renderNode(cfgState.data, []);
  root.appendChild(tree);
  enableTreePan();
}

function renderYAML() {
  $("yaml-preview").value = jsyaml.dump(cfgState.data, { indent: 2 });
}

function selectPath(pathArr, value) {
  cfgState.selectedPath = pathArr;
  const pathStr = pathToString(pathArr);
  $("selected-path").textContent = pathStr || "<root>";
  const type = typeOf(value);
  const input = $("quick-input");
  input.value = type === "object" || type === "array" ? "" : value ?? "";
  input.disabled = type === "object" || type === "array";
  $("quick-hint").textContent = type === "object" || type === "array" ? "Expand and edit nested fields." : `Editing (${type})`;
}

function applyQuickEdit() {
  if (!cfgState.selectedPath) return;
  let cursor = cfgState.data;
  for (let i = 0; i < cfgState.selectedPath.length - 1; i++) {
    cursor = cursor[cfgState.selectedPath[i]];
  }
  const key = cfgState.selectedPath[cfgState.selectedPath.length - 1];
  const original = cursor[key];
  const newVal = parseInput($("quick-input").value, original);
  cursor[key] = newVal;
  renderTree();
  renderYAML();
  setStatus(`Updated ${pathToString(cfgState.selectedPath)}`);
}

function downloadYaml() {
  const blob = new Blob([$(`yaml-preview`).value], { type: "text/yaml" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  const fname = cfgState.path.split("/").pop() || "config.yaml";
  a.download = `edited_${fname}`;
  a.click();
  URL.revokeObjectURL(url);
}

function foldAll(shouldFold) {
  cfgState.folded = new Set();
  if (shouldFold) {
    const walk = (obj, path = []) => {
      const t = typeOf(obj);
      if (t === "object") {
        Object.keys(obj).forEach((k) => {
          const p = [...path, k];
          cfgState.folded.add(pathToString(p));
          walk(obj[k], p);
        });
      } else if (t === "array") {
        obj.forEach((v, idx) => {
          const p = [...path, idx];
          cfgState.folded.add(pathToString(p));
          walk(v, p);
        });
      }
    };
    walk(cfgState.data);
  }
  cfgState.selectedPath = null;
  const qi = $("quick-input");
  if (qi) qi.value = "";
  $("selected-path").textContent = "-";
  renderTree();
  setStatus(shouldFold ? "Collapsed all" : "Expanded all");
}

function bindEvents() {
  $("config-source").addEventListener("change", (e) => {
    const src = configSources.find((s) => s.id === e.target.value);
    if (src) {
      cfgState.selected = src.id;
      cfgState.path = src.path;
      renderSourcePicker();
      loadYaml(cfgState.path);
    }
  });
  $("config-path").addEventListener("change", (e) => {
    cfgState.path = normalizePath(e.target.value.trim());
  });
  $("load-btn").addEventListener("click", () => loadYaml(cfgState.path));
  $("fold-all").addEventListener("click", () => foldAll(true));
  $("expand-all").addEventListener("click", () => foldAll(false));
  $("quick-apply").addEventListener("click", applyQuickEdit);
  $("download-btn").addEventListener("click", downloadYaml);
}

let treePanBound = false;
function enableTreePan() {
  if (treePanBound) return;
  const shell = $("tree-shell");
  if (!shell) return;
  let isPanning = false;
  let startX = 0;
  let startY = 0;
  let scrollLeft = 0;
  let scrollTop = 0;
  shell.addEventListener("pointerdown", (e) => {
    if (e.target.closest(".tree-row") || e.target.closest(".tree-controls") || e.target.closest(".tree-label")) {
      // let clicks on rows/controls handle folding/selecting
      return;
    }
    isPanning = true;
    shell.classList.add("panning");
    startX = e.clientX;
    startY = e.clientY;
    scrollLeft = shell.scrollLeft;
    scrollTop = shell.scrollTop;
    shell.setPointerCapture(e.pointerId);
  });
  shell.addEventListener("pointermove", (e) => {
    if (!isPanning) return;
    const dx = e.clientX - startX;
    const dy = e.clientY - startY;
    shell.scrollLeft = scrollLeft - dx;
    shell.scrollTop = scrollTop - dy;
  });
  const stopPan = () => {
    isPanning = false;
    shell.classList.remove("panning");
  };
  shell.addEventListener("pointerup", stopPan);
  shell.addEventListener("pointercancel", stopPan);
  shell.addEventListener("pointerleave", stopPan);
  treePanBound = true;
}

function initConfigPage() {
  fetch("yaml_index.json")
    .then((res) => (res.ok ? res.json() : []))
    .then((items) => {
      const normalized =
        Array.isArray(items) && items.length
          ? items.map((i, idx) => ({
              id: i.id || `yaml_${idx}`,
              name: i.name || (i.path || "").split("/").pop() || `config_${idx}`,
              path: normalizePath(i.path || ""),
            }))
          : [];
      // Always add a Custom entry at the end.
      normalized.push({ id: "custom", name: "Custom path", path: "models/path/to/config.yaml" });
      configSources = normalized.filter((s) => s.path);
    })
    .catch(() => {})
    .finally(() => {
      if (!configSources.length) {
        configSources = [{ id: "custom", name: "Custom path", path: "models/path/to/config.yaml" }];
      }
      cfgState.selected = configSources[0].id;
      cfgState.path = configSources[0].path;
      renderSourcePicker();
      bindEvents();
      loadYaml(cfgState.path);
    });
}

document.addEventListener("DOMContentLoaded", initConfigPage);

