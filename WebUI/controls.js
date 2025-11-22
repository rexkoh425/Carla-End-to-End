const logBox = document.getElementById("ctrl-log");

function log(msg) {
  const time = new Date().toLocaleTimeString();
  if (logBox) logBox.textContent = `[${time}] ${msg}\n${logBox.textContent}`;
}

function buildFileUrl(path) {
  return `/api/file?path=${encodeURIComponent(path)}`;
}

function refreshImage(imgId, pathInputId) {
  const img = document.getElementById(imgId);
  const path = document.getElementById(pathInputId)?.value || "";
  if (img && path) {
    img.src = `${buildFileUrl(path)}&ts=${Date.now()}`;
    log(`Preview updated: ${path}`);
  }
}

function getCarlaHostPort() {
  const host = document.getElementById("carla-host")?.value || "host.docker.internal";
  const port = parseInt(document.getElementById("carla-port")?.value || "2000", 10);
  return { host, port };
}

function getLidarOpts() {
  const rangeVal = parseFloat(document.getElementById("lidar-range")?.value || "50");
  const sensor = document.getElementById("lidar-mode")?.value || "sensor.lidar.ray_cast";
  return {
    range: isNaN(rangeVal) ? 50 : rangeVal,
    sensor,
  };
}

function setText(id, text) {
  const el = document.getElementById(id);
  if (el) {
    el.textContent = text;
  }
}

function updateSpawnVisibility() {
  const mode = document.getElementById("spawn-mode")?.value || "scene";
  const numbersRow = document.querySelector(".grid-3");
  const autopilotLabel = document.querySelector("#btn-spawn + label");
  if (numbersRow) {
    numbersRow.style.display = mode === "hero_only" || mode === "sensor_rig" || mode === "hero_sensors" ? "none" : "grid";
  }
  if (autopilotLabel) {
    autopilotLabel.style.display = mode === "sensor_rig" ? "none" : "inline-flex";
  }
}

async function runScript(script, payload = {}) {
  try {
    const res = await fetch("/api/tools/run_script", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ script, ...payload }),
    });
    if (!res.ok) {
      const text = await res.text();
      log(`Error (${res.status}): ${text}`);
      return null;
    }
    const data = await res.json();
    log(`Started ${script} pid=${data.pid || "n/a"}`);
    return data;
  } catch (err) {
    log(`Error: ${err.message}`);
    return null;
  }
}

function bindControls() {
  const btnSpawn = document.getElementById("btn-spawn");
  if (btnSpawn) {
    btnSpawn.addEventListener("click", () => {
      const { host, port } = getCarlaHostPort();
      const vehicles = parseInt(document.getElementById("ctrl-vehicles")?.value || "20", 10);
      const walkers = parseInt(document.getElementById("ctrl-walkers")?.value || "40", 10);
      const town = document.getElementById("ctrl-town")?.value || "";
      const autopilotSelect = document.getElementById("autopilot-select");
      const autopilot = autopilotSelect ? autopilotSelect.value === "on" : true;
      const mode = document.getElementById("spawn-mode")?.value || "scene";
      const followAction = document.getElementById("follow-action")?.value || "none";
      const wantFollow = followAction === "follow_hero";

      if (mode === "sensor_rig") {
        const { range, sensor } = getLidarOpts();
        runScript("spawn_sensor_rig", { host, port, autopilot: true, sync: true, range, sensor });
        return;
      }
      if (mode === "hero_sensors") {
        const { range } = getLidarOpts();
        runScript("spawn_hero_sensors", { host, port, range, fps: 10, pps: 200000, channels: 32, no_autopilot: !autopilot }).then(() => {
          if (wantFollow) {
            setTimeout(() => runScript("follow_hero", { host, port }), 800);
          }
        });
        return;
      }

      if (mode === "scene") {
        runScript("spawn_scene", { host, port, vehicles, walkers, town, no_autopilot: !autopilot });
        return;
      }

      const payload = {
        host,
        port,
        vehicles: mode === "hero_only" ? 0 : vehicles,
        walkers: mode === "hero_only" ? 0 : walkers,
        no_autopilot: !autopilot,
      };
      // spawn_custom_npc already spawns a hero by default; we keep spectator follow on by default.
      runScript("spawn_custom_npc", payload).then(() => {
        if (wantFollow) {
          setTimeout(() => runScript("follow_hero", { host, port }), 1200);
        }
      });
    });
  }

  const btnSensorRig = document.getElementById("btn-sensor-rig");
  if (btnSensorRig) {
    btnSensorRig.addEventListener("click", () => {
      const { host, port } = getCarlaHostPort();
      const { range, sensor } = getLidarOpts();
      runScript("spawn_sensor_rig", { host, port, autopilot: true, sync: true, range, sensor });
    });
  }

  const btnLane = document.getElementById("btn-lane");
  if (btnLane) {
    btnLane.addEventListener("click", () => {
      const { host, port } = getCarlaHostPort();
      runScript("lane_follow", { host, port });
    });
  }

  const btnFollowHero = document.getElementById("btn-follow-hero");
  if (btnFollowHero) {
    btnFollowHero.addEventListener("click", () => {
      const { host, port } = getCarlaHostPort();
      runScript("follow_hero", { host, port });
    });
  }
  const btnStopFpv = document.getElementById("btn-stop-fpv");
  if (btnStopFpv) {
    btnStopFpv.addEventListener("click", () => {
      const { host, port } = getCarlaHostPort();
      runScript("follow_hero", { host, port, stop: true });
    });
  }

  const btnRecStart = document.getElementById("btn-rec-start");
  if (btnRecStart) {
    btnRecStart.addEventListener("click", () => {
      const { host, port } = getCarlaHostPort();
      const duration = parseFloat(document.getElementById("rec-duration")?.value || "15");
      const fps = parseFloat(document.getElementById("rec-fps")?.value || "15");
      const range = parseFloat(document.getElementById("rec-range")?.value || "50");
      const autopilotSelect = document.getElementById("autopilot-select");
      const autopilot = autopilotSelect ? autopilotSelect.value === "on" : true;
      runScript("record_sensors", {
        host,
        port,
        duration,
        fps,
        range,
        no_autopilot: !autopilot,
      });
    });
  }
  const btnRecStop = document.getElementById("btn-rec-stop");
  if (btnRecStop) {
    btnRecStop.addEventListener("click", () => {
      runScript("record_sensors", { stop: true });
    });
  }

  const btnRobotaxi = document.getElementById("btn-robotaxi-start");
  if (btnRobotaxi) {
    btnRobotaxi.addEventListener("click", () => {
      const { host, port } = getCarlaHostPort();
      const duration = parseFloat(document.getElementById("rec-duration")?.value || "15");
      const fps = parseFloat(document.getElementById("rec-fps")?.value || "10");
      const range = parseFloat(document.getElementById("rec-range")?.value || "100");
      const autopilotSelect = document.getElementById("autopilot-select");
      const autopilot = autopilotSelect ? autopilotSelect.value === "on" : true;
      runScript("record_robotaxi", {
        host,
        port,
        duration,
        fps,
        range,
        no_autopilot: !autopilot,
      });
    });
  }
  const btnRobotaxiStop = document.getElementById("btn-robotaxi-stop");
  if (btnRobotaxiStop) {
    btnRobotaxiStop.addEventListener("click", () => {
      runScript("record_robotaxi", { stop: true });
    });
  }

  const btnCarla = document.getElementById("btn-carla-start");
  if (btnCarla) {
    btnCarla.addEventListener("click", () => {
      runScript("carla_launch", {});
    });
  }

  const btnConnectCheck = document.getElementById("btn-connect-check");
  if (btnConnectCheck) {
    btnConnectCheck.addEventListener("click", async () => {
      const { host, port } = getCarlaHostPort();
      const streamingPortRaw = document.getElementById("carla-streaming-port")?.value || "";
      const streaming_port = streamingPortRaw === "" ? null : parseInt(streamingPortRaw, 10);
      try {
        const res = await fetch("/api/carla/connect_check", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ host, port, streaming_port }),
        });
        const data = await res.json();
        if (!res.ok) {
          log(`Connectivity error (${res.status}): ${data.detail || res.statusText}`);
          setText("connect-output", `Error: ${data.detail || res.statusText}`);
          return;
        }
        const msg = `RPC ${data.rpc_port_ok ? "ok" : "fail"}`
          + (streaming_port !== null ? ` / Stream ${data.streaming_port_ok ? "ok" : "fail"}` : "");
        log(`Connectivity: host=${host} port=${port} ${msg}`);
        setText("connect-output", JSON.stringify(data, null, 2));
      } catch (err) {
        log(`Connectivity error: ${err.message}`);
        setText("connect-output", `Error: ${err.message}`);
      }
    });
  }

  const btnActorCounts = document.getElementById("btn-actor-counts");
  if (btnActorCounts) {
    btnActorCounts.addEventListener("click", async () => {
      const { host, port } = getCarlaHostPort();
      try {
        const res = await fetch("/api/carla/actor_counts", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ host, port }),
        });
        const data = await res.json();
        if (!res.ok) {
          log(`Actor counts error (${res.status}): ${data.detail || res.statusText}`);
          setText("actors-output", `Error: ${data.detail || res.statusText}`);
          return;
        }
        log(`Actors: vehicles=${data.vehicles} walkers=${data.walkers}`);
        setText("actors-output", JSON.stringify(data, null, 2));
      } catch (err) {
        log(`Actor counts error: ${err.message}`);
        setText("actors-output", `Error: ${err.message}`);
      }
    });
  }

  const btnLidar = document.getElementById("btn-lidar");
  if (btnLidar) {
    btnLidar.addEventListener("click", () => {
      const { host, port } = getCarlaHostPort();
      const { range, sensor } = getLidarOpts();
      // Capture, then build map, then refresh preview.
      runScript("lidar_capture", { host, port, range, sensor }).then(() => {
        setTimeout(() => {
          runScript("lidar_to_map", {}).then(() => {
            setTimeout(() => refreshImage("preview-lidar-img", "preview-lidar-path"), 1500);
          });
        }, 1200);
      });
    });
  }

  const btnMap = document.getElementById("btn-map");
  if (btnMap) {
    btnMap.addEventListener("click", () => {
      runScript("lidar_to_map", {});
      // Try to auto-refresh the preview a moment after map generation kicks off.
      setTimeout(() => refreshImage("preview-lidar-img", "preview-lidar-path"), 1500);
    });
  }

  const btnPreviewLidar = document.getElementById("btn-preview-lidar");
  if (btnPreviewLidar) {
    btnPreviewLidar.addEventListener("click", () => {
      refreshImage("preview-lidar-img", "preview-lidar-path");
    });
  }

  const btnPreviewDebug = document.getElementById("btn-preview-debug");
  if (btnPreviewDebug) {
    btnPreviewDebug.addEventListener("click", () => {
      const path = document.getElementById("preview-debug-path")?.value || "";
      const img = document.getElementById("preview-debug-img");
      if (img && path) img.src = buildFileUrl(path);
      log(`Preview debug: ${path || "no path"}`);
    });
  }

  const btnAgentSend = document.getElementById("btn-agent-send");
  if (btnAgentSend) {
    btnAgentSend.addEventListener("click", async () => {
      const prompt = document.getElementById("agent-prompt")?.value || "";
      const replyBox = document.getElementById("agent-reply");
      if (!prompt.trim()) {
        log("Agent: empty prompt");
        return;
      }
      try {
        const res = await fetch("/api/agent/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ prompt }),
        });
        if (!res.ok) {
          const t = await res.text();
          log(`Agent error (${res.status}): ${t}`);
          if (replyBox) replyBox.textContent = `Error: ${t}`;
          return;
        }
        const data = await res.json();
        log("Agent reply received");
        if (replyBox) replyBox.textContent = data.reply || JSON.stringify(data, null, 2);
      } catch (err) {
        log(`Agent error: ${err.message}`);
        if (replyBox) replyBox.textContent = `Error: ${err.message}`;
      }
    });
  }
}

document.addEventListener("DOMContentLoaded", () => {
  bindControls();
  const modeSelect = document.getElementById("spawn-mode");
  if (modeSelect) {
    modeSelect.addEventListener("change", updateSpawnVisibility);
    updateSpawnVisibility();
  }
});
