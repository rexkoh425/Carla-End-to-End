from __future__ import annotations

import re
from typing import Dict, List


DEFAULT_HOST = "host.docker.internal"
DEFAULT_PORT = 2000


def _extract_int(text: str, keywords: List[str], default: int | None = None) -> int | None:
  for kw in keywords:
    m = re.search(rf"(\d+)\s+{kw}", text)
    if m:
      try:
        return int(m.group(1))
      except Exception:
        pass
  return default


def _extract_number_after_word(text: str, word: str, default: int | None = None) -> int | None:
  m = re.search(rf"{word}\s+(\d+)", text)
  if m:
    try:
      return int(m.group(1))
    except Exception:
      pass
  return default


def _extract_bool(text: str, positive: List[str], negative: List[str], default: bool | None) -> bool | None:
  if any(p in text for p in positive):
    return True
  if any(n in text for n in negative):
    return False
  return default


def parse_text_to_actions(
  text: str,
  default_host: str = DEFAULT_HOST,
  default_port: int = DEFAULT_PORT,
) -> Dict[str, List[Dict]]:
  t = text.lower()
  host = default_host
  port = default_port

  host_match = re.search(r"host\s+([a-z0-9\.\-:_]+)", t)
  if host_match:
    host = host_match.group(1)
  port_match = re.search(r"port\s+(\d+)", t)
  if port_match:
    try:
      port = int(port_match.group(1))
    except Exception:
      pass

  actions: List[Dict] = []

  # Connectivity
  if any(k in t for k in ["connect", "ping", "reach", "latency", "check connection"]):
    streaming_port = _extract_number_after_word(t, "stream", None)
    act = {"type": "connect_check", "host": host, "port": port}
    if streaming_port is not None:
      act["streaming_port"] = streaming_port
    actions.append(act)

  # Actor counts
  if any(k in t for k in ["actor count", "how many actors", "count actors", "count vehicles", "count walkers"]):
    actions.append({"type": "actor_counts", "host": host, "port": port})

  # Lane follow
  if "lane follow" in t or "lane_follow" in t:
    actions.append({"type": "lane_follow", "host": host, "port": port})

  # LiDAR
  if "lidar" in t and ("capture" in t or "frame" in t):
    actions.append({"type": "lidar_capture", "host": host, "port": port})
  if "lidar" in t and ("map" in t or "project" in t):
    actions.append({"type": "lidar_to_map"})

  # Spawn logic
  if "spawn" in t or "sensor rig" in t or "camera and lidar" in t:
    vehicles = _extract_int(t, ["vehicle", "vehicles", "car", "cars"], None)
    walkers = _extract_int(t, ["walker", "walkers", "pedestrian", "pedestrians"], None)
    town_match = re.search(r"town\s*([0-9a-z_]+)", t)
    town = town_match.group(1) if town_match else None
    autopilot = _extract_bool(t, ["autopilot on", "with autopilot"], ["no autopilot", "autopilot off"], True)
    sync = _extract_bool(t, ["sync", "synchronous"], ["async", "asynchronous"], None)
    seed = _extract_number_after_word(t, "seed", None)
    respawn = "respawn" in t
    hybrid = "hybrid physics" in t or "hybrid" in t
    safe = not ("unsafe" in t)
    hero = not ("no hero" in t or "skip hero" in t)
    spectator_follow = "first person" in t or "fpv" in t or "follow camera" in t

    vehicles = vehicles if vehicles is not None else 5
    walkers = walkers if walkers is not None else 5

    is_sensor_rig = "sensor rig" in t or "camera and lidar" in t
    is_custom = ("custom npc" in t or "tm" in t or "traffic manager" in t) and not is_sensor_rig
    action_type = "spawn_sensor_rig" if is_sensor_rig else ("spawn_custom_npc" if is_custom else "spawn_scene")
    act = {
      "type": action_type,
      "host": host,
      "port": port,
    }
    if not is_sensor_rig:
      act.update({"vehicles": vehicles, "walkers": walkers})
    if town:
      act["town"] = town
    if autopilot is not None:
      act["autopilot"] = bool(autopilot)
    if sync is not None:
      act["sync"] = bool(sync)
    if seed is not None:
      act["seed"] = seed
    if is_custom:
      act["safe"] = bool(safe)
      act["respawn"] = bool(respawn)
      act["hybrid_physics"] = bool(hybrid)
      act["hero"] = bool(hero)
      act["spectator_follow"] = bool(spectator_follow)
    if is_sensor_rig:
      act["sync"] = True if sync is None else bool(sync)
      act["autopilot"] = True if autopilot is None else bool(autopilot)
    actions.append(act)

  # Follow hero (spectator FPV)
  if "follow hero" in t or "fpv" in t or "first person" in t or "spectator follow" in t:
    actions.append({"type": "follow_hero", "host": host, "port": port})

  # Start CARLA container (only if explicitly asked)
  if "start carla container" in t or "launch carla container" in t or "carla launch" in t:
    actions.append({"type": "carla_launch"})

  # Stop sim
  if "stop sim" in t or "stop simulation" in t or "stop carla" in t:
    actions.append({"type": "stop_sim", "host": host, "port": port})

  return {"actions": actions}
