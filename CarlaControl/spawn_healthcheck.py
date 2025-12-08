#!/usr/bin/env python3
"""
Quick CARLA spawn health check for vehicles and walkers.

It connects to a CARLA server, spawns a few vehicles and walkers, ticks the
world briefly, and then cleans everything up. Useful as a lightweight API
check to confirm that spawning still works end-to-end.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from typing import List, Tuple

import carla
from carla import command as carla_cmd


@dataclass
class SpawnCheckConfig:
  host: str = "127.0.0.1"
  port: int = 2000
  tm_port: int = 8000
  vehicles: int = 3
  walkers: int = 6
  timeout: float = 8.0
  tick_sleep: float = 0.05
  attempts: int = 8
  sync: bool = True
  seed: int | None = None


def _set_sync(world: carla.World, sync: bool, delta: float = 0.05):
  settings = world.get_settings()
  settings.synchronous_mode = sync
  settings.fixed_delta_seconds = delta if sync else None
  world.apply_settings(settings)
  return settings


def _spawn_vehicles(world: carla.World, bp_lib, spawn_points: List[carla.Transform], count: int, tm_port: int) -> Tuple[List[carla.Actor], int]:
  vehicles: List[carla.Actor] = []
  failures = 0
  vehicle_bps = list(bp_lib.filter("vehicle.*"))
  if not vehicle_bps or count <= 0:
    return vehicles, failures

  points = list(spawn_points)
  random.shuffle(points)
  for sp in points[:count]:
    bp = random.choice(vehicle_bps)
    actor = world.try_spawn_actor(bp, sp)
    if actor:
      if hasattr(actor, "set_autopilot"):
        actor.set_autopilot(True, tm_port)
      vehicles.append(actor)
      print(f"[spawn_healthcheck] vehicle id={actor.id} at {sp}")
    else:
      failures += 1
  return vehicles, failures


def _spawn_walkers(client: carla.Client, world: carla.World, bp_lib, count: int) -> Tuple[List[carla.Actor], List[carla.Actor]]:
  walkers: List[carla.Actor] = []
  controllers: List[carla.Actor] = []
  walker_bps = list(bp_lib.filter("walker.pedestrian.*"))
  if not walker_bps or count <= 0:
    return walkers, controllers

  spawn_points: List[carla.Transform] = []
  for _ in range(count):
    loc = world.get_random_location_from_navigation()
    if loc:
      spawn_points.append(carla.Transform(loc))

  if not spawn_points:
    return walkers, controllers

  batch = [carla_cmd.SpawnActor(random.choice(walker_bps), tr) for tr in spawn_points]
  results = client.apply_batch_sync(batch, True)
  walker_ids = [r.actor_id for r in results if not r.error]
  if not walker_ids:
    return walkers, controllers

  controller_bp = bp_lib.find("controller.ai.walker")
  batch_ctrl = [carla_cmd.SpawnActor(controller_bp, carla.Transform(), wid) for wid in walker_ids]
  results_ctrl = client.apply_batch_sync(batch_ctrl, True)
  controller_ids = [r.actor_id for r in results_ctrl if not r.error]

  for wid, cid in zip(walker_ids, controller_ids):
    walker = world.get_actor(wid)
    controller = world.get_actor(cid)
    if not walker or not controller:
      continue
    walkers.append(walker)
    controllers.append(controller)
    controller.start()
    speed = 1.5
    try:
      speeds = walker.attributes.get("speed", "")
      if speeds:
        vals = [float(v) for v in speeds.split() if v]
        if vals:
          speed = vals[0]
    except Exception:
      pass
    controller.set_max_speed(speed)
    dest = world.get_random_location_from_navigation()
    if dest:
      controller.go_to_location(dest)
    print(f"[spawn_healthcheck] walker id={walker.id} ctrl={controller.id}")

  return walkers, controllers


def _cleanup(actors: List[carla.Actor]) -> None:
  for a in actors:
    try:
      a.destroy()
    except Exception:
      pass


def run_spawn_check(cfg: SpawnCheckConfig) -> dict:
  start = time.monotonic()
  client = carla.Client(cfg.host, cfg.port)
  client.set_timeout(cfg.timeout)

  last_exc: Exception | None = None
  world: carla.World | None = None
  for i in range(cfg.attempts):
    try:
      world = client.get_world()
      world.get_map()
      break
    except Exception as exc:
      last_exc = exc
      time.sleep(0.25)
  if world is None:
    raise RuntimeError(f"Failed to get CARLA world at {cfg.host}:{cfg.port}: {last_exc}")

  original_settings = world.get_settings()
  tm = client.get_trafficmanager(cfg.tm_port)
  tm.set_synchronous_mode(cfg.sync)

  bp_lib = world.get_blueprint_library()
  spawn_points = world.get_map().get_spawn_points()
  if not spawn_points:
    raise RuntimeError("No spawn points available in current map.")
  if cfg.seed is not None:
    random.seed(cfg.seed)

  if cfg.sync:
    _set_sync(world, True)

  vehicles: List[carla.Actor] = []
  walkers: List[carla.Actor] = []
  controllers: List[carla.Actor] = []
  notes: List[str] = []
  issues: List[str] = []

  try:
    vehicles, vehicle_failures = _spawn_vehicles(world, bp_lib, spawn_points, cfg.vehicles, cfg.tm_port)
    walkers, controllers = _spawn_walkers(client, world, bp_lib, cfg.walkers)

    for _ in range(5):
      if cfg.sync:
        world.tick()
        if cfg.tick_sleep > 0:
          time.sleep(cfg.tick_sleep)
      else:
        time.sleep(0.2)

    if not vehicles:
      issues.append("No vehicles spawned.")
    elif vehicle_failures:
      notes.append(f"{vehicle_failures} vehicle spawn attempts failed.")
    if cfg.walkers > 0 and not walkers:
      issues.append("No walkers spawned.")
    elif cfg.walkers > 0 and len(walkers) < cfg.walkers:
      notes.append(f"Requested {cfg.walkers} walkers, got {len(walkers)}.")
  finally:
    _cleanup(controllers)
    _cleanup(walkers)
    _cleanup(vehicles)
    tm.set_synchronous_mode(False)
    world.apply_settings(original_settings)

  result = {
    "ok": not issues,
    "vehicles_requested": cfg.vehicles,
    "vehicles_spawned": len(vehicles),
    "walkers_requested": cfg.walkers,
    "walkers_spawned": len(walkers),
    "duration_sec": round(time.monotonic() - start, 3),
    "issues": issues,
    "notes": notes,
    "config": asdict(cfg),
  }
  print(f"[spawn_healthcheck] result ok={result['ok']} vehicles={result['vehicles_spawned']} walkers={result['walkers_spawned']}")
  return result


def main() -> int:
  parser = argparse.ArgumentParser(description="CARLA spawn health check (vehicles + walkers).")
  parser.add_argument("--host", type=str, default="127.0.0.1")
  parser.add_argument("--port", type=int, default=2000)
  parser.add_argument("--tm-port", type=int, default=8000)
  parser.add_argument("--vehicles", type=int, default=3)
  parser.add_argument("--walkers", type=int, default=6)
  parser.add_argument("--timeout", type=float, default=8.0)
  parser.add_argument("--tick-sleep", type=float, default=0.05)
  parser.add_argument("--attempts", type=int, default=8, help="Attempts to fetch world/map before giving up.")
  parser.add_argument("--seed", type=int, default=None)
  parser.add_argument("--no-sync", action="store_true", help="Run without synchronous mode.")
  args = parser.parse_args()

  cfg = SpawnCheckConfig(
    host=args.host,
    port=args.port,
    tm_port=args.tm_port,
    vehicles=args.vehicles,
    walkers=args.walkers,
    timeout=args.timeout,
    tick_sleep=args.tick_sleep,
    attempts=args.attempts,
    seed=args.seed,
    sync=not args.no_sync,
  )
  try:
    result = run_spawn_check(cfg)
  except Exception as exc:
    print(f"[spawn_healthcheck] error: {exc}")
    return 1

  print(json.dumps(result, indent=2))
  return 0 if result.get("ok") else 2


if __name__ == "__main__":
  sys.exit(main())
