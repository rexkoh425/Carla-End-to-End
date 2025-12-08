#!/usr/bin/env python3
"""
Custom CARLA spawner (vehicles + walkers) using Traffic Manager.

This script follows the official CARLA samples but lives outside the bundled examples.
It can spawn a hero vehicle plus traffic, optional walkers, and (optionally) run in
sync mode while restoring world settings on exit.

Usage examples (from repo root):
  # Against a locally running CARLA server (Windows binary)
  python CarlaControl/spawn_custom_npc.py --host 127.0.0.1 --port 2000 --vehicles 10 --walkers 20

  # Against the Docker carla service from inside compose
  docker compose exec backend micromamba run -n app python CarlaControl/spawn_custom_npc.py --host carla --port 2000 --vehicles 10 --walkers 20 --sync
"""

from __future__ import annotations

import argparse
import random
import time
from dataclasses import dataclass
from typing import List, Tuple

import carla
from carla import command as carla_cmd


@dataclass
class SpawnCfg:
  host: str = "127.0.0.1"
  port: int = 2000
  tm_port: int = 8000
  vehicles: int = 10
  walkers: int = 20
  autopilot: bool = True
  safe: bool = True
  respawn: bool = False
  hybrid_physics: bool = False
  sync: bool = False
  hero: bool = True
  seed: int | None = None
  spectator_follow: bool = True
  timeout: float = 10.0
  tick_sleep: float = 0.05
  duration: float = 0.0  # seconds to run before cleanup; 0 = run until stopped


def _set_sync(world: carla.World, sync: bool, delta: float = 0.05):
  settings = world.get_settings()
  settings.synchronous_mode = sync
  settings.fixed_delta_seconds = delta if sync else None
  world.apply_settings(settings)
  return settings


def _spawn_hero(world: carla.World, bp_lib, spawn_points: List[carla.Transform]) -> carla.Actor | None:
  for sp in spawn_points:
    bp = random.choice(bp_lib.filter("vehicle.*"))
    try:
      if bp.has_attribute("role_name"):
        bp.set_attribute("role_name", "hero")
    except Exception:
      pass
    hero = world.try_spawn_actor(bp, sp)
    if hero:
      print(f"[spawn] hero id={hero.id} at {sp}")
      return hero
  print("[spawn] failed to spawn hero")
  return None


def _spawn_vehicles(world: carla.World, bp_lib, spawn_points: List[carla.Transform], count: int, autopilot: bool, tm_port: int) -> List[carla.Actor]:
  actors: List[carla.Actor] = []
  random.shuffle(spawn_points)
  for sp in spawn_points[:count]:
    bp = random.choice(bp_lib.filter("vehicle.*"))
    actor = world.try_spawn_actor(bp, sp)
    if actor:
      if autopilot and hasattr(actor, "set_autopilot"):
        actor.set_autopilot(True, tm_port)
      actors.append(actor)
      print(f"[spawn] vehicle id={actor.id} at {sp}")
    else:
      print(f"[spawn] vehicle spawn failed at {sp}")
  return actors


def _spawn_walkers(client: carla.Client, world: carla.World, bp_lib, count: int, tm: carla.TrafficManager) -> Tuple[List[carla.Actor], List[carla.Actor]]:
  walkers: List[carla.Actor] = []
  controllers: List[carla.Actor] = []
  walker_bps = bp_lib.filter("walker.pedestrian.*")

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
    # Set speed from blueprint if available; default to 1.5 m/s
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
    print(f"[spawn] walker id={walker.id} ctrl={controller.id}")

  tm.global_percentage_speed_difference(-30.0)

  return walkers, controllers


def _cleanup(actors):
  for a in actors:
    try:
      a.destroy()
    except Exception:
      pass


def main(cfg: SpawnCfg) -> None:
  client = carla.Client(cfg.host, cfg.port)
  client.set_timeout(cfg.timeout)
  # Wait for world to be ready
  world = None
  for i in range(10):
    try:
      world = client.get_world()
      world.get_map()
      break
    except Exception as exc:
      print(f"[spawn_custom_npc] world attempt {i+1}/10 failed: {exc}")
      time.sleep(0.5)
  if world is None:
    raise RuntimeError("Failed to get CARLA world (server not ready)")
  original_settings = world.get_settings()
  tm = client.get_trafficmanager(cfg.tm_port)
  if cfg.seed is not None:
    tm.set_random_device_seed(cfg.seed)
  tm.set_synchronous_mode(cfg.sync)
  tm.set_global_distance_to_leading_vehicle(2.0 if cfg.safe else 1.0)
  tm.set_respawn_dormant_vehicles(cfg.respawn)
  tm.set_hybrid_physics_mode(cfg.hybrid_physics)

  bp_lib = world.get_blueprint_library()
  spawn_points = world.get_map().get_spawn_points()
  if not spawn_points:
    raise RuntimeError("No spawn points available in current map.")
  if cfg.seed is not None:
    random.seed(cfg.seed)

  previous_settings = None
  if cfg.sync:
    previous_settings = _set_sync(world, True)

  all_actors: List[carla.Actor] = []
  controllers: List[carla.Actor] = []
  hero = None
  try:
    if cfg.hero:
      hero = _spawn_hero(world, bp_lib, spawn_points)
      if hero and cfg.autopilot and hasattr(hero, "set_autopilot"):
        hero.set_autopilot(True, cfg.tm_port)
        all_actors.append(hero)
        if cfg.spectator_follow:
          spectator = world.get_spectator()
          spectator.set_transform(hero.get_transform())

    vehicles_needed = max(cfg.vehicles - (1 if hero else 0), 0)
    max_vehicles = max(0, len(spawn_points) - 1)
    vehicles_needed = min(vehicles_needed, max_vehicles)
    vehicles = _spawn_vehicles(world, bp_lib, spawn_points, vehicles_needed, cfg.autopilot, cfg.tm_port)
    walkers, walker_controllers = _spawn_walkers(client, world, bp_lib, cfg.walkers, tm) if cfg.walkers > 0 else ([], [])

    all_actors.extend(vehicles)
    all_actors.extend(walkers)
    controllers.extend(walker_controllers)

    print(f"[summary] hero={'yes' if hero else 'no'}, vehicles={len(vehicles)}, walkers={len(walkers)}, controllers={len(controllers)}")
    # Let the simulation run; keep ticking in sync mode.
    if cfg.duration and cfg.duration > 0:
      start = time.monotonic()
      while time.monotonic() - start < cfg.duration:
        if cfg.sync:
          world.tick()
          if cfg.tick_sleep > 0:
            time.sleep(cfg.tick_sleep)
        else:
          time.sleep(0.5)
    else:
      while True:
        if cfg.sync:
          world.tick()
          if cfg.tick_sleep > 0:
            time.sleep(cfg.tick_sleep)
        else:
          time.sleep(0.5)
  except KeyboardInterrupt:
    pass
  finally:
    _cleanup(controllers)
    _cleanup(all_actors)
    tm.set_synchronous_mode(False)
    if previous_settings:
      world.apply_settings(previous_settings)
    else:
      world.apply_settings(original_settings)
    print("[cleanup] destroyed actors and restored settings.")


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--host", type=str, default="127.0.0.1")
  parser.add_argument("--port", type=int, default=2000)
  parser.add_argument("--tm-port", type=int, default=8000)
  parser.add_argument("--vehicles", type=int, default=10)
  parser.add_argument("--walkers", type=int, default=20)
  parser.add_argument("--no-autopilot", action="store_true", help="Disable TM autopilot for vehicles.")
  parser.add_argument("--unsafe", action="store_true", help="Disable safe distance in TM.")
  parser.add_argument("--respawn", action="store_true", help="Enable TM dormant vehicle respawn.")
  parser.add_argument("--hybrid-physics", action="store_true", help="Enable TM hybrid physics.")
  parser.add_argument("--sync", action="store_true", help="Run world and TM in synchronous mode.")
  parser.add_argument("--no-hero", action="store_true", help="Skip spawning a hero vehicle.")
  parser.add_argument("--no-follow", action="store_true", help="Do not move spectator to hero.")
  parser.add_argument("--seed", type=int, default=None)
  parser.add_argument("--timeout", type=float, default=10.0)
  parser.add_argument("--tick-sleep", type=float, default=0.05, help="Seconds to sleep after each tick (sync mode).")
  parser.add_argument("--duration", type=float, default=0.0, help="Seconds to run before auto-cleanup (0 = infinite).")
  args = parser.parse_args()

  cfg = SpawnCfg(
    host=args.host,
    port=args.port,
    tm_port=args.tm_port,
    vehicles=args.vehicles,
    walkers=args.walkers,
    autopilot=not args.no_autopilot,
    safe=not args.unsafe,
    respawn=args.respawn,
    hybrid_physics=args.hybrid_physics,
    sync=args.sync,
    hero=not args.no_hero,
    spectator_follow=not args.no_follow,
    seed=args.seed,
    timeout=args.timeout,
    tick_sleep=args.tick_sleep,
    duration=args.duration,
  )
  main(cfg)
