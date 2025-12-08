#!/usr/bin/env python3
"""
Spawn a hero vehicle with attached RGB camera + LiDAR and leave them running (autopilot optional).

Usage:
  python CarlaControl/spawn_hero_sensors.py --host 127.0.0.1 --port 2000 --range 50 --fps 10
Stop with Ctrl+C (or via run_script stop signal if added externally).
"""

from __future__ import annotations

import argparse
import random
import time

import carla


def _set_sync(world: carla.World, sync: bool, delta: float = 0.05):
  settings = world.get_settings()
  settings.synchronous_mode = sync
  settings.fixed_delta_seconds = delta if sync else None
  world.apply_settings(settings)
  return settings


def main() -> None:
  parser = argparse.ArgumentParser(description="Spawn hero with camera + LiDAR and keep running.")
  parser.add_argument("--host", type=str, default="127.0.0.1")
  parser.add_argument("--port", type=int, default=2000)
  parser.add_argument("--tm-port", type=int, default=8000)
  parser.add_argument("--range", type=float, default=50.0, help="LiDAR range (m)")
  parser.add_argument("--fps", type=float, default=10.0, help="LiDAR rotation frequency / target FPS (sync)")
  parser.add_argument("--pps", type=int, default=200000, help="LiDAR points per second")
  parser.add_argument("--channels", type=int, default=32)
  parser.add_argument("--cam-width", type=int, default=960)
  parser.add_argument("--cam-height", type=int, default=540)
  parser.add_argument("--cam-fov", type=float, default=90.0)
  parser.add_argument("--sync", action="store_true", help="Enable synchronous mode.")
  parser.add_argument("--no-autopilot", action="store_true", help="Disable autopilot for hero.")
  parser.add_argument("--vehicle-bp", type=str, default="vehicle.*model3*")
  args = parser.parse_args()

  client = carla.Client(args.host, args.port)
  client.set_timeout(10.0)
  world = client.get_world()
  bp_lib = world.get_blueprint_library()

  vehicle_bp = bp_lib.filter(args.vehicle_bp)[0]
  if vehicle_bp.has_attribute("role_name"):
    vehicle_bp.set_attribute("role_name", "hero")

  spawn_points = world.get_map().get_spawn_points()
  if not spawn_points:
    raise RuntimeError("No spawn points available.")
  random.shuffle(spawn_points)
  hero = world.try_spawn_actor(vehicle_bp, spawn_points[0])
  if not hero:
    raise RuntimeError("Failed to spawn hero vehicle.")
  print(f"[hero_sensors] spawned hero id={hero.id}")

  tm = client.get_trafficmanager(args.tm_port)
  tm.set_synchronous_mode(args.sync)
  if not args.no_autopilot and hasattr(hero, "set_autopilot"):
    hero.set_autopilot(True, args.tm_port)

  # Camera
  cam_bp = bp_lib.find("sensor.camera.rgb")
  cam_bp.set_attribute("image_size_x", str(args.cam_width))
  cam_bp.set_attribute("image_size_y", str(args.cam_height))
  cam_bp.set_attribute("fov", str(args.cam_fov))
  cam_tf = carla.Transform(carla.Location(x=0.0, y=0.0, z=2.0))
  camera = world.spawn_actor(cam_bp, cam_tf, attach_to=hero)

  # LiDAR
  lidar_bp = bp_lib.find("sensor.lidar.ray_cast")
  lidar_bp.set_attribute("range", str(args.range))
  lidar_bp.set_attribute("points_per_second", str(args.pps))
  lidar_bp.set_attribute("rotation_frequency", str(args.fps))
  lidar_bp.set_attribute("channels", str(args.channels))
  lidar_tf = carla.Transform(carla.Location(x=0.0, y=0.0, z=2.5))
  lidar = world.spawn_actor(lidar_bp, lidar_tf, attach_to=hero)

  original_settings = world.get_settings()
  previous_settings = None
  if args.sync:
    previous_settings = _set_sync(world, True, delta=1.0 / max(args.fps, 1.0))

  try:
    print("[hero_sensors] running... Ctrl+C to stop")
    while True:
      if args.sync:
        world.tick()
        time.sleep(0.001)
      else:
        time.sleep(0.05)
  except KeyboardInterrupt:
    pass
  finally:
    try:
      camera.stop()
      lidar.stop()
      camera.destroy()
      lidar.destroy()
      hero.destroy()
    except Exception:
      pass
    try:
      tm.set_synchronous_mode(False)
      world.apply_settings(previous_settings or original_settings)
    except Exception:
      pass
    print("[hero_sensors] cleanup done")


if __name__ == "__main__":
  main()
