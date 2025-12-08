"""
Capture and save a single LiDAR point cloud from CARLA.

Run:
    python CarlaControl/lidar_capture.py --host 127.0.0.1 --port 2000 --outfile lidar.npy

This spawns a vehicle + LiDAR, captures one frame, saves Nx4 numpy array (x,y,z,intensity),
and cleans up actors.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
import os

import carla
import numpy as np
import yaml


def save_lidar_points(point_cloud: carla.LidarMeasurement, outfile: Path) -> None:
  # point_cloud is an iterable of carla.LidarDetection (x, y, z, intensity)
  pts = np.frombuffer(point_cloud.raw_data, dtype=np.float32).reshape(-1, 4)
  outfile.parent.mkdir(parents=True, exist_ok=True)
  np.save(outfile, pts)
  print(f"Saved {pts.shape[0]} points to {outfile}")


def main(args) -> None:
  client = carla.Client(args.host, args.port)
  client.set_timeout(args.timeout)
  world = client.get_world()
  bp = world.get_blueprint_library()

  vehicle_bp = bp.filter(args.vehicle_bp)[0]
  spawn_points = world.get_map().get_spawn_points()
  spawn = spawn_points[0] if spawn_points else carla.Transform()
  vehicle = world.try_spawn_actor(vehicle_bp, spawn)
  if vehicle is None:
    raise RuntimeError("Failed to spawn vehicle.")
  if args.autopilot and hasattr(vehicle, "set_autopilot"):
    vehicle.set_autopilot(True)

  lidar_bp = bp.find(args.sensor)
  if args.planar:
    args.channels = 1
    args.upper_fov = 0.0
    args.lower_fov = 0.0
  lidar_bp.set_attribute("range", str(args.range))
  lidar_bp.set_attribute("points_per_second", str(args.pps))
  lidar_bp.set_attribute("rotation_frequency", str(args.rot_freq))
  lidar_bp.set_attribute("channels", str(args.channels))
  lidar_bp.set_attribute("upper_fov", str(args.upper_fov))
  lidar_bp.set_attribute("lower_fov", str(args.lower_fov))
  lidar_tf = carla.Transform(
    carla.Location(x=args.lidar_x, y=args.lidar_y, z=args.lidar_z),
    carla.Rotation(pitch=args.lidar_pitch, yaw=args.lidar_yaw, roll=args.lidar_roll),
  )
  lidar = world.spawn_actor(lidar_bp, lidar_tf, attach_to=vehicle)

  outfile = Path(args.outfile).expanduser()
  if not outfile.is_absolute():
    env_root = os.environ.get("STORAGE_ROOT")
    if env_root:
      outfile = Path(env_root) / outfile
  captured = {"done": False}

  def on_lidar(point_cloud: carla.LidarMeasurement):
    if captured["done"]:
      return
    save_lidar_points(point_cloud, outfile)
    captured["done"] = True

  lidar.listen(on_lidar)

  try:
    print("Waiting for LiDAR frame...")
    # Allow a couple of frames to arrive
    for _ in range(30):
      if captured["done"]:
        break
      time.sleep(0.1)
    if not captured["done"]:
      print("No LiDAR frame captured within timeout.")
  finally:
    lidar.stop()
    lidar.destroy()
    vehicle.destroy()
    print("Cleanup done.")


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--host", type=str, default="127.0.0.1")
  parser.add_argument("--port", type=int, default=2000)
  parser.add_argument("--outfile", type=str, default=None)
  parser.add_argument("--cfg", type=str, default="CarlaControl/lidar_capture_config.yaml")
  parser.add_argument("--sensor", type=str, default=None, help="sensor.lidar.ray_cast or sensor.lidar.ray_cast_semantic")
  parser.add_argument("--range", type=float, default=50.0)
  parser.add_argument("--pps", type=int, default=150000)
  parser.add_argument("--rot-freq", type=float, default=10.0)
  parser.add_argument("--channels", type=int, default=32)
  parser.add_argument("--upper-fov", type=float, default=10.0)
  parser.add_argument("--lower-fov", type=float, default=-30.0)
  parser.add_argument("--vehicle-bp", type=str, default="vehicle.*model3*")
  parser.add_argument("--timeout", type=float, default=5.0)
  parser.add_argument("--planar", action="store_true", help="Force 2D lidar: channels=1, upper/lower FOV=0.")
  parser.add_argument("--lidar-x", type=float, default=0.0)
  parser.add_argument("--lidar-y", type=float, default=0.0)
  parser.add_argument("--lidar-z", type=float, default=2.5)
  parser.add_argument("--lidar-pitch", type=float, default=0.0)
  parser.add_argument("--lidar-yaw", type=float, default=0.0)
  parser.add_argument("--lidar-roll", type=float, default=0.0)
  parser.add_argument("--autopilot", action="store_true", help="Enable vehicle autopilot while capturing.")
  cli = parser.parse_args()

  cfg = {}
  cfg_path = Path(cli.cfg).expanduser()
  if cfg_path.is_file():
    try:
      raw = yaml.safe_load(cfg_path.read_text()) or {}
      if isinstance(raw, dict):
        cfg = raw
    except Exception:
      pass

  def pick(name, default):
    val = getattr(cli, name, None)
    return val if val not in (None, "", 0) else cfg.get(name, default)

  merged = argparse.Namespace(
    host=pick("host", "127.0.0.1"),
    port=int(pick("port", 2000)),
    outfile=pick("outfile", "Output/lidar/lidar.npy"),
    sensor=pick("sensor", "sensor.lidar.ray_cast"),
    range=float(pick("range", 50.0)),
    pps=int(pick("pps", 150000)),
    rot_freq=float(pick("rot_freq", 10.0)),
    channels=int(pick("channels", 32)),
    upper_fov=float(pick("upper_fov", 10.0)),
    lower_fov=float(pick("lower_fov", -30.0)),
    vehicle_bp=pick("vehicle_bp", "vehicle.*model3*"),
    timeout=float(pick("timeout", 5.0)),
    planar=bool(pick("planar", False)),
    lidar_x=float(pick("lidar_x", 0.0)),
    lidar_y=float(pick("lidar_y", 0.0)),
    lidar_z=float(pick("lidar_z", 2.5)),
    lidar_pitch=float(pick("lidar_pitch", 0.0)),
    lidar_yaw=float(pick("lidar_yaw", 0.0)),
    lidar_roll=float(pick("lidar_roll", 0.0)),
    autopilot=bool(pick("autopilot", False)),
  )

  main(merged)
