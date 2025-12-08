#!/usr/bin/env python3
"""
Spawn a single vehicle with attached RGB camera and LiDAR, capture one frame from each, then clean up.

Usage:
  python CarlaControl/spawn_sensor_rig.py --host 127.0.0.1 --port 2000 \
    --lidar-out Output/lidar/lidar.npy --image-out Output/rgb/cam.png
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Tuple

import carla
import numpy as np


def save_lidar_points(point_cloud: carla.LidarMeasurement, outfile: Path) -> None:
  pts = np.frombuffer(point_cloud.raw_data, dtype=np.float32).reshape(-1, 4)
  outfile.parent.mkdir(parents=True, exist_ok=True)
  np.save(outfile, pts)
  print(f"[sensor_rig] saved {pts.shape[0]} lidar points -> {outfile}")


def save_camera_image(image: carla.Image, outfile: Path) -> None:
  # Convert BGRA uint8 to RGB numpy and save via PIL to preserve portability
  array = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)
  rgb = array[:, :, :3][:, :, ::-1]  # BGR -> RGB
  try:
    from PIL import Image
    outfile.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(outfile)
    print(f"[sensor_rig] saved camera image -> {outfile}")
  except Exception:
    pass


def _set_sync(world: carla.World, sync: bool, delta: float = 0.05):
  settings = world.get_settings()
  settings.synchronous_mode = sync
  settings.fixed_delta_seconds = delta if sync else None
  world.apply_settings(settings)
  return settings


def main(args: argparse.Namespace) -> None:
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

  # Camera
  cam_bp = bp.find("sensor.camera.rgb")
  cam_bp.set_attribute("image_size_x", str(args.cam_width))
  cam_bp.set_attribute("image_size_y", str(args.cam_height))
  cam_bp.set_attribute("fov", str(args.cam_fov))
  cam_tf = carla.Transform(
    carla.Location(x=args.cam_x, y=args.cam_y, z=args.cam_z),
    carla.Rotation(pitch=args.cam_pitch, yaw=args.cam_yaw, roll=args.cam_roll),
  )
  cam = world.spawn_actor(cam_bp, cam_tf, attach_to=vehicle)

  # LiDAR
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

  # Resolve output paths (respect STORAGE_ROOT if provided)
  def resolve_out(path_raw: str, default: str) -> Path:
    path = Path(path_raw or default)
    if not path.is_absolute():
      env_root = os.environ.get("STORAGE_ROOT")
      if env_root:
        path = Path(env_root) / path
    return path

  lidar_out = resolve_out(args.lidar_out, "Output/lidar/lidar.npy")
  img_out = resolve_out(args.image_out, "Output/rgb/cam.png")

  captured = {"lidar": False, "cam": False}

  def on_lidar(point_cloud: carla.LidarMeasurement):
    if captured["lidar"]:
      return
    save_lidar_points(point_cloud, lidar_out)
    captured["lidar"] = True

  def on_cam(image: carla.Image):
    if captured["cam"]:
      return
    save_camera_image(image, img_out)
    captured["cam"] = True

  lidar.listen(on_lidar)
  cam.listen(on_cam)

  original_settings = world.get_settings()
  previous_settings = None
  if args.sync:
    previous_settings = _set_sync(world, True, delta=args.fixed_delta)

  try:
    print("[sensor_rig] waiting for camera/lidar frames...")
    start = time.monotonic()
    while time.monotonic() - start < args.timeout:
      if args.sync:
        world.tick()
      time.sleep(0.05)
      if captured["lidar"] and captured["cam"]:
        break
    if not captured["lidar"]:
      print("[sensor_rig] lidar frame not captured before timeout")
    if not captured["cam"]:
      print("[sensor_rig] camera frame not captured before timeout")
  finally:
    lidar.stop()
    cam.stop()
    lidar.destroy()
    cam.destroy()
    vehicle.destroy()
    if args.sync and previous_settings:
      world.apply_settings(previous_settings)
    else:
      world.apply_settings(original_settings)
    print("[sensor_rig] cleanup done")


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--host", type=str, default="127.0.0.1")
  parser.add_argument("--port", type=int, default=2000)
  parser.add_argument("--timeout", type=float, default=10.0)
  parser.add_argument("--sensor", type=str, default="sensor.lidar.ray_cast")
  parser.add_argument("--range", type=float, default=50.0)
  parser.add_argument("--pps", type=int, default=150000)
  parser.add_argument("--rot-freq", type=float, default=10.0)
  parser.add_argument("--channels", type=int, default=32)
  parser.add_argument("--upper-fov", type=float, default=10.0)
  parser.add_argument("--lower-fov", type=float, default=-30.0)
  parser.add_argument("--vehicle-bp", type=str, default="vehicle.*model3*")
  parser.add_argument("--planar", action="store_true", help="Force 2D lidar: channels=1, upper/lower FOV=0.")
  parser.add_argument("--lidar-x", type=float, default=0.0)
  parser.add_argument("--lidar-y", type=float, default=0.0)
  parser.add_argument("--lidar-z", type=float, default=2.5)
  parser.add_argument("--lidar-pitch", type=float, default=0.0)
  parser.add_argument("--lidar-yaw", type=float, default=0.0)
  parser.add_argument("--lidar-roll", type=float, default=0.0)
  parser.add_argument("--autopilot", action="store_true", help="Enable vehicle autopilot.")
  parser.add_argument("--sync", action="store_true", help="Enable synchronous mode for sensor capture.")
  parser.add_argument("--fixed-delta", type=float, default=0.05)
  parser.add_argument("--lidar-out", type=str, default="Output/lidar/lidar.npy")
  parser.add_argument("--image-out", type=str, default="Output/rgb/cam.png")
  parser.add_argument("--cam-width", type=int, default=800)
  parser.add_argument("--cam-height", type=int, default=600)
  parser.add_argument("--cam-fov", type=float, default=90.0)
  parser.add_argument("--cam-x", type=float, default=0.0)
  parser.add_argument("--cam-y", type=float, default=0.0)
  parser.add_argument("--cam-z", type=float, default=2.2)
  parser.add_argument("--cam-pitch", type=float, default=0.0)
  parser.add_argument("--cam-yaw", type=float, default=0.0)
  parser.add_argument("--cam-roll", type=float, default=0.0)
  cli = parser.parse_args()

  main(cli)
