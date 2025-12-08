"""
Simple lane-follow controller for CARLA.

Run:
    python CarlaControl/lane_follow.py --host 127.0.0.1 --port 2000

Requires:
    pip install numpy opencv-python carla  # or use the CARLA-provided egg
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import carla
import yaml


LANE_CLASS = 6  # CARLA semantic segmentation class id for lane markings


@dataclass
class Params:
  host: str = "127.0.0.1"
  port: int = 2000
  cam_w: int = 640
  cam_h: int = 360
  fov: float = 90.0
  sensor: str = "semantic"  # semantic|rgb
  throttle: float = 0.35
  kp: float = 0.8
  kd: float = 0.2
  lane_shift: float = 0.0


def extract_lane_mask(seg_image: carla.Image, cam_h: int, cam_w: int) -> np.ndarray:
  """Return a 0/1 mask for lane markings from a semantic segmentation frame."""
  arr = np.frombuffer(seg_image.raw_data, dtype=np.uint8).reshape((cam_h, cam_w, 4))
  tags = arr[:, :, 2]  # class ids live in one channel
  return (tags == LANE_CLASS).astype(np.uint8)


def lane_center_offset(mask: np.ndarray, offset_norm: float = 0.0) -> float:
  """
  Compute normalized horizontal offset of lane center (-1..1) and apply a bias.
  offset_norm: negative = shift left, positive = shift right (fraction of half-width).
  """
  band = mask[int(mask.shape[0] * 0.6) :, :]  # lower part of image
  ys, xs = np.nonzero(band)
  if xs.size == 0:
    return 0.0  # no lanes found
  center = xs.mean()
  img_center = mask.shape[1] / 2
  offset = (center - img_center) / img_center
  # Apply user bias (offset_norm is relative to half-width)
  offset += offset_norm
  return offset


def main(args: Params) -> None:
  client = carla.Client(args.host, args.port)
  client.set_timeout(5.0)
  world = client.get_world()
  bp = world.get_blueprint_library()

  vehicle_bp = bp.filter("vehicle.*model3*")[0]
  spawn = world.get_map().get_spawn_points()[0]
  vehicle = world.spawn_actor(vehicle_bp, spawn)

  cam_bp = bp.find("sensor.camera.semantic_segmentation" if args.sensor == "semantic" else "sensor.camera.rgb")
  cam_bp.set_attribute("image_size_x", str(args.cam_w))
  cam_bp.set_attribute("image_size_y", str(args.cam_h))
  cam_bp.set_attribute("fov", str(args.fov))
  cam_tf = carla.Transform(carla.Location(x=1.5, z=1.6))
  cam = world.spawn_actor(cam_bp, cam_tf, attach_to=vehicle)

  prev_err = 0.0

  def on_image(img: carla.Image) -> None:
    nonlocal prev_err
    mask = extract_lane_mask(img, args.cam_h, args.cam_w)
    err = lane_center_offset(mask, offset_norm=args.lane_shift)
    derr = err - prev_err
    steer = float(np.clip(args.kp * err + args.kd * derr, -1.0, 1.0))
    prev_err = err

    vehicle.apply_control(carla.VehicleControl(throttle=args.throttle, steer=steer, brake=0.0))

    # Debug overlay
    overlay = cv2.cvtColor(mask * 255, cv2.COLOR_GRAY2BGR)
    cv2.line(
      overlay,
      (int(mask.shape[1] / 2), mask.shape[0]),
      (int(mask.shape[1] / 2), int(mask.shape[0] * 0.6)),
      (255, 0, 0),
      2,
    )
    cv2.imshow("lane mask", overlay)
    cv2.waitKey(1)

  cam.listen(on_image)

  try:
    print(f"Driving... connect to {args.host}:{args.port}. Ctrl+C to stop.")
    while True:
      time.sleep(0.05)
  except KeyboardInterrupt:
    pass
  finally:
    cam.stop()
    cam.destroy()
    vehicle.destroy()
    cv2.destroyAllWindows()


def parse_args() -> Params:
  parser = argparse.ArgumentParser()
  parser.add_argument("--host", type=str, default="127.0.0.1")
  parser.add_argument("--port", type=int, default=2000)
  parser.add_argument("--cam-w", type=int, default=640)
  parser.add_argument("--cam-h", type=int, default=360)
  parser.add_argument("--fov", type=float, default=90.0)
  parser.add_argument("--sensor", type=str, default=None, choices=["semantic", "rgb"], help="Camera sensor type")
  parser.add_argument("--throttle", type=float, default=0.35)
  parser.add_argument("--kp", type=float, default=0.8)
  parser.add_argument("--kd", type=float, default=0.2)
  parser.add_argument(
    "--lane-shift",
    type=float,
    default=0.0,
    help="Normalized horizontal bias: -0.1 shifts left, +0.1 shifts right (relative to half-width).",
  )
  parser.add_argument("--cfg", type=str, default="CarlaControl/lane_follow_config.yaml", help="Optional YAML config")
  ns = parser.parse_args()
  cfg = {}
  cfg_path = Path(ns.cfg).expanduser()
  if cfg_path.is_file():
    try:
      raw = yaml.safe_load(cfg_path.read_text()) or {}
      if isinstance(raw, dict):
        cfg = raw
    except Exception:
      pass

  def pick(key, default):
    return ns.__dict__.get(key) if ns.__dict__.get(key) not in (None, "") else cfg.get(key, default)

  return Params(
    host=pick("host", "127.0.0.1"),
    port=int(pick("port", 2000)),
    cam_w=int(pick("cam_w", 640)),
    cam_h=int(pick("cam_h", 360)),
    fov=float(pick("fov", 90.0)),
    sensor=str(pick("sensor", "semantic")),
    throttle=float(pick("throttle", 0.35)),
    kp=float(pick("kp", 0.8)),
    kd=float(pick("kd", 0.2)),
    lane_shift=float(pick("lane_shift", 0.0)),
  )


if __name__ == "__main__":
  main(parse_args())
