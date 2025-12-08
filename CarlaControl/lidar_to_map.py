"""
Project a LiDAR point cloud (Nx4 x,y,z,intensity) into a 2D top-down occupancy map.

Run:
    python CarlaControl/lidar_to_map.py --in lidar.npy --out map.png --res 0.2 --x-range -50 50 --y-range -50 50
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import yaml


def load_cfg(path: Path) -> dict:
  if not path or not path.is_file():
    return {}
  data = yaml.safe_load(path.read_text()) or {}
  return data if isinstance(data, dict) else {}


def project_to_map(
  pts: np.ndarray,
  x_range: tuple[float, float],
  y_range: tuple[float, float],
  res: float,
) -> np.ndarray:
  w = int(round((x_range[1] - x_range[0]) / res))
  h = int(round((y_range[1] - y_range[0]) / res))
  img = np.zeros((h, w), np.uint8)

  mask = (
    (pts[:, 0] >= x_range[0])
    & (pts[:, 0] <= x_range[1])
    & (pts[:, 1] >= y_range[0])
    & (pts[:, 1] <= y_range[1])
  )
  pts = pts[mask]
  if pts.shape[0] == 0:
    return img

  u = ((pts[:, 0] - x_range[0]) / res).astype(int)
  v = ((pts[:, 1] - y_range[0]) / res).astype(int)
  # Flip v for image coords (row 0 at top)
  img[h - 1 - v, u] = 255
  return img


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--in", dest="infile", type=str, default="Output/lidar/lidar.npy")
  parser.add_argument("--out", dest="outfile", type=str, default="Output/lidar/lidar_topdown.png")
  parser.add_argument("--cfg", type=str, default="CarlaControl/lidar_to_map_config.yaml")
  parser.add_argument("--res", type=float, default=None, help="Meters per pixel")
  parser.add_argument("--x-range", nargs=2, type=float, default=None, help="Min max forward range (m)")
  parser.add_argument("--y-range", nargs=2, type=float, default=None, help="Min max lateral range (m)")
  args = parser.parse_args()

  cfg = load_cfg(Path(args.cfg))
  res = args.res if args.res is not None else float(cfg.get("res", 0.2))
  x_range = tuple(args.x_range) if args.x_range else tuple(cfg.get("x_range", [-50.0, 50.0]))
  y_range = tuple(args.y_range) if args.y_range else tuple(cfg.get("y_range", [-50.0, 50.0]))

  infile = Path(args.infile).expanduser()
  outfile = Path(args.outfile).expanduser()
  env_root = os.environ.get("STORAGE_ROOT")
  if env_root:
    if not infile.is_absolute():
      infile = Path(env_root) / infile
    if not outfile.is_absolute():
      outfile = Path(env_root) / outfile
  if not infile.is_file():
    raise FileNotFoundError(f"Input file not found: {infile}")

  pts = np.load(infile)
  if pts.ndim != 2 or pts.shape[1] < 3:
    raise ValueError("Expected Nx4 or Nx3 array of points (x,y,z,intensity)")

  img = project_to_map(pts[:, :3], x_range=x_range, y_range=y_range, res=res)
  outfile.parent.mkdir(parents=True, exist_ok=True)
  cv2.imwrite(str(outfile), img)
  print(f"Saved top-down map to {outfile} (res={res} m/px, x_range={x_range}, y_range={y_range}, points={pts.shape[0]})")


if __name__ == "__main__":
  main()
