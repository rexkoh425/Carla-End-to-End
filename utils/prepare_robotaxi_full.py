"""
Convert recorded robotaxi sessions into a JSONL index for the full multimodal model.

Assumptions about each recording folder:
  - camera_front.mp4            (RGB video)
  - lidar_roof.mp4              (BEV lidar video; semantic optional)
  - controls.json               (frames list with steer/throttle/pose/imu/speed)

Output:
  - Extracted images to {out_dir}/images/<stamp>_frameXXXX.jpg
  - Extracted lidar BEV frames to {out_dir}/lidar/<stamp>_frameXXXX.png
  - JSONL at {out_dir}/index.jsonl with per-frame entries:
      {
        "recording": "<stamp>",
        "frame": <int>,
        "camera_path": "images/<stamp>_frameXXXX.jpg",
        "lidar_bev_path": "lidar/<stamp>_frameXXXX.png",
        "steer": <float>,
        "throttle": <float>,
        "speed_mps": <float>,
        "imu_accel": {"x":..., "y":..., "z":...} | null,
        "imu_gyro":  {"x":..., "y":..., "z":...} | null
      }

Notes:
  - This consumes the rendered lidar BEV video, not raw point clouds.
  - Frames are aligned by index: frame 0 in videos pairs with controls[0], etc.
    Mismatched lengths are truncated to the shortest.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

import cv2


def load_controls(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text())
    frames = data.get("frames") or []
    return frames


def extract_video_frames(video_path: Path, out_dir: Path, prefix: str) -> List[Path]:
    cap = cv2.VideoCapture(str(video_path))
    idx = 0
    paths: List[Path] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        fname = f"{prefix}_frame{idx:04d}.jpg"
        fpath = out_dir / fname
        cv2.imwrite(str(fpath), frame)
        paths.append(fpath)
        idx += 1
    cap.release()
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recordings-root", required=True, help="Folder containing recording subfolders (with camera_front.mp4, lidar_roof.mp4, controls.json)")
    ap.add_argument("--out-dir", required=True, help="Output directory for extracted frames and index.jsonl")
    ap.add_argument("--max-recordings", type=int, default=None, help="Optional limit of recordings to process")
    ap.add_argument("--every-n", type=int, default=1, help="Sample every Nth frame to reduce size")
    args = ap.parse_args()

    root = Path(args.recordings_root)
    out_root = Path(args.out_dir)
    img_out = out_root / "images"
    lidar_out = out_root / "lidar"
    out_root.mkdir(parents=True, exist_ok=True)

    rec_folders = sorted([p for p in root.iterdir() if p.is_dir()])
    if args.max_recordings:
        rec_folders = rec_folders[: args.max_recordings]

    index_path = out_root / "index.jsonl"
    with index_path.open("w", encoding="utf-8") as jf:
        for rec in rec_folders:
            camera_mp4 = rec / "camera_front.mp4"
            lidar_mp4 = rec / "lidar_roof.mp4"
            controls_json = rec / "controls.json"
            if not (camera_mp4.exists() and lidar_mp4.exists() and controls_json.exists()):
                print(f"[skip] missing files in {rec}")
                continue

            stamp = rec.name
            cam_frames = extract_video_frames(camera_mp4, img_out, stamp)
            lidar_frames = extract_video_frames(lidar_mp4, lidar_out, stamp)
            controls = load_controls(controls_json)

            n = min(len(cam_frames), len(lidar_frames), len(controls))
            if n == 0:
                print(f"[warn] zero aligned frames in {rec}")
                continue

            for i in range(0, n, args.every_n):
                ctrl = controls[i]
                imu = ctrl.get("imu") or {}
                accel = imu.get("accel")
                gyro = imu.get("gyro")
                entry = {
                    "recording": stamp,
                    "frame": int(ctrl.get("frame", i)),
                    "camera_path": str(cam_frames[i].relative_to(out_root)),
                    "lidar_bev_path": str(lidar_frames[i].relative_to(out_root)),
                    "steer": float(ctrl.get("control", {}).get("steer", 0.0)),
                    "throttle": float(ctrl.get("control", {}).get("throttle", 0.0)),
                    "speed_mps": float(ctrl.get("state", {}).get("speed_mps", 0.0)),
                    "imu_accel": accel if accel else None,
                    "imu_gyro": gyro if gyro else None,
                }
                jf.write(json.dumps(entry) + "\n")
            print(f"[ok] processed {rec} -> {n} frames (sampled every {args.every_n})")

    print(f"[done] wrote {index_path}")


if __name__ == "__main__":
    main()
