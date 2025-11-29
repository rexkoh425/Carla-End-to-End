"""
Optimized frame + control extraction for CARLA recordings.

What it does:
- Streams frames from camera_front.mp4 (does NOT load whole video into RAM)
- Aligns each saved frame with the corresponding controls.json frame entry
- Saves images as JPEG for speed + smaller storage
- Supports frame skipping (e.g., save every 3rd frame)
- Writes per-recording index.jsonl and aggregated all_camera_steer.jsonl

Usage:
  python utils/prepare_recordings_camera_only_optimized.py \
    --recording-dirs /Storage/recordings/20251202_090054 \
    --output-root /Storage/recordings/processed_camera_steer \
    --frame-skip 3 \
    --jpeg-quality 95
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

import cv2


def load_controls(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    frames = data.get("frames", [])
    return frames if isinstance(frames, list) else []


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def process_recording(
    rec_dir: Path,
    out_root: Path,
    frame_skip: int,
    jpeg_quality: int,
    verbose_every: int = 500,
) -> List[Dict[str, Any]]:
    video_path = rec_dir / "camera_front.mp4"
    controls_path = rec_dir / "controls.json"

    controls = load_controls(controls_path)
    if len(controls) == 0:
        print(f"skip {rec_dir}: no control frames found")
        return []

    stamp = rec_dir.name
    out_dir = out_root / stamp
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"skip {rec_dir}: cannot open {video_path}")
        return []

    entries: List[Dict[str, Any]] = []

    i = 0  # frame index in video
    saved = 0
    max_i = len(controls)  # we only have labels up to this

    # Open per-recording jsonl now so we can stream-write (less RAM too)
    index_path = out_dir / "index.jsonl"
    with index_path.open("w", encoding="utf-8") as fidx:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i >= max_i:
                break

            # save only every Nth frame
            if (i % frame_skip) == 0:
                ctrl = controls[i].get("control", {}) if isinstance(controls[i], dict) else {}
                steer = safe_float(ctrl.get("steer", 0.0))
                accel = safe_float(ctrl.get("throttle", 0.0))

                fname = f"frame_{i:09d}.jpg"
                out_path = img_dir / fname

                # Faster + smaller than PNG
                ok_write = cv2.imwrite(
                    str(out_path),
                    frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
                )
                if not ok_write:
                    print(f"warn {rec_dir}: failed to write {out_path}")
                else:
                    e = {"camera": str(out_path), "steer": steer, "accel": accel}
                    entries.append(e)
                    fidx.write(json.dumps(e, ensure_ascii=False) + "\n")
                    saved += 1

                    if verbose_every > 0 and (saved % verbose_every) == 0:
                        print(f"{stamp}: saved {saved} frames (video_idx={i})")

            i += 1

    cap.release()
    print(f"{stamp}: done. saved {saved} frames (skip={frame_skip}) -> {index_path}")
    return entries


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--recording-dirs",
        nargs="+",
        required=True,
        help="Recording folders containing camera_front.mp4 and controls.json",
    )
    ap.add_argument("--output-root", required=True, help="Root output folder")
    ap.add_argument(
        "--frame-skip",
        type=int,
        default=3,
        help="Save every Nth frame (default: 3). Example: 3 means keep frames 0,3,6,...",
    )
    ap.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="JPEG quality (1-100). Higher = better quality, larger files (default: 95)",
    )
    ap.add_argument(
        "--verbose-every",
        type=int,
        default=500,
        help="Print progress every N saved frames (default: 500). Set 0 to disable.",
    )
    args = ap.parse_args()

    if args.frame_skip < 1:
        raise SystemExit("--frame-skip must be >= 1")
    if not (1 <= args.jpeg_quality <= 100):
        raise SystemExit("--jpeg-quality must be between 1 and 100")

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    all_entries: List[Dict[str, Any]] = []

    for rec in args.recording_dirs:
        rec_dir = Path(rec)
        if not (rec_dir / "camera_front.mp4").exists():
            print(f"skip {rec_dir}: no camera_front.mp4")
            continue
        if not (rec_dir / "controls.json").exists():
            print(f"skip {rec_dir}: no controls.json")
            continue

        print(f"Processing {rec_dir} ...")
        entries = process_recording(
            rec_dir=rec_dir,
            out_root=out_root,
            frame_skip=args.frame_skip,
            jpeg_quality=args.jpeg_quality,
            verbose_every=args.verbose_every,
        )
        all_entries.extend(entries)

    agg_path = out_root / "all_camera_steer.jsonl"
    with agg_path.open("w", encoding="utf-8") as f:
        for e in all_entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    print(f"Wrote {len(all_entries)} samples to {agg_path}")


if __name__ == "__main__":
    main()
