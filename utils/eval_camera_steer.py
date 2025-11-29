"""
Evaluate steering MAE/MSE on camera-only data JSONL.

Supports:
- Simple camera steer model (CameraSteerNet)
- Full BEV model with frozen LiDAR/IMU (EndToEndPartialBEVNet), feeding dummy BEV/IMU

Usage example:
python utils/eval_camera_steer.py \
  --data-jsonl /Storage/recordings/processed_camera_steer/all_camera_steer.jsonl \
  --ckpt /app/Output/full_camera_only_convnext/full_camera_only.pt \
  --full-model \
  --img-height 256 --img-width 512 --batch-size 32 --num-workers 8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T

# Ensure project root on path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
from models.multimodal.partial_bev.train_camera_steer import CameraSteerNet
from models.multimodal.partial_bev.model import EndToEndPartialBEVNet, BEVConfig


class CameraJsonlDataset(Dataset):
    def __init__(self, jsonl_path: Path, img_size=(256, 512)):
        self.items = self._load(jsonl_path)
        self.transform = T.Compose(
            [
                T.Resize(img_size),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    @staticmethod
    def _load(path: Path) -> List[dict]:
        rows: List[dict] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        rec = self.items[idx]
        img = Image.open(rec["camera"]).convert("RGB")
        img_t = self.transform(img)
        steer = torch.tensor(float(rec.get("steer", 0.0)), dtype=torch.float32)
        accel = torch.tensor(float(rec.get("accel", 0.0)), dtype=torch.float32)
        return img_t, steer, accel


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-jsonl", required=True, help="JSONL with camera/steer (and optional accel).")
    ap.add_argument("--ckpt", required=True, help="Path to model checkpoint (.pt).")
    ap.add_argument("--full-model", action="store_true", help="Use full BEV model with dummy BEV/IMU.")
    ap.add_argument("--img-height", type=int, default=256)
    ap.add_argument("--img-width", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=4)
    return ap.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = CameraJsonlDataset(Path(args.data_jsonl), img_size=(args.img_height, args.img_width))
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=bool(args.num_workers > 0),
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    if args.full_model:
        model = EndToEndPartialBEVNet(
            img_channels=3,
            imu_input_dim=6,
            freeze_bev=True,
            freeze_imu=True,
        ).to(device)
        bev_cfg = BEVConfig()
    else:
        model = CameraSteerNet().to(device)
        bev_cfg = None

    model.load_state_dict(torch.load(Path(args.ckpt), map_location=device))
    model.eval()

    mae = mse = mae_accel = mse_accel = n = n_accel = 0
    with torch.no_grad():
        for img, steer, accel in loader:
            img = img.to(device)
            steer = steer.to(device)
            if args.full_model:
                # dummy BEV/IMU
                h, w = bev_cfg.shape
                bev = torch.zeros((img.size(0), 1, h, w), device=device)
                imu = torch.zeros((img.size(0), 6), device=device)
                pred_steer, pred_accel = model(img, bev, imu)
            else:
                pred_steer = model(img)
                pred_accel = None

            diff = pred_steer - steer
            mae += diff.abs().sum().item()
            mse += (diff ** 2).sum().item()
            n += steer.numel()

            if pred_accel is not None:
                accel = accel.to(device)
                diff_a = pred_accel - accel
                mae_accel += diff_a.abs().sum().item()
                mse_accel += (diff_a ** 2).sum().item()
                n_accel += accel.numel()

    print(f"Steer MAE: {mae / n:.4f}, MSE: {mse / n:.4f}")
    if n_accel > 0:
        print(f"Accel MAE: {mae_accel / n_accel:.4f}, MSE: {mse_accel / n_accel:.4f}")


if __name__ == "__main__":
    main()
