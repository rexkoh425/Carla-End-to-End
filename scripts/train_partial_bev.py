"""
Train the EndToEndPartialBEVNet on a file-backed dataset.

Expected index file: JSONL with one record per line, e.g.:
{
  "camera": "/path/frame_0001.png",
  "lidar_ranges": "/path/ranges_0001.npy",   # or inline list/array
  "lidar_angles": "/path/angles_0001.npy",   # or inline list/array
  "imu": [ax, ay, yaw_rate, pitch, roll],
  "speed": 6.5,
  "steer": -0.05,
  "accel": 0.42
}
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import List, Sequence, Tuple, Union

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from models.multimodal.partial_bev.model import (
    BEVConfig,
    EndToEndPartialBEVNet,
    lidar_to_bev,
    train_one_epoch,
    evaluate,
)


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def load_array(field: Union[str, Sequence[float], np.ndarray]) -> np.ndarray:
    """Load from .npy path or convert inline list/array to np.ndarray."""
    if isinstance(field, str):
        return np.load(field)
    if isinstance(field, np.ndarray):
        return field
    return np.asarray(field, dtype=np.float32)


class FileDrivingDataset(Dataset):
    def __init__(
        self,
        records: List[dict],
        img_size: Tuple[int, int] = (128, 256),
        bev_config: BEVConfig = BEVConfig(),
        imu_scale: Sequence[float] = (5.0, 5.0, 2.0, 1.0, 1.0, 30.0),
    ):
        self.records = records
        self.img_size = img_size
        self.bev_config = bev_config
        self.imu_scale = torch.tensor(imu_scale, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]

        # Camera
        img = Image.open(rec["camera"]).convert("RGB")
        img = img.resize((self.img_size[1], self.img_size[0]), resample=Image.BILINEAR)
        img_t = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        img_t = (img_t - IMAGENET_MEAN) / IMAGENET_STD

        # LiDAR -> BEV
        lidar_ranges = load_array(rec["lidar_ranges"])
        lidar_angles = load_array(rec["lidar_angles"])
        bev = lidar_to_bev(lidar_ranges, lidar_angles, self.bev_config)

        # IMU + speed (normalize by scale)
        imu = torch.tensor(list(rec["imu"]) + [rec["speed"]], dtype=torch.float32)
        imu_norm = imu / self.imu_scale

        steer = torch.tensor(rec["steer"], dtype=torch.float32)
        accel = torch.tensor(rec["accel"], dtype=torch.float32)
        return img_t, bev, imu_norm, steer, accel


def read_jsonl(path: Union[str, Path]) -> List[dict]:
    path = Path(path)
    records: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def split_train_val(records: List[dict], val_frac: float = 0.1):
    n = len(records)
    n_val = max(1, int(math.ceil(n * val_frac))) if n > 10 else max(1, int(n * val_frac)) or 1
    return records[n_val:], records[:n_val]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True, help="Path to JSONL index file.")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--img-h", type=int, default=128)
    ap.add_argument("--img-w", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--save", type=str, default="partial_bev_best.pt")
    args = ap.parse_args()

    records = read_jsonl(args.index)
    train_recs, val_recs = split_train_val(records, args.val_frac)

    bev_cfg = BEVConfig()
    train_ds = FileDrivingDataset(train_recs, img_size=(args.img_h, args.img_w), bev_config=bev_cfg)
    val_ds = FileDrivingDataset(val_recs, img_size=(args.img_h, args.img_w), bev_config=bev_cfg)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EndToEndPartialBEVNet().to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        loss, ls, la = train_one_epoch(model, train_loader, optim, device)
        val_mae_steer, val_mae_accel = evaluate(model, val_loader, device)
        val_score = val_mae_steer + val_mae_accel
        print(
            f"[Epoch {epoch}] train_loss={loss:.4f} steer_loss={ls:.4f} accel_loss={la:.4f} "
            f"val_mae_steer={val_mae_steer:.4f} val_mae_accel={val_mae_accel:.4f}"
        )
        if val_score < best_val:
            best_val = val_score
            torch.save(model.state_dict(), args.save)
            print(f"Saved best model -> {args.save}")


if __name__ == "__main__":
    main()
