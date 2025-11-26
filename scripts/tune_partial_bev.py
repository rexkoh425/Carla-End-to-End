"""
Lightweight hyperparameter tuning for EndToEndPartialBEVNet.

Runs a small number of trials over learning rate / weight decay / loss weights
and reports the best config based on validation MAE (steer + accel).

Usage (example):
  python scripts/tune_partial_bev.py --index /data/index.jsonl --trials 6 --epochs-per-trial 3
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import List, Sequence, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from models.multimodal.partial_bev.model import (
    BEVConfig,
    EndToEndPartialBEVNet,
    evaluate,
    lidar_to_bev,
    train_one_epoch,
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

        img = Image.open(rec["camera"]).convert("RGB")
        img = img.resize((self.img_size[1], self.img_size[0]), resample=Image.BILINEAR)
        img_t = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        img_t = (img_t - IMAGENET_MEAN) / IMAGENET_STD

        lidar_ranges = load_array(rec["lidar_ranges"])
        lidar_angles = load_array(rec["lidar_angles"])
        bev = lidar_to_bev(lidar_ranges, lidar_angles, self.bev_config)

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


def sample_config() -> dict:
    lr = 10 ** random.uniform(-4.5, -3.0)       # ~3e-5 to 1e-3
    weight_decay = 10 ** random.uniform(-6.0, -3.5)  # ~1e-6 to 3e-4
    steer_w = random.choice([0.7, 1.0, 1.3])
    accel_w = random.choice([0.7, 1.0, 1.3])
    return {"lr": lr, "weight_decay": weight_decay, "steer_w": steer_w, "accel_w": accel_w}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True, help="Path to JSONL index file.")
    ap.add_argument("--trials", type=int, default=6)
    ap.add_argument("--epochs-per-trial", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--img-h", type=int, default=128)
    ap.add_argument("--img-w", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--save-best", type=str, default="partial_bev_tuned_best.pt")
    ap.add_argument("--results-csv", type=str, default="tuning_results.csv")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    records = read_jsonl(args.index)
    train_recs, val_recs = split_train_val(records, args.val_frac)
    bev_cfg = BEVConfig()

    train_ds = FileDrivingDataset(train_recs, img_size=(args.img_h, args.img_w), bev_config=bev_cfg)
    val_ds = FileDrivingDataset(val_recs, img_size=(args.img_h, args.img_w), bev_config=bev_cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []
    best_score = float("inf")
    best_cfg = None

    for trial in range(1, args.trials + 1):
        cfg = sample_config()
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True
        )
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True
        )

        model = EndToEndPartialBEVNet().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

        for epoch in range(1, args.epochs_per_trial + 1):
            loss, ls, la = train_one_epoch(
                model, train_loader, optimizer, device, steer_w=cfg["steer_w"], accel_w=cfg["accel_w"]
            )
            val_mae_steer, val_mae_accel = evaluate(model, val_loader, device)
            print(
                f"[Trial {trial}/{args.trials} | Epoch {epoch}/{args.epochs_per_trial}] "
                f"lr={cfg['lr']:.2e} wd={cfg['weight_decay']:.2e} "
                f"steer_w={cfg['steer_w']:.2f} accel_w={cfg['accel_w']:.2f} "
                f"train_loss={loss:.4f} val_mae_steer={val_mae_steer:.4f} val_mae_accel={val_mae_accel:.4f}"
            )

        score = val_mae_steer + val_mae_accel
        results.append(
            {
                "trial": trial,
                "lr": cfg["lr"],
                "weight_decay": cfg["weight_decay"],
                "steer_w": cfg["steer_w"],
                "accel_w": cfg["accel_w"],
                "val_mae_steer": val_mae_steer,
                "val_mae_accel": val_mae_accel,
                "score": score,
            }
        )
        if score < best_score:
            best_score = score
            best_cfg = cfg
            torch.save(model.state_dict(), args.save_best)
            print(f"New best score={score:.4f} saved -> {args.save_best}")

    # Write results
    if args.results_csv:
        import csv

        with Path(args.results_csv).open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "trial",
                    "lr",
                    "weight_decay",
                    "steer_w",
                    "accel_w",
                    "val_mae_steer",
                    "val_mae_accel",
                    "score",
                ],
            )
            writer.writeheader()
            writer.writerows(results)
        print(f"Wrote results -> {args.results_csv}")

    print("Best config:", best_cfg, "score:", best_score)


if __name__ == "__main__":
    main()
