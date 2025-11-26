"""
Optuna-based hyperparameter tuning for EndToEndPartialBEVNet.

Searches over lr, weight_decay, steer/accel loss weights.
Best trial is selected by validation MAE (steer + accel).

Requirements: pip install optuna

Example:
  python scripts/tune_partial_bev_optuna.py --index /data/index.jsonl --trials 10 --epochs 3
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import List, Sequence, Tuple, Union

import numpy as np
import optuna
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


def objective(trial: optuna.Trial, train_ds: Dataset, val_ds: Dataset, device: torch.device, args) -> float:
    lr = trial.suggest_float("lr", 3e-5, 3e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 3e-4, log=True)
    steer_w = trial.suggest_float("steer_w", 0.7, 1.3)
    accel_w = trial.suggest_float("accel_w", 0.7, 1.3)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = EndToEndPartialBEVNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(1, args.epochs + 1):
        train_one_epoch(model, train_loader, optimizer, device, steer_w=steer_w, accel_w=accel_w)
        val_mae_steer, val_mae_accel = evaluate(model, val_loader, device)
        score = val_mae_steer + val_mae_accel
        trial.report(score, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    # Save best model per trial if desired
    if args.save_dir:
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), Path(args.save_dir) / f"trial_{trial.number}_score_{score:.4f}.pt")

    return score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True, help="Path to JSONL index file.")
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--img-h", type=int, default=128)
    ap.add_argument("--img-w", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--study-name", type=str, default="partial_bev_optuna")
    ap.add_argument("--storage", type=str, default=None, help="Optuna storage URI (e.g., sqlite:///optuna.db)")
    ap.add_argument("--save-dir", type=str, default=None, help="Optional dir to save trial checkpoints")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    optuna.logging.set_verbosity(optuna.logging.INFO)
    if args.storage:
        study = optuna.create_study(
            study_name=args.study_name,
            storage=args.storage,
            load_if_exists=True,
            direction="minimize",
            pruner=optuna.pruners.MedianPruner(n_startup_trials=2, n_warmup_steps=1),
        )
    else:
        study = optuna.create_study(
            direction="minimize",
            pruner=optuna.pruners.MedianPruner(n_startup_trials=2, n_warmup_steps=1),
        )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    records = read_jsonl(args.index)
    train_recs, val_recs = split_train_val(records, args.val_frac)
    bev_cfg = BEVConfig()
    train_ds = FileDrivingDataset(train_recs, img_size=(args.img_h, args.img_w), bev_config=bev_cfg)
    val_ds = FileDrivingDataset(val_recs, img_size=(args.img_h, args.img_w), bev_config=bev_cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _obj(trial: optuna.Trial) -> float:
        return objective(trial, train_ds, val_ds, device, args)

    study.optimize(_obj, n_trials=args.trials, timeout=None)

    print("Best trial:")
    print("  number:", study.best_trial.number)
    print("  value:", study.best_trial.value)
    print("  params:", study.best_trial.params)
    if args.storage:
        print(f"Study stored at {args.storage}")
    else:
        print("Use --storage sqlite:///optuna.db to persist across runs.")


if __name__ == "__main__":
    main()
