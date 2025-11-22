from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import fnmatch
import shutil
from functools import lru_cache
from datetime import datetime
import subprocess
import sys
import requests
from typing import Dict, List, Optional, Sequence, Tuple
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
import argparse
from backend.carla_actions import parse_text_to_actions

logger = logging.getLogger("storage.index")
logger.setLevel(logging.INFO)
pipeline_logger = logging.getLogger("pipeline")
pipeline_logger.setLevel(logging.INFO)
# Make sure custom loggers emit to the same handler as uvicorn.
_uvicorn_logger = logging.getLogger("uvicorn.error")
if _uvicorn_logger.handlers:
  logger.handlers = _uvicorn_logger.handlers
  logger.propagate = False
  pipeline_logger.handlers = _uvicorn_logger.handlers
  pipeline_logger.propagate = False
else:
  # Fallback so pipeline logs are not dropped when uvicorn handlers are missing.
  _handler = logging.StreamHandler()
  _handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
  logger.addHandler(_handler)
  logger.propagate = False
  pipeline_logger.addHandler(_handler)
  pipeline_logger.propagate = False

class ModelInfo(BaseModel):
    id: str
    name: str
    config: Optional[str] = None
    weights: Optional[str] = None


class YamlInfo(BaseModel):
    id: str
    name: str
    path: str


class PipelineRun(BaseModel):
    name: Optional[str] = None
    model: Optional[str] = None
    config: Optional[dict] = None
    graph: Optional[dict] = None
    inputs: Optional[list] = None
    outputs: Optional[list] = None
    middleware: Optional[list] = None
    tracking: Optional[dict] = None


class SpawnCheckRequest(BaseModel):
    host: str = "127.0.0.1"
    port: int = 2000
    tm_port: int = 8000
    vehicles: int = 3
    walkers: int = 6
    timeout: float = 8.0
    tick_sleep: float = 0.05
    attempts: int = 8
    sync: bool = True
    seed: Optional[int] = None


class ConnectCheckRequest(BaseModel):
    host: str = "127.0.0.1"
    port: int = 2000
    streaming_port: Optional[int] = None
    timeout: float = 2.0
    attempts: int = 5
    delay: float = 1.0


class ActorCountsRequest(BaseModel):
    host: str = "127.0.0.1"
    port: int = 2000
    timeout: float = 8.0


def load_yaml_index() -> List[YamlInfo]:
    """Load WebUI/yaml_index.json if present; otherwise return an empty list."""
    root = Path(__file__).resolve().parent.parent
    idx_path = root / "WebUI" / "yaml_index.json"
    if not idx_path.exists():
        return []
    try:
        raw = json.loads(idx_path.read_text(encoding="utf-8"))
        items = []
        for i, entry in enumerate(raw):
            # normalize shape
            items.append(
                YamlInfo(
                    id=str(entry.get("id") or f"yaml_{i}"),
                    name=str(
                        entry.get("name")
                        or Path(entry.get("path") or f"yaml_{i}.yaml").name
                    ),
                    path=str(entry.get("path") or ""),
                )
            )
        return items
    except Exception:
        return []


def default_models() -> List[ModelInfo]:
    """Static models list roughly mirroring WebUI config cards."""
    return [
        ModelInfo(id="mask2former", name="Mask2Former", config="../Mask2Former/mask2former_finetune_config.yaml"),
        ModelInfo(id="yolop", name="YOLOP", config="../YOLOP/configs/yolop.yaml"),
        ModelInfo(id="yolopv2", name="YOLOPv2", weights="models/vision/YOLOPv2/data/weights/yolopv2.pt"),
        ModelInfo(id="laneatt", name="LaneATT", config="models/vision/LaneATT/laneatt_model_config.yaml"),
        ModelInfo(id="yoloe", name="YOLOE", config="../YOLOE/yoloe_bbox_pipeline_config.yaml"),
    ]


app = FastAPI(title="Local Model Studio API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"
# Add models/ and category subfolders to sys.path so relocated projects import.
for p in [MODELS_DIR] + [d for d in (MODELS_DIR.glob("*")) if d.is_dir()]:
  sp = str(p)
  if sp not in sys.path:
    sys.path.insert(0, sp)
STORAGE_CONFIG_PATH = ROOT_DIR / "config.json"


def _load_storage_config() -> Dict[str, str]:
  if not STORAGE_CONFIG_PATH.exists():
    return {}
  try:
    data = json.loads(STORAGE_CONFIG_PATH.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}
  except Exception:
    logger.warning("Failed to read storage config at %s; falling back to defaults.", STORAGE_CONFIG_PATH)
    return {}


# Storage index for fast filename lookups (supports custom roots via STORAGE_ROOT or storage.config.json).
_storage_config = _load_storage_config()
_default_storage_root = ROOT_DIR / "Storage"
_raw_storage_root = os.environ.get("STORAGE_ROOT") or _storage_config.get("storage_root") or _default_storage_root
STORAGE_ROOT = Path(_raw_storage_root).expanduser()
if not STORAGE_ROOT.is_absolute():
  STORAGE_ROOT = (ROOT_DIR / STORAGE_ROOT).resolve()
else:
  STORAGE_ROOT = STORAGE_ROOT.resolve()
_config_alias = (_storage_config.get("storage_alias") or "").strip() if isinstance(_storage_config, dict) else ""
STORAGE_ALIAS = (os.environ.get("STORAGE_ALIAS") or _config_alias or "Storage").strip() or "Storage"
_config_index = _storage_config.get("storage_index_path") if isinstance(_storage_config, dict) else None
_default_index_path = _config_index or (STORAGE_ROOT / "storage_index.json")
_raw_index_path = os.environ.get("STORAGE_INDEX_PATH") or _default_index_path
STORAGE_INDEX_PATH = Path(_raw_index_path).expanduser()
if not STORAGE_INDEX_PATH.is_absolute():
  STORAGE_INDEX_PATH = (ROOT_DIR / STORAGE_INDEX_PATH).resolve()
else:
  STORAGE_INDEX_PATH = STORAGE_INDEX_PATH.resolve()
STORAGE_IGNORE_PATH = STORAGE_ROOT / ".storageignore"
_storage_index: Dict[str, List[str]] = {}


@app.get("/healthz")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/models", response_model=List[ModelInfo])
def list_models() -> List[ModelInfo]:
    return default_models()


@app.get("/api/yamls", response_model=List[YamlInfo])
def list_yamls() -> List[YamlInfo]:
    return load_yaml_index()


def _resolve_path(raw: str) -> Path:
  """Resolve paths relative to storage/root so UI inputs can stay short."""
  path = Path(str(raw)).expanduser()
  if path.is_absolute():
    return path
  # Map legacy top-level project names to their new locations under models/.
  remap = {
    "LaneATT": "models/vision/LaneATT",
    "YOLOP": "models/vision/YOLOP",
    "YOLOPv2": "models/vision/YOLOPv2",
    "YOLOE": "models/vision/YOLOE",
    "DepthAnything": "models/vision/DepthAnything",
    "DINO-X": "models/vision/DINO-X",
    "PersFormer": "models/vision/PersFormer",
    "Mask2Former": "models/segmentation/Mask2Former",
    "SegFormer": "models/segmentation/SegFormer",
    "SAM3": "models/segmentation/SAM3",
    "CycleGAN": "models/generative/CycleGAN",
    "SDXL": "models/generative/SDXL",
    "DeepSeek": "models/llm/DeepSeek",
    "Qwen3": "models/llm/Qwen3",
    "InternVL": "models/multimodal/InternVL",
    "BLIP": "models/multimodal/BLIP",
    "MoE": "models/time_series/MoE",
    "TCNTransformerMoETrader": "models/time_series/TCNTransformerMoETrader",
    "VideoHarvester": "models/video/VideoHarvester",
    "VideoProcessingLab": "models/video/VideoProcessingLab",
    "WebHarvesting": "models/other/WebHarvesting",
  }
  parts = path.parts
  if parts and parts[0] in remap:
    mapped_root = Path(remap[parts[0]])
    rel = Path(*parts[1:]) if len(parts) > 1 else Path()
    path = mapped_root / rel
  parts = path.parts
  alias_lower = (STORAGE_ALIAS or "Storage").lower()
  if parts and parts[0].lower() in {"storage", alias_lower}:
    rel = Path(*parts[1:]) if len(parts) > 1 else Path()
    return (STORAGE_ROOT / rel).resolve()
  # Handle legacy paths like ../YOLOPv2/... relative to backend directory.
  if parts and parts[0] == "..":
    candidate = (Path(__file__).resolve().parent / path).resolve()
    if candidate.exists():
      return candidate
  return (ROOT_DIR / path).resolve()


class StorageConflictError(Exception):
  def __init__(self, name: str, matches: List[str]):
    super().__init__(name)
    self.name = name
    self.matches = matches


def _apply_overlay(
    frame_bgr: np.ndarray,
    mask: np.ndarray,
    color_bgr: Tuple[int, int, int],
    alpha: float,
) -> np.ndarray:
    if not mask.any():
        return frame_bgr
    overlay = frame_bgr.copy()
    overlay[mask] = (
        overlay[mask].astype(np.float32) * (1.0 - alpha)
        + np.array(color_bgr, dtype=np.float32) * alpha
    )
    overlay[mask] = np.clip(overlay[mask], 0, 255)
    return overlay.astype(np.uint8)


def _build_lane_mask(
    segmentation: torch.Tensor,
    segments_info: Sequence[Dict],
    label_lookup: Dict[int, str],
    target_labels: Sequence[str],
    min_score: float,
) -> np.ndarray:
    desired = {label.lower() for label in target_labels}
    eligible_ids: List[int] = []
    for segment in segments_info:
        label_id = int(segment.get("label_id", -1))
        label_name = label_lookup.get(label_id, str(label_id))
        if label_name.lower() not in desired:
            continue
        score = float(segment.get("score", 1.0))
        if score < min_score:
            continue
        eligible_ids.append(int(segment["id"]))

    if not eligible_ids:
        return np.zeros(tuple(segmentation.shape), dtype=bool)
    segmentation_np = segmentation.cpu().numpy()
    return np.isin(segmentation_np, eligible_ids)


def _resize_mask(mask: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    target_h, target_w = target_size
    if mask.shape[0] == target_h and mask.shape[1] == target_w:
        return mask
    resized = cv2.resize(mask.astype(np.uint8), (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    return resized.astype(bool)


def _dilate_mask(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    if kernel_size <= 1:
        return mask
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
    return dilated.astype(bool)


def _maybe_resize(frame_bgr: np.ndarray, max_long_edge: Optional[int]) -> Tuple[np.ndarray, float]:
    if not max_long_edge:
        return frame_bgr, 1.0
    height, width = frame_bgr.shape[:2]
    long_edge = max(height, width)
    if long_edge <= max_long_edge:
        return frame_bgr, 1.0
    scale = max_long_edge / float(long_edge)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    resized = cv2.resize(frame_bgr, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized, scale


def _render_lanes(
    proposals: torch.Tensor,
    base_bgr: np.ndarray,
    img_w: int,
    img_h: int,
) -> np.ndarray:
    if proposals is None or proposals.numel() == 0:
        return cv2.resize(base_bgr, (img_w, img_h))

    lanes = proposals.detach().cpu()
    if lanes.ndim == 1:
        lanes = lanes.unsqueeze(0)

    steps = lanes.shape[1] - 5
    if steps <= 0:
        return cv2.resize(base_bgr, (img_w, img_h))

    ys = np.linspace(img_h, 0, steps)
    overlay = cv2.resize(base_bgr, (img_w, img_h))
    # High-contrast colors to stay visible over the Mask2Former overlay.
    colors = [
        (0, 0, 255),      # red
        (255, 0, 0),      # blue
        (0, 255, 255),    # yellow
        (255, 0, 255),    # magenta
        (0, 165, 255),    # orange
    ]

    for lane_idx, lane in enumerate(lanes):
        xs = lane[5:].numpy()
        max_len = xs.shape[0]
        length_raw = float(lane[4].item())
        length = int(max(0, min(abs(length_raw), max_len)))
        if length < 2:
            continue

        color = colors[lane_idx % len(colors)]
        prev_x, prev_y = xs[0], ys[0]
        for idx in range(length):
            x, y = xs[idx], ys[idx]
            cv2.line(overlay, (int(prev_x), int(prev_y)), (int(x), int(y)), color, 3)
            prev_x, prev_y = x, y

    return overlay


@lru_cache(maxsize=2)
def _load_mask2former(model_id: str, device: str) -> Tuple[AutoImageProcessor, Mask2FormerForUniversalSegmentation, torch.device]:
    if device == "auto":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for mask2former (device=auto) but no GPU is available.")
        resolved_device = torch.device("cuda")
    else:
        resolved_device = torch.device(device)
        if resolved_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"Requested CUDA device '{device}' but no GPU is available.")
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(model_id)
    model.to(resolved_device)
    model.eval()
    return processor, model, resolved_device


def execute_mask2former_laneatt(req: PipelineRun, run_id: str) -> Dict:
    """Execute a simple Input -> Mask2Former -> LaneATT -> Output image pipeline."""
    cfg = req.config or {}
    model_cfg = (cfg.get("model") or {})
    runtime_cfg = (cfg.get("runtime") or {})
    data_cfg = (cfg.get("data") or {})
    downstream = ((cfg.get("downstream") or {}).get("laneatt") or {})

    # Flexible input/output resolution so the same pipeline works with new paths.
    input_candidates = []
    if req.inputs:
        input_candidates.extend(req.inputs)
    if data_cfg.get("source"):
        input_candidates.append(data_cfg["source"])
    if req.graph:
        for node in req.graph.get("nodes", []):
            if node.get("type") in ("Input", "Data"):
                src = node.get("meta", {}).get("source")
                if src:
                    input_candidates.append(src)

    input_path_raw = next((p for p in input_candidates if p), None)
    if not input_path_raw:
        raise RuntimeError("No input path provided in pipeline.")
    input_path = _resolve_path(input_path_raw)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input not found: {input_path}")

    output_candidates = []
    if req.outputs:
        output_candidates.extend(req.outputs)
    if runtime_cfg.get("output"):
        output_candidates.append(runtime_cfg["output"])
    if data_cfg.get("output"):
        output_candidates.append(data_cfg["output"])
    if req.graph:
        for node in req.graph.get("nodes", []):
            if node.get("type") == "Post":
                out = node.get("meta", {}).get("output")
                if out:
                    output_candidates.append(out)

    output_path_raw = next((p for p in output_candidates if p), None)
    default_name = f"{input_path.stem}_mask2former_laneatt.png"
    if not output_path_raw:
        output_path = (STORAGE_ROOT / "Output" / default_name)
    else:
        output_candidate = _resolve_path(output_path_raw)
        if output_candidate.suffix:  # looks like a file
            output_path = output_candidate
        else:
            output_path = output_candidate / default_name
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model_id = model_cfg.get("id", "facebook/mask2former-swin-large-cityscapes-panoptic")
    device = model_cfg.get("device", "auto")
    overlay_alpha = float(runtime_cfg.get("overlay_alpha", 0.6))
    min_score = float(runtime_cfg.get("min_score", 0.35))
    dilate_kernel = int(runtime_cfg.get("dilate_kernel", 5))
    max_long_edge = runtime_cfg.get("max_long_edge")
    max_long_edge = int(max_long_edge) if max_long_edge is not None else None
    overlay_color_raw = runtime_cfg.get("overlay_color", (64, 255, 112))
    overlay_color = tuple(int(c) for c in overlay_color_raw) if isinstance(overlay_color_raw, (list, tuple)) else (64, 255, 112)
    highlight_labels_raw = runtime_cfg.get("highlight_labels", ("road", "lane-marking", "ground", "parking"))
    if isinstance(highlight_labels_raw, str):
        highlight_labels = tuple(p.strip() for p in highlight_labels_raw.split(",") if p.strip())
    else:
        highlight_labels = tuple(highlight_labels_raw)

    # Load and preprocess image.
    frame_bgr = cv2.imread(str(input_path))
    if frame_bgr is None:
        raise RuntimeError(f"Failed to read image: {input_path}")
    original_h, original_w = frame_bgr.shape[:2]
    resized_frame, scale = _maybe_resize(frame_bgr, max_long_edge)
    rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

    processor, model, resolved_device = _load_mask2former(model_id, device)
    inputs = processor(images=Image.fromarray(rgb_frame), return_tensors="pt")
    inputs = {k: v.to(resolved_device) for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = model(**inputs)
        processed = processor.post_process_panoptic_segmentation(outputs, target_sizes=[rgb_frame.shape[:2]])[0]

    segmentation = processed["segmentation"]
    segments_info = processed["segments_info"]
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    mask = _build_lane_mask(segmentation, segments_info, id2label, highlight_labels, min_score)
    mask_full_res = _resize_mask(mask, (resized_frame.shape[0], resized_frame.shape[1]))
    if scale != 1.0:
        mask_full_res = _resize_mask(mask_full_res, (frame_bgr.shape[0], frame_bgr.shape[1]))
    mask_full_res = _dilate_mask(mask_full_res, dilate_kernel)
    overlay_frame = _apply_overlay(frame_bgr, mask_full_res, overlay_color, overlay_alpha)

    # Optional LaneATT refinement/drawing.
    laneatt_rendered = None
    if downstream:
        try:
            from laneatt import LaneATT  # import locally to keep optional
        except Exception as exc:  # pragma: no cover - optional dep
            raise RuntimeError(f"LaneATT not available: {exc}") from exc

        lane_cfg_path = _resolve_path(downstream.get("config", "models/vision/LaneATT/laneatt_model_config.yaml"))
        lane_weights_path = _resolve_path(downstream.get("weights", "models/vision/LaneATT/laneatt_100.pt"))
        if not lane_cfg_path.is_file():
            raise FileNotFoundError(f"LaneATT config not found: {lane_cfg_path}")
        if not lane_weights_path.is_file():
            raise FileNotFoundError(f"LaneATT weights not found: {lane_weights_path}")

        if not torch.cuda.is_available():
            raise RuntimeError("LaneATT requires CUDA but no GPU is available.")
        lane_model = LaneATT(str(lane_cfg_path))
        device = torch.device("cuda")
        try:
            state = torch.load(str(lane_weights_path), map_location=device, weights_only=False)
            if isinstance(state, dict) and "model" in state:
                lane_model.load_state_dict(state["model"], strict=False)
            else:
                lane_model.load_state_dict(state, strict=False)
        except Exception as exc:
            raise RuntimeError(f"Failed to load LaneATT weights {lane_weights_path}: {exc}") from exc
        pos_thr = downstream.get("positive_threshold")
        if pos_thr is not None:
            try:
                lane_model._LaneATT__positive_threshold = float(pos_thr)  # type: ignore[attr-defined]
            except Exception:
                pass
        if hasattr(lane_model, "model"):
            lane_model.model.to(device)
        if hasattr(lane_model, "device"):
            lane_model.device = device
        lane_model.eval()
        nms = bool(downstream.get("nms", True))
        nms_threshold = float(downstream.get("nms_threshold", 40))
        with torch.inference_mode():
            proposals = lane_model.cv2_inference(frame_bgr)
            if nms:
                proposals = lane_model.nms(proposals, nms_threshold)
        lane_overlay = _render_lanes(proposals, overlay_frame, lane_model.img_w, lane_model.img_h)
        laneatt_rendered = cv2.resize(lane_overlay, (original_w, original_h))
    final_frame = laneatt_rendered if laneatt_rendered is not None else overlay_frame

    cv2.imwrite(str(output_path), final_frame)
    return {
        "status": "ok",
        "output": str(output_path),
        "steps": {
            "mask2former": {"model_id": model_id, "device": str(resolved_device)},
            "laneatt": {"used": bool(downstream), "config": str(downstream.get("config", "")), "weights": str(downstream.get("weights", ""))},
        },
        "run_id": run_id,
        "input": str(input_path),
    }


@app.post("/api/pipeline/run")
def run_pipeline(req: PipelineRun) -> dict:
    run_id = str(uuid4())
    root = Path(__file__).resolve().parent.parent
    out_dir = root / "pipeline_runs"
    out_dir.mkdir(exist_ok=True)
    payload_path = out_dir / f"{run_id}.json"
    payload_path.write_text(req.model_dump_json(indent=2), encoding="utf-8")

    pipeline_logger.info(
      "run start id=%s model=%s nodes=%d",
      run_id,
      req.model,
      len((req.graph or {}).get("nodes", []) or []),
    )
    try:
        result = execute_graph_pipeline(req, run_id)
        result["path"] = str(payload_path)
        result["id"] = run_id
        if result.get("steps"):
            pipeline_logger.info("run steps id=%s count=%d", run_id, len(result.get("steps", [])))
        pipeline_logger.info("run done id=%s status=%s output=%s", run_id, result.get("status"), result.get("output"))
        return result
    except Exception as exc:
        pipeline_logger.error("run fail id=%s err=%s", run_id, exc)
        return {
            "id": run_id,
            "status": "error",
            "message": f"Pipeline execution failed: {exc}",
            "path": str(payload_path),
        }


@app.get("/api/pipeline/run/{run_id}", response_model=PipelineRun)
def read_pipeline(run_id: str) -> PipelineRun:
    root = Path(__file__).resolve().parent.parent
    run_path = root / "pipeline_runs" / f"{run_id}.json"
    if not run_path.exists():
        raise RuntimeError(f"run {run_id} not found")
    data = json.loads(run_path.read_text(encoding="utf-8"))
    return PipelineRun(**data)


@app.post("/api/cvat/start")
def start_cvat() -> dict:
  """
  Attempt to launch CVAT via docker compose (expects CVAT/docker-compose.yml).
  """
  root = Path(__file__).resolve().parent.parent
  compose_file = root / "CVAT" / "docker-compose.yml"
  if not compose_file.exists():
    return {"status": "error", "message": "CVAT docker-compose.yml not found"}
  try:
    subprocess.run(
      ["docker", "compose", "-f", str(compose_file), "up", "-d"],
      check=True,
      cwd=root,
    )
    return {"status": "ok", "message": "CVAT starting"}
  except subprocess.CalledProcessError as err:
    return {"status": "error", "message": f"Failed to start CVAT: {err}"}


def load_storage_index() -> Dict[str, List[str]]:
  if STORAGE_INDEX_PATH.exists():
    try:
      data = json.loads(STORAGE_INDEX_PATH.read_text(encoding="utf-8"))
      if isinstance(data, dict):
        return {k: v for k, v in data.items() if isinstance(v, list)}
    except Exception:
      logger.warning("Failed to load storage index from %s", STORAGE_INDEX_PATH)
  return {}


def load_storage_ignore() -> List[str]:
  """
  Read .storageignore under the configured storage root (one glob pattern per line, # comments allowed).
  Patterns are matched against paths relative to STORAGE_ROOT, using POSIX slashes.
  """
  if not STORAGE_IGNORE_PATH.exists():
    return []
  patterns: List[str] = []
  try:
    for line in STORAGE_IGNORE_PATH.read_text(encoding="utf-8").splitlines():
      line = line.strip()
      if not line or line.startswith("#"):
        continue
      patterns.append(line.replace("\\", "/").lstrip("/"))
  except Exception:
    return []
  return patterns


def should_ignore_path(rel_to_storage: Path, patterns: Sequence[str]) -> bool:
  if not patterns:
    return False
  rel_posix = rel_to_storage.as_posix()
  top = rel_posix.split("/", 1)[0]
  for pat in patterns:
    norm = pat.strip()
    if not norm:
      continue
    if fnmatch.fnmatch(rel_posix, norm) or fnmatch.fnmatch(top, norm):
      return True
  return False


def _index_entry_to_path(entry: str) -> Optional[Path]:
  raw = Path(str(entry))
  if raw.is_absolute():
    return raw
  parts = raw.parts
  rel = raw
  alias_lower = (STORAGE_ALIAS or "Storage").lower()
  if parts and parts[0].lower() in {alias_lower, "storage"}:
    rel = Path(*parts[1:]) if len(parts) > 1 else Path()
  candidate = (STORAGE_ROOT / rel).resolve()
  if candidate.is_file():
    return candidate
  legacy_candidate = (ROOT_DIR / raw).resolve()
  try:
    legacy_candidate.relative_to(ROOT_DIR)
  except Exception:
    return None
  return legacy_candidate if legacy_candidate.is_file() else None


def save_storage_index(index: Dict[str, List[str]]) -> None:
  try:
    STORAGE_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    STORAGE_INDEX_PATH.write_text(json.dumps(index), encoding="utf-8")
  except Exception as exc:
    logger.warning("Failed to persist storage index to %s: %s", STORAGE_INDEX_PATH, exc)


def build_storage_index() -> Dict[str, List[str]]:
  index: Dict[str, List[str]] = {}
  if not STORAGE_ROOT.exists():
    logger.warning("Storage root not found at %s; index is empty.", STORAGE_ROOT)
    return index
  ignore_patterns = load_storage_ignore()
  logger.info(
    "Building storage index (root=%s, output=%s, ignores=%s)",
    STORAGE_ROOT,
    STORAGE_INDEX_PATH,
    ", ".join(ignore_patterns) if ignore_patterns else "none",
  )
  prefix = Path(STORAGE_ALIAS or "Storage")
  for path in STORAGE_ROOT.rglob("*"):
    if not path.is_file():
      continue
    rel_storage = path.relative_to(STORAGE_ROOT)
    if should_ignore_path(rel_storage, ignore_patterns):
      continue
    name = path.name
    rel_entry = (prefix / rel_storage).as_posix()
    index.setdefault(name, []).append(rel_entry)
  save_storage_index(index)
  _storage_index.clear()
  _storage_index.update(index)
  total_files = sum(len(v) for v in index.values())
  logger.info(
    "Storage index built: %d unique filenames, %d file entries (ignore patterns: %s)",
    len(index),
    total_files,
    ", ".join(ignore_patterns) if ignore_patterns else "none",
  )
  return index

# --- Pipeline execution helpers ---

def _topo_sort_nodes(nodes: List[dict], links: List[dict]) -> List[dict]:
  id_to_node = {n.get("id"): n for n in nodes if n.get("id")}
  indeg: Dict[str, int] = {nid: 0 for nid in id_to_node}
  for l in links:
    f, t = l.get("from"), l.get("to")
    if f in indeg and t in indeg:
      indeg[t] += 1
  queue = [nid for nid, d in indeg.items() if d == 0]
  ordered: List[dict] = []
  while queue:
    nid = queue.pop(0)
    node = id_to_node.get(nid)
    if node:
      ordered.append(node)
    for l in links:
      if l.get("from") == nid and l.get("to") in indeg:
        indeg[l["to"]] -= 1
        if indeg[l["to"]] == 0:
          queue.append(l["to"])
  # Fallback: if cycle, append remaining
  remaining = [id_to_node[nid] for nid, d in indeg.items() if d > 0 and id_to_node.get(nid)]
  return ordered + remaining


def _derive_output_path(input_path: Path, output_root: Path) -> Path:
  stem = input_path.stem
  try:
    if hasattr(input_path, "is_relative_to") and input_path.is_relative_to(output_root):
      rel = input_path.relative_to(output_root)
    else:
      rel = input_path.relative_to(output_root.parent)
    # Avoid nesting Output/Output when chaining model nodes.
    if rel.parts and rel.parts[0].lower() == "output":
      rel = Path(*rel.parts[1:])
    rel_dir = rel.parent
    return output_root / rel_dir / f"{stem}_mask.png"
  except Exception:
    return output_root / f"{stem}_mask.png"


def _derive_debug_path(input_path: Path, output_root: Path, node_id: str) -> Path:
  base = _derive_output_path(input_path, output_root)
  return base.with_name(f"{input_path.stem}_debug_{node_id}.png")


def _copy_file(src: Path, dst: Path) -> Path:
  dst.parent.mkdir(parents=True, exist_ok=True)
  shutil.copy2(src, dst)
  return dst


def _run_mask2former_stage(input_path: Path, output_path: Path, config: dict, graph: Optional[dict], run_id: str) -> Path:
  cfg = config or {}
  pr = PipelineRun(
    model="mask2former",
    config=cfg,
    graph=graph,
    inputs=[str(input_path)],
    outputs=[str(output_path)],
  )
  result = execute_mask2former_laneatt(pr, run_id)
  out = Path(result.get("output", output_path))
  pipeline_logger.info("Mask2Former stage: input=%s output=%s", input_path, out)
  return out


def _run_laneatt_stage(input_path: Path, output_path: Path, meta: dict, raw_input_path: Optional[Path] = None) -> Path:
  try:
    from laneatt import LaneATT
  except Exception as exc:
    raise RuntimeError(f"LaneATT not available: {exc}") from exc

  cfg_raw = meta.get("config", "models/vision/LaneATT/laneatt_model_config.yaml")
  weights_raw = meta.get("weights", "models/vision/LaneATT/laneatt_100.pt")
  # Auto-correct experiment config that includes !!python/tuple (pip laneatt can't parse it).
  if isinstance(cfg_raw, str) and "experiments/laneatt_r34_tusimple" in cfg_raw:
    pipeline_logger.info("LaneATT config %s is incompatible; falling back to models/vision/LaneATT/laneatt_model_config.yaml", cfg_raw)
    cfg_raw = "models/vision/LaneATT/laneatt_model_config.yaml"
  if isinstance(weights_raw, str) and "experiments/" in weights_raw:
    pipeline_logger.info("LaneATT weights %s look like experiment paths; falling back to models/vision/LaneATT/laneatt_100.pt", weights_raw)
    weights_raw = "models/vision/LaneATT/laneatt_100.pt"

  lane_cfg_path = _resolve_path(cfg_raw)
  lane_weights_path = _resolve_path(weights_raw)
  if not lane_cfg_path.is_file():
    raise FileNotFoundError(f"LaneATT config not found: {lane_cfg_path}")
  if not lane_weights_path.is_file():
    raise FileNotFoundError(f"LaneATT weights not found: {lane_weights_path}")

  overlay_bgr = cv2.imread(str(input_path))
  if overlay_bgr is None:
    raise RuntimeError(f"Failed to read image: {input_path}")
  detect_path = raw_input_path if raw_input_path and raw_input_path.is_file() else input_path
  detect_bgr = overlay_bgr if detect_path == input_path else cv2.imread(str(detect_path))
  if detect_bgr is None:
    raise RuntimeError(f"Failed to read image for LaneATT: {detect_path}")

  positive_threshold = float(meta.get("positive_threshold", 0.5))
  try:
    lane_model = LaneATT(str(lane_cfg_path))
  except Exception as exc:
    # Common failure: yaml.safe_load cannot parse configs with !!python/tuple.
    hint = "Try models/vision/LaneATT/laneatt_model_config.yaml (pip-compatible) instead of experiment configs with python tuples."
    raise RuntimeError(f"Failed to load LaneATT config {lane_cfg_path}: {exc}. {hint}") from exc

  # Custom load with CPU map_location because laneatt.load() uses weights_only=True and fails on CUDA checkpoints when CUDA is absent.
  if not torch.cuda.is_available():
    raise RuntimeError("LaneATT requires CUDA but no GPU is available.")
  device = torch.device("cuda")
  try:
    state = torch.load(str(lane_weights_path), map_location=device, weights_only=False)
    if isinstance(state, dict) and "model" in state:
      lane_model.load_state_dict(state["model"], strict=False)
    else:
      lane_model.load_state_dict(state, strict=False)
  except Exception as exc:
    raise RuntimeError(f"Failed to load LaneATT weights {lane_weights_path}: {exc}") from exc
  # Override positive threshold if provided.
  if positive_threshold is not None:
    try:
      lane_model._LaneATT__positive_threshold = float(positive_threshold)  # type: ignore[attr-defined]
    except Exception:
      pass

  if hasattr(lane_model, "model"):
    lane_model.model.to(device)
  if hasattr(lane_model, "device"):
    lane_model.device = device
  lane_model.eval()
  nms = bool(meta.get("nms", True))
  nms_threshold = float(meta.get("nms_threshold", 40))
  with torch.inference_mode():
    proposals = lane_model.cv2_inference(detect_bgr)
    raw_count = 0 if proposals is None else int(proposals.shape[0])
    if nms and proposals is not None and proposals.numel():
      proposals = lane_model.nms(proposals, nms_threshold)
    nms_count = 0 if proposals is None else int(proposals.shape[0])
    first_len = float(proposals[0][4].item()) if proposals is not None and proposals.numel() else 0.0
  pipeline_logger.info(
    "LaneATT proposals: raw=%d after_nms=%d nms=%s thr=%.2f pos_thr=%.3f first_len=%.2f",
    raw_count,
    nms_count,
    nms,
    nms_threshold,
    positive_threshold,
    first_len,
  )
  lane_overlay = _render_lanes(proposals, overlay_bgr, lane_model.img_w, lane_model.img_h)
  overlay = cv2.resize(lane_overlay, (overlay_bgr.shape[1], overlay_bgr.shape[0]))
  output_path.parent.mkdir(parents=True, exist_ok=True)
  cv2.imwrite(str(output_path), overlay)
  pipeline_logger.info(
    "LaneATT stage: input=%s detect_source=%s output=%s device=%s",
    input_path,
    detect_path,
    output_path,
    device,
  )
  return output_path


@lru_cache(maxsize=1)
def _load_yolopv2_model(weights_path: str, device_raw: str):
  from YOLOPv2.utils import utils as yutils

  device = yutils.select_device(device_raw)
  half = device.type != "cpu"
  model = torch.jit.load(weights_path, map_location=device)
  model.to(device)
  if half:
    model.half()
  model.eval()
  return model, device, half


def _render_yolopv2_overlay(
  base_bgr: np.ndarray,
  da_mask: np.ndarray,
  ll_mask: np.ndarray,
  area_alpha: float = 0.45,
  lane_alpha: float = 0.55,
  lane_dilate: int = 3,
) -> np.ndarray:
  """Blend drivable-area (green) and lane-line (red) masks onto the frame."""
  h, w = base_bgr.shape[:2]
  overlay = base_bgr.copy()

  def _maybe_resize(mask: np.ndarray) -> np.ndarray:
    if mask.shape[:2] == (h, w):
      return mask
    return cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

  if da_mask is not None:
    da = _maybe_resize(da_mask)
    overlay[da > 0] = (
      overlay[da > 0].astype(np.float32) * (1.0 - area_alpha)
      + np.array([0, 255, 0], dtype=np.float32) * area_alpha
    )
  if ll_mask is not None:
    ll = _maybe_resize(ll_mask)
    if lane_dilate > 1:
      kernel = np.ones((lane_dilate, lane_dilate), dtype=np.uint8)
      ll = cv2.dilate(ll.astype(np.uint8), kernel, iterations=1)
    overlay[ll > 0] = (
      overlay[ll > 0].astype(np.float32) * (1.0 - lane_alpha)
      + np.array([0, 0, 255], dtype=np.float32) * lane_alpha
    )
  return np.clip(overlay, 0, 255).astype(np.uint8)


def _run_yolopv2_stage(input_path: Path, output_path: Path, meta: dict, raw_input_path: Optional[Path] = None) -> Path:
  """
  Single-image YOLOPv2 inference (torchscript weights).
  """
  try:
    from YOLOPv2.utils import utils as yutils
  except Exception as exc:
    raise RuntimeError(f"YOLOPv2 not available: {exc}") from exc

  weights_raw = meta.get("weights", "YOLOPv2/data/weights/yolopv2.pt")
  device_raw = str(meta.get("device", "cuda:0" if torch.cuda.is_available() else "cpu"))
  conf_threshold = float(meta.get("conf_threshold", 0.3))
  iou_threshold = float(meta.get("iou_threshold", 0.45))
  img_size = int(meta.get("img_size", 640))
  lane_dilate = int(meta.get("lane_dilate", 3))
  area_alpha = float(meta.get("area_alpha", 0.45))
  lane_alpha = float(meta.get("lane_alpha", 0.55))

  default_weights_path = _resolve_path("YOLOPv2/data/weights/yolopv2.pt")
  weights_path = _resolve_path(weights_raw)
  # Prevent accidental use of non-YOLOPv2 checkpoints (e.g., LaneATT).
  if "laneatt" in str(weights_path).lower():
    pipeline_logger.warning("yolopv2 weights pointed to LaneATT; resetting to default YOLOPv2 weights.")
    weights_path = default_weights_path
  pipeline_logger.info(
    "yolopv2 start id=%s dev=%s weights=%s",
    input_path,
    device_raw,
    weights_path,
  )
  if not weights_path.is_file():
    raise FileNotFoundError(f"YOLOPv2 weights not found: {weights_path}")

  detect_path = raw_input_path if raw_input_path and raw_input_path.is_file() else input_path
  frame_bgr = cv2.imread(str(detect_path))
  if frame_bgr is None:
    raise RuntimeError(f"Failed to read image: {detect_path}")

  # YOLOPv2 expects a 1280x720 road-view; keep alignment with the original demo.
  resized_frame = cv2.resize(frame_bgr, (1280, 720), interpolation=cv2.INTER_LINEAR)
  img, _, _ = yutils.letterbox(resized_frame, new_shape=img_size, stride=32)
  img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR -> RGB CHW
  img = np.ascontiguousarray(img)

  # Graceful device fallback if requested CUDA is unavailable/invalid.
  load_device = device_raw
  if "cuda" in device_raw and not torch.cuda.is_available():
    load_device = "cpu"
    pipeline_logger.info("yolopv2 fallback to CPU (CUDA unavailable)")
  if "cuda" in load_device:
    # Ensure requested CUDA index is valid.
    try:
      idx = int(load_device.split(":")[1]) if ":" in load_device else 0
    except Exception:
      idx = 0
    if idx >= torch.cuda.device_count():
      pipeline_logger.warning("yolopv2 requested invalid CUDA device %s; falling back to cuda:0", load_device)
      load_device = "cuda:0" if torch.cuda.device_count() > 0 else "cpu"

  try:
    model, device, half = _load_yolopv2_model(str(weights_path), load_device)
  except Exception as exc:
    # If weights were overridden and failed to load, retry with default YOLOPv2 weights.
    if str(weights_path) != str(default_weights_path):
      pipeline_logger.warning("yolopv2 load failed for %s (%s); retrying default %s", weights_path, exc, default_weights_path)
      weights_path = default_weights_path
      model, device, half = _load_yolopv2_model(str(weights_path), "cpu")
    elif load_device != "cpu":
      pipeline_logger.error("yolopv2 device error on %s: %s", load_device, exc)
      pipeline_logger.info("yolopv2 fallback to CPU after device error")
      model, device, half = _load_yolopv2_model(str(weights_path), "cpu")
    else:
      pipeline_logger.error("yolopv2 load failed: %s", exc)
      raise

  tensor = torch.from_numpy(img).to(device)
  tensor = tensor.half() if half else tensor.float()
  tensor /= 255.0
  if tensor.ndimension() == 3:
    tensor = tensor.unsqueeze(0)

  with torch.inference_mode():
    pred_anchor, seg, ll = model(tensor)
    if isinstance(pred_anchor, (list, tuple)) and len(pred_anchor) == 2:
      pred, anchor_grid = pred_anchor
    else:
      pred, anchor_grid = pred_anchor, None
    pred = yutils.split_for_trace_model(pred, anchor_grid)
    # Run NMS mainly to exercise the detection head; boxes are not drawn for now.
    _ = yutils.non_max_suppression(pred, conf_threshold, iou_threshold, agnostic=False)
    da_mask = yutils.driving_area_mask(seg)
    ll_mask = yutils.lane_line_mask(ll)

  overlay = _render_yolopv2_overlay(
    resized_frame,
    da_mask,
    ll_mask,
    area_alpha=area_alpha,
    lane_alpha=lane_alpha,
    lane_dilate=lane_dilate,
  )
  if overlay.shape[:2] != frame_bgr.shape[:2]:
    overlay = cv2.resize(overlay, (frame_bgr.shape[1], frame_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)

  output_path.parent.mkdir(parents=True, exist_ok=True)
  cv2.imwrite(str(output_path), overlay)
  pipeline_logger.info(
    "YOLOPv2 stage: input=%s output=%s device=%s weights=%s",
    input_path,
    output_path,
    device,
    weights_path,
  )
  return output_path


def execute_graph_pipeline(req: PipelineRun, run_id: str) -> Dict:
  """Minimal executor: walk graph order, run mask2former stages, pass files along."""
  output_root = STORAGE_ROOT / "Output"
  graph = req.graph or {}
  nodes = graph.get("nodes", []) or []
  links = graph.get("links", []) or []
  ordered = _topo_sort_nodes(nodes, links)
  pipeline_logger.info("graph start id=%s nodes=%d links=%d", run_id, len(nodes), len(links))

  id_to_outputs: Dict[str, Path] = {}
  id_to_meta: Dict[str, dict] = {n.get("id"): (n.get("meta") or {}) for n in nodes if n.get("id")}
  id_to_sources: Dict[str, Path] = {}

  steps: List[Dict[str, str]] = []

  # Seed inputs
  for node in ordered:
    nid = node.get("id")
    ntype = node.get("type")
    meta = id_to_meta.get(nid, {})
    if ntype in ("Input", "Data"):
      src_raw = meta.get("source") or ""
      if not src_raw:
        continue
      src_path = _resolve_path(src_raw)
      if not src_path.is_file():
        raise FileNotFoundError(f"Input not found: {src_path}")
      id_to_outputs[nid] = src_path
      id_to_sources[nid] = src_path

  for node in ordered:
    nid = node.get("id")
    ntype = node.get("type")
    if ntype in ("Input", "Data"):
      continue
    incoming = [l for l in links if l.get("to") == nid]
    if not incoming:
      continue
    upstream_path = None
    upstream_source = None
    for l in incoming:
      cand = id_to_outputs.get(l.get("from"))
      if cand:
        upstream_path = cand
        upstream_source = id_to_sources.get(l.get("from"))
        break
    if not upstream_path:
      continue

    if ntype == "Model":
      meta = id_to_meta.get(nid, {})
      model_id = (meta.get("modelId") or req.model or "").lower()
      pipeline_logger.info("stage start node=%s type=%s input=%s", nid, model_id, upstream_path)
      try:
        if model_id.startswith("mask2former"):
          out_path = _derive_output_path(upstream_path, output_root)
          produced = _run_mask2former_stage(upstream_path, out_path, req.config or {}, req.graph, run_id)
          id_to_outputs[nid] = produced
          id_to_sources[nid] = upstream_source or upstream_path
          steps.append({"node": nid, "type": "mask2former", "input": str(upstream_path), "output": str(produced)})
        elif model_id.startswith("yolopv2"):
          out_path = _derive_output_path(upstream_path, output_root)
          produced = _run_yolopv2_stage(upstream_path, out_path, meta, upstream_source)
          id_to_outputs[nid] = produced
          id_to_sources[nid] = upstream_source or upstream_path
          steps.append({"node": nid, "type": "yolopv2", "input": str(upstream_path), "output": str(produced)})
        elif model_id.startswith("laneatt"):
          out_path = _derive_output_path(upstream_path, output_root)
          produced = _run_laneatt_stage(upstream_path, out_path, meta, upstream_source)
          id_to_outputs[nid] = produced
          id_to_sources[nid] = upstream_source or upstream_path
          steps.append({"node": nid, "type": "laneatt", "input": str(upstream_path), "output": str(produced)})
        else:
          out_path = _derive_output_path(upstream_path, output_root)
          id_to_outputs[nid] = _copy_file(upstream_path, out_path)
          id_to_sources[nid] = upstream_source or upstream_path
          steps.append({"node": nid, "type": "copy", "input": str(upstream_path), "output": str(out_path)})
        pipeline_logger.info("stage done node=%s type=%s output=%s", nid, model_id, id_to_outputs.get(nid))
      except Exception as exc:
        pipeline_logger.error("stage fail node=%s type=%s err=%s", nid, model_id, exc)
        raise
    elif ntype == "Debug":
      debug_out = _derive_debug_path(upstream_path, output_root, nid or "debug")
      debug_out = _copy_file(upstream_path, debug_out)
      steps.append({"node": nid, "type": "debug", "input": str(upstream_path), "output": str(debug_out)})
      id_to_outputs[nid] = upstream_path  # pass-through for downstream
      id_to_sources[nid] = upstream_source or upstream_path
    elif ntype == "Post":
      meta = id_to_meta.get(nid, {})
      out_raw = meta.get("output") or ""
      if out_raw:
        out_path = _resolve_path(out_raw)
      else:
        out_path = _derive_output_path(upstream_path, output_root)
      id_to_outputs[nid] = _copy_file(upstream_path, out_path)
      id_to_sources[nid] = upstream_source or upstream_path
      steps.append({"node": nid, "type": "post", "input": str(upstream_path), "output": str(out_path)})

  final_output = None
  # Prefer the last Post node output if any.
  for node in reversed(ordered):
    if node.get("type") == "Post":
      out = id_to_outputs.get(node.get("id"))
      if out:
        final_output = out
        break
  if not final_output and id_to_outputs:
    final_output = list(id_to_outputs.values())[-1]

  pipeline_logger.info(
    "graph done id=%s status=%s steps=%d",
    run_id,
    "ok" if final_output else "queued",
    len(steps),
  )
  return {
    "status": "ok" if final_output else "queued",
    "output": str(final_output) if final_output else "",
    "run_id": run_id,
    "steps": steps,
    "message": "" if final_output else "No output produced",
  }


def ensure_storage_index() -> Dict[str, List[str]]:
  if _storage_index:
    return _storage_index
  idx = load_storage_index()
  if idx:
    _storage_index.update(idx)
  return _storage_index


@app.on_event("startup")
def warm_storage_index() -> None:
  """
  Kick off Storage indexing without blocking startup. The index is refreshed
  immediately in a background thread so /api/file and /api/storage/search can
  serve shallow paths shortly after boot.
  """
  ensure_storage_index()
  # Only rebuild if no index is loaded from disk.
  if _storage_index:
    return
  # Build asynchronously to avoid delaying container readiness if Storage is large.
  try:
    import threading
    threading.Thread(target=build_storage_index, name="storage-indexer", daemon=True).start()
  except Exception:
    # Fall back to a synchronous build if thread spawn fails (rare).
    build_storage_index()


def _resolve_storage_from_index(raw: str) -> Optional[Path]:
  """
  If given a shallow storage path (e.g., Storage/file.jpg or just file.jpg),
  use the cached index to locate the real file inside the configured STORAGE_ROOT.
  """
  path = Path(str(raw))
  if path.is_absolute():
    return None
  parts = path.parts
  is_shallow = len(parts) == 1 or (len(parts) == 2 and parts[0].lower() == "storage")
  if not is_shallow:
    return None
  name = path.name
  if not name:
    return None
  index = ensure_storage_index()
  if not index:
    index = build_storage_index()
  matches = index.get(name, [])
  if not matches:
    return None
  if len(matches) > 1:
    raise StorageConflictError(name, matches)
  for rel in matches:
    candidate = _index_entry_to_path(rel)
    if candidate and candidate.is_file():
      return candidate
  return None


@app.get("/api/file")
def read_file(path: str) -> FileResponse:
  """
  Serve a local file (e.g., input/output image) from the repo root.
  """
  target = _resolve_path(path)
  if not target.exists():
    try:
      fallback = _resolve_storage_from_index(path)
    except StorageConflictError as exc:
      raise HTTPException(
        status_code=409,
        detail={
          "error": "multiple_matches",
          "message": f"Multiple files named '{exc.name}' found under Storage; please provide the full relative path.",
          "candidates": exc.matches[:5],
        },
      ) from exc
    if fallback:
      target = fallback
  if not target.exists():
    raise HTTPException(status_code=404, detail=f"file not found: {target}")
  return FileResponse(str(target))


@app.get("/api/storage/search")
def search_storage(name: str) -> JSONResponse:
  """
  Search under Storage/ for a given filename and return repo-relative matches.
  Uses a cached index so repeated lookups are fast.
  """
  if not name:
    return JSONResponse([])
  index = ensure_storage_index()
  if not index:
    index = build_storage_index()
  matches = index.get(name, [])
  return JSONResponse(matches[:25])


@app.post("/api/storage/reindex")
def rebuild_storage_index() -> JSONResponse:
  index = build_storage_index()
  return JSONResponse({"status": "ok", "files_indexed": sum(len(v) for v in index.values()), "unique_names": len(index)})


@app.post("/api/agent/chat")
def agent_chat(payload: dict) -> dict:
  """
  Deterministic Carla agent: map user prompt to actions without calling an LLM.
  Expects: {"prompt": "...", "host": "...", "port": 2000}
  """
  prompt = payload.get("prompt") or ""
  if not prompt.strip():
    raise HTTPException(status_code=400, detail="Missing prompt")
  try:
    host = payload.get("host") or "host.docker.internal"
    port = int(payload.get("port") or 2000)
    actions = parse_text_to_actions(prompt, default_host=host, default_port=port)
    pipeline_logger.info(
      "agent_chat (deterministic) host=%s port=%s prompt_len=%d actions=%s",
      host,
      port,
      len(prompt),
      actions.get("actions"),
    )
    executed = []
    for act in actions.get("actions") or []:
      try:
        res = _execute_agent_action(act)
        executed.append({"action": act, "result": res, "status": "ok"})
      except Exception as exc:
        pipeline_logger.error("agent_chat action failed: %s", exc)
        executed.append({"action": act, "error": str(exc), "status": "error"})

    # Return structured actions, execution results, and a reply string for the UI.
    reply = json.dumps({"actions": actions.get("actions"), "executed": executed}, indent=2)
    return {"status": "ok", "actions": actions.get("actions"), "executed": executed, "reply": reply}
  except Exception as exc:
    pipeline_logger.error("agent_chat parse failed: %s", exc)
    raise HTTPException(status_code=500, detail=f"Failed to parse prompt: {exc}") from exc


def _execute_agent_action(action: dict) -> dict:
  """
  Execute a parsed agent action. Supports:
  - connect_check -> carla_connect_check
  - actor_counts -> carla_actor_counts
  - spawn_scene, spawn_custom_npc, lane_follow, lidar_capture, lidar_to_map, carla_launch -> run_script
  - stop_sim -> no-op acknowledgement
  """
  atype = (action.get("type") or "").lower()
  if atype == "connect_check":
    body = ConnectCheckRequest(
      host=action.get("host") or "host.docker.internal",
      port=int(action.get("port") or 2000),
      streaming_port=action.get("streaming_port"),
      timeout=float(action.get("timeout") or 2.0),
      attempts=int(action.get("attempts") or 5),
      delay=float(action.get("delay") or 1.0),
    )
    return carla_connect_check(body)

  if atype == "actor_counts":
    body = ActorCountsRequest(
      host=action.get("host") or "host.docker.internal",
      port=int(action.get("port") or 2000),
      timeout=float(action.get("timeout") or 8.0),
    )
    return carla_actor_counts(body)

  if atype in {"spawn_scene", "spawn_custom_npc", "spawn_sensor_rig", "spawn_hero_sensors", "record_robotaxi", "lane_follow", "lidar_capture", "lidar_to_map", "carla_launch", "record_sensors"}:
    payload = {k: v for k, v in action.items() if k != "type"}
    payload["script"] = atype
    return run_script(payload)

  if atype == "stop_sim":
    return {"status": "ok", "message": "stop_sim acknowledged (no-op)"}

  raise ValueError(f"Unsupported action type {atype}")


@app.post("/api/agent/parse_text")
def agent_parse_text(payload: dict) -> dict:
  """
  Rule-based parser to map natural language Carla requests into action JSON.
  This bypasses the LLM and uses regex patterns to produce {"actions":[...]}.
  """
  text = (payload.get("text") or "").strip()
  if not text:
    raise HTTPException(status_code=400, detail="Missing text")
  host = payload.get("host") or "host.docker.internal"
  port = int(payload.get("port") or 2000)
  try:
    result = parse_text_to_actions(text, default_host=host, default_port=port)
    pipeline_logger.info(
      "agent_parse_text ok host=%s port=%s text_len=%d actions=%s",
      host,
      port,
      len(text),
      result.get("actions"),
    )
    return result
  except Exception as exc:
    pipeline_logger.error("agent_parse_text failed: %s", exc)
    raise HTTPException(status_code=500, detail=f"Failed to parse text: {exc}") from exc


def _run_detached(cmd: List[str], workdir: Optional[Path] = None) -> int:
  """
  Launch a background process; mirror stdout/stderr into Storage/logs/<script>-<ts>.log when possible.
  """
  log_path: Optional[Path] = None
  try:
    # Prefer the script path (second arg when cmd starts with python)
    if len(cmd) >= 2 and cmd[0].lower().endswith(("python", "python3", "python.exe")):
      script_name = Path(cmd[1]).stem
    else:
      script_name = Path(cmd[0]).stem if cmd else "proc"
    log_dir = STORAGE_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    log_path = log_dir / f"{script_name}-{stamp}.log"
  except Exception:
    log_path = None

  proc = None
  if log_path:
    try:
      lf = open(log_path, "ab")
      proc = subprocess.Popen(
        cmd,
        cwd=str(workdir) if workdir else None,
        stdout=lf,
        stderr=subprocess.STDOUT,
      )
    except Exception:
      proc = None

  if proc is None:
    proc = subprocess.Popen(
      cmd,
      cwd=str(workdir) if workdir else None,
      stdout=subprocess.DEVNULL,
      stderr=subprocess.DEVNULL,
    )
  return proc.pid


@app.post("/api/carla/spawn_check")
def carla_spawn_check(body: SpawnCheckRequest) -> dict:
  """
  Lightweight API check that spawns a few vehicles and walkers, then cleans up.
  """
  try:
    from CarlaControl.spawn_healthcheck import SpawnCheckConfig, run_spawn_check
  except Exception as exc:
    raise HTTPException(status_code=500, detail=f"spawn healthcheck unavailable: {exc}") from exc

  cfg = SpawnCheckConfig(
    host=body.host,
    port=body.port,
    tm_port=body.tm_port,
    vehicles=body.vehicles,
    walkers=body.walkers,
    timeout=body.timeout,
    tick_sleep=body.tick_sleep,
    attempts=body.attempts,
    sync=body.sync,
    seed=body.seed,
  )
  try:
    return run_spawn_check(cfg)
  except Exception as exc:
    raise HTTPException(status_code=500, detail=f"Spawn check failed: {exc}") from exc


@app.post("/api/carla/connect_check")
def carla_connect_check(body: ConnectCheckRequest) -> dict:
  """
  TCP reachability probe for CARLA RPC/streaming ports (fast readiness check).
  """
  try:
    from CarlaControl.carla_connect_check import ConnectCheckConfig, run_check
  except Exception as exc:
    raise HTTPException(status_code=500, detail=f"connect check unavailable: {exc}") from exc

  cfg = ConnectCheckConfig(
    host=body.host,
    port=body.port,
    streaming_port=body.streaming_port,
    timeout=body.timeout,
    attempts=body.attempts,
    delay=body.delay,
  )
  try:
    return run_check(cfg)
  except Exception as exc:
    raise HTTPException(status_code=500, detail=f"Connect check failed: {exc}") from exc


@app.post("/api/carla/actor_counts")
def carla_actor_counts(body: ActorCountsRequest) -> dict:
  """
  Return counts (and sample ids) of vehicles and walkers currently in the world.
  """
  try:
    import carla  # type: ignore
  except Exception as exc:
    raise HTTPException(status_code=500, detail=f"carla client not available: {exc}") from exc

  client = carla.Client(body.host, body.port)
  client.set_timeout(body.timeout)
  try:
    world = client.get_world()
    actors = world.get_actors()
  except Exception as exc:
    raise HTTPException(status_code=502, detail=f"Failed to fetch actors: {exc}") from exc

  vehicles = [a for a in actors if a.type_id.startswith("vehicle.")]
  walkers = [a for a in actors if a.type_id.startswith("walker.pedestrian.")]
  vehicle_ids = [a.id for a in vehicles]
  walker_ids = [a.id for a in walkers]
  hero_ids = [
    a.id
    for a in vehicles
    if getattr(a, "attributes", {}).get("role_name") == "hero"
  ]
  return {
    "ok": True,
    "host": body.host,
    "port": body.port,
    "vehicles": len(vehicles),
    "walkers": len(walkers),
    "hero_ids": hero_ids[:10],
    "vehicle_ids": vehicle_ids[:25],
    "walker_ids": walker_ids[:25],
  }


@app.post("/api/tools/run_script")
def run_script(payload: dict) -> dict:
  """
  Run selected helper scripts in background (best-effort).
  Supported scripts: spawn_scene, spawn_custom_npc, spawn_healthcheck, spawn_sensor_rig, spawn_hero_sensors, record_robotaxi, carla_connect_check, lane_follow, lidar_capture, lidar_to_map, carla_launch, follow_hero, record_sensors.
  """
  script = (payload.get("script") or "").lower()
  allowed = {"spawn_scene", "spawn_custom_npc", "spawn_healthcheck", "spawn_sensor_rig", "spawn_hero_sensors", "record_robotaxi", "carla_connect_check", "lane_follow", "lidar_capture", "lidar_to_map", "carla_launch", "follow_hero", "record_sensors"}
  if script not in allowed:
    raise HTTPException(status_code=400, detail=f"Unsupported script {script}")

  if script == "carla_launch":
    # Try to start CARLA server container (best-effort).
    name = payload.get("container_name") or "carla-server"
    image = payload.get("image") or "carlasim/carla:0.9.16"
    ports = payload.get("ports") or [
      "127.0.0.1:2000-2002:2000-2002",
      "127.0.0.1:2000-2002:2000-2002/udp",
    ]
    # If already running, no-op.
    try:
      status = subprocess.check_output(
        ["docker", "ps", "-f", f"name={name}", "--format", "{{.Status}}"],
        text=True,
      ).strip()
      if status:
        return {"status": "ok", "message": f"{name} already running", "container": name}
    except Exception:
      pass
    # If exists but not running, remove.
    try:
      subprocess.run(["docker", "rm", "-f", name], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
      pass
    cmd = [
      "docker",
      "run",
      "-d",
      "--gpus",
      "all",
      "--name",
      name,
      "-p",
      *ports,
      "-e",
      "SDL_VIDEODRIVER=offscreen",
      "-e",
      "SDL_HINT_VIDEODRIVER=offscreen",
      "-e",
      "NVIDIA_VISIBLE_DEVICES=all",
      "-e",
      "NVIDIA_DRIVER_CAPABILITIES=all",
      image,
      "bash",
      "CarlaUE4.sh",
      "-opengl",
      "-nosound",
      "-quality-level=Low",
      "-carla-rpc-port=2000",
      "-carla-streaming-port=0",
    ]
    try:
      cid = subprocess.check_output(cmd, text=True).strip()
      pipeline_logger.info("Started CARLA container %s id=%s", name, cid)
      return {"status": "ok", "container": name, "id": cid}
    except subprocess.CalledProcessError as exc:
      pipeline_logger.error("Failed to start CARLA: %s", exc)
      raise HTTPException(status_code=500, detail=f"Failed to launch CARLA: {exc}") from exc

  args: List[str] = [sys.executable]
  script_path = ROOT_DIR / "CarlaControl" / f"{script}.py"
  if not script_path.is_file():
    raise HTTPException(status_code=404, detail=f"Script not found: {script_path}")
  args.append(str(script_path))

  # Map payload fields to CLI flags (lightweight).
  for key in (
    "host",
    "port",
    "vehicles",
    "walkers",
    "town",
    "outfile",
    "infile",
    "res",
    "tm_port",
    "seed",
    "sensor",
    "range",
    "fps",
    "out_dir",
    "pps",
    "rot_freq",
    "channels",
    "upper_fov",
    "lower_fov",
    "vehicle_bp",
    "timeout",
    "lidar_x",
    "lidar_y",
    "lidar_z",
    "lidar_pitch",
    "lidar_yaw",
    "lidar_roll",
    "tick_sleep",
    "attempts",
    "streaming_port",
    "delay",
    "stop",
    "duration",
    "z_offset",
  ):
    if key in payload and payload[key] not in (None, ""):
      flag = f"--{key.replace('_', '-')}"
      args.extend([flag, str(payload[key])])
  if payload.get("no_autopilot"):
    args.append("--no-autopilot")
  if payload.get("unsafe"):
    args.append("--unsafe")
  if payload.get("respawn"):
    args.append("--respawn")
  if payload.get("hybrid_physics"):
    args.append("--hybrid-physics")
  if script == "spawn_healthcheck":
    if payload.get("sync") is False:
      args.append("--no-sync")
  elif payload.get("sync"):
    args.append("--sync")
  if payload.get("no_hero"):
    args.append("--no-hero")
  if payload.get("no_follow"):
    args.append("--no-follow")
  if payload.get("planar"):
    args.append("--planar")

  try:
    pid = _run_detached(args, workdir=ROOT_DIR)
    pipeline_logger.info("Launched script %s pid=%s cmd=%s", script, pid, " ".join(args))
    return {"status": "ok", "pid": pid, "script": script}
  except Exception as exc:
    raise HTTPException(status_code=500, detail=f"Failed to launch {script}: {exc}")


def main_cli() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--reindex", action="store_true", help="Rebuild the Storage index and exit.")
  args, _ = parser.parse_known_args()
  if args.reindex:
    idx = build_storage_index()
    print(f"Reindexed: {len(idx)} unique filenames")
    return
  # If run normally, start the FastAPI app (no-op here; handled by uvicorn)


if __name__ == "__main__":
  main_cli()
