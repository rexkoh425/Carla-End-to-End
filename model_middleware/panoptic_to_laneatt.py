"""
Adapter: Mask2Former panoptic output -> LaneATT-friendly input.

Inputs (sample dict):
{
  "panoptic_seg": HxW int array (segment ids),
  "segments_info": [
      {"id": int, "category_id": int, "score": float, "bbox": [...], "isthing": bool},
      ...
  ],
  "category_map": {category_id: "label_name"},
  "frame": HxWx3 array (uint8), optional,
  "meta": {...}
}

Config (cfg dict):
{
  "lane_labels": ["lane-marking", "road_marking"],
  "roi": [[x_rel, y_rel], ...]  # optional polygon in relative coords,
  "output_mode": "mask_channel" | "overlay",
  "resize": [width, height]  # optional
}

Returns:
{
  "frame": processed frame (overlay or stacked channel),
  "lane_mask": HxW uint8 mask (255 where lane-marking),
  "meta": {..., "lane_mask_empty": bool}
}
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
  import numpy as np
except ImportError as err:  # pragma: no cover
  raise ImportError("numpy is required for panoptic_to_laneatt adapter") from err


def _label_ids(category_map: Dict[int, str], lane_labels: Iterable[str]) -> set:
  wanted = set(lane_labels)
  return {cid for cid, name in category_map.items() if name in wanted}


def _poly_to_mask(shape: Tuple[int, int], roi: Optional[Sequence[Sequence[float]]]) -> Optional[np.ndarray]:
  if not roi:
    return None
  try:
    import cv2  # optional, used for fillPoly
  except ImportError:
    return None
  h, w = shape
  pts = np.array([[int(x * w), int(y * h)] for x, y in roi], dtype=np.int32)
  mask = np.zeros((h, w), dtype=np.uint8)
  cv2.fillPoly(mask, [pts], 255)
  return mask


def adapt_panoptic_to_laneatt(sample: Dict, cfg: Optional[Dict] = None) -> Dict:
  cfg = cfg or {}
  lane_labels = cfg.get("lane_labels", ["lane-marking", "road_marking"])
  seg = sample.get("panoptic_seg")
  segments_info = sample.get("segments_info") or []
  category_map = sample.get("category_map") or {}
  frame = sample.get("frame")

  if seg is None:
    raise ValueError("panoptic_seg missing from sample")

  seg_ids = set(np.unique(seg))
  lane_ids = _label_ids(category_map, lane_labels)

  mask = np.zeros_like(seg, dtype=np.uint8)
  for s in segments_info:
    cid = s.get("category_id")
    sid = s.get("id")
    if cid in lane_ids and sid in seg_ids:
      mask[seg == sid] = 255

  roi_mask = _poly_to_mask(mask.shape, cfg.get("roi"))
  if roi_mask is not None:
    mask = np.bitwise_and(mask, roi_mask)

  out_frame = frame
  if frame is not None:
    if cfg.get("output_mode", "mask_channel") == "overlay":
      if frame.shape[:2] != mask.shape:
        mask_resized = np.zeros(frame.shape[:2], dtype=np.uint8)
        mh, mw = mask.shape
        fh, fw = frame.shape[:2]
        hh = min(mh, fh)
        ww = min(mw, fw)
        mask_resized[:hh, :ww] = mask[:hh, :ww]
        mask = mask_resized
      colored = frame.copy()
      colored[mask > 0] = (0, 255, 0)
      out_frame = colored
    else:
      if mask.ndim == 2:
        ch_mask = mask[..., None]
      else:
        ch_mask = mask
      out_frame = np.concatenate([frame, ch_mask], axis=2)

  if cfg.get("resize"):
    try:
      import cv2

      target_w, target_h = cfg["resize"]
      mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
      if out_frame is not None:
        out_frame = cv2.resize(out_frame, (target_w, target_h))
    except Exception:
      pass

  meta = dict(sample.get("meta") or {})
  meta["lane_mask_empty"] = bool(mask.max() == 0)
  return {"frame": out_frame, "lane_mask": mask, "meta": meta}

