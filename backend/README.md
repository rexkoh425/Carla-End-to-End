# Local Model Studio API (FastAPI)

A lightweight backend to stop `/api/models` 404s in the WebUI and optionally serve the YAML index.

## Run
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 7000
```

Then open the WebUI (e.g., `http://localhost:8000/WebUI/index.html` if you serve statics separately) and the inventory requests to `/api/models` will succeed. If you prefer a different port, update the frontend fallback or use a reverse proxy to `/api`.

## Mask2Former + LaneATT execution (single image)
- POST `/api/pipeline/run` with a Mask2Former pipeline payload (the Flow Builder already sends this).  
- The backend will resolve relative paths from the repo root, run Mask2Former panoptic seg, optionally draw LaneATT lanes if config/weights exist, and write the final overlay.
- Outputs land under `Output/<input>_mask2former_laneatt.png` unless you provide a file path in the pipeline `outputs`.

Required python deps (besides FastAPI): `torch`, `transformers`, `pillow`, `opencv-python`, `numpy`, and `laneatt` with its weights.

## Endpoints
- `GET /healthz` — simple health check.
- `GET /api/models` — static model list mirroring the default cards.
- `GET /api/yamls` — returns entries from `WebUI/yaml_index.json` when present.
