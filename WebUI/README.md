# WebUI (local)

A static, local-only UI for wiring model pipelines, exporting configs, and planning training runs. No backend is required; if you want tracking, point it at your own MLflow/ClearML server.

## Run locally
From repo root:
```bash
python -m http.server 8000
# (optional) in another terminal, start the backend for tracking/model discovery
python WebUI/server.py --port 7000
```
Then open `http://localhost:8000/WebUI/index.html` (Flow Builder) or `http://localhost:8000/WebUI/training.html` (Training planner).
The backend exposes:
- `GET /api/models` for auto-populating config/weights paths (Flow Builder + Config Editor call this).
- `POST /api/track` to forward events to MLflow/ClearML (Flow Builder + Training planner call this).

## Flow Builder
- Pick a model from the dropdown (cards remain for quick info), edit JSON, and save to localStorage.
- Scales with many models: dropdown + search bar + total count so you can filter quickly.
- Drag nodes (Input/Model/Post) and click connectors to link them; export pipeline JSON.
- Add Connector node to route/parse inputs to outputs visually.
- Tracking panel lets you set a provider (MLflow/ClearML/None) plus endpoint/experiment/run. Click **Ping** to test reachability; **Save tracking** persists to localStorage.
- “Simulate” logs the run and sends a POST to `<endpoint>/api/track` with `{provider, experiment, run, event, payload}` when tracking is enabled.
 - Validation: basic schema checks ensure required config/weight fields before simulating/exporting.

## Config Editor
- Open `configs.html` to get a form-based editor for every model (BLIP, CycleGAN, DepthAnything, DINO-X, InternVL, LaneATT, Mask2Former, MoE, PersFormer, Qwen3, SDXL, TCNTransformerMoETrader, VideoHarvester, VideoProcessingLab, WebHarvesting, YOLOE, YOLOP, Custom).
- Search/filter, pick a model, edit form fields (config/weights/device/batch/lr/output/notes), see the JSON view, save to localStorage, or export JSON.
 - Auto-populates config/weights from `GET /api/models` when the backend is running, with simple schema validation for required fields.

## Training planner
- Fill model/experiment/dataset/epochs/batch/lr/device; load config via file upload or by entering a path and clicking “Load path” (shows preview).
- Save multiple plans locally and reload them; export a plan to JSON.
- Tracking panel mirrors Flow Builder and posts to `<endpoint>/api/track` on start/export/dry run.

## Wiring to MLflow / ClearML
- Run your tracking server locally (examples):
  - MLflow: `mlflow ui --host 0.0.0.0 --port 5000`
  - ClearML: set `CLEARML_API_HOST`, `CLEARML_WEB_HOST`, `CLEARML_FILES_HOST` and run the server or use the SaaS endpoints.
- Point `Tracking endpoint` to your gateway (e.g., `http://localhost:5000`). The UI expects a simple POST at `/api/track`; add a small bridge that forwards events to MLflow/ClearML APIs (`/api/2.0/mlflow/runs/create` or ClearML task creation).

## Notes
- Everything is static; no credentials are stored or transmitted unless you wire a backend.
- Styling is defined in `styles.css`; scripts in `app.js` (Flow) and `training.js` (training planner).
