Project Overview
- Robotaxi data pipeline: record, replay, and export synchronized RGB, LiDAR, GNSS, IMU from CARLA; configurable via `CarlaControl/record_robotaxi.py`.
- Multimodal ML: partial-BEV models (camera + LiDAR + state) under `models/multimodal/partial_bev/` and camera-only baselines; training scripts live in `models/multimodal/partial_bev/`.
- LLM tooling: TinyLlama LoRA fine-tuning for command-to-CLI mapping in `models/llm/tinyllama/`.

Quickstart (record → replay)
- Start CARLA server on Windows: `CarlaWin/CARLA_0.9.16/CarlaUE4.exe -quality-level=Low -carla-rpc-port=2000 -carla-streaming-port=0`
- Record (WSL/bash):
  `docker compose -f docker-compose.yml -f docker-compose.override.yml exec backend micromamba run -n app python CarlaControl/record_robotaxi.py record --host host.docker.internal --port 2000 --fps 20 --range 50 --duration 60 --out-dir "C:/NUS/MachineLearning/GeneralML/recordings"`
- Replay/export (camera+lidar):
  `docker compose -f docker-compose.yml -f docker-compose.override.yml exec backend micromamba run -n app python CarlaControl/record_robotaxi.py replay --host host.docker.internal --port 2000 --rec-file "C:/NUS/MachineLearning/GeneralML/recordings/<STAMP>/run.rec" --out-dir "C:/NUS/MachineLearning/GeneralML/recordings"`
- Optional flags to reduce load: `--no-camera`, `--no-lidar`, `--no-gnss`, `--no-imu`.

Paths and mounts
- Windows recordings folder: `C:/NUS/MachineLearning/GeneralML/recordings` (mapped as `/recordings` inside backend container).
- Datasets on D: `D:/Datasets` are mapped as `/Storage` in container.

Training entry points
- Camera-only steer regression: `models/multimodal/partial_bev/train_camera_steer.py`
  Example (WSL): `docker compose ... exec backend micromamba run -n app python models/multimodal/partial_bev/train_camera_steer.py --data-jsonl "/recordings/CameraFront_Steer.jsonl" --out-dir "/app/Output/camera_steer_model_hi" --epochs 5 --batch-size 8 --lr 1e-4 --img-height 256 --img-width 512 --num-workers 12`
- Full multimodal (camera+LiDAR+state): `models/multimodal/partial_bev/train_full_camera_only.py` (supports freezing unused branches).
- TinyLlama LoRA fine-tune: `models/llm/tinyllama/finetune_tinyllama.py --config models/llm/tinyllama/finetune_config.yaml`
- Evaluate TinyLlama: `models/llm/tinyllama/eval_tinyllama.py --test-jsonl models/llm/tinyllama/splits/test.jsonl --adapter-path "/app/models/llm/tinyllama_finetuned_v3" --model-name TinyLlama/TinyLlama-1.1B-Chat-v1.0 --max-length 256 --batch-size 4`

Notable scripts
- `CarlaControl/record_robotaxi.py` – record/replay/export with per-sensor toggles (`--no-camera`, `--no-lidar`, `--no-gnss`, `--no-imu`).
- `CarlaControl/spawn_scene.py` and `CarlaControl/spawn_traffic.py` – spawn vehicles/walkers with TM sync.
- `models/llm/tinyllama/split_dataset.py` – split JSONL datasets.
- `utils/prepare_recordings_camera_only.py` – convert camera+steer recordings to JSONL for training.

Tips for smoother replay
- Use `--no-lidar` or `--no-camera` to debug performance; reduce `--fps` (e.g., 15–20) and/or camera resolution if needed.
- Keep CARLA server on Low quality/offscreen and ensure mounts point to local SSD (e.g., `/recordings`).
