Setup & Prereqs
- Install CARLA 0.9.16 on Windows (e.g., `CarlaWin/CARLA_0.9.16/CarlaUE4.exe`).
- Ensure Docker Desktop + WSL2 are enabled; this repo’s backend container maps:
  - `/recordings` ↔ `C:/NUS/MachineLearning/GeneralML/recordings`
  - `/Storage` ↔ `D:/Datasets` (if present)
- Start CARLA server (example):
  `CarlaWin/CARLA_0.9.16/CarlaUE4.exe -quality-level=Low -carla-rpc-port=2000 -carla-streaming-port=0`

Record a run (WSL/bash)
```
docker compose -f docker-compose.yml -f docker-compose.override.yml exec backend \
  micromamba run -n app python CarlaControl/record_robotaxi.py \
  record --host host.docker.internal --port 2000 \
  --fps 20 --range 50 --duration 60 \
  --out-dir "C:/NUS/MachineLearning/GeneralML/recordings"
```
Outputs: `<out-dir>/<STAMP>/run.rec`, `meta.json`.

Replay/export (choose sensors)
```
docker compose -f docker-compose.yml -f docker-compose.override.yml exec backend \
  micromamba run -n app python CarlaControl/record_robotaxi.py \
  replay --host host.docker.internal --port 2000 \
  --rec-file "C:/NUS/MachineLearning/GeneralML/recordings/<STAMP>/run.rec" \
  --out-dir "C:/NUS/MachineLearning/GeneralML/recordings" \
  [--no-camera] [--no-lidar] [--no-gnss] [--no-imu]
```
Outputs: camera_front.mp4, lidar_roof.mp4 (unless skipped), controls.json.

Spawn traffic (inside container, optional)
```
docker compose -f docker-compose.yml -f docker-compose.override.yml exec backend \
  micromamba run -n app python CarlaControl/spawn_scene.py \
  --host host.docker.internal --port 2000 --vehicles 5 --walkers 5
```

Training entry points
- Camera-only steer regression:
  `models/multimodal/partial_bev/train_camera_steer.py --data-jsonl "/recordings/CameraFront_Steer.jsonl" --out-dir "/app/Output/camera_steer_model_hi" --epochs 5 --batch-size 8 --lr 1e-4 --img-height 256 --img-width 512 --num-workers 12`
- Multimodal (camera+LiDAR+state): `models/multimodal/partial_bev/train_full_camera_only.py` (supports freezing unused branches).
- TinyLlama LoRA: `models/llm/tinyllama/finetune_tinyllama.py --config models/llm/tinyllama/finetune_config.yaml`
- Evaluate TinyLlama: `models/llm/tinyllama/eval_tinyllama.py --test-jsonl ... --adapter-path ...`

Performance tips
- To debug stutter: use `--no-camera` or `--no-lidar`, lower `--fps` to 15–20, or reduce camera resolution in the script.
- Keep CARLA quality low/offscreen; ensure output paths are on fast storage (SSD).

Notes on mounts
- `/recordings` inside the backend container maps to `C:/NUS/MachineLearning/GeneralML/recordings`.
- `/Storage` maps to `D:/Datasets` (adjust in docker-compose overrides if your layout differs).
