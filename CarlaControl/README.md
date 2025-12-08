# CarlaControl

Minimal lane-follow control using CARLA’s semantic segmentation camera. Spawns a vehicle, reads the lane-marking mask, and steers toward the lane center.

## Run
Ensure CARLA server is running on port 2000, then:
```bash
python CarlaControl/lane_follow.py --host 127.0.0.1 --port 2000
```

## Requirements
- `carla` (the Python API/egg that matches your CARLA server)
- `numpy`
- `opencv-python`

Install locally (if you don’t use the CARLA egg):
```bash
pip install numpy opencv-python carla
```
Or set `PYTHONPATH` to the CARLA egg:
```bash
export PYTHONPATH=PythonAPI/carla/dist/carla-*py3.*-linux-x86_64.egg:$PYTHONPATH
```

## Notes
- Uses semantic segmentation camera; `LANE_CLASS=6` marks lane lines. If you want drivable area, change to class 7 or replace `extract_lane_mask` with your own model (e.g., YOLOPv2) on an RGB camera.
- PD steering gains (`kp`, `kd`) and throttle are CLI flags. Use Ctrl+C to stop; actors are cleaned up on exit.
- Lateral shift: `--lane-shift` lets you bias left/right (e.g., `--lane-shift -0.1` to hug left, `+0.1` to hug right). It’s normalized to half-width; keep it small (|0.2| or less).

## LiDAR capture
Save a single LiDAR frame (Nx4 x,y,z,intensity) to `lidar.npy`:
```bash
python CarlaControl/lidar_capture.py --host 127.0.0.1 --port 2000 --outfile lidar.npy
```

## LiDAR to 2D map
Project a LiDAR point cloud to a top-down occupancy image:
```bash
python CarlaControl/lidar_to_map.py --in lidar.npy --out lidar_topdown.png --cfg CarlaControl/lidar_to_map_config.yaml
```
You can override ranges/resolution with CLI flags (`--res`, `--x-range`, `--y-range`).
