#!/usr/bin/env python3
"""
Spawn a hero with camera + LiDAR, start server-side recording, then replay the log to export camera/lidar MP4s + controls.
This avoids live stalls: recording is server-side; export happens after capture.
"""

from __future__ import annotations

import argparse
import json
import queue
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import carla
import imageio.v2 as imageio
import numpy as np

from CarlaControl.log_utils import get_logger

logger = get_logger("record_sensors")


def resolve_out_dir(raw: Optional[str]) -> Path:
    base = Path(raw or "Output/recordings")
    if not base.is_absolute():
        env_root = Path("/Storage")
        if env_root.exists():
            base = env_root / base
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out = base / stamp
    out.mkdir(parents=True, exist_ok=True)
    return out


def find_or_spawn_hero(world: carla.World, vehicle_bp_filter: str) -> carla.Actor:
    for actor in world.get_actors():
        try:
            if actor.attributes.get("role_name") in {"hero", "ego", "ego_robotaxi"}:
                return actor
        except Exception:
            continue
    bp = world.get_blueprint_library().filter(vehicle_bp_filter)[0]
    if bp.has_attribute("role_name"):
        bp.set_attribute("role_name", "hero")
    spawns = world.get_map().get_spawn_points()
    hero = world.try_spawn_actor(bp, spawns[0])
    if not hero:
        raise RuntimeError("Failed to spawn hero vehicle.")
    return hero


def log(msg: str) -> None:
    message = f"[record] {msg}"
    print(message, flush=True)
    try:
        logger.info(msg)
    except Exception:
        pass


def replay_and_export(client: carla.Client, log_file: Path, out_dir: Path, fps: float, lidar_range: float, width: int, height: int):
    client.replay_file(str(log_file), 0.0, 0.0, 0)
    world = client.get_world()

    # Wait for hero
    hero = None
    for _ in range(50):
        hero = find_or_spawn_hero(world, "vehicle.*") if False else None
        for actor in world.get_actors():
            try:
                if actor.attributes.get("role_name") in {"hero", "ego", "ego_robotaxi"}:
                    hero = actor
                    break
            except Exception:
                continue
        if hero:
            break
        world.tick()
        time.sleep(0.05)
    if not hero:
        # fallback: spectator
        hero = world.get_spectator()

    bp_lib = world.get_blueprint_library()
    cam_bp = bp_lib.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", str(width))
    cam_bp.set_attribute("image_size_y", str(height))
    cam_bp.set_attribute("fov", "90")
    cam = world.spawn_actor(cam_bp, carla.Transform(carla.Location(x=1.5, z=1.6)), attach_to=hero)

    lidar_bp = bp_lib.find("sensor.lidar.ray_cast")
    lidar_bp.set_attribute("range", str(lidar_range))
    lidar_bp.set_attribute("points_per_second", "120000")
    lidar_bp.set_attribute("rotation_frequency", str(fps))
    lidar_bp.set_attribute("channels", "32")
    lidar = world.spawn_actor(lidar_bp, carla.Transform(carla.Location(x=0.0, z=2.5)), attach_to=hero)

    cam_q: queue.Queue[carla.Image] = queue.Queue()
    lidar_q: queue.Queue[carla.LidarMeasurement] = queue.Queue()
    cam.listen(cam_q.put)
    lidar.listen(lidar_q.put)

    settings = world.get_settings()
    original_settings = settings
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / max(fps, 1.0)
    world.apply_settings(settings)

    cam_frames = []
    lidar_frames = []
    controls = []

    try:
        frame = 0
        while True:
            world.tick()
            try:
                cam_img = cam_q.get(timeout=1.0)
                lidar_meas = lidar_q.get(timeout=1.0)
            except queue.Empty:
                break

            arr = np.frombuffer(cam_img.raw_data, dtype=np.uint8).reshape(cam_img.height, cam_img.width, 4)
            cam_frames.append(arr[:, :, :3][:, :, ::-1])

            pts = np.frombuffer(lidar_meas.raw_data, dtype=np.float32).reshape(-1, 4)
            x = pts[:, 0]; y = pts[:, 1]
            mask = (np.abs(x) <= lidar_range) & (np.abs(y) <= lidar_range)
            x = x[mask]; y = y[mask]
            size = 512; bev = np.zeros((size, size), dtype=np.uint8); scale = size / (2 * lidar_range)
            u = (x * scale + size / 2).astype(int); v = (y * scale + size / 2).astype(int)
            valid = (u >= 0) & (u < size) & (v >= 0) & (v < size)
            bev[v[valid], u[valid]] = 255
            lidar_frames.append(np.stack([bev, bev, bev], axis=2))

            ctrl = hero.get_control() if hasattr(hero, "get_control") else None
            tr = hero.get_transform() if hasattr(hero, "get_transform") else None
            vel = hero.get_velocity() if hasattr(hero, "get_velocity") else None
            ang = hero.get_angular_velocity() if hasattr(hero, "get_angular_velocity") else None
            controls.append(
                {
                    "frame": frame,
                    "time": frame * settings.fixed_delta_seconds,
                    "controls": {
                        "throttle": getattr(ctrl, "throttle", None),
                        "steer": getattr(ctrl, "steer", None),
                        "brake": getattr(ctrl, "brake", None),
                        "hand_brake": getattr(ctrl, "hand_brake", None),
                        "reverse": getattr(ctrl, "reverse", None),
                        "manual_gear_shift": getattr(ctrl, "manual_gear_shift", None),
                        "gear": getattr(ctrl, "gear", None),
                    },
                    "pose": {
                        "location": {"x": tr.location.x, "y": tr.location.y, "z": tr.location.z} if tr else None,
                        "rotation": {"pitch": tr.rotation.pitch, "yaw": tr.rotation.yaw, "roll": tr.rotation.roll} if tr else None,
                        "velocity": {"x": vel.x, "y": vel.y, "z": vel.z} if vel else None,
                        "angular_velocity": {"x": ang.x, "y": ang.y, "z": ang.z} if ang else None,
                    },
                }
            )
            frame += 1
    finally:
        try:
            cam.stop(); lidar.stop()
            cam.destroy(); lidar.destroy()
        except Exception:
            pass
        try:
            world.apply_settings(original_settings)
        except Exception:
            pass

    cam_path = out_dir / "camera.mp4"
    lidar_path = out_dir / "lidar.mp4"
    controls_path = out_dir / "controls.json"

    with imageio.get_writer(cam_path, fps=fps) as w:
        for f in cam_frames:
            w.append_data(f)
    with imageio.get_writer(lidar_path, fps=fps) as w:
        for f in lidar_frames:
            w.append_data(f)

    meta = {
        "frames": len(controls),
        "fps": fps,
        "range_m": lidar_range,
        "camera": str(cam_path),
        "lidar": str(lidar_path),
        "log": str(log_file),
    }
    with controls_path.open("w", encoding="utf-8") as f:
        json.dump({"meta": meta, "frames": controls}, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--duration", type=float, default=15.0)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--range", type=float, default=50.0)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--vehicle-bp", type=str, default="vehicle.*model3*")
    parser.add_argument("--no-autopilot", action="store_true")
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    out_dir = resolve_out_dir(args.out_dir)
    log_file = out_dir / "session.log"

    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    world = client.get_world()

    hero = find_or_spawn_hero(world, args.vehicle_bp)
    tm = client.get_trafficmanager(8000)
    tm.set_synchronous_mode(False)
    if not args.no_autopilot and hasattr(hero, "set_autopilot"):
        hero.set_autopilot(True, 8000)

    log(f"starting server recorder -> {log_file}")
    client.start_recorder(str(log_file), True)
    time.sleep(args.duration)
    client.stop_recorder()
    log("stopped recorder, exporting ...")

    replay_and_export(client, log_file, out_dir, fps=args.fps, lidar_range=args.range, width=args.width, height=args.height)
    log(f"export complete -> {out_dir}")


if __name__ == "__main__":
    main()
