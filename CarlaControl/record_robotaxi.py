#!/usr/bin/env python3
"""
Record + replay workflow for clean ML dataset generation (robotaxi sensors).

Phase 1 (record):
  - Spawn hero, enable autopilot, fixed FPS sync, start_recorder -> run.rec

Phase 2 (replay):
  - replay_file(run.rec), attach sensors to hero, capture synchronized:
      camera_front.mp4
      camera_depth.mp4
      lidar_roof.mp4 (BEV)
      lidar_semantic.mp4 (BEV)
      controls.json (controls + pose + GNSS/IMU + meta)

Compatibility:
- If invoked with no subcommand, defaults to "both" (record then replay/export) to
  match the WebUI start-recording button.
- A --stop flag will kill any running instance using the PID file.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import os
import queue
import signal
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PureWindowsPath
from typing import Optional, Dict, Any, Tuple, List

import numpy as np
import cv2
import carla

# --------------------------
# PID helpers (backward compat with WebUI stop button)
# --------------------------

PID_FILE = Path("/tmp/record_robotaxi.pid")


def _read_pid() -> int | None:
    try:
        if PID_FILE.is_file():
            return int(PID_FILE.read_text().strip())
    except Exception:
        return None
    return None


def _write_pid() -> None:
    try:
        PID_FILE.write_text(str(os.getpid()))
    except Exception:
        pass


def _stop_previous() -> bool:
    pid = _read_pid()
    if not pid:
        return False
    try:
        os.kill(pid, signal.SIGTERM)
        return True
    except Exception:
        return False


def _clear_pid() -> None:
    try:
        PID_FILE.unlink(missing_ok=True)  # type: ignore
    except Exception:
        pass


# --------------------------
# Utilities
# --------------------------

def now_stamp_utc() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def _storage_root_from_config() -> Optional[Path]:
    try:
        cfg_path = Path(__file__).resolve().parent.parent / "config.json"
        if cfg_path.exists():
            cfg_data = json.loads(cfg_path.read_text())
            root = cfg_data.get("storage_root")
            if root and Path(root).is_absolute():
                return Path(root)
    except Exception:
        return None
    return None


def resolve_out_paths(base: Optional[str]) -> Tuple[str, Path]:
    """
    Returns (remote_path_str, local_path) where:
    - remote_path_str: passed to CARLA server (may be Windows path)
    - local_path: where we write meta and look for run.rec via mounts
    """
    cfg_root = _storage_root_from_config()
    storage_mount = Path(os.environ.get("STORAGE_ROOT", "/Storage"))

    # Remote path (string sent to CARLA server)
    if base is None:
        remote_str = str(cfg_root / "Output/recordings") if cfg_root else "Output/recordings"
    else:
        remote_str = str(base)

    remote_pw = PureWindowsPath(remote_str)
    local_path: Path

    if remote_pw.drive:
        # Windows absolute path. Try to map to known mounts.
        parts = list(remote_pw.parts)
        mount_root = Path(os.environ.get("RECORDINGS_MOUNT", "/recordings"))
        tail = parts[1:]
        if "recordings" in parts:
            try:
                idx = parts.index("recordings")
                tail = parts[idx + 1 :]
            except Exception:
                tail = parts[1:]
        local_path = mount_root / Path(*tail)
    else:
        # POSIX path
        lp = Path(remote_str)
        if lp.is_absolute():
            local_path = lp
        else:
            local_path = storage_mount / lp

    local_path.mkdir(parents=True, exist_ok=True)
    return remote_str, local_path


def safe_destroy(actor: Optional[carla.Actor]) -> None:
    if actor is None:
        return
    try:
        actor.stop()
    except Exception:
        pass
    try:
        actor.destroy()
    except Exception:
        pass


def apply_sync(world: carla.World, tm: carla.TrafficManager, fps: float) -> Tuple[carla.WorldSettings, float]:
    original = world.get_settings()
    fixed_dt = 1.0 / float(fps)

    s = world.get_settings()
    s.synchronous_mode = True
    s.fixed_delta_seconds = fixed_dt
    s.max_substeps = 1
    world.apply_settings(s)

    tm.set_synchronous_mode(True)
    return original, fixed_dt


def restore_world(world: carla.World, tm: carla.TrafficManager, original: carla.WorldSettings) -> None:
    try:
        tm.set_synchronous_mode(False)
    except Exception:
        pass
    try:
        world.apply_settings(original)
    except Exception:
        pass

def tick_and_snapshot(world: carla.World) -> Tuple[int, Optional[carla.WorldSnapshot]]:
    """
    CARLA 0.9.16 world.tick() returns a frame id (int). Snapshot must be fetched via get_snapshot().
    This helper returns (frame_id, snapshot|None) to keep code version-agnostic.
    """
    frame_id = world.tick()
    snap = None
    try:
        snap = world.get_snapshot()
    except Exception:
        pass
    return frame_id, snap


def make_queue(sensor: carla.Actor) -> queue.Queue:
    q = queue.Queue()
    sensor.listen(q.put)
    return q


def get_by_frame(q: queue.Queue, frame: int, timeout_s: float = 2.0):
    """Pull from queue until we find matching frame. Discard older frames. Returns None on timeout."""
    t0 = time.time()
    while True:
        remaining = max(0.0, timeout_s - (time.time() - t0))
        if remaining == 0.0:
            return None
        try:
            item = q.get(timeout=remaining)
        except queue.Empty:
            return None
        if hasattr(item, "frame") and item.frame == frame:
            return item
        # discard and continue


def bgr_from_carla_image(image: carla.Image) -> np.ndarray:
    arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)
    return arr[:, :, :3].copy()  # BGRA -> BGR


def depth_vis_bgr(depth_image: carla.Image) -> np.ndarray:
    depth_image.convert(carla.ColorConverter.LogarithmicDepth)
    return bgr_from_carla_image(depth_image)


def lidar_bev(points_xy: np.ndarray, labels: Optional[np.ndarray], size: int, max_range_m: float) -> np.ndarray:
    """
    points_xy: Nx2 (x, y) in meters. Render simple BEV occupancy (optionally colored by labels).
    """
    img = np.zeros((size, size, 3), dtype=np.uint8)
    scale = (size - 1) / (2.0 * max_range_m)
    u = (points_xy[:, 0] * scale + size / 2.0).astype(np.int32)
    v = (points_xy[:, 1] * scale + size / 2.0).astype(np.int32)

    valid = (u >= 0) & (u < size) & (v >= 0) & (v < size)
    u = u[valid]
    v = v[valid]

    if u.size == 0:
        return img

    if labels is None:
        img[v, u] = (255, 255, 255)
    else:
        lab = labels[valid].astype(np.int32)
        colors = np.zeros((len(lab), 3), dtype=np.uint8)
        colors[lab == 10] = (255, 255, 0)   # vehicles -> cyan-ish (BGR)
        colors[lab == 4] = (255, 0, 255)    # pedestrians -> magenta
        colors[(lab != 10) & (lab != 4)] = (180, 180, 180)
        img[v, u] = colors

    img = np.flipud(img)  # optional flip so forward is "up"
    return img


def find_hero(world: carla.World, role_name: str = "hero", target_id: Optional[int] = None, timeout_frames: int = 200) -> Optional[carla.Vehicle]:
    """During replay, actors may appear after a few ticks. Try by id, then by role_name."""
    for _ in range(timeout_frames):
        actors = world.get_actors().filter("vehicle.*")
        for a in actors:
            try:
                if target_id and a.id == target_id:
                    return a
                if a.attributes.get("role_name") == role_name:
                    return a
            except Exception:
                continue
        world.tick()
    return None


# --------------------------
# Sensor rig config
# --------------------------

@dataclass
class RigConfig:
    fps: float
    lidar_range: float
    image_w: int = 1280
    image_h: int = 720
    fov: float = 90.0
    bev_size: int = 512

    def sensor_tick(self) -> str:
        return str(1.0 / float(self.fps))


# --------------------------
# Phase 1: Record .rec
# --------------------------

def do_record(args) -> Path:
    remote_base_str, local_base = resolve_out_paths(args.out_dir)
    stamp = now_stamp_utc()
    rec_path_remote = Path(remote_base_str) / stamp / "run.rec"
    out_dir_local = local_base / stamp
    out_dir_local.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir_local / "meta.json"

    client = carla.Client(args.host, args.port)
    client.set_timeout(20.0)

    world = client.get_world()
    bp_lib = world.get_blueprint_library()
    tm = client.get_trafficmanager(args.tm_port)
    try:
        if args.tm_speed_diff is not None:
            tm.global_percentage_speed_difference(float(args.tm_speed_diff))
    except Exception:
        pass

    original_settings, fixed_dt = apply_sync(world, tm, args.fps)

    hero = None
    try:
        candidates = bp_lib.filter(args.vehicle_bp)
        if not candidates:
            raise RuntimeError(f"No vehicle blueprint matches: {args.vehicle_bp}")
        veh_bp = candidates[0]
        if veh_bp.has_attribute("role_name"):
            veh_bp.set_attribute("role_name", "hero")

        spawn_points = world.get_map().get_spawn_points()
        if not spawn_points:
            raise RuntimeError("No spawn points available in map.")
        hero = None
        for sp in random.sample(spawn_points, len(spawn_points)):
            hero = world.try_spawn_actor(veh_bp, sp)
            if hero:
                break
        if hero is None:
            raise RuntimeError("Failed to spawn hero after trying all spawn points.")

        hero.set_autopilot(True, tm.get_port())
        try:
            if args.target_speed_kph is not None:
                tm.set_desired_speed(hero, float(args.target_speed_kph))
            if args.vehicle_speed_diff is not None:
                tm.vehicle_percentage_speed_difference(hero, float(args.vehicle_speed_diff))
        except Exception:
            pass

        duration_s = float(args.duration)
        total_frames = int(round(duration_s * float(args.fps)))

        client.start_recorder(str(rec_path_remote))
        print(f"[record] start_recorder -> {rec_path_remote}")

        tick_and_snapshot(world)  # settle one frame
        for i in range(total_frames):
            frame_id, snapshot = tick_and_snapshot(world)
            if i % int(max(1, args.fps)) == 0:
                sim_t = snapshot.timestamp.elapsed_seconds if snapshot else i / float(args.fps)
                print(f"[record] frame {i}/{total_frames} sim_t={sim_t:.2f}s (frame={frame_id})")

        client.stop_recorder()
        print("[record] stop_recorder")

        meta = {
            "host": args.host,
            "port": args.port,
            "tm_port": args.tm_port,
            "map": world.get_map().name,
            "vehicle_bp": args.vehicle_bp,
            "fps": float(args.fps),
            "fixed_dt": float(fixed_dt),
            "duration": float(duration_s),
            "total_frames": int(total_frames),
            "lidar_range": float(args.range),
            "recorded_hero_id": int(hero.id),
            "rec_file_remote": str(rec_path_remote),
            "rec_file_local": str(out_dir_local / "run.rec"),
            "target_speed_kph": args.target_speed_kph,
            "tm_speed_diff": args.tm_speed_diff,
            "vehicle_speed_diff": args.vehicle_speed_diff,
            "timestamp_utc": now_stamp_utc(),
        }
        meta_path.write_text(json.dumps(meta, indent=2))
        print(f"[record] wrote meta -> {meta_path}")
        print(f"[record] done -> {out_dir_local}")
        return out_dir_local
    finally:
        safe_destroy(hero)
        restore_world(world, tm, original_settings)


# --------------------------
# Phase 2: Replay + attach sensors + capture
# --------------------------

def spawn_sensor_rig(
    world: carla.World,
    bp_lib: carla.BlueprintLibrary,
    hero: carla.Vehicle,
    cfg: RigConfig,
    include_camera: bool = True,
    include_lidar: bool = True,
    include_gnss: bool = True,
    include_imu: bool = True,
) -> Dict[str, carla.Actor]:
    rig: Dict[str, carla.Actor] = {}

    if include_camera:
        cam_bp = bp_lib.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(cfg.image_w))
        cam_bp.set_attribute("image_size_y", str(cfg.image_h))
        cam_bp.set_attribute("fov", str(cfg.fov))
        cam_bp.set_attribute("sensor_tick", cfg.sensor_tick())
        cam_tf = carla.Transform(carla.Location(x=1.5, z=1.6))
        rig["rgb_front"] = world.spawn_actor(cam_bp, cam_tf, attach_to=hero)

    # Depth camera (disabled for now)
    # depth_bp = bp_lib.find("sensor.camera.depth")
    # depth_bp.set_attribute("image_size_x", str(cfg.image_w))
    # depth_bp.set_attribute("image_size_y", str(cfg.image_h))
    # depth_bp.set_attribute("fov", str(cfg.fov))
    # depth_bp.set_attribute("sensor_tick", cfg.sensor_tick())
    # rig["depth_front"] = world.spawn_actor(depth_bp, cam_tf, attach_to=hero)

    if include_lidar:
        lidar_bp = bp_lib.find("sensor.lidar.ray_cast")
        lidar_bp.set_attribute("channels", "64")
        lidar_bp.set_attribute("points_per_second", "200000")
        lidar_bp.set_attribute("rotation_frequency", str(cfg.fps))
        lidar_bp.set_attribute("range", str(cfg.lidar_range))
        lidar_bp.set_attribute("upper_fov", "10")
        lidar_bp.set_attribute("lower_fov", "-30")
        lidar_bp.set_attribute("sensor_tick", cfg.sensor_tick())
        lidar_tf = carla.Transform(carla.Location(x=0.0, z=2.5))
        rig["lidar_roof"] = world.spawn_actor(lidar_bp, lidar_tf, attach_to=hero)

        # Semantic LiDAR disabled for now (keep conditional for future enablement)
        # sem_bp = bp_lib.find("sensor.lidar.ray_cast_semantic")
        # sem_bp.set_attribute("channels", "32")
        # sem_bp.set_attribute("points_per_second", "120000")
        # sem_bp.set_attribute("rotation_frequency", str(cfg.fps))
        # sem_range = max(cfg.lidar_range - 20.0, 20.0)
        # sem_bp.set_attribute("range", str(sem_range))
        # sem_bp.set_attribute("upper_fov", "10")
        # sem_bp.set_attribute("lower_fov", "-30")
        # sem_bp.set_attribute("sensor_tick", cfg.sensor_tick())
        # rig["lidar_semantic"] = world.spawn_actor(sem_bp, lidar_tf, attach_to=hero)

    # Radar (disabled for now)
    # try:
    #     radar_bp = bp_lib.find("sensor.other.radar")
    #     radar_bp.set_attribute("horizontal_fov", "90")
    #     radar_bp.set_attribute("vertical_fov", "30")
    #     radar_bp.set_attribute("range", "100")
    #     radar_bp.set_attribute("sensor_tick", cfg.sensor_tick())
    #     radar_tf = carla.Transform(carla.Location(x=2.0, z=1.0))
    #     rig["radar"] = world.spawn_actor(radar_bp, radar_tf, attach_to=hero)
    # except Exception:
    #     pass

    if include_gnss:
        gnss_bp = bp_lib.find("sensor.other.gnss")
        gnss_bp.set_attribute("sensor_tick", cfg.sensor_tick())
        rig["gnss"] = world.spawn_actor(gnss_bp, carla.Transform(carla.Location(z=2.0)), attach_to=hero)

    if include_imu:
        imu_bp = bp_lib.find("sensor.other.imu")
        imu_bp.set_attribute("sensor_tick", cfg.sensor_tick())
        rig["imu"] = world.spawn_actor(imu_bp, carla.Transform(carla.Location(z=2.0)), attach_to=hero)

    rig = {k: v for k, v in rig.items() if v is not None}
    return rig


def do_replay(args) -> Path:
    remote_base_str = args.rec_file
    remote_pw = PureWindowsPath(remote_base_str)
    cfg_root = _storage_root_from_config()
    storage_mount = Path(os.environ.get("STORAGE_ROOT", "/Storage"))

    if remote_pw.drive:
        parts = list(remote_pw.parts)
        mount_root = Path(os.environ.get("RECORDINGS_MOUNT", "/recordings"))
        tail = parts[1:]
        if "recordings" in parts:
            try:
                idx = parts.index("recordings")
                tail = parts[idx + 1 :]
            except Exception:
                tail = parts[1:]
        local_base = mount_root / Path(*tail)
    else:
        lp = Path(remote_base_str)
        local_base = lp if lp.is_dir() else lp.parent
        if not local_base.is_absolute():
            local_base = storage_mount / local_base

    # If the mapped path includes the .rec filename, strip it to get the folder
    if local_base.suffix.lower() == ".rec":
        local_base = local_base.parent

    remote_rec = Path(remote_base_str)
    meta_path = local_base / "meta.json"
    local_rec = local_base / "run.rec"

    if not local_rec.exists():
        print(f"[replay] WARNING: local rec not found at {local_rec}; proceeding to replay remote path {remote_rec}")

    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    fps = float(args.fps or meta.get("fps", 20.0))
    duration = float(args.duration or meta.get("duration", 30.0))
    lidar_range = float(args.range or meta.get("lidar_range", 100.0))
    recorded_hero_id = int(meta.get("recorded_hero_id", 0))

    # Choose an output folder with stamp (use rec folder name if available)
    stamp = local_base.name or now_stamp_utc()
    _, out_local_root = resolve_out_paths(args.out_dir)
    out_local = out_local_root / stamp
    out_local.mkdir(parents=True, exist_ok=True)
    (out_local / "source").mkdir(parents=True, exist_ok=True)
    (out_local / "source" / "meta_source.json").write_text(json.dumps(meta, indent=2))

    client = carla.Client(args.host, args.port)
    client.set_timeout(20.0)

    world = client.get_world()
    bp_lib = world.get_blueprint_library()
    tm = client.get_trafficmanager(args.tm_port)

    original_settings, fixed_dt = apply_sync(world, tm, fps)

    rig: Dict[str, carla.Actor] = {}
    queues: Dict[str, queue.Queue] = {}
    writers = {}

    try:
        replay_speed = float(getattr(args, "replay_speed_factor", 1.0))
        if replay_speed <= 0:
            replay_speed = 1.0
        try:
            client.set_replayer_time_factor(replay_speed)
        except Exception:
            pass

        client.replay_file(str(remote_rec), 0.0, float(duration), int(recorded_hero_id))
        print(f"[replay] replay_file({remote_rec}) duration={duration}s fps={fps} speed_factor={replay_speed}")

        hero = find_hero(world, role_name="hero", target_id=recorded_hero_id or None, timeout_frames=300)
        if hero is None:
            raise RuntimeError("Could not find hero vehicle during replay (role_name=hero).")

        cfg = RigConfig(fps=fps, lidar_range=lidar_range)
        rig = spawn_sensor_rig(
            world,
            bp_lib,
            hero,
            cfg,
            include_camera=not args.no_camera,
            include_lidar=not args.no_lidar,
            include_gnss=not args.no_gnss,
            include_imu=not args.no_imu,
        )
        print("[replay] spawned rig:", {k: v.id for k, v in rig.items()})

        queues = {name: make_queue(actor) for name, actor in rig.items()}

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        cam_size = (cfg.image_w, cfg.image_h)
        bev_size = (cfg.bev_size, cfg.bev_size)

        if not args.no_camera:
            writers["rgb"] = cv2.VideoWriter(str(out_local / "camera_front.mp4"), fourcc, fps, cam_size)
        if not args.no_lidar:
            writers["lidar"] = cv2.VideoWriter(str(out_local / "lidar_roof.mp4"), fourcc, fps, bev_size)

        controls: List[Dict[str, Any]] = []
        total_frames = int(round(duration * fps))
        print(f"[replay] capturing {total_frames} frames")

        world.tick()  # settle after spawning sensors

        sem_dtype = np.dtype([
            ("x", np.float32),
            ("y", np.float32),
            ("z", np.float32),
            ("cos_inc", np.float32),
            ("obj_idx", np.uint32),
            ("obj_tag", np.uint32),
        ])

        for i in range(total_frames):
            frame_id, snapshot = tick_and_snapshot(world)
            f = snapshot.frame if snapshot else frame_id

            rgb = get_by_frame(queues["rgb_front"], f, timeout_s=2.0) if not args.no_camera else None
            lid = get_by_frame(queues["lidar_roof"], f, timeout_s=2.0) if not args.no_lidar else None
            sem = None
            gnss = get_by_frame(queues["gnss"], f, timeout_s=2.0) if not args.no_gnss else None
            imu = get_by_frame(queues["imu"], f, timeout_s=2.0) if not args.no_imu else None

            if (not args.no_camera and rgb is None) or (not args.no_lidar and lid is None):
                print(f"[WARN] missing sensor data at frame={f}; skipping")
                continue
            if not args.no_camera and rgb is not None:
                writers["rgb"].write(bgr_from_carla_image(rgb))

            if not args.no_lidar and lid is not None:
                pts = np.frombuffer(lid.raw_data, dtype=np.float32).reshape(-1, 4)
                bev = lidar_bev(pts[:, :2], labels=None, size=cfg.bev_size, max_range_m=lidar_range)
                writers["lidar"].write(bev)

            ctrl = hero.get_control()
            loc = hero.get_location()
            rot = hero.get_transform().rotation
            vel = hero.get_velocity()
            ang = hero.get_angular_velocity()
            speed = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)

            entry = {
                "frame": int(f),
                "sim_time": float(snapshot.timestamp.elapsed_seconds if snapshot else i / float(fps)),
                "control": {
                    "throttle": float(ctrl.throttle),
                    "steer": float(ctrl.steer),
                    "brake": float(ctrl.brake),
                    "hand_brake": bool(ctrl.hand_brake),
                    "reverse": bool(ctrl.reverse),
                    "gear": int(ctrl.gear),
                },
                "state": {
                    "speed_mps": float(speed),
                    "location": {"x": float(loc.x), "y": float(loc.y), "z": float(loc.z)},
                    "rotation": {"pitch": float(rot.pitch), "yaw": float(rot.yaw), "roll": float(rot.roll)},
                    "velocity": {"x": float(vel.x), "y": float(vel.y), "z": float(vel.z)},
                    "angular_velocity": {"x": float(ang.x), "y": float(ang.y), "z": float(ang.z)},
                },
                "gnss": None if args.no_gnss or gnss is None else {
                    "lat": float(gnss.latitude),
                    "lon": float(gnss.longitude),
                    "alt": float(gnss.altitude),
                },
                "imu": None if args.no_imu or imu is None else {
                    "accel": {"x": float(imu.accelerometer.x), "y": float(imu.accelerometer.y), "z": float(imu.accelerometer.z)},
                    "gyro": {"x": float(imu.gyroscope.x), "y": float(imu.gyroscope.y), "z": float(imu.gyroscope.z)},
                    "compass": float(getattr(imu, "compass", 0.0)),
                },
            }
            controls.append(entry)

            if i % int(max(1, fps)) == 0:
                print(f"[replay] {i}/{total_frames} steer={ctrl.steer:+.3f} thr={ctrl.throttle:.3f} spd={speed:.2f} m/s")

        out_json = out_local / "controls.json"
        payload = {
            "meta": {
                "fps": float(fps),
                "fixed_dt": float(fixed_dt),
                "duration": float(duration),
                "frames_requested": int(total_frames),
                "frames_written": int(len(controls)),
                "lidar_range": float(lidar_range),
                "source_rec": str(remote_rec),
                "recorded_hero_id": int(recorded_hero_id),
                "replay_speed_factor": float(replay_speed),
                "outputs": {
                    **({"camera_front": "camera_front.mp4"} if not args.no_camera else {}),
                    **({"lidar_roof": "lidar_roof.mp4"} if not args.no_lidar else {}),
                },
            },
            "frames": controls,
        }
        out_json.write_text(json.dumps(payload, indent=2))
        print(f"[replay] wrote {out_json}")
        print(f"[replay] done -> {out_local}")
        return out_local
    finally:
        for w in writers.values():
            try:
                w.release()
            except Exception:
                pass
        for a in rig.values():
            safe_destroy(a)
        restore_world(world, tm, original_settings)


# --------------------------
# CLI
# --------------------------

def build_parser():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=False)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--host", default="127.0.0.1")
    common.add_argument("--port", type=int, default=2000)
    common.add_argument("--tm-port", type=int, default=8000)
    common.add_argument("--out-dir", default=None)
    common.add_argument("--stop", action="store_true", help="Stop a running session (PID file).")

    pr = sub.add_parser("record", parents=[common])
    pr.add_argument("--duration", type=float, default=15.0)
    pr.add_argument("--fps", type=float, default=10.0)
    pr.add_argument("--range", type=float, default=100.0)
    pr.add_argument("--vehicle-bp", default="vehicle.*model3*")
    pr.add_argument("--target-speed-kph", type=float, default=None, help="Autopilot target speed during recording (km/h). Leave unset to use CARLA defaults.")
    pr.add_argument("--tm-speed-diff", type=float, default=0.0, help="Traffic manager global speed delta in percent (positive slows). 0 matches CARLA default.")
    pr.add_argument("--vehicle-speed-diff", type=float, default=0.0, help="Per-vehicle speed delta in percent (positive slows). 0 matches CARLA default.")
    pr.add_argument("--no-camera", action="store_true", help="(replay/export) skip camera spawn/write")
    pr.add_argument("--no-lidar", action="store_true", help="(replay/export) skip lidar spawn/write")
    pr.add_argument("--no-gnss", action="store_true", help="(replay/export) skip GNSS")
    pr.add_argument("--no-imu", action="store_true", help="(replay/export) skip IMU")

    pp = sub.add_parser("replay", parents=[common])
    pp.add_argument("--rec-file", required=False, help="Path to run.rec OR the recording folder containing run.rec/meta.json")
    pp.add_argument("--duration", type=float, default=None, help="Override meta.json duration")
    pp.add_argument("--fps", type=float, default=None, help="Override meta.json fps")
    pp.add_argument("--range", type=float, default=None, help="Override meta.json lidar range")
    pp.add_argument("--replay-speed-factor", type=float, default=1.0, help="Time factor for replay (<1 slows, >1 speeds up). 1.0 matches recorded timing.")
    pp.add_argument("--no-camera", action="store_true", help="Skip camera spawn/write on replay/export")
    pp.add_argument("--no-lidar", action="store_true", help="Skip lidar spawn/write on replay/export")
    pp.add_argument("--no-gnss", action="store_true", help="Skip GNSS on replay/export")
    pp.add_argument("--no-imu", action="store_true", help="Skip IMU on replay/export")

    pb = sub.add_parser("both", parents=[common])
    pb.add_argument("--duration", type=float, default=15.0)
    pb.add_argument("--fps", type=float, default=10.0)
    pb.add_argument("--range", type=float, default=100.0)
    pb.add_argument("--vehicle-bp", default="vehicle.*model3*")
    pb.add_argument("--target-speed-kph", type=float, default=None, help="Autopilot target speed during recording (km/h). Leave unset to use CARLA defaults.")
    pb.add_argument("--tm-speed-diff", type=float, default=0.0, help="Traffic manager global speed delta in percent (positive slows). 0 matches CARLA default.")
    pb.add_argument("--vehicle-speed-diff", type=float, default=0.0, help="Per-vehicle speed delta in percent (positive slows). 0 matches CARLA default.")
    pb.add_argument("--replay-speed-factor", type=float, default=1.0, help="Time factor for replay of the freshly recorded run (<1 slows). 1.0 matches recorded timing.")
    pb.add_argument("--no-camera", action="store_true", help="Skip camera spawn/write on replay/export")
    pb.add_argument("--no-lidar", action="store_true", help="Skip lidar spawn/write on replay/export")
    pb.add_argument("--no-gnss", action="store_true", help="Skip GNSS on replay/export")
    pb.add_argument("--no-imu", action="store_true", help="Skip IMU on replay/export")

    return p


def main():
    args = build_parser().parse_args()

    if args.stop:
        stopped = _stop_previous()
        print(f"[stop] {'stopped' if stopped else 'nothing running'}")
        _clear_pid()
        return

    if args.cmd is None:
        args.cmd = "both"

    _write_pid()
    try:
        if args.cmd == "record":
            do_record(args)
        elif args.cmd == "replay":
            if not getattr(args, "rec_file", None):
                raise RuntimeError("replay requires --rec-file")
            do_replay(args)
        elif args.cmd == "both":
            out_dir = do_record(args)
            rec_path = Path(out_dir) / "run.rec"
            if not rec_path.exists():
                print(
                    f"[replay] run.rec not found at {rec_path}. "
                    "CARLA writes .rec on the server side; if your server runs elsewhere (e.g., Windows host), "
                    "copy run.rec into this folder and rerun with the 'replay' command."
                )
                return
            # replay the folder we just wrote
            newest = Path(out_dir)
            class R:  # shallow args-like object
                pass
            r = R()
            r.host = args.host
            r.port = args.port
            r.tm_port = args.tm_port
            r.out_dir = args.out_dir
            r.rec_file = str(newest)  # folder containing run.rec/meta.json
            r.duration = None
            r.fps = None
            r.range = None
            r.replay_speed_factor = args.replay_speed_factor
            r.no_camera = args.no_camera
            r.no_lidar = args.no_lidar
            r.no_gnss = args.no_gnss
            r.no_imu = args.no_imu
            do_replay(r)
        else:
            raise RuntimeError(f"Unknown cmd {args.cmd}")
    finally:
        _clear_pid()


if __name__ == "__main__":
    main()
