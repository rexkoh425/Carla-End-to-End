"""
Spawn a scene with a player vehicle, traffic vehicles, and pedestrians in CARLA.

Usage:
    python CarlaControl/spawn_scene.py --host 127.0.0.1 --port 2000 \
        --vehicles 30 --walkers 60 --town Town10HD_Opt

Notes:
- Requires CARLA server running on the specified host/port.
- Uses sync mode to ensure consistent spawning; switches back to original settings on exit.
"""

from __future__ import annotations

import argparse
import random
import time
from dataclasses import dataclass
import time
from typing import List, Tuple

import carla
from carla import command as carla_cmd


@dataclass
class SceneConfig:
    host: str = "127.0.0.1"
    port: int = 2000
    tm_port: int = 8000
    vehicles: int = 20
    walkers: int = 40
    town: str | None = None
    autopilot: bool = True
    timeout: float = 10.0
    tick_sleep: float = 0.05  # seconds to sleep after each tick to approximate realtime
    duration: float = 0.0     # seconds to run before auto-cleanup; 0 means run until stopped


def set_sync_mode(world: carla.World, sync: bool, delta: float = 0.05):
    settings = world.get_settings()
    settings.synchronous_mode = sync
    settings.fixed_delta_seconds = delta if sync else None
    world.apply_settings(settings)
    print(f"[spawn_scene] sync_mode={sync} delta={delta}")


def wait_for_world(client: carla.Client, attempts: int = 10, delay: float = 0.5) -> carla.World:
    for i in range(attempts):
        try:
            world = client.get_world()
            world.get_map()
            return world
        except Exception as exc:
            print(f"[spawn_scene] world attempt {i+1}/{attempts} failed: {exc}")
            time.sleep(delay)
    raise RuntimeError("Failed to get CARLA world (server not ready)")


def spawn_player(world: carla.World, bp_lib, spawn_points: List[carla.Transform]) -> carla.Actor:
    vehicle_bp = random.choice(bp_lib.filter("vehicle.*"))
    spawn = random.choice(spawn_points)
    player = world.try_spawn_actor(vehicle_bp, spawn)
    if player:
        print(f"[spawn_scene] spawned player id={player.id} at {spawn}")
    else:
        print("[spawn_scene] failed to spawn player")
    return player


def spawn_vehicles(world: carla.World, bp_lib, spawn_points: List[carla.Transform], count: int, autopilot: bool, tm_port: int) -> List[carla.Actor]:
    actors = []
    random.shuffle(spawn_points)
    for sp in spawn_points[:count]:
        bp = random.choice(bp_lib.filter("vehicle.*"))
        actor = world.try_spawn_actor(bp, sp)
        if actor:
            if autopilot and hasattr(actor, "set_autopilot"):
                actor.set_autopilot(True, tm_port)
            actors.append(actor)
            print(f"[spawn_scene] vehicle id={actor.id} spawn={sp}")
        else:
            print(f"[spawn_scene] vehicle spawn failed at {sp}")
    return actors


def spawn_walkers(client: carla.Client, world: carla.World, bp_lib, count: int) -> Tuple[List[carla.Actor], List[carla.Actor]]:
    """
    Batch-spawn walkers and controllers per CARLA docs to avoid collisions/crashes.
    """
    walker_bps = bp_lib.filter("walker.pedestrian.*")
    spawn_points = []
    for _ in range(count):
        loc = world.get_random_location_from_navigation()
        if loc:
            spawn_points.append(carla.Transform(loc))
    if not spawn_points:
        return [], []

    # Batch spawn walkers
    batch = [carla_cmd.SpawnActor(random.choice(walker_bps), tr) for tr in spawn_points]
    results = client.apply_batch_sync(batch, True)
    walker_ids = [r.actor_id for r in results if not r.error]

    if not walker_ids:
        return [], []

    controller_bp = bp_lib.find("controller.ai.walker")
    batch_ctrl = [carla_cmd.SpawnActor(controller_bp, carla.Transform(), wid) for wid in walker_ids]
    results_ctrl = client.apply_batch_sync(batch_ctrl, True)
    controller_ids = [r.actor_id for r in results_ctrl if not r.error]

    walkers: List[carla.Actor] = []
    controllers: List[carla.Actor] = []
    for wid, cid in zip(walker_ids, controller_ids):
        walker = world.get_actor(wid)
        controller = world.get_actor(cid)
        if not walker or not controller:
            continue
        walkers.append(walker)
        controllers.append(controller)
        controller.start()
        # Set speed using recommended values if present
        speed = 1.5
        try:
            speeds = walker.type_id and walker.attributes.get("speed", "")
            if speeds:
                vals = [float(v) for v in speeds.split() if v]
                if vals:
                    speed = vals[0]
        except Exception:
            pass
        controller.set_max_speed(speed)
        dest = world.get_random_location_from_navigation()
        if dest:
            controller.go_to_location(dest)
        print(f"[spawn_scene] walker id={walker.id} ctrl={controller.id}")

    return walkers, controllers


def cleanup(actors):
    for a in actors:
        try:
            a.destroy()
        except Exception:
            pass


def main(cfg: SceneConfig):
    client = carla.Client(cfg.host, cfg.port)
    client.set_timeout(cfg.timeout)
    world = wait_for_world(client)

    tm = client.get_trafficmanager(cfg.tm_port)
    tm.set_synchronous_mode(True)
    tm.set_global_distance_to_leading_vehicle(2.0)
    tm.global_percentage_speed_difference(0.0)

    if cfg.town and world.get_map().name != cfg.town:
        world = client.load_world(cfg.town)

    bp_lib = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        raise RuntimeError("No spawn points available in current map.")
    random.shuffle(spawn_points)

    # Store original settings
    original_settings = world.get_settings()
    set_sync_mode(world, True)

    all_actors = []
    walker_controllers = []
    try:
        player = spawn_player(world, bp_lib, spawn_points)
        if not player:
            raise RuntimeError("Failed to spawn player vehicle.")
        all_actors.append(player)
        if cfg.autopilot and hasattr(player, "set_autopilot"):
            player.set_autopilot(True, cfg.tm_port)

        # Reserve one spot for player
        max_vehicles = max(0, len(spawn_points) - 1)
        vehicle_target = min(cfg.vehicles, max_vehicles)
        vehicles = spawn_vehicles(world, bp_lib, spawn_points, vehicle_target, cfg.autopilot, cfg.tm_port)
        all_actors.extend(vehicles)

        walkers, controllers = spawn_walkers(client, world, bp_lib, cfg.walkers)
        all_actors.extend(walkers)
        walker_controllers.extend(controllers)

        # Tick a few frames to let everything settle
        for _ in range(10):
            world.tick()

        print(f"Spawned: player={player.id}, vehicles={len(vehicles)}, walkers={len(walkers)}")
        if cfg.duration and cfg.duration > 0:
            print(f"Running for {cfg.duration} seconds... Ctrl+C to stop sooner.")
            start = time.monotonic()
            while time.monotonic() - start < cfg.duration:
                try:
                    world.tick()
                except Exception as exc:
                    print(f"[spawn_scene] tick error: {exc}")
                    time.sleep(0.1)
                    continue
                if cfg.tick_sleep > 0:
                    time.sleep(cfg.tick_sleep)
        else:
            print("Running until stopped (Ctrl+C)...")
            while True:
                try:
                    world.tick()
                except Exception as exc:
                    print(f"[spawn_scene] tick error: {exc}")
                    time.sleep(0.1)
                    continue
                if cfg.tick_sleep > 0:
                    time.sleep(cfg.tick_sleep)
    except KeyboardInterrupt:
        pass
    finally:
        cleanup(walker_controllers)
        cleanup(all_actors)
        set_sync_mode(world, original_settings.synchronous_mode, original_settings.fixed_delta_seconds or 0.05)
        world.apply_settings(original_settings)
        print("Cleanup done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1", help="CARLA host (use host.docker.internal when targeting host from container).")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--vehicles", type=int, default=20)
    parser.add_argument("--walkers", type=int, default=40)
    parser.add_argument("--town", type=str, default=None, help="e.g., Town10HD_Opt")
    parser.add_argument("--tm-port", type=int, default=8000, help="Traffic Manager port.")
    parser.add_argument("--no-autopilot", action="store_true", help="Disable autopilot on vehicles")
    parser.add_argument("--timeout", type=float, default=10.0, help="Client timeout in seconds.")
    parser.add_argument("--tick-sleep", type=float, default=0.05, help="Seconds to sleep after each tick (sync mode).")
    parser.add_argument("--duration", type=float, default=0.0, help="Seconds to run before auto-cleanup (0 = infinite).")
    args = parser.parse_args()

    cfg = SceneConfig(
        host=args.host,
        port=args.port,
        tm_port=args.tm_port,
        vehicles=args.vehicles,
        walkers=args.walkers,
        town=args.town,
        autopilot=not args.no_autopilot,
        timeout=args.timeout,
        tick_sleep=args.tick_sleep,
        duration=args.duration,
    )
    main(cfg)
