#!/usr/bin/env python3
"""
Spawn background traffic (vehicles + walkers) via CARLA client.
Uses navmesh locations and controller batching for walkers to avoid spawning in roads.

Example:
  python CarlaControl/spawn_traffic.py --host host.docker.internal --port 2000 --tm-port 8000 \
    --vehicles 10 --walkers 10 --safe
"""
from __future__ import annotations

import argparse
import random
import carla
from carla import command as carla_cmd


def spawn_vehicles(world: carla.World, tm: carla.TrafficManager, bp_lib: carla.BlueprintLibrary, n: int, safe: bool):
    veh_bps = bp_lib.filter("vehicle.*")
    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)
    spawned = []
    for sp in spawn_points[:n]:
        bp = random.choice(veh_bps)
        bp.set_attribute("role_name", "autopilot")
        if safe and bp.has_attribute("color"):
            bp.set_attribute("color", bp.get_attribute("color").recommended_values[0])
        veh = world.try_spawn_actor(bp, sp)
        if veh:
            veh.set_autopilot(True, tm.get_port())
            if safe:
                tm.ignore_lights_percentage(veh, 0.0)
            spawned.append(veh)
    return spawned


def spawn_walkers_batch(client: carla.Client, world: carla.World, bp_lib: carla.BlueprintLibrary, n: int):
    """Spawn walkers on navmesh with controllers, similar to CARLA's spawn_npc.py."""
    walker_bps = bp_lib.filter("walker.pedestrian.*")
    spawn_transforms = []
    for _ in range(n):
        loc = world.get_random_location_from_navigation()
        if loc:
            spawn_transforms.append(carla.Transform(loc))
    if not spawn_transforms:
        return [], []

    batch = [carla_cmd.SpawnActor(random.choice(walker_bps), tr) for tr in spawn_transforms]
    results = client.apply_batch_sync(batch, True)
    walker_ids = [r.actor_id for r in results if not r.error]
    if not walker_ids:
        return [], []

    controller_bp = bp_lib.find("controller.ai.walker")
    batch_ctrl = [carla_cmd.SpawnActor(controller_bp, carla.Transform(), wid) for wid in walker_ids]
    results_ctrl = client.apply_batch_sync(batch_ctrl, True)
    controller_ids = [r.actor_id for r in results_ctrl if not r.error]

    walkers = []
    controllers = []
    for wid, cid in zip(walker_ids, controller_ids):
        w = world.get_actor(wid)
        c = world.get_actor(cid)
        if not w or not c:
            continue
        walkers.append(w)
        controllers.append(c)
        c.start()
        speed = 1.4
        try:
            attr = w.attributes.get("speed", "")
            if attr:
                vals = [float(v) for v in attr.split() if v]
                if vals:
                    speed = vals[0]
        except Exception:
            pass
        c.set_max_speed(speed)
        dest = world.get_random_location_from_navigation()
        if dest:
            c.go_to_location(dest)
    return walkers, controllers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--tm-port", type=int, default=8000)
    ap.add_argument("--vehicles", type=int, default=10)
    ap.add_argument("--walkers", type=int, default=10)
    ap.add_argument("--safe", action="store_true", help="Enable safer TM settings.")
    args = ap.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    world = client.get_world()
    tm = client.get_trafficmanager(args.tm_port)
    # Align TM sync with a sync-world (record_robotaxi uses sync); keep neutral speed unless overridden.
    tm.set_synchronous_mode(True)
    if args.safe:
        tm.global_percentage_speed_difference(0.0)

    bp_lib = world.get_blueprint_library()
    vehicles = spawn_vehicles(world, tm, bp_lib, args.vehicles, args.safe)
    walkers, ctrls = spawn_walkers_batch(client, world, bp_lib, args.walkers)

    print(f"Spawned vehicles: {len(vehicles)}, walkers: {len(walkers)} (controllers: {len(ctrls)})")


if __name__ == "__main__":
    main()
