#!/usr/bin/env python3
"""
Lock the spectator to the hero vehicle (role_name == "hero") to view FPV.

Usage:
  python CarlaControl/follow_hero.py --host 127.0.0.1 --port 2000 --z-offset 1.6 --duration 0

Press Ctrl+C to stop. If --duration > 0, the script auto-exits after that many seconds.
"""

from __future__ import annotations

import argparse
import os
import signal
import time
from pathlib import Path

import carla

PID_FILE = Path("/tmp/follow_hero.pid")


def _read_pid() -> int | None:
  try:
    if PID_FILE.is_file():
      return int(PID_FILE.read_text().strip())
  except Exception:
    pass
  return None


def _stop_previous():
  pid = _read_pid()
  if not pid:
    return False
  try:
    os.kill(pid, signal.SIGTERM)
    return True
  except Exception:
    return False


def find_hero(world: carla.World) -> carla.Actor | None:
  for actor in world.get_actors():
    try:
      if actor.attributes.get("role_name") == "hero":
        return actor
    except Exception:
      continue
  return None


def main() -> None:
  parser = argparse.ArgumentParser(description="Attach spectator to hero vehicle for FPV viewing.")
  parser.add_argument("--host", type=str, default="127.0.0.1")
  parser.add_argument("--port", type=int, default=2000)
  parser.add_argument("--timeout", type=float, default=10.0)
  parser.add_argument("--z-offset", type=float, default=1.6, help="Camera height above the hero.")
  parser.add_argument("--duration", type=float, default=0.0, help="Seconds to run; 0 = until Ctrl+C.")
  parser.add_argument("--tick-sleep", type=float, default=0.05, help="Sleep between updates.")
  parser.add_argument("--stop", action="store_true", help="Stop any running follow_hero session and exit.")
  args = parser.parse_args()

  if args.stop:
    stopped = _stop_previous()
    if stopped:
      print("[follow_hero] stopped previous session")
      try:
        PID_FILE.unlink(missing_ok=True)  # type: ignore
      except Exception:
        pass
    else:
      print("[follow_hero] no running session to stop")
    return

  # Stop any prior session before starting a new one.
  _stop_previous()

  client = carla.Client(args.host, args.port)
  client.set_timeout(args.timeout)
  world = client.get_world()
  spectator = world.get_spectator()

  hero = find_hero(world)
  if not hero:
    raise RuntimeError("No hero vehicle found (role_name == 'hero'). Spawn a hero first (e.g., spawn_custom_npc).")

  print(f"[follow_hero] following hero id={hero.id}")
  try:
    PID_FILE.write_text(str(os.getpid()))
  except Exception:
    pass
  start = time.monotonic()
  try:
    while True:
      tr = hero.get_transform()
      spectator.set_transform(
        carla.Transform(
          carla.Location(x=tr.location.x, y=tr.location.y, z=tr.location.z + args.z_offset),
          tr.rotation,
        )
      )
      if args.duration > 0 and (time.monotonic() - start) >= args.duration:
        break
      if args.tick_sleep > 0:
        time.sleep(args.tick_sleep)
  except KeyboardInterrupt:
    pass
  finally:
    try:
      PID_FILE.unlink(missing_ok=True)  # type: ignore
    except Exception:
      pass
    print("[follow_hero] done.")


if __name__ == "__main__":
  main()
