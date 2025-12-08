#!/usr/bin/env python3
"""
CARLA connectivity check (TCP) for RPC/streaming ports.

Intended to be run inside a container or host to validate network reachability
before launching heavier components (ROS bridge, health checks, etc.).
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from dataclasses import asdict, dataclass
from typing import Optional


@dataclass
class ConnectCheckConfig:
  host: str = "127.0.0.1"
  port: int = 2000
  streaming_port: Optional[int] = None
  timeout: float = 2.0  # seconds per attempt
  attempts: int = 5
  delay: float = 1.0  # seconds between attempts


def _probe(host: str, port: int, timeout: float) -> tuple[bool, str]:
  s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  s.settimeout(timeout)
  try:
    s.connect((host, port))
    return True, ""
  except Exception as exc:
    return False, str(exc)
  finally:
    try:
      s.close()
    except Exception:
      pass


def run_check(cfg: ConnectCheckConfig) -> dict:
  attempts = []
  ok_rpc = False
  ok_stream = None

  for i in range(cfg.attempts):
    success, err = _probe(cfg.host, cfg.port, cfg.timeout)
    attempts.append({"attempt": i + 1, "port": cfg.port, "ok": success, "error": err})
    if success:
      ok_rpc = True
      break
    time.sleep(cfg.delay)

  stream_attempts = []
  if cfg.streaming_port is not None:
    for i in range(cfg.attempts):
      success, err = _probe(cfg.host, cfg.streaming_port, cfg.timeout)
      stream_attempts.append({"attempt": i + 1, "port": cfg.streaming_port, "ok": success, "error": err})
      if success:
        ok_stream = True
        break
      time.sleep(cfg.delay)
    if ok_stream is None:
      ok_stream = False

  result = {
    "ok": ok_rpc and (ok_stream is not False),
    "rpc_port_ok": ok_rpc,
    "streaming_port_ok": ok_stream,
    "host": cfg.host,
    "port": cfg.port,
    "streaming_port": cfg.streaming_port,
    "attempts": attempts,
    "stream_attempts": stream_attempts,
    "config": asdict(cfg),
  }
  return result


def main() -> int:
  parser = argparse.ArgumentParser(description="Check TCP reachability to CARLA RPC/streaming ports.")
  parser.add_argument("--host", type=str, default="127.0.0.1")
  parser.add_argument("--port", type=int, default=2000, help="CARLA RPC port")
  parser.add_argument("--streaming-port", type=int, default=None, help="Optional CARLA streaming port to probe")
  parser.add_argument("--timeout", type=float, default=2.0, help="Seconds per attempt")
  parser.add_argument("--attempts", type=int, default=5, help="Max attempts per port")
  parser.add_argument("--delay", type=float, default=1.0, help="Delay between attempts (seconds)")
  args = parser.parse_args()

  cfg = ConnectCheckConfig(
    host=args.host,
    port=args.port,
    streaming_port=args.streaming_port,
    timeout=args.timeout,
    attempts=args.attempts,
    delay=args.delay,
  )
  result = run_check(cfg)
  print(json.dumps(result, indent=2))
  return 0 if result["ok"] else 2


if __name__ == "__main__":
  sys.exit(main())
