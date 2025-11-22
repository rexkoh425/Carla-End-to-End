# FAQ: CARLA + Local Stack Issues

## 1) CARLA server keeps crashing / container restarts
- Symptoms: `docker compose ps` shows `carla-server` restarting; logs mention `lavapipe` or segfault.
- Cause: Container lacks GPU driver libs and falls back to Mesa/soft rendering.
- Fix: Do **not** run the CARLA container on this host. Instead, run the Windows CARLA binary locally and point clients at it (see FAQ #2). If you must use the container, you need NVIDIA libs mounted into the container (not available here by default), or force software GL at a big performance cost.

## 2) Using Windows CARLA binary with Dockerized backend/web
- Run Windows CARLA server locally (CarlaWin). Do **not** start the `carla` compose service.
- Start only backend + web: `docker compose up -d backend web`.
- In WebUI/controls, set Host to `host.docker.internal`, Port `2000`.
- From backend container scripts, also use host `host.docker.internal` and port `2000`.
- Connectivity check: `python CarlaControl/carla_connect_check.py --host host.docker.internal --port 2000`.
- Spawn test (inside backend): `docker compose exec backend micromamba run -n app python CarlaControl/spawn_healthcheck.py --host host.docker.internal --port 2000 --vehicles 5 --walkers 5`.

## 3) Actor counts endpoint returns 502
- Cause: Backend cannot reach CARLA or CARLA crashed.
- Check CARLA is running and reachable (see #2). If using Windows CARLA, ensure firewall allows TCP/UDP 2000.
- Verify via `carla_connect_check` and a direct client snippet inside backend:
  ```
  docker compose exec backend micromamba run -n app python - <<'PY'
  import carla
  c=carla.Client("host.docker.internal",2000); c.set_timeout(5)
  w=c.get_world(); print("actors", len(w.get_actors()))
  PY
  ```

## 4) Networking/compose got stuck (missing network)
- Symptom: `failed to set up container networking: network <id> not found`.
- Fix: `docker compose down` then `docker network prune` (optional) and restart with `docker compose up -d backend web` (no carla profile when using Windows CARLA).

## 5) Default hosts to use
- WebUI default host was switched to `carla-server` (for the container). When using Windows CARLA, always override to `host.docker.internal`.
- For scripts run directly on the Windows host, use `127.0.0.1`.

## 6) LocalAI models and pipeline (LLM -> TTS)
- Models live under `models/localai/`:
  - LLM: `tinyllama-1.1b-chat` (GGUF)
  - STT: `whisper-tiny`
  - TTS: `en-us-amy-tts` (Piper)
- Helper pipeline: `python utils/llm_to_speech.py --prompt "..." --out out.wav --base-url http://localhost:8080`
- Ensure LocalAI mounts `models/localai` at `/models` and is running on `localhost:8080`.

## 7) If you must retry the CARLA container (not recommended here)
- Requirements: GPU driver libs inside container; mount `/usr/lib/wsl/lib` if present, set `LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/lib/x86_64-linux-gnu`, keep offscreen flags (`-RenderOffScreen -opengl -nosound`, `SDL_VIDEODRIVER=offscreen`).
- Start in foreground to see crashes: `docker compose --profile carla up carla`.
- If it still segfaults, fall back to the Windows binary (see #2).
