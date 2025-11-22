"""
Minimal local backend for the WebUI.

Endpoints:
- GET  /api/models  -> scans known project folders for configs/weights
- POST /api/track   -> forwards events to MLflow or ClearML endpoints (best-effort)

Run:
    python WebUI/server.py --port 7000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_FOLDERS = {
    "mask2former": "Mask2Former",
    "laneatt": "LaneATT",
    "yolop": "YOLOP",
    "yoloe": "YOLOE",
    "blip": "BLIP",
    "cyclegan": "CycleGAN",
    "depthanything": "DepthAnything",
    "dinox": "DINO-X",
    "internvl": "InternVL",
    "moe": "MoE",
    "persformer": "PersFormer",
    "qwen3": "Qwen3",
    "sdxl": "SDXL",
    "tcntrader": "TCNTransformerMoETrader",
    "videoharvester": "VideoHarvester",
    "videoproclab": "VideoProcessingLab",
    "webharvesting": "WebHarvesting",
}


def find_first(root: Path, patterns: Tuple[str, ...]) -> Optional[str]:
    for pattern in patterns:
        for path in root.rglob(pattern):
            if path.is_file():
                return str(path.relative_to(PROJECT_ROOT))
    return None


def scan_models() -> List[Dict]:
    results = []
    for model_id, folder_name in MODEL_FOLDERS.items():
        root = PROJECT_ROOT / folder_name
        if not root.exists():
            continue
        config = find_first(root, ("*config*.yaml", "*config*.yml", "*.json"))
        weights = find_first(root, ("*.pth", "*.pt", "*.bin", "*.safetensors"))
        results.append(
            {
                "id": model_id,
                "name": folder_name,
                "config": config,
                "weights": weights,
            }
        )
    return results


def http_json(url: str, payload: Dict, method: str = "POST", headers: Optional[Dict] = None) -> Tuple[int, str]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method=method.upper())
    req.add_header("Content-Type", "application/json")
    if headers:
        for k, v in headers.items():
            req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, resp.read().decode("utf-8")
    except Exception as exc:  # pragma: no cover - network errors
        return 599, str(exc)


def ensure_mlflow_experiment(base: str, name: str) -> Optional[str]:
    url = f"{base}/api/2.0/mlflow/experiments/get-by-name?experiment_name={urllib.parse.quote(name)}"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.loads(resp.read())
            return data["experiment"]["experiment_id"]
    except Exception:
        pass

    # create experiment
    status, body = http_json(f"{base}/api/2.0/mlflow/experiments/create", {"name": name})
    if status == 200:
        try:
            return json.loads(body)["experiment_id"]
        except Exception:
            return None
    return None


def forward_to_mlflow(endpoint: str, experiment: str, run_name: str, event: str, payload: Dict) -> Tuple[int, str]:
    exp_id = ensure_mlflow_experiment(endpoint, experiment)
    if not exp_id:
        return 500, "Could not get/create experiment"

    run_body = {
        "experiment_id": exp_id,
        "run_name": run_name,
        "start_time": int(time.time() * 1000),
        "tags": [{"key": "event", "value": event}, {"key": "source", "value": "local-ui"}],
    }
    status, body = http_json(f"{endpoint}/api/2.0/mlflow/runs/create", run_body)
    if status != 200:
        return status, body

    try:
        run_id = json.loads(body)["run"]["info"]["run_id"]
    except Exception:
        return status, body

    params = []
    cfg = payload.get("config") or {}
    runtime = cfg.get("runtime") or {}
    for key, value in runtime.items():
        params.append({"key": f"runtime.{key}", "value": str(value)})

    if params:
        http_json(
            f"{endpoint}/api/2.0/mlflow/runs/log-batch",
            {"run_id": run_id, "params": params},
        )

    return status, body


def forward_to_clearml(endpoint: str, experiment: str, run_name: str, event: str, payload: Dict) -> Tuple[int, str]:
    # ClearML server APIs are more involved; here we forward the payload as-is for a custom bridge.
    return http_json(f"{endpoint}/api/track", {"experiment": experiment, "run": run_name, "event": event, "payload": payload})


class RequestHandler(BaseHTTPRequestHandler):
    def _set_headers(self, code: int = 200, extra: Optional[Dict[str, str]] = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        if extra:
            for k, v in extra.items():
                self.send_header(k, v)
        self.end_headers()

    def do_OPTIONS(self):  # pragma: no cover - preflight
        self._set_headers(204)

    def do_GET(self):
        if self.path.startswith("/api/models"):
            models = scan_models()
            self._set_headers(200)
            self.wfile.write(json.dumps(models).encode("utf-8"))
            return

        self._set_headers(404)
        self.wfile.write(json.dumps({"error": "Not found"}).encode("utf-8"))

    def do_POST(self):
        if self.path.startswith("/api/track"):
            content_length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(content_length or 0)
            try:
                payload = json.loads(raw.decode("utf-8"))
            except Exception:
                self._set_headers(400)
                self.wfile.write(json.dumps({"error": "Invalid JSON"}).encode("utf-8"))
                return

            provider = payload.get("provider", "none")
            endpoint = payload.get("endpoint")
            experiment = payload.get("experiment", "default")
            run_name = payload.get("run", "run")
            event = payload.get("event", "event")

            if provider == "none" or not endpoint:
                self._set_headers(200)
                self.wfile.write(json.dumps({"status": "ok", "forwarded": False}).encode("utf-8"))
                return

            if provider == "mlflow":
                status, body = forward_to_mlflow(endpoint, experiment, run_name, event, payload.get("payload", {}))
            elif provider == "clearml":
                status, body = forward_to_clearml(endpoint, experiment, run_name, event, payload.get("payload", {}))
            else:
                status, body = 400, "Unsupported provider"

            self._set_headers(status if status else 500)
            self.wfile.write(json.dumps({"status": status, "body": body}).encode("utf-8"))
            return

        self._set_headers(404)
        self.wfile.write(json.dumps({"error": "Not found"}).encode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Local backend for WebUI")
    parser.add_argument("--port", type=int, default=7000)
    args = parser.parse_args()
    server = HTTPServer(("0.0.0.0", args.port), RequestHandler)
    print(f"Serving local backend on http://0.0.0.0:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
