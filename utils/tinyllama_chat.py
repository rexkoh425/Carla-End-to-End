#!/usr/bin/env python3
"""
Minimal chat client for the TinyLlama model served by the llama.cpp container.

Requirements: requests (available in the backend image) and the llama.cpp server
running at localhost:8081 (see docker-compose.yml service "llama").

Example:
  python utils/tinyllama_chat.py --prompt "Hello, how are you?"
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import List

import requests


def chat(prompt: str, host: str = "localhost", port: int = 8081, max_tokens: int = 128, temperature: float = 0.2) -> str:
  url = f"http://{host}:{port}/completion"
  body = {
    "prompt": prompt,
    "temperature": temperature,
    "n_predict": max_tokens,
  }
  r = requests.post(url, json=body, timeout=60)
  r.raise_for_status()
  data = r.json()
  return data.get("content", "") or data.get("response", "")


def main():
  parser = argparse.ArgumentParser(description="Chat with TinyLlama served by llama.cpp.")
  parser.add_argument("--prompt", required=True, help="User prompt.")
  parser.add_argument("--host", default="localhost")
  parser.add_argument("--port", type=int, default=8081)
  parser.add_argument("--max-tokens", type=int, default=128)
  parser.add_argument("--temperature", type=float, default=0.2)
  args = parser.parse_args()
  try:
    reply = chat(args.prompt, host=args.host, port=args.port, max_tokens=args.max_tokens, temperature=args.temperature)
    print(json.dumps({"prompt": args.prompt, "reply": reply}, indent=2))
  except Exception as exc:
    print(f"Error: {exc}", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
  main()
