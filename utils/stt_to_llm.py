#!/usr/bin/env python3
"""
Pipeline: speech-to-text with whisper.cpp server, then send text to llama.cpp server.

Assumes:
  - Whisper server running at http://localhost:8082 (whisper-server service)
  - Llama server running at http://localhost:8081 (mcp-server service)

Usage:
  python utils/stt_to_llm.py --audio path/to/audio.wav
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import requests


def transcribe(audio_path: Path, host: str, port: int) -> str:
  url = f"http://{host}:{port}/inference"
  with audio_path.open("rb") as f:
    files = {"file": (audio_path.name, f, "audio/wav")}
    r = requests.post(url, files=files, timeout=120)
  r.raise_for_status()
  data = r.json()
  return data.get("text") or data.get("transcription") or ""


def chat_llama(prompt: str, host: str, port: int, max_tokens: int, temperature: float) -> str:
  url = f"http://{host}:{port}/completion"
  body = {
    "prompt": prompt,
    "n_predict": max_tokens,
    "temperature": temperature,
  }
  r = requests.post(url, json=body, timeout=60)
  r.raise_for_status()
  data = r.json()
  return data.get("content") or data.get("response") or ""


def main() -> int:
  parser = argparse.ArgumentParser(description="Speech-to-text then LLM pipeline.")
  parser.add_argument("--audio", required=True, help="Path to audio WAV file.")
  parser.add_argument("--whisper-host", default="localhost")
  parser.add_argument("--whisper-port", type=int, default=8082)
  parser.add_argument("--llm-host", default="localhost")
  parser.add_argument("--llm-port", type=int, default=8081)
  parser.add_argument("--max-tokens", type=int, default=128)
  parser.add_argument("--temperature", type=float, default=0.2)
  args = parser.parse_args()

  audio_path = Path(args.audio)
  if not audio_path.is_file():
    print(f"Audio file not found: {audio_path}", file=sys.stderr)
    return 1

  try:
    text = transcribe(audio_path, args.whisper_host, args.whisper_port)
    reply = chat_llama(text, args.llm_host, args.llm_port, args.max_tokens, args.temperature)
    print(json.dumps({"transcript": text, "llm_reply": reply}, indent=2))
    return 0
  except Exception as exc:
    print(f"Error: {exc}", file=sys.stderr)
    return 1


if __name__ == "__main__":
  sys.exit(main())
