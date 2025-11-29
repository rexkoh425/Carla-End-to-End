#!/usr/bin/env python3
"""
Minimal pipeline: send a prompt to LocalAI chat, then synthesize the reply to audio via LocalAI TTS.

Requires LocalAI running with:
- Chat model name: tinyllama-1.1b-chat (GGUF loaded under /models)
- TTS model name: en-us-amy-tts (Piper ONNX)

Usage (from repo root):
  python utils/llm_to_speech.py --prompt "Hello there" \
    --out audio.wav \
    --chat-model tinyllama-1.1b-chat \
    --tts-model en-us-amy-tts \
    --base-url http://localhost:8080
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path
from typing import Optional

import requests


def call_chat(prompt: str, model: str, base_url: str, max_tokens: int = 128, temperature: float = 0.2) -> str:
  url = f"{base_url.rstrip('/')}/v1/chat/completions"
  body = {
    "model": model,
    "messages": [
      {"role": "system", "content": "You are a concise assistant for Carla tasks."},
      {"role": "user", "content": prompt},
    ],
    "max_tokens": max_tokens,
    "temperature": temperature,
  }
  r = requests.post(url, json=body, timeout=60)
  r.raise_for_status()
  data = r.json()
  return data["choices"][0]["message"]["content"]


def call_tts(text: str, model: str, base_url: str) -> bytes:
  url = f"{base_url.rstrip('/')}/v1/audio/speech"
  body = {"model": model, "input": text}
  r = requests.post(url, json=body, timeout=120)
  r.raise_for_status()
  return r.content


def main() -> int:
  parser = argparse.ArgumentParser(description="LLM -> speech pipeline via LocalAI.")
  parser.add_argument("--prompt", required=True, help="User text to send to LLM.")
  parser.add_argument("--out", default="llm_output.wav", help="Path to save synthesized audio.")
  parser.add_argument("--chat-model", default="tinyllama-1.1b-chat", help="LocalAI chat model name.")
  parser.add_argument("--tts-model", default="en-us-amy-tts", help="LocalAI TTS model name (Piper).")
  parser.add_argument("--base-url", default="http://localhost:8080", help="LocalAI base URL.")
  parser.add_argument("--max-tokens", type=int, default=128)
  parser.add_argument("--temperature", type=float, default=0.2)
  args = parser.parse_args()

  try:
    reply = call_chat(args.prompt, args.chat_model, args.base_url, max_tokens=args.max_tokens, temperature=args.temperature)
    audio = call_tts(reply, args.tts_model, args.base_url)
    out_path = Path(args.out)
    out_path.write_bytes(audio)
    print(json.dumps({"text": reply, "audio": str(out_path), "bytes": len(audio)}, indent=2))
    return 0
  except Exception as exc:
    print(f"Error: {exc}", file=sys.stderr)
    return 1


if __name__ == "__main__":
  sys.exit(main())
