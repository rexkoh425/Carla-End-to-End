from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from whispercpp import Whisper

WHISPER_MODEL_PATH = Path(os.environ.get("WHISPER_MODEL", "/models/audio/whisper/ggml-base.en.bin"))

app = FastAPI(title="Whisper STT API")

_whisper: Optional[Whisper] = None


def get_model() -> Whisper:
  global _whisper
  if _whisper is None:
    if not WHISPER_MODEL_PATH.is_file():
      raise RuntimeError(f"Whisper model not found at {WHISPER_MODEL_PATH}")
    _whisper = Whisper(model_path=str(WHISPER_MODEL_PATH))
  return _whisper


@app.post("/inference")
async def inference(file: UploadFile = File(...)) -> JSONResponse:
  try:
    audio_bytes = await file.read()
    tmp_path = Path("/tmp") / file.filename
    tmp_path.write_bytes(audio_bytes)
    model = get_model()
    result = model.transcribe(str(tmp_path))
    text = result.get("text", "").strip() if isinstance(result, dict) else str(result).strip()
    tmp_path.unlink(missing_ok=True)
    return JSONResponse({"text": text})
  except Exception as exc:
    raise HTTPException(status_code=500, detail=f"Transcription failed: {exc}") from exc
