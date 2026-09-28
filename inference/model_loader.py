# -*- coding: utf-8 -*-
"""
model_loader.py — Carga del artefacto de inferencia (Task 9, demo).

Carga `output/ml_inference/task8b_rf.joblib` (modelo demostrativo ajustado
sobre ML Dataset v0 con los parámetros fijos de Task 8B). Si el artefacto no
existe, levanta InferenceError indicando cómo generarlo (script 20).
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib

from inference.schemas import InferenceError

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "output" / "ml_inference"
MODEL_ARTIFACT = ARTIFACT_DIR / "task8b_rf.joblib"
ARTIFACT_INFO = ARTIFACT_DIR / "artifact_info.json"

_LOADED = None


def load_model():
    """Devuelve el modelo serializado (caché en vuelo)."""
    global _LOADED
    if _LOADED is not None:
        return _LOADED
    if not MODEL_ARTIFACT.exists():
        raise InferenceError(
            f"no existe artefacto de modelo en {MODEL_ARTIFACT}; ejecuta "
            "`.venv\\Scripts\\python scripts\\20_build_inference_artifact.py`")
    _LOADED = joblib.load(MODEL_ARTIFACT)
    return _LOADED


def model_info() -> dict:
    """Metadatos del artefacto (artifact_info.json) si existen."""
    if not ARTIFACT_INFO.exists():
        return {}
    return json.loads(ARTIFACT_INFO.read_text(encoding="utf-8"))