# -*- coding: utf-8 -*-
"""
schemas.py — Contrato de la capa de inferencia (Task 9, demo).

Ejecuta un contrato EXACTO con el modelo demostrativo de Task 8B:
  - 10 features (Experimento B: SIN snr)
  - target: technique (S02-S05)
  - grupo/atletas: fuera de la inferencia (no se usan como variables)

snr existe en el dataset (ml_dataset_v0) pero fue excluida del modelo
(Experimento B / Task 8B); por eso NO forma parte del contrato de entrada.
"""

INFERENCE_VERSION = "0.1.0"
MODEL_VERSION = "task8b_random_forest_baseline"

TARGET = "technique"
CLASSES = ["S02", "S03", "S04", "S05"]

# Las 10 features del modelo de Task 8B (Experimento B, sin SNR).
FEATURES = [
    "duration_s", "time_to_peak_s", "vmax", "vmean", "amax",
    "displacement", "path_length", "hip_rom", "knee_rom", "ankle_rom",
]

# Nombres legibles (mismo vocabulario del Data Mart/dashboard; copiados para
# que la capa de inferencia NO dependa del dashboard).
TECHNIQUE_NAMES = {
    "S02": "Mae-Geri",
    "S03": "Mawashi-Geri gedan",
    "S04": "Mawashi-Geri jodan",
    "S05": "Ushiro-Mawashi-Geri",
}


class InferenceError(ValueError):
    """Error controlado de la capa de inferencia (entrada inválida)."""


def build_result(predicted: str, probs: dict, features: dict) -> dict:
    """Estructura estable del resultado de inferencia."""
    return {
        "predicted_technique": predicted,
        "technique_name": TECHNIQUE_NAMES.get(predicted, predicted),
        "probabilities": {c: round(float(probs[c]), 4) for c in CLASSES},
        "confidence": round(float(probs[predicted]), 4),
        # 'confidence' = probabilidad de la clase predicha entregada por el
        # clasificador. NO equivale a una probabilidad validada de éxito.
        "features": {f: float(features[f]) for f in FEATURES},
    }