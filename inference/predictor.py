# -*- coding: utf-8 -*-
"""
predictor.py — Interfaz de inferencia (Task 9, demo).

predict_execution(features) -> {predicted_technique, technique_name,
                                probabilities, confidence, features}

Validación ESTRICTA (sin correcciones silenciosas):
  - exactamente las 10 features del contrato (faltantes -> error; extra -> error)
  - valores numéricos convertibles; NaN / Inf -> error
"""

from __future__ import annotations

import math

import numpy as np

from inference.model_loader import load_model
from inference.schemas import (CLASSES, FEATURES, InferenceError, build_result)


def predict_execution(features: dict) -> dict:
    """Predice la técnica para una ejecución dada por sus 10 features."""
    if not isinstance(features, dict):
        raise InferenceError("entrada debe ser un diccionario de features")

    missing = [f for f in FEATURES if f not in features]
    if missing:
        raise InferenceError(f"features faltantes: {missing}")

    extra = [k for k in features if k not in FEATURES]
    if extra:
        raise InferenceError(f"features inesperadas (fuera del contrato): {extra}")

    feats = {}
    for f in FEATURES:
        try:
            v = float(features[f])
        except (TypeError, ValueError):
            raise InferenceError(f"feature {f} no convertible a número")
        if math.isnan(v):
            raise InferenceError(f"feature {f} es NaN")
        if math.isinf(v):
            raise InferenceError(f"feature {f} es infinito")
        feats[f] = v

    X = np.asarray([[feats[f] for f in FEATURES]], dtype=float)
    model = load_model()
    probs_full = model.predict_proba(X)[0]
    pred = model.predict(X)[0]
    probs = {c: float(probs_full[list(model.classes_).index(c)])
             if c in model.classes_ else 0.0 for c in CLASSES}
    return build_result(str(pred), probs, feats)