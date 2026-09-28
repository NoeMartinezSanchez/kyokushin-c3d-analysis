# -*- coding: utf-8 -*-
"""
inference — Capa de predicción/inferencia del demo (Task 9).

Independiente del dashboard. Uso:

    from inference import predict_execution, load_model, INFERENCE_VERSION
"""

from inference.predictor import predict_execution
from inference.model_loader import load_model, model_info
from inference.schemas import (INFERENCE_VERSION, MODEL_VERSION, FEATURES,
                               CLASSES, TARGET, InferenceError)

__all__ = ["predict_execution", "load_model", "model_info",
           "INFERENCE_VERSION", "MODEL_VERSION", "FEATURES", "CLASSES",
           "TARGET", "InferenceError"]