#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
20_build_inference_artifact.py
==============================
Task 9 — Materializa el artefacto DEMOSTRATIVO de inferencia.

Ajusta UNA VEZ un RandomForestClassifier sobre el ML Dataset v0 COMPLETO
(419 ejecuciones) con los MISMOS parámetros fijos de Task 8B y lo serializa
en output/ml_inference/task8b_rf.joblib + artifact_info.json.

IMPORTANTE (documentación del demo):
  - Este modelo es un DEMOSTRADOR de capacidad de inferencia, NO un estimador
    de generalización. Sus métricas internas no deben usarse como evidencia;
    las métricas oficiales quedan en las predicciones OOF de Task 8B.
  - No es tuning, no es búsqueda; usa exactamente los parámetros de 8B.
  - No modifica: ml_dataset_v0, ml_results*, ml_results_task8b*, Data Mart,
    segmentación, configs ni dashboard.

Uso:
  .venv\\Scripts\\python scripts\\20_build_inference_artifact.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parents[1]
V0_FILE = ROOT / "output" / "ml_dataset_v0" / "ml_dataset_v0.csv"
OUT_DIR = ROOT / "output" / "ml_inference"
ARTIFACT = OUT_DIR / "task8b_rf.joblib"
INFO = OUT_DIR / "artifact_info.json"

sys.path.insert(0, str(ROOT))
from inference.schemas import FEATURES, CLASSES, TARGET, INFERENCE_VERSION, MODEL_VERSION  # noqa: E402

RF_PARAMS = {
    "n_estimators": 300, "max_depth": None, "min_samples_split": 2,
    "min_samples_leaf": 1, "max_features": "sqrt", "bootstrap": True,
    "random_state": 42, "n_jobs": -1, "class_weight": None,
}


def main() -> None:
    v0 = pd.read_csv(V0_FILE)
    assert len(v0) == 419
    assert set(v0[TARGET]) == set(CLASSES)
    assert v0[FEATURES].isna().sum().sum() == 0

    X = v0[FEATURES].to_numpy(dtype=float)
    y = v0[TARGET].to_numpy()

    rfc = RandomForestClassifier(**RF_PARAMS)
    rfc.fit(X, y)  # ajuste DEMOSTRATIVO sobre el dataset completo (no validación)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(rfc, ARTIFACT)
    data = ARTIFACT.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    info = {
        "model_version": MODEL_VERSION,
        "inference_version": INFERENCE_VERSION,
        "predictor": "RandomForestClassifier",
        "params": RF_PARAMS,
        "dataset": "ML Dataset v0",
        "dataset_file": "output/ml_dataset_v0/ml_dataset_v0.csv",
        "ml_dataset_v0_md5": hashlib.md5(V0_FILE.read_bytes()).hexdigest(),
        "population": {"rows": int(len(v0)),
                       "athletes": int(v0["athlete_id"].nunique()),
                       "techniques": sorted(v0[TARGET].unique()),
                       "sampling_rate_hz": 250, "condition": "E01",
                       "trial": "T01"},
        "features": FEATURES, "target": TARGET, "classes": CLASSES,
        "artifact_path": str(ARTIFACT.relative_to(ROOT)),
        "size_bytes": len(data), "sha256": sha,
        "created": date.today().isoformat(),
        "note": ("MODELO DEMOSTRATIVO: ajustado sobre el dataset completo para "
                 "inferencia; NO estima generalización (métricas oficiales en "
                 "predicciones OOF de Task 8B). No es un modelo de producción."),
    }
    INFO.write_text(json.dumps(info, indent=2, ensure_ascii=False),
                    encoding="utf-8")
    print(f"[T9] Artefacto: {ARTIFACT}")
    print(f"[T9] sha256: {sha}")
    print(f"[T9] tamaño: {len(data)} bytes | features: {len(FEATURES)}")


if __name__ == "__main__":
    main()