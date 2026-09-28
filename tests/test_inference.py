# -*- coding: utf-8 -*-
r"""
Tests de la capa de inferencia (Task 9, demo).

Ejecutar:
    .venv\Scripts\python -m pytest tests\test_inference.py -q
"""

from __future__ import annotations

import hashlib
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from inference import predict_execution, model_info, load_model  # noqa: E402
from inference.schemas import (FEATURES, CLASSES, InferenceError,  # noqa: E402
                               INFERENCE_VERSION, MODEL_VERSION)

V0 = ROOT / "output" / "ml_dataset_v0" / "ml_dataset_v0.csv"
ART = ROOT / "output" / "ml_inference" / "task8b_rf.joblib"
INFO = ROOT / "output" / "ml_inference" / "artifact_info.json"


def _real_features(technique: str) -> dict:
    v0 = pd.read_csv(V0)
    row = v0[v0["technique"] == technique].iloc[0]
    return {f: float(row[f]) for f in FEATURES}


def test_9_artifact_exists():
    assert ART.exists()
    assert INFO.exists()
    inf = model_info()
    assert inf["model_version"] == MODEL_VERSION
    assert inf["inference_version"] == INFERENCE_VERSION


def test_9_artifact_hash_matches():
    inf = model_info()
    cur = hashlib.sha256(ART.read_bytes()).hexdigest()
    assert cur == inf["sha256"]


def test_9_valid_input_produces_result():
    res = predict_execution(_real_features("S04"))
    assert set(res.keys()) == {"predicted_technique", "technique_name",
                               "probabilities", "confidence", "features"}


def test_9_missing_feature_error():
    feats = _real_features("S02")
    feats.pop("vmax")
    try:
        predict_execution(feats)
    except InferenceError:
        return
    raise AssertionError("debería lanzar InferenceError por feature faltante")


def test_9_extra_feature_error():
    feats = _real_features("S02")
    feats["snr"] = 100.0  # snr fuera del contrato del modelo (Exp B)
    try:
        predict_execution(feats)
    except InferenceError:
        return
    raise AssertionError("debería lanzar InferenceError por feature inesperada")


def test_9_nan_error():
    feats = _real_features("S02")
    feats["vmax"] = math.nan
    try:
        predict_execution(feats)
    except InferenceError:
        return
    raise AssertionError("debería lanzar InferenceError por NaN")


def test_9_inf_error():
    feats = _real_features("S02")
    feats["amax"] = math.inf
    try:
        predict_execution(feats)
    except InferenceError:
        return
    raise AssertionError("debería lanzar InferenceError por Inf")


def test_9_probs_sum_one_and_class_valid():
    for t in CLASSES:
        res = predict_execution(_real_features(t))
        s = sum(res["probabilities"].values())
        assert abs(s - 1.0) < 1e-3
        assert res["predicted_technique"] in CLASSES
        assert set(res["probabilities"].keys()) == set(CLASSES)


def test_9_result_has_ten_features():
    res = predict_execution(_real_features("S03"))
    assert set(res["features"].keys()) == set(FEATURES)
    assert len(FEATURES) == 10 and "snr" not in FEATURES


def test_9_deterministic():
    feats = _real_features("S05")
    r1 = predict_execution(feats)
    r2 = predict_execution(feats)
    assert r1["predicted_technique"] == r2["predicted_technique"]
    assert r1 == r2


def test_9_model_loads():
    m = load_model()
    assert hasattr(m, "predict")
    assert set(m.classes_) <= set(CLASSES)
    assert m.n_features_in_ == 10


def test_9_v0_unchanged():
    inf = model_info()
    cur = hashlib.md5(V0.read_bytes()).hexdigest()
    assert cur == inf["ml_dataset_v0_md5"]


def test_9_all_four_real_executions_valid():
    for t in CLASSES:
        res = predict_execution(_real_features(t))
        assert res["confidence"] > 0.0
        assert np.isfinite(list(res["probabilities"].values())).all()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))  # noqa: F821