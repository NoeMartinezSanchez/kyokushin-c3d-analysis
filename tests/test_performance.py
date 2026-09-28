# -*- coding: utf-8 -*-
r"""
Tests del Athlete Performance Dashboard (Task 10, demo).

Ejecutar:
    .venv\Scripts\python -m pytest tests\test_performance.py -q
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dashboard"))
sys.path.insert(0, str(ROOT))

from inference.schemas import FEATURES, CLASSES  # noqa: E402
import inference  # noqa: E402


def _data():
    import performance_data as d
    return d


def _ui():
    import performance_ui as u
    return u


def test_10_dashboard_modules_import():
    d = _data()
    assert callable(d.load_v0)
    u = _ui()
    assert len(u.GROUPS) == 4


def test_10_dataset_loads():
    d = _data()
    df = d.load_v0()
    assert len(df) == 419
    assert {"athlete_id", "execution_id", "technique"}.issubset(df.columns)


def test_10_execution_selection():
    d = _data()
    athletes = d.available_athletes()
    assert len(athletes) == 33
    execs = d.executions_for(athletes[0])
    assert execs
    row = d.execution_row(execs[0])
    assert row["execution_id"] == execs[0]


def test_10_features_extracted():
    d = _data()
    row = d.execution_row(d.default_execution())
    feats = d.features_of(row)
    assert set(feats.keys()) == set(FEATURES)
    assert all(np.isfinite(v) for v in feats.values())


def test_10_predict_execution_receives_features():
    d = _data()
    row = d.execution_row(d.default_execution())
    res = inference.predict_execution(d.features_of(row))
    assert "predicted_technique" in res


def test_10_result_complete():
    d = _data()
    res = inference.predict_execution(d.features_of(
        d.execution_row(d.default_execution())))
    assert set(res.keys()) == {"predicted_technique", "technique_name",
                               "probabilities", "confidence", "features"}


def test_10_probabilities_s02_s05():
    d = _data()
    res = inference.predict_execution(d.features_of(
        d.execution_row(d.default_execution())))
    assert set(res["probabilities"].keys()) == set(CLASSES)
    assert abs(sum(res["probabilities"].values()) - 1.0) < 1e-3


def test_10_confidence_in_range():
    d = _data()
    res = inference.predict_execution(d.features_of(
        d.execution_row(d.default_execution())))
    assert 0.0 <= res["confidence"] <= 1.0


def test_10_profile_has_ten_features():
    d = _data()
    res = inference.predict_execution(d.features_of(
        d.execution_row(d.default_execution())))
    assert set(res["features"].keys()) == set(FEATURES)


def test_10_initial_execution_valid():
    d = _data()
    eid = d.default_execution()
    row = d.execution_row(eid)
    assert row["technique"] == "S04"
    res = inference.predict_execution(d.features_of(row))
    assert res["predicted_technique"] in CLASSES


def test_10_dashboard_no_sklearn_no_training_imports():
    """el dashboard NO importa sklearn ni scripts de entrenamiento."""
    bad = ["import sklearn", "from sklearn",
           "scripts/02_execution_segmentation",
           "scripts/16_task7b_baseline_ml",
           "scripts/19_task8b_nonlinear_baseline"]
    for fname in ("app_performance.py", "performance_data.py",
                  "performance_ui.py"):
        src = (ROOT / "dashboard" / fname).read_text(encoding="utf-8")
        for needle in bad:
            assert needle not in src, f"{fname} contiene: {needle}"
    # la única entrada al modelo debe ser inference
    app_src = (ROOT / "dashboard" / "app_performance.py").read_text(encoding="utf-8")
    assert "from inference import predict_execution" in app_src


def test_10_v0_integrity():
    """v0 no cambia (md5 == el registrado en el artefacto de inferencia)."""
    inf = inference.model_info()
    md5 = inf["ml_dataset_v0_md5"]
    cur = hashlib.md5((ROOT / "output" / "ml_dataset_v0" /
                       "ml_dataset_v0.csv").read_bytes()).hexdigest()
    assert cur == md5


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))  # noqa: F821