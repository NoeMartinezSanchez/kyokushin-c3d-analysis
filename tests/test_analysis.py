# -*- coding: utf-8 -*-
r"""
Tests de la capa de análisis (Task 11): perfil de referencia, comparación y
coach insights del Athlete Performance Dashboard.

Ejecutar:
    .venv\Scripts\python -m pytest tests\test_analysis.py -q
"""

from __future__ import annotations

import hashlib
import math
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dashboard"))
sys.path.insert(0, str(ROOT))

from inference.schemas import FEATURES, CLASSES  # noqa: E402
import inference  # noqa: E402
import performance_analysis as pa  # noqa: E402


def _df():
    return pd.read_csv(ROOT / "output" / "ml_dataset_v0" / "ml_dataset_v0.csv")


def _ref():
    return pa.reference_profile(_df())


def test_11_reference_profile_for_each_technique():
    ref = _ref()
    assert set(ref.keys()) == set(CLASSES)


def test_11_reference_has_ten_features():
    ref = _ref()
    for t in CLASSES:
        assert set(ref[t].keys()) == set(FEATURES)


def test_11_median_q1_q3_computed():
    ref = _ref()
    for t in CLASSES:
        for f in FEATURES:
            row = ref[t][f]
            assert {"median", "q1", "q3"} <= set(row.keys())
            assert row["q1"] <= row["median"] <= row["q3"]


def test_11_compare_to_reference_fields():
    df = _df()
    row = df[df["technique"] == "S04"].iloc[0]
    feats = {f: float(row[f]) for f in FEATURES}
    comp = pa.compare_to_reference(feats, _ref()["S04"])
    assert len(comp) == len(FEATURES)
    it = {c["feature"]: c for c in comp}["hip_rom"]
    assert "athlete" in it and "reference_median" in it
    assert abs(it["difference_abs"] - (it["athlete"] - it["reference_median"])) < 1e-9


def test_11_difference_percentage_formula():
    comp = pa.compare_to_reference({"vmax": 10.0, "duration_s": 1.0,
                                    "time_to_peak_s": 1.0, "vmean": 2.0,
                                    "amax": 3.0, "displacement": 0.1,
                                    "path_length": 1.0, "hip_rom": 100.0,
                                    "knee_rom": 100.0, "ankle_rom": 50.0},
                                   {"vmax": {"median": 8.0, "q1": 7.0, "q3": 9.0},
                                    "duration_s": {"median": 1.0, "q1": .8, "q3": 1.2},
                                    "time_to_peak_s": {"median": .3, "q1": .2, "q3": .4},
                                    "vmean": {"median": 2.0, "q1": 1.5, "q3": 2.5},
                                    "amax": {"median": 3.0, "q1": 2.5, "q3": 3.5},
                                    "displacement": {"median": .1, "q1": .05, "q3": .15},
                                    "path_length": {"median": 1.0, "q1": .8, "q3": 1.2},
                                    "hip_rom": {"median": 100.0, "q1": 90.0, "q3": 110.0},
                                    "knee_rom": {"median": 100.0, "q1": 90.0, "q3": 110.0},
                                    "ankle_rom": {"median": 50.0, "q1": 45.0, "q3": 55.0}})
    it = {c["feature"]: c for c in comp}["vmax"]
    assert it["difference_pct"] == pytest.approx(25.0)  # (10-8)/8*100


def test_11_division_by_zero_protected():
    comp = pa.compare_to_reference({"vmax": 0.0, "duration_s": 1.0,
                                    "time_to_peak_s": 1.0, "vmean": 2.0,
                                    "amax": 3.0, "displacement": 0.1,
                                    "path_length": 1.0, "hip_rom": 100.0,
                                    "knee_rom": 100.0, "ankle_rom": 50.0},
                                   {"vmax": {"median": 0.0, "q1": 0.0, "q3": 0.0},
                                    "duration_s": {"median": 1.0, "q1": .8, "q3": 1.2},
                                    "time_to_peak_s": {"median": .3, "q1": .2, "q3": .4},
                                    "vmean": {"median": 2.0, "q1": 1.5, "q3": 2.5},
                                    "amax": {"median": 3.0, "q1": 2.5, "q3": 3.5},
                                    "displacement": {"median": .1, "q1": .05, "q3": .15},
                                    "path_length": {"median": 1.0, "q1": .8, "q3": 1.2},
                                    "hip_rom": {"median": 100.0, "q1": 90.0, "q3": 110.0},
                                    "knee_rom": {"median": 100.0, "q1": 90.0, "q3": 110.0},
                                    "ankle_rom": {"median": 50.0, "q1": 45.0, "q3": 55.0}})
    it = {c["feature"]: c for c in comp}["vmax"]
    assert it["difference_pct"] is None  # mediana 0 → solo abs


def test_11_iqr_status():
    assert pa.iqr_status(1.2, 0.8, 1.0) == "ABOVE_REFERENCE_RANGE"
    assert pa.iqr_status(0.7, 0.8, 1.0) == "BELOW_REFERENCE_RANGE"
    assert pa.iqr_status(0.9, 0.8, 1.0) == "WITHIN_REFERENCE_RANGE"


def test_11_compare_executions():
    a = {f: 1.0 for f in FEATURES}
    b = {f: 2.0 for f in FEATURES}
    rows = pa.compare_executions(a, b)
    assert len(rows) == len(FEATURES)
    r = rows[0]
    assert r["difference_abs"] == pytest.approx(1.0)
    assert r["difference_pct"] == pytest.approx(100.0)


def test_11_observed_differences():
    df = _df()
    row = df[df["technique"] == "S04"].iloc[0]
    feats = {f: float(row[f]) for f in FEATURES}
    comp = pa.compare_to_reference(feats, _ref()["S04"])
    obs = pa.observed_differences(comp)
    assert len(obs) <= 4
    assert all(o["status"] != "WITHIN_REFERENCE_RANGE" for o in obs)


def test_11_coach_insights_neutral_language():
    insights = pa.coach_insights([{"feature": "hip_rom", "label": "ROM cadera",
                                   "status": "BELOW_REFERENCE_RANGE",
                                   "difference_pct": -8.0, "difference_abs": -9.0}])
    assert insights
    it = insights[0]
    assert "por debajo de" in it["observation"]
    assert "Posible área de observación" in it["coach_review"]
    for banned in ("mala técnica", "necesita ", "causará", "incorrecta"):
        assert banned not in it["observation"] + it["coach_review"]


def test_11_v0_unchanged():
    inf = inference.model_info()
    cur = hashlib.md5((ROOT / "output" / "ml_dataset_v0" /
                       "ml_dataset_v0.csv").read_bytes()).hexdigest()
    assert cur == inf["ml_dataset_v0_md5"]


def test_11_dashboard_still_no_sklearn():
    bad = ["import sklearn", "from sklearn",
           "scripts/02_execution_segmentation",
           "scripts/16_task7b_baseline_ml", "scripts/19_task8b_nonlinear_baseline"]
    for fname in ("app_performance.py", "performance_data.py",
                  "performance_ui.py", "performance_analysis.py"):
        src = (ROOT / "dashboard" / fname).read_text(encoding="utf-8")
        for needle in bad:
            assert needle not in src, f"{fname} contiene: {needle}"


def test_11_inference_layer_not_modified_by_analysis():
    """performance_analysis importa inference sin escribirlo y sigue funcionando."""
    inf_files = [f.name for f in (ROOT / "inference").glob("*.py")]
    assert "__init__.py" in inf_files
    feats = {f: 8.0 for f in FEATURES}
    res = inference.predict_execution(feats)
    assert res["predicted_technique"] in CLASSES  # la carga funciona tras importar análisis


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))  # noqa: F821