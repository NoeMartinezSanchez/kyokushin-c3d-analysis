# -*- coding: utf-8 -*-
r"""
Tests específicos de Task 8B (baseline no lineal RandomForest).

Ejecutar desde la raíz:
    .venv\Scripts\python -m pytest tests -q
    .venv\Scripts\python -m pytest tests\test_task8b.py -q
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]

ML8B = ROOT / "output" / "ml_results_task8b"
V0 = ROOT / "output" / "ml_dataset_v0" / "ml_dataset_v0.csv"
OOF7 = ROOT / "output" / "ml_results" / "oof_predictions.csv"
FOLD_ASN = ROOT / "output" / "ml_results" / "fold_assignments.csv"

FEAT_B = ["duration_s", "time_to_peak_s", "vmax", "vmean", "amax",
          "displacement", "path_length", "hip_rom", "knee_rom", "ankle_rom"]
CLASSES = ["S02", "S03", "S04", "S05"]

_T19 = None


def _m19():
    global _T19
    if _T19 is None:
        import importlib.util as _ilu
        _spec = _ilu.spec_from_file_location(
            "t8b", str(ROOT / "scripts" / "19_task8b_nonlinear_baseline.py"))
        _T19 = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_T19)
    return _T19


def _oof8b():
    return pd.read_csv(ML8B / "oof_predictions.csv")


def test_8b_dataset_419_rows():
    v0 = pd.read_csv(V0)
    assert len(v0) == 419


def test_8b_ten_features_no_snr():
    fi = pd.read_csv(ML8B / "feature_importance.csv")
    assert set(fi["feature"]) == set(FEAT_B)
    assert "snr" not in set(fi["feature"])


def test_8b_no_nan_no_inf():
    v0 = pd.read_csv(V0)
    assert v0[FEAT_B].isna().sum().sum() == 0
    assert not np.isinf(v0[FEAT_B].to_numpy(dtype=float)).any()


def test_8b_four_classes_and_33_athletes():
    v0 = pd.read_csv(V0)
    assert set(v0["technique"]) == set(CLASSES)
    assert v0["athlete_id"].nunique() == 33


def test_8b_five_folds_and_no_leakage():
    fm = pd.read_csv(ML8B / "fold_metrics.csv")
    assert set(fm["fold"]) == {1, 2, 3, 4, 5}
    assert set(fm["experiment"]) == {"8B"}
    for _, r in fm.iterrows():
        assert r["n_train_athletes"] + r["n_validation_athletes"] == 33


def test_8b_oof_complete_and_unique():
    oof = _oof8b()
    assert len(oof) == 419
    assert oof["execution_id"].is_unique


def test_8b_oof_only_validation_fold():
    oof = _oof8b()
    fa = pd.read_csv(FOLD_ASN)
    fold_of = dict(zip(fa["athlete_id"], fa["fold"]))
    bad = [r["execution_id"] for _, r in oof.iterrows()
           if r["fold"] != fold_of[r["athlete_id"]]]
    assert not bad


def test_8b_classes_and_probs():
    oof = _oof8b()
    assert set(oof["true_technique"]) == set(CLASSES)
    assert set(oof["predicted_technique"]) <= set(CLASSES)
    prob_cols = [c for c in oof.columns if c.startswith("prob_")]
    assert prob_cols
    s = oof[prob_cols].sum(axis=1)
    assert (np.abs(s - 1.0) < 1e-3).all()


def test_8b_feature_importance_has_folds():
    fi = pd.read_csv(ML8B / "feature_importance.csv")
    assert len(fi) == len(FEAT_B)
    assert fi["mean"].notna().all()


def test_8b_global_metrics_range():
    g = pd.read_csv(ML8B / "global_metrics.csv").set_index("metric")
    assert 0.5 < g.loc["accuracy", "mean"] < 0.9
    assert 0.5 < g.loc["f1_macro", "mean"] < 0.9


def test_8b_model_comparison_present():
    mc = pd.read_csv(ML8B / "model_comparison.csv")
    assert set(mc["metric"]) == {"accuracy", "balanced_accuracy",
                                 "precision_macro", "recall_macro",
                                 "f1_macro", "f1_weighted"}
    assert mc["logreg_mean"].notna().all() and mc["rf_mean"].notna().all()


def test_8b_error_comparison_consistent():
    ec = pd.read_csv(ML8B / "error_comparison.csv")
    err7 = int(ec.loc[ec["error_set"] == "task7b_B_errors", "n"].iloc[0])
    err8 = int(ec.loc[ec["error_set"] == "task8b_RF_errors", "n"].iloc[0])
    shared = int(ec["shared"].iloc[0])
    oof = _oof8b()
    assert err8 == int((oof["true_technique"] != oof["predicted_technique"]).sum())
    assert shared <= min(err7, err8)


def test_8b_integrity_pass():
    ih = pd.read_csv(ML8B / "integrity_hashes.csv")
    assert set(ih["status"]) == {"PASS"}
    # recomprobar en tiempo presente: los hashes registrados coinciden con el disco
    for _, r in ih.iterrows():
        p = ROOT / r["file"]
        assert p.exists()
        assert hashlib.md5(p.read_bytes()).hexdigest() == r["md5_before"]


def test_8b_v0_unchanged():
    import json as _json
    cfg = _json.loads((ML8B / "experiment_config.json").read_text(encoding="utf-8"))
    cur = hashlib.md5(V0.read_bytes()).hexdigest()
    assert cur == cfg["ml_dataset_v0_md5"]


def test_8b_determinism_byte_identity():
    """dos ejecuciones producen byte-identidad de los artefactos principales."""
    mod = _m19()
    hashes = []
    for _ in range(2):
        mod.run_task8b()
        hashes.append(hashlib.md5((ML8B / "oof_predictions.csv")
                                  .read_bytes()).hexdigest())
    assert hashes[0] == hashes[1]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))  # noqa: F821