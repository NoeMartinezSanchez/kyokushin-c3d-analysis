#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
19_task8b_nonlinear_baseline.py
===============================
TASK 8B — BASELINE NO LINEAL INTERPRETABLE (RandomForestClassifier)
con exactamente la misma validación agrupada por atleta que Task 7B.

  Dataset : output/ml_dataset_v0/ml_dataset_v0.csv (419 × 33 atletas, S02-S05)
  Features: las 10 del Experimento B de Task 7B (SIN snr)
  Target  : technique     Group: athlete_id
  Folds   : reuses output/ml_results/fold_assignments.csv (congelados) y
            valida contra GroupKFold(5) re-derivado sobre v0.
  Modelo  : RandomForestClassifier con parámetros FIJOS (sin tuning).

NO: tuning, Grid/Random/Optuna, XGB, SVM, DL, SMOTE, PCA, feature selection,
     features nuevas, SNR, C3D, modificar v0 / Task 7B / Task 8A / Data Mart /
     dashboard.
Escribe SOLO en output/ml_results_task8b/. Determinista (random_state=42).

Comparación: vs LogisticRegression (Task 7B, Experimento B) y vs Task 8A
(errores por execution_id). Lenguaje descriptivo, sin ranking ni causalidad.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import GroupKFold

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parents[1]
V0_FILE = ROOT / "output" / "ml_dataset_v0" / "ml_dataset_v0.csv"
FOLD_ASN = ROOT / "output" / "ml_results" / "fold_assignments.csv"
OUT_DIR = ROOT / "output" / "ml_results_task8b"
FIG_DIR = OUT_DIR / "figures"

CLASSES = ["S02", "S03", "S04", "S05"]
FEATURES_B = ["duration_s", "time_to_peak_s", "vmax", "vmean", "amax",
              "displacement", "path_length", "hip_rom", "knee_rom",
              "ankle_rom"]
N_SPLITS = 5
RANDOM_STATE = 42

RF_PARAMS = {
    "n_estimators": 300, "max_depth": None, "min_samples_split": 2,
    "min_samples_leaf": 1, "max_features": "sqrt", "bootstrap": True,
    "random_state": RANDOM_STATE, "n_jobs": -1, "class_weight": None,
}

METRICS = ["accuracy", "balanced_accuracy", "precision_macro",
           "recall_macro", "f1_macro", "f1_weighted"]

# protegidos (md5 antes/después)
PROTECTED = [
    "output/ml_dataset_v0/ml_dataset_v0.csv",
    "output/ml_results/oof_predictions.csv",
    "output/ml_results/fold_metrics.csv",
    "output/ml_results/experiment_config.json",
    "output/ml_results/fold_assignments.csv",
    "output/task8a_error_analysis/error_summary.csv",
    "output/task8a_error_analysis/confusion_pairs.csv",
    "scripts/16_task7b_baseline_ml.py",
    "dashboard/app.py",
    "dashboard/app_ml.py",
]


def _md5(rel: str) -> str:
    p = ROOT / rel
    return hashlib.md5(p.read_bytes()).hexdigest() if p.exists() else "MISSING"


def _metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro",
                                           zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro",
                                     zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted",
                                zero_division=0),
    }


def run_task8b():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    before = {rel: _md5(rel) for rel in PROTECTED}

    v0 = pd.read_csv(V0_FILE)
    assert len(v0) == 419
    assert set(v0["technique"]) == set(CLASSES)
    assert v0["execution_id"].is_unique
    assert v0[FEATURES_B].isna().sum().sum() == 0
    assert not np.isinf(v0[FEATURES_B].to_numpy(dtype=float)).any()

    # ---------- folds congelados (validar GroupKFold == fold_assignments) ----------
    fa = pd.read_csv(FOLD_ASN)
    fold_of = dict(zip(fa["athlete_id"], fa["fold"]))
    assert len(fold_of) == v0["athlete_id"].nunique()
    gkf = GroupKFold(n_splits=N_SPLITS)
    derived = {}
    for _, (tr_idx, va_idx) in enumerate(gkf.split(v0, v0["technique"],
                                                   v0["athlete_id"]), start=1):
        pass
    for k, (tr_idx, va_idx) in enumerate(gkf.split(v0, v0["technique"],
                                                   v0["athlete_id"]), start=1):
        va_ath = set(v0["athlete_id"].iloc[va_idx])
        for a in va_ath:
            assert a not in derived, f"atleta {a} en >1 fold (derivado)"
            derived[a] = k
    assert derived == fold_of, "fold_assignments no coincide con GroupKFold"

    y = v0["technique"].to_numpy()
    groups = v0["athlete_id"].to_numpy()
    X = v0[FEATURES_B].to_numpy(dtype=float)

    # ---------- OOF por fold (atleta -> fold congelado) ----------
    oof_rows, fm_rows, fi_rows = [], [], []
    for k in range(1, N_SPLITS + 1):
        test_mask = v0["athlete_id"].map(lambda a: fold_of[a] == k).to_numpy()
        train_mask = ~test_mask
        tr_ath = set(groups[train_mask]); va_ath = set(groups[test_mask])
        assert tr_ath.isdisjoint(va_ath), f"leakage fold {k}"
        rfc = RandomForestClassifier(**RF_PARAMS)
        rfc.fit(X[train_mask], y[train_mask])
        pred = rfc.predict(X[test_mask])
        proba = rfc.predict_proba(X[test_mask])
        prob_cols = {c: proba[:, list(rfc.classes_).index(c)] for c in CLASSES}

        fm_rows.append({
            "experiment": "8B", "fold": k,
            "n_train": int(train_mask.sum()), "n_validation": int(test_mask.sum()),
            "n_train_athletes": len(tr_ath), "n_validation_athletes": len(va_ath),
            **{m: round(_metrics(y[test_mask], pred)[m], 4) for m in METRICS},
        })
        for f, imp in zip(FEATURES_B, rfc.feature_importances_):
            fi_rows.append({"experiment": "8B", "fold": k, "feature": f,
                            "importance": round(float(imp), 6)})
        pos = test_mask.nonzero()[0]
        for j, idx in enumerate(pos):
            eid = v0["execution_id"].iloc[idx]
            pr = [float(prob_cols[c][j]) for c in CLASSES]
            oof_rows.append({
                "experiment": "8B", "execution_id": eid,
                "athlete_id": v0["athlete_id"].iloc[idx],
                "true_technique": y[idx], "predicted_technique": pred[j],
                "fold": k, **{f"prob_{c}": round(float(prob_cols[c][j]), 4)
                              for c in CLASSES}})

    oof = pd.DataFrame(oof_rows)
    assert len(oof) == len(v0)
    assert oof["execution_id"].is_unique
    oof["correct"] = oof["true_technique"] == oof["predicted_technique"]

    fm_df = pd.DataFrame(fm_rows)
    fi_df = pd.DataFrame(fi_rows)

    # ---------- métricas globales ----------
    glo_rows = []
    for m in METRICS:
        v = fm_df[m].astype(float)
        glo_rows.append({"metric": m, "mean": round(v.mean(), 4),
                         "std": round(v.std(), 4), "min": round(v.min(), 4),
                         "max": round(v.max(), 4)})
    glo_df = pd.DataFrame(glo_rows)

    # ---------- por técnica (OOF pooled) ----------
    mt_rows = []
    for c in CLASSES:
        tp = int(((oof["true_technique"] == c) & (oof["predicted_technique"] == c)).sum())
        fp = int(((oof["true_technique"] != c) & (oof["predicted_technique"] == c)).sum())
        fn = int(((oof["true_technique"] == c) & (oof["predicted_technique"] != c)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        mt_rows.append({"technique": c, "precision": round(prec, 4),
                        "recall": round(rec, 4), "f1": round(f1, 4),
                        "support": int((oof["true_technique"] == c).sum())})
    mt_df = pd.DataFrame(mt_rows)

    # ---------- confusión ----------
    cm_rows = []
    for t in CLASSES:
        tot = int((oof["true_technique"] == t).sum())
        for p in CLASSES:
            cnt = int(((oof["true_technique"] == t) &
                       (oof["predicted_technique"] == p)).sum())
            cm_rows.append({"true": t, "predicted": p, "count": cnt,
                            "normalized_recall": round(cnt / tot, 4)
                            if tot else 0.0})
    cm_df = pd.DataFrame(cm_rows)

    # ---------- feature importance (media±std) ----------
    fis_rows = []
    for f in FEATURES_B:
        vals = fi_df.loc[fi_df["feature"] == f, "importance"].to_numpy(dtype=float)
        fis_rows.append({"feature": f, "mean": round(float(vals.mean()), 6),
                         "std": round(float(vals.std()), 6),
                         "min": round(float(vals.min()), 6),
                         "max": round(float(vals.max()), 6)})
    fis_df = pd.DataFrame(fis_rows)

    # ---------- por atleta ----------
    ath_rows = []
    for aid in sorted(oof["athlete_id"].unique()):
        d = oof[oof["athlete_id"] == aid]
        ath_rows.append({"athlete_id": aid, "n": int(len(d)),
                         "correct": int(d["correct"].sum()),
                         "errors": int(len(d) - d["correct"].sum()),
                         "accuracy": round(d["correct"].mean(), 4),
                         "techniques_present":
                         ",".join(sorted(d["true_technique"].unique()))})
    ath_df = pd.DataFrame(ath_rows)

    # ---------- comparación de errores vs Task 7B (exp B) y 8A ----------
    oof7b = pd.read_csv(ROOT / "output" / "ml_results" / "oof_predictions.csv")
    b7 = oof7b[oof7b["experiment"] == "B"]
    err7 = set(b7.loc[b7["true_technique"] != b7["predicted_technique"],
                      "execution_id"])
    err8 = set(oof.loc[~oof["correct"], "execution_id"])
    err_df = pd.DataFrame([{
        "error_set": "task7b_B_errors", "n": len(err7),
        "shared": len(err7 & err8),
        "corrected_by_8b": len(err7 - err8),
        "new_in_8b": len(err8 - err7),
    }, {
        "error_set": "task8b_RF_errors", "n": len(err8),
        "shared": len(err7 & err8),
        "corrected_by_8b": len(err7 - err8),
        "new_in_8b": len(err8 - err7),
    }])

    # persistencia del mismo par (7B-B vs 8B)
    pair7 = set(zip(b7.loc[b7["true_technique"] != b7["predicted_technique"],
                           "true_technique"],
                    b7.loc[b7["true_technique"] != b7["predicted_technique"],
                           "predicted_technique"]))
    e8 = oof.loc[~oof["correct"]]
    pair8 = set(zip(e8["true_technique"], e8["predicted_technique"]))
    pair_df = pd.DataFrame([{
        "pair_group": "same_pair_persist",
        "n_7b": len(pair7), "n_8b": len(pair8),
        "n_both": len(pair7 & pair8)}])

    # ---------- comparación de modelos (7B-B vs 8B) ----------
    bc = pd.read_csv(ROOT / "output" / "ml_results" / "baseline_comparison.csv")
    b7c = bc[(bc["experiment"] == "B")].set_index("metric")
    mc_rows = []
    for m in METRICS:
        mc_rows.append({
            "metric": m,
            "logreg_mean": b7c.loc[m, "mean"], "logreg_std": b7c.loc[m, "std"],
            "logreg_min": b7c.loc[m, "min"], "logreg_max": b7c.loc[m, "max"],
            "rf_mean": glo_df.loc[glo_df["metric"] == m, "mean"].iloc[0],
            "rf_std": glo_df.loc[glo_df["metric"] == m, "std"].iloc[0],
            "rf_min": glo_df.loc[glo_df["metric"] == m, "min"].iloc[0],
            "rf_max": glo_df.loc[glo_df["metric"] == m, "max"].iloc[0],
        })
    mc_df = pd.DataFrame(mc_rows)

    # ---------- config + hashes ----------
    config = {
        "experiment_id": "TASK8B_NONLINEAR_002",
        "model": "RandomForestClassifier",
        "params": RF_PARAMS,
        "dataset": "ML Dataset v0",
        "dataset_file": "output/ml_dataset_v0/ml_dataset_v0.csv",
        "ml_dataset_v0_md5": before["output/ml_dataset_v0/ml_dataset_v0.csv"],
        "features": FEATURES_B,
        "target": "technique", "group": "athlete_id",
        "cv": {"name": "GroupKFold", "n_splits": N_SPLITS,
               "folds_source": "output/ml_results/fold_assignments.csv"},
        "n_splits": N_SPLITS, "random_state": RANDOM_STATE,
        "experiment_version": "002",
        "population": {"rows": int(len(v0)), "athletes": int(v0["athlete_id"].nunique()),
                       "techniques": sorted(v0["technique"].unique()),
                       "sampling_rate_hz": 250, "condition": "E01", "trial": "T01"},
    }
    (OUT_DIR / "experiment_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---------- integridad después ----------
    after = {rel: _md5(rel) for rel in PROTECTED}
    changed = [rel for rel in PROTECTED if before[rel] != after[rel]]
    integ = pd.DataFrame([{"file": rel, "md5_before": before[rel],
                           "md5_after": after[rel],
                           "status": "PASS" if before[rel] == after[rel] else "FAIL"}
                          for rel in PROTECTED])
    if changed:
        raise RuntimeError(f"archivos protegidos modificados en 8B: {changed}")
    integ.to_csv(OUT_DIR / "integrity_hashes.csv", index=False, encoding="utf-8")

    # ---------- escritura ----------
    oof.to_csv(OUT_DIR / "oof_predictions.csv", index=False, encoding="utf-8")
    fm_df.to_csv(OUT_DIR / "fold_metrics.csv", index=False, encoding="utf-8")
    glo_df.to_csv(OUT_DIR / "global_metrics.csv", index=False, encoding="utf-8")
    mt_df.to_csv(OUT_DIR / "metrics_by_technique.csv", index=False, encoding="utf-8")
    cm_df.to_csv(OUT_DIR / "confusion_matrix.csv", index=False, encoding="utf-8")
    fis_df.to_csv(OUT_DIR / "feature_importance.csv", index=False, encoding="utf-8")
    ath_df.to_csv(OUT_DIR / "athlete_metrics.csv", index=False, encoding="utf-8")
    err_df.to_csv(OUT_DIR / "error_comparison.csv", index=False, encoding="utf-8")
    pair_df.to_csv(OUT_DIR / "error_comparison_pairs.csv", index=False,
                   encoding="utf-8")
    mc_df.to_csv(OUT_DIR / "model_comparison.csv", index=False, encoding="utf-8")

    _plot_figures(cm_df, mc_df, mt_df, fis_df, oof)
    return {"oof": oof, "global": glo_df, "mc": mc_df, "err": err_df}


# --------------------------------------------------------------------------- #
# Figuras
# --------------------------------------------------------------------------- #

def _plot_figures(cm_df, mc_df, mt_df, fis_df, oof):
    # 1) matriz de confusión RF
    mat = np.zeros((4, 4))
    for _, r in cm_df.iterrows():
        mat[CLASSES.index(r["true"]), CLASSES.index(r["predicted"])] = r["count"]
    fig, ax = plt.subplots(figsize=(6.5, 5))
    im = ax.imshow(mat, cmap="Greens")
    ax.set_xticks(range(4)); ax.set_xticklabels(CLASSES, rotation=45)
    ax.set_yticks(range(4)); ax.set_yticklabels(CLASSES)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, int(mat[i, j]), ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title("Matriz de confusión OOF — Random Forest (8B)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "confusion_matrix_random_forest.png", dpi=130,
                bbox_inches="tight")
    plt.close(fig)

    # 2) comparación de métricas 7B vs 8B
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(mc_df))
    w = 0.35
    ax.bar(x - w / 2, mc_df["logreg_mean"], w, yerr=mc_df["logreg_std"],
           capsize=3, label="LogisticRegression (7B)", color="#3a7ca5")
    ax.bar(x + w / 2, mc_df["rf_mean"], w, yerr=mc_df["rf_std"], capsize=3,
           label="RandomForest (8B)", color="#e0a83a")
    ax.set_xticks(x); ax.set_xticklabels(mc_df["metric"], rotation=30, fontsize=8)
    ax.set_ylabel("media ± desviación (OOF, 5 folds)")
    ax.set_title("Comparación de métricas 7B vs 8B (media ± std)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "metrics_comparison_7b_8b.png", dpi=130,
                bbox_inches="tight")
    plt.close(fig)

    # 3) métricas por técnica (7B-B vs 8B)
    oof7 = pd.read_csv(ROOT / "output" / "ml_results" / "oof_predictions.csv")
    b7 = oof7[oof7["experiment"] == "B"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    for ax, (df_, name) in zip(axes, ((b7, "LogReg (7B)"), (oof, "RF (8B)"))):
        rows = []
        for c in CLASSES:
            sub = df_[df_["true_technique"] == c]
            ok = (sub["true_technique"] == sub["predicted_technique"]).sum()
            true_c, pred_c = ok, int((sub["predicted_technique"] == c).sum())
            prec = ok / pred_c if pred_c else 0
            rec = ok / len(sub) if len(sub) else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
            rows.append({"t": c, "precision": prec, "recall": rec, "f1": f1})
        rr = pd.DataFrame(rows)
        x = np.arange(4); w = 0.25
        for j, m in enumerate(["precision", "recall", "f1"]):
            ax.bar(x + (j - 1) * w, rr[m], width=w, label=m)
        ax.set_xticks(x); ax.set_xticklabels(CLASSES)
        ax.set_ylim(0, 1); ax.legend(fontsize=7)
        ax.set_title(name)
    fig.suptitle("Métricas por técnica (OOF pooled) — comparación")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(FIG_DIR / "metrics_by_technique_comparison.png", dpi=130,
                bbox_inches="tight")
    plt.close(fig)

    # 4) feature importance
    sub = fis_df.sort_values("mean", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.barh(sub["feature"], sub["mean"], xerr=sub["std"], capsize=3, color="#5b8db8")
    ax.set_title("Random Forest — feature importance (media ± std entre folds)")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "feature_importance_random_forest.png", dpi=130,
                bbox_inches="tight")
    plt.close(fig)

    # 5) pares de confusión 7B vs 8B
    def pair_counts(df_):
        e = df_[df_["true_technique"] != df_["predicted_technique"]]
        return e.groupby(["true_technique", "predicted_technique"]).size()
    p7 = pair_counts(b7)
    p8 = pair_counts(oof)
    labels = sorted(set(list(p7.index)) | set(list(p8.index)))
    lab = [f"{t}→{p}" for t, p in labels]
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    w = 0.4
    xs = np.arange(len(labels))
    ax.bar(xs - w / 2, [p7.get(k, 0) for k in labels], w, label="LogReg (7B)",
           color="#3a7ca5")
    ax.bar(xs + w / 2, [p8.get(k, 0) for k in labels], w, label="RF (8B)",
           color="#e0a83a")
    ax.set_xticks(xs); ax.set_xticklabels(lab, rotation=60, fontsize=7)
    ax.set_ylabel("n (errores OOF)")
    ax.set_title("Pares de confusión: 7B vs 8B")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "confusion_pairs_comparison.png", dpi=130,
                bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    res = run_task8b()
    print("[8B] RandomForest (parámetros fijos) — OOF")
    print(res["global"].to_string(index=False))
    print()
    print("Comparación 7B vs 8B:")
    print(res["mc"][["metric", "logreg_mean", "rf_mean", "logreg_std", "rf_std"]]
          .to_string(index=False))
    print()
    print("Errores:", res["err"].to_dict("records"))
    print(f"[8B] Guardado en {OUT_DIR}")


if __name__ == "__main__":
    main()