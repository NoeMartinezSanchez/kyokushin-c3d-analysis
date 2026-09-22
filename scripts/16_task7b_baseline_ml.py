#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
16_task7b_baseline_ml.py
========================
TASK 7B — BASELINE ML + EVALUACIÓN POR ATLETA
Clasificación de técnica (S02-S05) a partir de features biomecánicas del
ML Dataset v0, con validación agrupada por atleta.

  X       = features biomecánicas del manifest de Task 7
  y       = technique (S02/S03/S04/S05)
  groups  = athlete_id
  CV      = GroupKFold(n_splits=5)   (los MISMOS folds para A y B)
  Modelo  = LogisticRegression (lbfgs, default multinomial, max_iter=2000, rng=0)
  Scale   = StandardScaler ajustado SOLO dentro de cada fold (Pipeline)

Experimentos PAR:
  A = 11 features (incluye snr)
  B = 10 features (excluye snr)

NO: tuning, SMOTE, PCA, clustering, eliminación de outliers, DL.
NO: modifica ml_dataset_v0.csv / Data Mart / segmentación / configs.
Escribe SOLO en output/ml_results/. Determinismo: sin RNG no controlado.

Uso:
  .venv\\Scripts\\python scripts\\16_task7b_baseline_ml.py
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

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parents[1]
V0_FILE = ROOT / "output" / "ml_dataset_v0" / "ml_dataset_v0.csv"
OUT_DIR = ROOT / "output" / "ml_results"
FIG_DIR = OUT_DIR / "figures"
LOG_DIR = OUT_DIR / "logs"

CLASSES = ["S02", "S03", "S04", "S05"]
FEATURES_ALL = ["duration_s", "time_to_peak_s", "vmax", "vmean", "amax",
                "displacement", "path_length", "hip_rom", "knee_rom",
                "ankle_rom", "snr"]
FEATURES_NO_SNR = [f for f in FEATURES_ALL if f != "snr"]

N_FOLDS = 5
RANDOM_STATE = 0
EXPERIMENT_ID = "TASK7B_BASELINE_001"

METRICS_GLOBAL = ["accuracy", "balanced_accuracy", "precision_macro",
                  "recall_macro", "f1_macro", "f1_weighted"]


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


def _per_class(y_true, y_pred):
    rows = []
    for c in CLASSES:
        tp = int(((y_true == c) & (y_pred == c)).sum())
        fp = int(((y_true != c) & (y_pred == c)).sum())
        fn = int(((y_true == c) & (y_pred != c)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        rows.append({"technique": c, "precision": round(prec, 4),
                     "recall": round(rec, 4), "f1": round(f1, 4),
                     "support": int((y_true == c).sum())})
    return rows


def run_task7b():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    v0 = pd.read_csv(V0_FILE)
    v0_md5 = hashlib.md5(V0_FILE.read_bytes()).hexdigest()
    assert set(v0["technique"]) == set(CLASSES)
    for f in FEATURES_ALL:
        assert f in v0.columns
    assert v0["execution_id"].is_unique

    y = v0["technique"].to_numpy()
    groups = v0["athlete_id"].to_numpy()

    # ---------- folds (una sola vez) ----------
    gkf = GroupKFold(n_splits=N_FOLDS)
    splits = list(gkf.split(v0, y, groups))
    assert len(splits) == N_FOLDS
    fold_of = {}
    for k, (tr_idx, va_idx) in enumerate(splits, start=1):
        va_ath = set(v0["athlete_id"].iloc[va_idx])
        for a in va_ath:
            assert a not in fold_of, f"atleta {a} en >1 fold"
            fold_of[a] = k
    assert len(fold_of) == v0["athlete_id"].nunique()  # todos asignados
    fold_assign = pd.DataFrame([{"athlete_id": a, "fold": k}
                                for a, k in sorted(fold_of.items())])

    # ---------- experiments ----------
    experiments = {
        "A": {"features": FEATURES_ALL, "note": "con_snr"},
        "B": {"features": FEATURES_NO_SNR, "note": "sin_snr"},
    }

    oof_all, fm_rows, ath_rows, err_rows = [], [], [], []
    conf_rows, coef_rows = [], []
    coef_sum = {exp: {f: [] for f in ex["features"]} for exp, ex in experiments.items()}

    for exp, ex in experiments.items():
        feats = ex["features"]
        X = v0[feats].to_numpy(dtype=float)
        for k, (tr_idx, va_idx) in enumerate(splits, start=1):
            # integridad de grupos
            tr_ath = set(groups[tr_idx])
            va_ath = set(groups[va_idx])
            assert tr_ath.isdisjoint(va_ath), f"leakage fold {k} ({exp})"
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(solver="lbfgs", max_iter=2000,
                                           random_state=RANDOM_STATE)),
            ])
            pipe.fit(X[tr_idx], y[tr_idx])
            pred = pipe.predict(X[va_idx])
            proba = pipe.predict_proba(X[va_idx])
            prob_cols = {c: proba[:, list(pipe.classes_).index(c)]
                         for c in CLASSES if c in pipe.classes_}

            m = _metrics(y[va_idx], pred)
            fm_rows.append({"experiment": exp, "fold": k,
                            "n_train": int(len(tr_idx)),
                            "n_validation": int(len(va_idx)),
                            "n_train_athletes": len(tr_ath),
                            "n_validation_athletes": len(va_ath),
                            **{key: round(m[key], 4) for key in METRICS_GLOBAL}})

            for row in _per_class(y[va_idx], pred):
                pass  # per-class pooling se hace sobre OOF (no por fold)

            clf = pipe.named_steps["clf"]
            for j, c in enumerate(CLASSES):
                for i, f in enumerate(feats):
                    coef = float(clf.coef_[LIST_POS(j, CLASSES, pipe.classes_),
                                            i]) if f in feats else 0.0
                    coef_rows.append({"experiment": exp, "fold": k, "class": c,
                                      "feature": f, "coefficient": round(coef, 5),
                                      "abs_coefficient": round(abs(coef), 5)})
                    coef_sum[exp][f].append(abs(coef))

            # OOF
            for pos, idx in enumerate(va_idx):
                eid = v0["execution_id"].iloc[idx]
                true_t = y[idx]
                oof_all.append({"experiment": exp, "execution_id": eid,
                                "athlete_id": v0["athlete_id"].iloc[idx],
                                "true_technique": true_t,
                                "predicted_technique": pred[pos], "fold": k,
                                **{f"prob_{c}": round(float(prob_cols[c][pos]), 4)
                                   for c in CLASSES}})
                if true_t != pred[pos]:
                    err_rows.append({
                        "experiment": exp, "execution_id": eid,
                        "athlete_id": v0["athlete_id"].iloc[idx],
                        "true_technique": true_t,
                        "predicted_technique": pred[pos], "fold": k,
                        "probability_predicted": round(float(prob_cols[pred[pos]][pos]), 4),
                        "probability_true": round(float(prob_cols[true_t][pos]), 4),
                    })

    # OOF integrity
    oof_df = pd.DataFrame(oof_all)
    for exp in experiments:
        sub = oof_df[oof_df["experiment"] == exp]
        assert len(sub) == len(v0), f"OOF incompletas en {exp}"
        assert sub["execution_id"].is_unique, f"OOF duplicadas en {exp}"
        # las OOF provienen solo de datos que el modelo no entrenó
        assert (sub["execution_id"].map(
            lambda e: fold_of[v0.loc[v0["execution_id"] == e, "athlete_id"].iloc[0]] ==
            sub.loc[sub["execution_id"] == e, "fold"].iloc[0])).all()

    fm_df = pd.DataFrame(fm_rows)
    coef_df = pd.DataFrame(coef_rows)
    err_df = pd.DataFrame(err_rows)

    # per-class sobre OOF pooled (por experimento, agrupando folds)
    per_class_rows = []
    for exp in experiments:
        sub = oof_df[oof_df["experiment"] == exp]
        for row in _per_class(sub["true_technique"].to_numpy(),
                              sub["predicted_technique"].to_numpy()):
            per_class_rows.append({"experiment": exp, **row})
    per_class_df = pd.DataFrame(per_class_rows)

    # per-athlete (OOF pooled)
    for exp in experiments:
        sub = oof_df[oof_df["experiment"] == exp]
        for aid, g in sub.groupby("athlete_id"):
            n = len(g)
            ok = int((g["true_technique"] == g["predicted_technique"]).sum())
            ath_rows.append({"experiment": exp, "athlete_id": aid,
                             "total": n, "correct": ok, "incorrect": n - ok,
                             "accuracy": round(ok / n, 4),
                             "techniques_present":
                             ",".join(sorted(g["true_technique"].unique()))})
    ath_df = pd.DataFrame(ath_rows)

    # confusion + error matrix
    for exp in experiments:
        sub = oof_df[oof_df["experiment"] == exp]
        for t in CLASSES:
            tot = (sub["true_technique"] == t).sum()
            for p in CLASSES:
                cnt = int(((sub["true_technique"] == t) &
                           (sub["predicted_technique"] == p)).sum())
                conf_rows.append({"experiment": exp, "true": t, "predicted": p,
                                  "count": cnt,
                                  "normalized_recall": round(cnt / tot, 4)
                                  if tot else 0.0})
    conf_df = pd.DataFrame(conf_rows)

    for exp in experiments:
        sub = err_df[err_df["experiment"] == exp]
        for t in CLASSES:
            for p in CLASSES:
                cnt = int(((sub["true_technique"] == t) &
                           (sub["predicted_technique"] == p)).sum())
                if cnt:
                    error_matrix_rows.append({"experiment": exp, "true": t,
                                              "predicted": p, "count": cnt})
    error_matrix_df = pd.DataFrame(error_matrix_rows)

    # feature coefficients summary (abs over folds×classes)
    coef_sum_rows = []
    for exp in experiments:
        for f in experiments[exp]["features"]:
            vals = np.asarray(coef_sum[exp][f])
            coef_sum_rows.append({
                "experiment": exp, "feature": f,
                "mean_abs_coefficient": round(float(vals.mean()), 5),
                "std_abs_coefficient": round(float(vals.std()), 5),
                "min_abs_coefficient": round(float(vals.min()), 5),
                "max_abs_coefficient": round(float(vals.max()), 5)})
    coef_sum_df = pd.DataFrame(coef_sum_rows)

    # baseline comparison
    cmp_rows = []
    for exp in experiments:
        sub = fm_df[fm_df["experiment"] == exp]
        for met in METRICS_GLOBAL:
            v = sub[met].astype(float)
            cmp_rows.append({"experiment": exp, "metric": met,
                             "mean": round(v.mean(), 4), "std": round(v.std(), 4),
                             "min": round(v.min(), 4), "max": round(v.max(), 4)})
    cmp_df = pd.DataFrame(cmp_rows)

    # ---------- escritura ----------
    fold_assign.to_csv(OUT_DIR / "fold_assignments.csv", index=False,
                       encoding="utf-8")
    fm_df.to_csv(OUT_DIR / "fold_metrics.csv", index=False, encoding="utf-8")
    cmp_df.to_csv(OUT_DIR / "baseline_comparison.csv", index=False, encoding="utf-8")
    oof_df.sort_values(["experiment", "athlete_id", "execution_id"]).to_csv(
        OUT_DIR / "oof_predictions.csv", index=False, encoding="utf-8")
    err_df.sort_values(["experiment", "athlete_id"]).to_csv(
        OUT_DIR / "classification_errors.csv", index=False, encoding="utf-8")
    per_class_df.to_csv(OUT_DIR / "metrics_by_class.csv", index=False,
                        encoding="utf-8")
    ath_df.to_csv(OUT_DIR / "metrics_by_athlete.csv", index=False,
                  encoding="utf-8")
    conf_df.to_csv(OUT_DIR / "confusion_matrix.csv", index=False, encoding="utf-8")
    error_matrix_df.to_csv(OUT_DIR / "error_matrix.csv", index=False,
                           encoding="utf-8")
    coef_df.to_csv(OUT_DIR / "feature_coefficients.csv", index=False,
                   encoding="utf-8")
    coef_sum_df.to_csv(OUT_DIR / "feature_coefficient_summary.csv", index=False,
                       encoding="utf-8")

    config = {
        "experiment_id": EXPERIMENT_ID,
        "dataset": "ML Dataset v0",
        "dataset_file": "output/ml_dataset_v0/ml_dataset_v0.csv",
        "ml_dataset_v0_md5": v0_md5,
        "features": {"A": FEATURES_ALL, "B": FEATURES_NO_SNR},
        "target": "technique", "group": "athlete_id",
        "model": {"name": "LogisticRegression", "solver": "lbfgs",
                  "multi_class": "default (multinomial)", "max_iter": 2000,
                  "C": 1.0},
        "scaler": {"name": "StandardScaler", "fit": "per-fold (pipeline)"},
        "cv": {"name": "GroupKFold", "n_splits": N_FOLDS, "groups": "athlete_id"},
        "random_state": RANDOM_STATE,
        "experiments": {"A": "con SNR", "B": "sin SNR"},
        "population": {"rows": int(len(v0)),
                       "athletes": int(v0["athlete_id"].nunique()),
                       "techniques": sorted(v0["technique"].unique()),
                       "sampling_rate_hz": 250, "condition": "E01", "trial": "T01"},
    }
    (OUT_DIR / "experiment_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    _plot_all(oof_df, conf_df, fm_df, per_class_df, coef_sum_df, error_matrix_df)
    return oof_df, fm_df, cmp_df


# --------------------------------------------------------------------------- #
# utilidades y figuras
# --------------------------------------------------------------------------- #

def LIST_POS(j, classes_all, classes_fit):
    """posición de la clase j (orden CLASSES) en classes_fit; fallback -1."""
    c = classes_all[j]
    return list(classes_fit).index(c) if c in classes_fit else 0


error_matrix_rows: list = []


def _plot_all(oof_df, conf_df, fm_df, per_class_df, coef_sum_df, error_matrix_df):
    exps = ["A", "B"]
    # confusion
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, exp in zip(axes, exps):
        sub = conf_df[conf_df["experiment"] == exp]
        mat = np.zeros((4, 4))
        for _, r in sub.iterrows():
            mat[CLASSES.index(r["true"]), CLASSES.index(r["predicted"])] = r["count"]
        im = ax.imshow(mat, cmap="Blues")
        ax.set_xticks(range(4)); ax.set_xticklabels(CLASSES, rotation=45)
        ax.set_yticks(range(4)); ax.set_yticklabels(CLASSES)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, int(mat[i, j]), ha="center", va="center", fontsize=8)
        ax.set_title(f"Exp {exp}")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Matriz de confusión (OOF pooled)")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(FIG_DIR / "confusion_matrix.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # normalized
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, exp in zip(axes, exps):
        sub = conf_df[conf_df["experiment"] == exp]
        mat = np.zeros((4, 4))
        for _, r in sub.iterrows():
            mat[CLASSES.index(r["true"]), CLASSES.index(r["predicted"])] = \
                r["normalized_recall"]
        ax.imshow(mat, cmap="Oranges", vmin=0, vmax=1)
        ax.set_xticks(range(4)); ax.set_xticklabels(CLASSES, rotation=45)
        ax.set_yticks(range(4)); ax.set_yticklabels(CLASSES)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=8)
        ax.set_title(f"Exp {exp}")
    fig.suptitle("Matriz de confusión normalizada por clase real")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(FIG_DIR / "confusion_matrix_normalized.png", dpi=130,
                bbox_inches="tight")
    plt.close(fig)

    # fold metrics
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for exp, color in zip(exps, ("#3a7ca5", "#e0a83a")):
        sub = fm_df[fm_df["experiment"] == exp]
        ax.plot(sub["fold"], sub["f1_macro"], "o-", color=color, label=f"{exp} f1_macro")
        ax.plot(sub["fold"], sub["balanced_accuracy"], "s--", color=color,
                label=f"{exp} bal_acc", alpha=0.6)
    ax.set_xlabel("fold"); ax.set_ylabel("métrica")
    ax.set_title("Métricas por fold (OOF)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fold_metrics.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # metrics by class
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, exp in zip(axes, exps):
        sub = per_class_df[per_class_df["experiment"] == exp]
        x = np.arange(4)
        w = 0.25
        for j, met in enumerate(["precision", "recall", "f1"]):
            vals = [sub.loc[sub["technique"] == c, met].iloc[0] if not
                    sub.loc[sub["technique"] == c].empty else 0 for c in CLASSES]
            ax.bar(x + (j - 1) * w, vals, width=w, label=met)
        ax.set_xticks(x); ax.set_xticklabels(CLASSES)
        ax.set_ylim(0, 1); ax.legend(fontsize=7); ax.set_title(f"Exp {exp}")
    fig.suptitle("Métricas por técnica (OOF pooled) — diagnóstico, sin rankings")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(FIG_DIR / "metrics_by_class.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # coefficients
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), sharey=True)
    for ax, exp in zip(axes, exps):
        sub = coef_sum_df[coef_sum_df["experiment"] == exp].set_index("feature")
        ax.barh(sub.index, sub["mean_abs_coefficient"], color="#5b8db8",
                xerr=sub["std_abs_coefficient"], capsize=2)
        ax.set_title(f"Exp {exp}")
        ax.invert_yaxis()
    fig.suptitle("Coeficientes |abs| (m±std, features escaladas)")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(FIG_DIR / "feature_coefficients.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # error matrix
    fig, ax = plt.subplots(figsize=(9, 4.6))
    mat = np.zeros((4, 4))
    for _, r in error_matrix_df.iterrows():
        mat[CLASSES.index(r["true"]), CLASSES.index(r["predicted"])] += r["count"]
    ax.imshow(mat, cmap="Reds")
    ax.set_xticks(range(4)); ax.set_xticklabels(CLASSES, rotation=45)
    ax.set_yticks(range(4)); ax.set_yticklabels(CLASSES)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, int(mat[i, j]), ha="center", va="center", fontsize=8)
    ax.set_title("Errores de clasificación (true×predicted, A+B)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "error_matrix.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    oof, fm, cmp = run_task7b()
    print(f"[7B] Experiment {EXPERIMENT_ID} — OOF {len(oof)} filas ({len(oof)//2} por exp)")
    print(cmp.to_string(index=False))
    v0_md5 = hashlib.md5(V0_FILE.read_bytes()).hexdigest()
    print(f"[7B] hash(ml_dataset_v0.csv) = {v0_md5}")
    print(f"[7B] Guardado en {OUT_DIR}")


if __name__ == "__main__":
    main()