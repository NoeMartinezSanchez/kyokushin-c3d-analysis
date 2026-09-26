#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
18_task8a_error_analysis.py
===========================
TASK 8A — ANÁLISIS DESCRIPTIVO DE ERRORES OOF DEL BASELINE (Task 7B)

Análisis POST-HOC, reproducible y exclusivamente descriptivo de los errores
del baseline LogisticRegression (Task 7B), usando SOLO:

  output/ml_results/oof_predictions.csv   (predicciones OOF, exp A y B)
  output/ml_dataset_v0/ml_dataset_v0.csv  (features del ML Dataset v0)

NO entrena, NO modifica Task 7B/v0/Data Mart/dashboard, NO lee C3D.

Foco: EXPERIMENTO B (10 features, sin SNR) como "baseline de referencia para
el análisis de errores"; A se mantiene para comparación por execution_id.

Reglas: correct = true==pred; error = true!=pred. Lenguaje descriptivo
(OBSERVACIÓN / DESCRIPTIVA / HIPÓTESIS). Sin rankings, sin causalidad.

Integridad: registra md5 antes/después de los archivos protegidos y ABORTA si
alguno cambia durante la ejecución.

Uso:
  .venv\\Scripts\\python scripts\\18_task8a_error_analysis.py
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parents[1]
OOF_FILE = ROOT / "output" / "ml_results" / "oof_predictions.csv"
V0_FILE = ROOT / "output" / "ml_dataset_v0" / "ml_dataset_v0.csv"
OUT_DIR = ROOT / "output" / "task8a_error_analysis"
FIG_DIR = OUT_DIR / "figures"

CLASSES = ["S02", "S03", "S04", "S05"]
FEATURES_B = ["duration_s", "time_to_peak_s", "vmax", "vmean", "amax",
              "displacement", "path_length", "hip_rom", "knee_rom",
              "ankle_rom"]
PROB_COLS = ["prob_S02", "prob_S03", "prob_S04", "prob_S05"]
MIN_PAIR_N = 5  # umbral (descriptivo) para perfilar pares de confusión

# Archivos protegidos que deben quedar idénticos (md5 antes/después).
PROTECTED = [
    "output/ml_dataset_v0/ml_dataset_v0.csv",
    "output/ml_results/oof_predictions.csv",
    "output/ml_results/fold_metrics.csv",
    "output/ml_results/experiment_config.json",
    "scripts/02_execution_segmentation.py",
    "config/segmentation.yaml",
    "output/data_mart/athlete_execution_features.csv",
    "dashboard/app.py",
    "dashboard/app_ml.py",
]


def _md5(rel: str) -> str:
    p = ROOT / rel
    return hashlib.md5(p.read_bytes()).hexdigest() if p.exists() else "MISSING"


def _desc(series: pd.Series) -> dict:
    s = series.dropna()
    return {
        "n": int(s.count()),
        "mean": round(float(s.mean()), 5) if s.count() else np.nan,
        "median": round(float(s.median()), 5) if s.count() else np.nan,
        "std": round(float(s.std(ddof=1)), 5) if s.count() > 1 else np.nan,
        "q25": round(float(s.quantile(.25)), 5) if s.count() else np.nan,
        "q75": round(float(s.quantile(.75)), 5) if s.count() else np.nan,
    }


def run_task8a():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # ---------- integridad: hash ANTES ----------
    before = {rel: _md5(rel) for rel in PROTECTED}

    oof = pd.read_csv(OOF_FILE)
    v0 = pd.read_csv(V0_FILE)
    assert len(v0) == 419

    # ---------- merge B (y A) 1:1 ----------
    frames = {}
    for exp in ("A", "B"):
        sub = oof[oof["experiment"] == exp].copy()
        assert len(sub) == len(v0)
        assert sub["execution_id"].is_unique
        m = sub.merge(v0[["execution_id"] + FEATURES_B], on="execution_id",
                      how="inner")
        assert len(m) == len(v0)
        tech_map = v0.set_index("execution_id")["technique"]
        ok_true = (m["true_technique"].to_numpy() ==
                   tech_map.loc[m["execution_id"]].to_numpy())
        assert ok_true.all(), "true_technique no coincide con el dataset v0"
        m["correct"] = m["true_technique"] == m["predicted_technique"]
        m["error"] = ~m["correct"]
        frames[exp] = m
    B = frames["B"]

    # ---------- resumen general ----------
    rows = []
    for exp in ("A", "B"):
        d = frames[exp]
        n_err = int(d["error"].sum())
        rows.append({"experiment": exp, "total": int(len(d)),
                     "correct": int(d["correct"].sum()), "errors": n_err,
                     "correct_pct": round(100 * d["correct"].mean(), 2),
                     "error_pct": round(100 * d["error"].mean(), 2)})
    error_summary = pd.DataFrame(rows)

    # ---------- feature correct vs error (B) ----------
    fce_rows = []
    for f in FEATURES_B:
        for grp, mask in (("correct", B["correct"]), ("error", B["error"])):
            d = _desc(B.loc[mask, f])
            fce_rows.append({"feature": f, "group": grp, **d})
    fce_df = pd.DataFrame(fce_rows)

    # ---------- por técnica (B) ----------
    tech_rows = []
    for t in CLASSES:
        d = B[B["true_technique"] == t]
        n_err = int(d["error"].sum())
        tech_rows.append({"technique": t, "total": int(len(d)),
                          "correct": int(d["correct"].sum()),
                          "errors": n_err,
                          "accuracy": round(d["correct"].mean(), 4),
                          "error_prop": round(d["error"].mean(), 4)})
    tech_df = pd.DataFrame(tech_rows)

    # ---------- pares de confusión ----------
    pair_rows = []
    for exp in ("B", "A"):
        d = frames[exp]
        err = d[d["error"]]
        n_err = len(err)
        counts = err.groupby(["true_technique", "predicted_technique"]).size()
        for (t, p), cnt in sorted(counts.items()):
            pair_rows.append({"experiment": exp, "true": t, "predicted": p,
                              "n": int(cnt),
                              "prop_of_errors": round(cnt / n_err, 4)
                              if n_err else 0.0})
    pairs_df = pd.DataFrame(pair_rows)

    # ---------- perfiles por par (B), mediana ----------
    top_pairs = []
    subB = pairs_df[pairs_df["experiment"] == "B"]
    for _, r in subB.iterrows():
        if r["n"] >= MIN_PAIR_N:
            top_pairs.append((str(r["true"]), str(r["predicted"])))
    profiles = []
    for t, p in top_pairs:
        true_correct = B[(B["true_technique"] == t) & (B["correct"])]
        confused = B[(B["true_technique"] == t) &
                     (B["predicted_technique"] == p)]
        target_correct = B[(B["true_technique"] == p) & (B["correct"])]
        if confused.empty:
            continue
        for f in FEATURES_B:
            tc = float(true_correct[f].median()) if not true_correct.empty else np.nan
            cm = float(confused[f].median())
            tcc = float(target_correct[f].median()) if not target_correct.empty else np.nan
            diff = cm - tc if np.isfinite(tc) else np.nan
            rel = (diff / tc * 100) if (np.isfinite(tc) and tc != 0) else np.nan
            profiles.append({
                "pair": f"{t}->{p}", "true": t, "predicted": p, "feature": f,
                "true_correct_median": round(tc, 5),
                "confused_median": round(cm, 5),
                "target_correct_median": round(tcc, 5),
                "absolute_difference": round(diff, 5) if np.isfinite(diff) else np.nan,
                "relative_difference_pct": round(rel, 2) if np.isfinite(rel) else np.nan,
            })
    profiles_df = pd.DataFrame(profiles)

    # ---------- confianza / margen (B) ----------
    conf_rows = []
    for _, r in B.iterrows():
        probs = {c: float(r[f"prob_{c}"]) for c in CLASSES}
        pt = probs[r["true_technique"]]
        pp = probs[r["predicted_technique"]]
        ordered = sorted(probs.values(), reverse=True)
        second = ordered[1] if len(ordered) > 1 else 0.0
        margin = pp - second
        conf_rows.append({"execution_id": r["execution_id"],
                          "group": "correct" if r["correct"] else "error",
                          "prob_true": round(pt, 4),
                          "prob_predicted": round(pp, 4),
                          "margin": round(margin, 4)})
    conf_df = pd.DataFrame(conf_rows)

    # ---------- por atleta (B), sin ranking ----------
    ath_rows = []
    for aid in sorted(B["athlete_id"].unique()):
        d = B[B["athlete_id"] == aid]
        ath_rows.append({"athlete_id": aid, "n": int(len(d)),
                         "correct": int(d["correct"].sum()),
                         "errors": int(d["error"].sum()),
                         "error_rate": round(d["error"].mean(), 4),
                         "techniques_present":
                         ",".join(sorted(d["true_technique"].unique()))})
    ath_df = pd.DataFrame(ath_rows)

    # ---------- por fold (B) ----------
    fold_rows = []
    for k in range(1, 6):
        d = B[B["fold"] == k]
        fold_rows.append({"fold": k, "n": int(len(d)),
                          "errors": int(d["error"].sum()),
                          "error_rate": round(d["error"].mean(), 4)})
    fold_df = pd.DataFrame(fold_rows)

    # ---------- comparación A vs B (por execution_id) ----------
    errA = set(frames["A"].loc[frames["A"]["error"], "execution_id"])
    errB = set(frames["B"].loc[frames["B"]["error"], "execution_id"])
    shared = errA & errB
    ab_df = pd.DataFrame([{
        "experiment": "A", "errors": len(errA),
        "shared_with_other": len(errA & errB),
        "unique": len(errA - errB),
    }, {
        "experiment": "B", "errors": len(errB),
        "shared_with_other": len(errB & errA),
        "unique": len(errB - errA),
    }])

    # ---------- integridad: hash DESPUÉS ----------
    after = {rel: _md5(rel) for rel in PROTECTED}
    changed = [rel for rel in PROTECTED if before[rel] != after[rel]]
    if changed:
        raise RuntimeError(f"archivos protegidos modificados durante 8A: {changed}")
    integ = pd.DataFrame([{"file": rel, "md5": before[rel], "exists":
                           before[rel] != "MISSING", "status":
                           "PASS" if before[rel] == after[rel] else "FAIL"}
                          for rel in PROTECTED])

    # ---------- escritura ----------
    mapping = {
        "error_summary.csv": error_summary,
        "feature_correct_vs_error.csv": fce_df,
        "technique_error_summary.csv": tech_df,
        "confusion_pairs.csv": pairs_df,
        "confusion_pair_profiles.csv": profiles_df,
        "confidence_analysis.csv": conf_df,
        "athlete_error_summary.csv": ath_df,
        "fold_error_summary.csv": fold_df,
        "ab_comparison.csv": ab_df,
        "integrity_hashes.csv": integ,
    }
    for name, df in mapping.items():
        df.to_csv(OUT_DIR / name, index=False, encoding="utf-8")

    _plot_figures(B, pairs_df, tech_df, conf_df, profiles_df)

    return {"error_summary": error_summary, "techniques": tech_df,
            "pairs": pairs_df, "athletes": ath_df, "folds": fold_df,
            "ab": ab_df}


# --------------------------------------------------------------------------- #
# Figuras (descriptivas, sin ranking)
# --------------------------------------------------------------------------- #

def _plot_figures(B, pairs_df, tech_df, conf_df, profiles_df):
    # A) matriz de confusión (B)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    mat = np.zeros((4, 4))
    for _, r in B.iterrows():
        mat[CLASSES.index(r["true_technique"]),
            CLASSES.index(r["predicted_technique"])] += 1
    im = ax.imshow(mat, cmap="Blues")
    ax.set_xticks(range(4)); ax.set_xticklabels(CLASSES, rotation=45)
    ax.set_yticks(range(4)); ax.set_yticklabels(CLASSES)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, int(mat[i, j]), ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title("Matriz de confusión OOF — Experimento B")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "confusion_matrix_b.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # B) error rate por técnica (B)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.bar(tech_df["technique"], tech_df["error_prop"], color="#5b8db8")
    for i, r in tech_df.iterrows():
        ax.text(i, r["error_prop"] + 0.01, f"{r['error_prop']:.2f}",
                ha="center", fontsize=9)
    ax.set_ylim(0, max(tech_df["error_prop"]) + 0.1)
    ax.set_ylabel("proporción de error (OOF, B)")
    ax.set_title("Tasa de error por técnica real")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "error_rate_by_technique.png", dpi=130,
                bbox_inches="tight")
    plt.close(fig)

    # C) correct vs error — distribuciones de las 10 features
    fig, axes = plt.subplots(2, 5, figsize=(18, 7))
    axes = axes.ravel()
    for i, f in enumerate(FEATURES_B):
        ax = axes[i]
        data = [B.loc[B["correct"], f].dropna().to_list(),
                B.loc[B["error"], f].dropna().to_list()]
        ax.boxplot(data, tick_labels=["correct", "error"], patch_artist=True,
                   boxprops=dict(facecolor="#dbe7f2"))
        ax.set_title(f, fontsize=8)
        ax.grid(alpha=0.3)
    for ax in axes[len(FEATURES_B):]:
        ax.axis("off")
    fig.suptitle("Distribuciones de features: correcto vs error (Experimento B)")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIG_DIR / "correct_vs_error_features.png", dpi=130,
                bbox_inches="tight")
    plt.close(fig)

    # D) perfiles de pares de confusión (|rel diff| top features)
    if not profiles_df.empty:
        pairs = sorted(profiles_df["pair"].unique())[:4]
        ncols = min(len(pairs), 2)
        fig, axes = plt.subplots(1, ncols, figsize=(6.5 * ncols, 5),
                                 squeeze=False)
        for k, p in enumerate(pairs[:ncols]):
            ax = axes[0][k]
            sub = profiles_df[profiles_df["pair"] == p].dropna(
                subset=["relative_difference_pct"])
            sub = sub.reindex(sub["relative_difference_pct"].abs().sort_values(
                ascending=False).index).head(6)
            ax.barh(sub["feature"], sub["relative_difference_pct"].abs(),
                    color="#e0a83a")
            ax.set_title(f"{p} — |Δ rel medio|")
            ax.set_xlabel("%")
        fig.suptitle("Perfiles de pares de confusión (diferencia descriptiva de "
                     "medianas)")
        fig.tight_layout(rect=(0, 0, 1, 0.9))
        fig.savefig(FIG_DIR / "confusion_pair_profiles.png", dpi=130,
                    bbox_inches="tight")
        plt.close(fig)

    # E) margen de predicción correct vs error (B)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for grp, color in (("correct", "#3a7ca5"), ("error", "#c2554f")):
        ax.hist(conf_df.loc[conf_df["group"] == grp, "margin"], bins=25,
                alpha=0.6, color=color, label=grp)
    ax.set_xlabel("margen de predicción (prob_pred - 2ª clase)")
    ax.set_ylabel("n")
    ax.set_title("Margen de predicción: correcto vs error (B)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "prediction_margin_correct_vs_error.png", dpi=130,
                bbox_inches="tight")
    plt.close(fig)

    # F) frecuencia de pares (B)
    sub = pairs_df[pairs_df["experiment"] == "B"]
    sub = sub[sub["n"] > 0].sort_values("n", ascending=False).head(12)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    labels = [f"{r['true']}→{r['predicted']}" for _, r in sub.iterrows()]
    ax.barh(labels, sub["n"], color="#5b8db8")
    ax.set_xlabel("n (OOF, B)")
    ax.set_title("Frecuencia de pares true→predicted (errores)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "confusion_pair_frequency.png", dpi=130,
                bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    res = run_task8a()
    print("[8A] Análisis de errores OOF (post-hoc, descriptivo)")
    print(res["error_summary"].to_string(index=False))
    print()
    print(res["pairs"][res["pairs"]["experiment"] == "B"].to_string(index=False))
    print()
    print("A vs B:", res["ab"].to_dict("records"))
    print(f"[8A] Guardado en {OUT_DIR}")


if __name__ == "__main__":
    main()