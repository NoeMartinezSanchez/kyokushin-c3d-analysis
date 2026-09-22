#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
15_build_ml_dataset_v0.py
=========================
TASK 7 — DISEÑO Y AUDITORÍA DEL ML DATASET v0 (NO entrenar modelos)

Construye de forma reproducible el primer dataset preparado para ML a partir
del Data Mart consolidado (428 filas, 39 columnas), limitado a la cohorte
250 Hz × S02-S05 × E01 × T01 × accepted.

NO entrena, NO normaliza para entrenamiento, NO hace feature selection,
NO toca el Data Mart/dashboard/segmentación. Escribe SOLO en
output/ml_dataset_v0/.

Unidad de observación: UNA EJECUCIÓN ACEPTADA (1 fila = 1 repetición).

Validación futura (Task 7B): agrupada por athlete_id (GroupKFold baseline;
se documentan StratifiedGroupKFold y LOGO). Prohibido split fila a fila.

Determinismo: orden explícito por athlete_id, technique, execution_id;
sin RNG, sin dependencias de fs/timestamps.
"""

from __future__ import annotations

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
MART_FILE = ROOT / "output" / "data_mart" / "athlete_execution_features.csv"
OUT_DIR = ROOT / "output" / "ml_dataset_v0"
FIG_DIR = OUT_DIR / "figures"

TECH_ML = ["S02", "S03", "S04", "S05"]
CONDITION, TRIAL = "E01", "T01"
RATE = 250.0

FEATURES = ["duration_s", "time_to_peak_s", "vmax", "vmean", "amax",
            "displacement", "path_length", "hip_rom", "knee_rom", "ankle_rom",
            "snr"]

COMPARABILITY = {
    "duration_s": ("DIRECTLY_COMPARABLE", False, "s"),
    "time_to_peak_s": ("DIRECTLY_COMPARABLE", False, "s"),
    "vmax": ("REQUIRES_NORMALIZATION", True, "m/s"),
    "vmean": ("REQUIRES_NORMALIZATION", True, "m/s"),
    "amax": ("REQUIRES_NORMALIZATION", True, "m/s²"),
    "displacement": ("COMPARABLE_WITH_CAVEAT", False, "m"),
    "path_length": ("REQUIRES_NORMALIZATION", True, "m"),
    "hip_rom": ("COMPARABLE_WITH_CAVEAT", False, "deg"),
    "knee_rom": ("COMPARABLE_WITH_CAVEAT", False, "deg"),
    "ankle_rom": ("COMPARABLE_WITH_CAVEAT", False, "deg"),
    "snr": ("COMPARABLE_WITH_CAVEAT", False, "dimensionless"),
}

# Categorías A-H del plan y razones/leakage por columna.
AUDIT_SPEC = {
    "technique": ("TARGET", "n/a", False, "LOW", "variable objetivo (clasificación de técnica)"),
    "athlete_id": ("METADATA", "n/a", False, "HIGH", "identificador de individuo: memorizar individuos"),
    "execution_id": ("METADATA", "n/a", False, "HIGH", "identificador de ejecución: memorizar ejecuciones"),
    "event_id": ("METADATA", "n/a", False, "LOW", "índice de evento (trazabilidad)"),
    "repetition": ("METADATA", "n/a", False, "LOW", "orden de repetición dentro del archivo"),
    "sampling_rate_hz": ("NON_FEATURE_CONSTANT", "Hz", False, "LOW", "constante 250 tras filtro"),
    "primary_signal": ("POTENTIAL_LEAKAGE", "n/a", False, "MEDIUM", "señal configurada por atleta×técnica"),
    "movement_side": ("POTENTIAL_LEAKAGE", "n/a", False, "MEDIUM", "lateralidad de config por celda"),
    "qc_status": ("QUALITY_CONTROL", "n/a", False, "LOW", "constante accepted"),
    "quality_flag": ("QUALITY_CONTROL", "n/a", False, "LOW", "constante OK"),
    "condition": ("NON_FEATURE_CONSTANT", "n/a", False, "LOW", "constante E01 tras filtro"),
    "trial": ("NON_FEATURE_CONSTANT", "n/a", False, "LOW", "constante T01 tras filtro"),
    "source_dataset": ("METADATA", "n/a", False, "MEDIUM", "provenance de cohorte/tarea"),
    "feature_version": ("VERSIONING", "n/a", False, "LOW", "versión de la capa de features"),
    "segmentation_version": ("VERSIONING", "n/a", False, "LOW", "versión del motor de segmentación"),
    "units_version": ("VERSIONING", "n/a", False, "LOW", "versión de unidades"),
    "mart_version": ("VERSIONING", "n/a", False, "LOW", "versión del Data Mart"),
}


def _load() -> pd.DataFrame:
    return pd.read_csv(MART_FILE)


def _population(mart: pd.DataFrame) -> pd.DataFrame:
    m = mart
    return m[(m["sampling_rate_hz"] == RATE)
             & (m["condition"] == CONDITION)
             & (m["trial"] == TRIAL)
             & (m["technique"].isin(TECH_ML))
             & (m["qc_status"] == "accepted")
             & (m["quality_flag"].isin(["OK", "WARN"]))].copy()


def build_ml_dataset_v0():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    mart = _load()
    pop = _population(mart)

    # ---------- 1) Población: trazabilidad de filtros ----------
    stages = []
    cur = mart
    for label, fn, reason in [
        ("total_mart", lambda d: d.index.isin(d.index), "población inicial del Data Mart"),
        ("sampling_250hz", lambda d: d["sampling_rate_hz"] == RATE, "excluye 200 Hz (B0367) y demás no-250"),
        ("condition_E01", lambda d: d["condition"] == CONDITION, "golden path E01"),
        ("trial_T01", lambda d: d["trial"] == TRIAL, "golden path T01"),
        ("technique_S02_S05", lambda d: d["technique"].isin(TECH_ML), "solo S02-S05 (S01 fuera)"),
        ("qc_accepted", lambda d: d["qc_status"] == "accepted", "solo ejecuciones aceptadas"),
        ("quality_ok_warn", lambda d: d["quality_flag"].isin(["OK", "WARN"]), "solo calidad OK/WARN"),
    ]:
        before = len(cur)
        nxt = cur[fn(cur)]
        stages.append({"stage": label, "filter": reason, "rows_before": before,
                       "rows_after": len(nxt), "rows_removed": before - len(nxt),
                       "reason": reason})
        cur = nxt
    pop_audit = pd.DataFrame(stages)
    assert len(cur) == len(pop)

    # ---------- 2) Clases ----------
    cls_rows = []
    for t in TECH_ML:
        sub = pop[pop["technique"] == t]
        nav = sub["athlete_id"].nunique()
        per = sub.groupby("athlete_id").size()
        cls_rows.append({"technique": t, "n": int(len(sub)),
                         "n_pct": round(100 * len(sub) / len(pop), 2),
                         "n_athletes": nav,
                         "exec_per_athlete_mean": round(float(per.mean()), 2),
                         "exec_per_athlete_min": int(per.min()),
                         "exec_per_athlete_max": int(per.max())})
    cls_df = pd.DataFrame(cls_rows)
    ratio = cls_df["n"].max() / cls_df["n"].min()
    cls_df["imbalance_ratio_max_min"] = round(ratio, 3)
    cls_df["imbalance_level"] = ("BALANCED" if ratio < 2
                                 else "MODERATELY_UNBALANCED" if ratio < 3
                                 else "HIGHLY_UNBALANCED")

    # ---------- 3) Cobertura por atleta ----------
    cov_rows = []
    for aid in sorted(pop["athlete_id"].unique()):
        sub = pop[pop["athlete_id"] == aid]
        counts = {t: int((sub["technique"] == t).sum()) for t in TECH_ML}
        cov_rows.append({"athlete_id": aid, "total_executions": int(len(sub)),
                         **{f"{t}_count": counts[t] for t in TECH_ML},
                         "techniques_present":
                         ",".join([t for t in TECH_ML if counts[t] > 0])})
    cov_df = pd.DataFrame(cov_rows)

    # ---------- 4) Distribution summary ----------
    dist_rows = []
    for f in FEATURES:
        s = pop[f]
        q = s.quantile([.01, .05, .25, .5, .75, .95, .99])
        dist_rows.append({"feature": f,
                          "count": int(s.count()), "mean": round(s.mean(), 5),
                          "median": round(s.median(), 5), "std": round(s.std(), 5),
                          "min": round(s.min(), 5), "max": round(s.max(), 5),
                          "p01": round(q[.01], 5), "p05": round(q[.05], 5),
                          "p25": round(q[.25], 5), "p50": round(q[.5], 5),
                          "p75": round(q[.75], 5), "p95": round(q[.95], 5),
                          "p99": round(q[.99], 5),
                          "IQR": round(q[.75] - q[.25], 5),
                          "missing": int(s.isna().sum())})
    dist_df = pd.DataFrame(dist_rows)

    # ---------- 5) Outliers (IQR, diagnóstico) ----------
    out_rows = []
    for f in FEATURES:
        s = pop[f].dropna()
        q1, q3 = s.quantile(.25), s.quantile(.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = int(((s < lo) | (s > hi)).sum())
        out_rows.append({"feature": f, "outlier_count": n_out,
                         "outlier_pct": round(100 * n_out / len(s), 2),
                         "lower_bound": round(lo, 5), "upper_bound": round(hi, 5)})
    outlier_df = pd.DataFrame(out_rows)

    # ---------- 6) Correlación ----------
    corr = pop[FEATURES].corr(method="pearson")
    corr_rows = []
    for i, a in enumerate(FEATURES):
        for b in FEATURES[i + 1:]:
            corr_rows.append({"feature_a": a, "feature_b": b,
                              "correlation": round(float(corr.loc[a, b]), 4)})
    corr_df = pd.DataFrame(corr_rows)

    # ---------- 7) Feature vs technique ----------
    by_rows = []
    for f in FEATURES:
        for t in TECH_ML:
            s = pop.loc[pop["technique"] == t, f].dropna()
            q = s.quantile([.25, .5, .75])
            by_rows.append({"feature": f, "technique": t,
                            "mean": round(s.mean(), 5), "median": round(q[.5], 5),
                            "std": round(s.std(), 5),
                            "IQR": round(q[.75] - q[.25], 5), "n": int(s.count())})
    by_df = pd.DataFrame(by_rows)

    # ---------- 8) Feature audit (39 columnas reales) ----------
    audit_rows = []
    for c in mart.columns:
        if c.startswith("comparability_"):
            cat, unit, cflag, leak, reason = \
                ("METADATA", "n/a", False, "LOW",
                 "etiqueta de comparabilidad (contrato)")
        elif c in FEATURES:
            cat = "CANDIDATE_BIOMECHANICAL_FEATURE"
            unit, cflag, leak, reason = COMPARABILITY[c][2], True, "LOW", \
                "feature biomecánica candidata"
        elif c in AUDIT_SPEC:
            cat, unit, cflag, leak, reason = AUDIT_SPEC[c]
        else:
            cat, unit, cflag, leak, reason = \
                ("OTHER", "n/a", False, "LOW", "no clasificada")
        n_unique = int(pop[c].nunique(dropna=True))
        missing = int(pop[c].isna().sum())
        audit_rows.append({
            "column": c, "dtype": str(pop[c].dtype), "category": cat,
            "unit": unit, "n_unique": n_unique,
            "missing_count": missing,
            "missing_pct": round(100 * missing / len(pop), 2),
            "constant": n_unique <= 1,
            "candidate_for_ml": cflag,
            "leakage_risk": leak, "reason": reason, "notes": "",
        })
    audit_df = pd.DataFrame(audit_rows)

    # ---------- 9) Comparabilidad por feature ----------
    cmp_rows = []
    for f in FEATURES:
        st, norm, unit = COMPARABILITY[f]
        allowed = True
        note = ("raw permitido como candidato; normalización dentro de cada fold en Task 7B"
                if norm else
                ("proxy señal/baseline: incluir con caveat (riesgo documentado)"
                 if f == "snr" else
                 "comparabilidad con caveat; raw permitido"))
        cmp_rows.append({"feature": f, "comparability_status": st, "unit": unit,
                         "normalization_required": norm,
                         "reason": "contrato 1.8B-1/" + st,
                         "allowed_in_raw_ml_v0": allowed, "notes": note})
    cmp_df = pd.DataFrame(cmp_rows)

    # ---------- 10) Dataset v0 ----------
    keep = ["execution_id", "athlete_id", "technique"] + FEATURES
    v0 = pop[keep].sort_values(["athlete_id", "technique", "execution_id"],
                               kind="stable").reset_index(drop=True)

    # ---------- 11) Manifest ----------
    man_rows = []
    for c in v0.columns:
        if c == "execution_id":
            role = "identifier"; src = "execution_id"
            reason = "trazabilidad (nunca predictor)"
            leak = "HIGH->metadata"; comp = "n/a"
        elif c == "athlete_id":
            role = "identifier"; src = "athlete_id"
            reason = "agrupación de validación futura (nunca predictor)"
            leak = "HIGH->metadata"; comp = "n/a"
        elif c == "technique":
            role = "target"; src = "technique"
            reason = "variable objetivo"; leak = "TARGET"; comp = "n/a"
        else:
            role = "feature"; src = c
            reason = "feature biomecánica candidata"
            leak = "SAFE (post-excl)"; comp = COMPARABILITY[c][0]
            if c == "snr":
                leak = "SAFE_WITH_CAVEAT"
        man_rows.append({"column": c, "role": role, "dtype": str(v0[c].dtype),
                         "unit": COMPARABILITY[c][2] if c in FEATURES else "n/a",
                         "source_column": src, "reason_included": reason,
                         "leakage_status": leak,
                         "comparability_status": comp})
    manifest_df = pd.DataFrame(man_rows)

    # ---------- 12) Estrategia de validación (documentada) ----------
    _write_validation_strategy(pop, cov_df, cls_df)

    # ---------- escritura ----------
    mapping = {
        "ml_dataset_v0.csv": v0,
        "dataset_manifest.csv": manifest_df,
        "feature_audit.csv": audit_df,
        "feature_comparability_audit.csv": cmp_df,
        "population_filter_audit.csv": pop_audit,
        "class_distribution.csv": cls_df,
        "athlete_coverage.csv": cov_df,
        "feature_distribution_summary.csv": dist_df,
        "feature_correlation.csv": corr_df,
        "feature_by_technique.csv": by_df,
        "outlier_audit.csv": outlier_df,
    }
    for name, df in mapping.items():
        df.to_csv(OUT_DIR / name, index=False, encoding="utf-8")

    _plot_distributions(pop, dist_df)
    _plot_correlation_heatmap(corr)
    _plot_by_technique(pop)

    return v0, audit_df


# --------------------------------------------------------------------------- #
# Figuras
# --------------------------------------------------------------------------- #

def _plot_distributions(pop: pd.DataFrame, dist_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(3, 4, figsize=(16, 9))
    axes = axes.ravel()
    for i, f in enumerate(FEATURES):
        ax = axes[i]
        s = pop[f].dropna()
        ax.hist(s, bins=30, color="#3a7ca5", alpha=0.8)
        ax.axvline(s.median(), color="#c0392b", ls="--", lw=1, label="mediana")
        ax.set_title(f"{f} ({COMPARABILITY[f][2]})", fontsize=9)
        ax.legend(fontsize=6)
    for ax in axes[len(FEATURES):]:
        ax.axis("off")
    fig.suptitle("Distribuciones de features — ML v0 (250 Hz, S02-S05)")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(FIG_DIR / "feature_distributions_hist.png", dpi=130,
                bbox_inches="tight")
    plt.close(fig)


def _plot_correlation_heatmap(corr: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(FEATURES))); ax.set_xticklabels(FEATURES, rotation=90)
    ax.set_yticks(range(len(FEATURES))); ax.set_yticklabels(FEATURES)
    for i in range(len(FEATURES)):
        for j in range(len(FEATURES)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center",
                    fontsize=6)
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title("Correlación de features — ML v0")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "feature_correlation_heatmap.png", dpi=130,
                bbox_inches="tight")
    plt.close(fig)


def _plot_by_technique(pop: pd.DataFrame) -> None:
    fig, axes = plt.subplots(3, 4, figsize=(16, 9))
    axes = axes.ravel()
    for i, f in enumerate(FEATURES):
        ax = axes[i]
        data = [pop.loc[pop["technique"] == t, f].dropna().to_list()
                for t in TECH_ML]
        ax.boxplot(data, tick_labels=TECH_ML, patch_artist=True,
                   boxprops=dict(facecolor="#dbe7f2"))
        ax.set_title(f, fontsize=9)
        ax.grid(alpha=0.3)
    for ax in axes[len(FEATURES):]:
        ax.axis("off")
    fig.suptitle("Feature × technique (descriptivo, sin rankings) — ML v0")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(FIG_DIR / "feature_by_technique_box.png", dpi=130,
                bbox_inches="tight")
    plt.close(fig)


def _write_validation_strategy(pop: pd.DataFrame, cov_df: pd.DataFrame,
                               cls_df: pd.DataFrame) -> None:
    n_groups = int(pop["athlete_id"].nunique())
    counts = {t: int(cls_df.loc[cls_df["technique"] == t, "n_athletes"].iloc[0])
              for t in TECH_ML}
    min_groups = min(counts.values())
    partial = [r["athlete_id"] for _, r in cov_df.iterrows()
               if len(r["techniques_present"].split(",")) < 4]
    text = f"""# Estrategia de validación futura — ML Dataset v0

**Grupo:** `athlete_id` (el atleta es la unidad de agrupación; NUNCA split fila a fila).

**Unidad de observación:** 1 ejecución aceptada (1 fila = 1 repetición).

**Objetivo (Task 7B):** clasificación multiclase de técnica (target = `technique`,
clases S02/S03/S04/S05).

**Riesgo de leakage por grupos:** múltiples ejecuciones por atleta -> un split
aleatorio por filas filtraría atleta (y su biomecánica) entre train/test.

**Datos de la población 250 Hz:** {len(pop)} ejecuciones, {n_groups} atletas,
grupos por clase: {counts}. Grupo minoritario por clase: S04 = {counts['S04']}.
Atletas con cobertura parcial (< 4 técnicas): {', '.join(partial) or 'ninguno'}.

## Estrategia propuesta (baseline)
**GroupKFold(n_splits=5)** sobre `athlete_id`: viable porque hay {n_groups} grupos
y la clase con menos grupos (S04) tiene {counts['S04']} >= 5. Cada fold separa
atletas completos; todas las repeticiones de un atleta quedan en el mismo fold.
La normalización/`y` de escalado se ajusta SOLO dentro de cada fold (train), para
evitar leakage hacia test.

## Alternativas documentadas
- **StratifiedGroupKFold**: viable con k <= ~{counts['S04']} pero la estratificación
  es PARCIAL porque {len(partial)} atletas no cubren las 4 clases (grupos
  incompletos); no se usa como baseline.
- **Leave-One-Group-Out (LOGO)**: {n_groups} folds (uno por atleta); exhaustivo
  pero de mayor coste; se mantiene como opción de análisis en Task 7B.

## Limitaciones
- n=33 grupos (pequeño): métricas con varianza alta; recomendar reportar
  distribución de métricas por fold + intervalo.
- Clase minoritaria S04 ({counts['S04']} grupos, {cls_df.loc[cls_df['technique']=='S04', 'n'].iloc[0]} filas): sin balanceo en esta tarea; se evaluará en Task 7B.
"""
    (OUT_DIR / "ml_validation_strategy.md").write_text(text, encoding="utf-8")


def main() -> None:
    v0, audit = build_ml_dataset_v0()
    print("[T7] ML Dataset v0 construido")
    print(f"[T7] Filas: {len(v0)} | atletas: {v0['athlete_id'].nunique()}")
    print(f"[T7] Clases: {v0['technique'].value_counts().to_dict()}")
    feats = [c for c in v0.columns if c in FEATURES]
    print(f"[T7] Features finales ({len(feats)}): {', '.join(feats)}")
    ident = [c for c in v0.columns if c not in feats and c != 'technique']
    print(f"[T7] Identifiers: {ident} | target: technique")
    print(f"[T7] Guardado en {OUT_DIR}")
    n_na = int(v0[feats].isna().sum().sum())
    n_inf = int(np.isinf(v0[feats].to_numpy(dtype=float)).sum())
    print(f"[T7] NaN features={n_na} | inf={n_inf}")


if __name__ == "__main__":
    main()