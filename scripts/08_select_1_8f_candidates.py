#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
08_select_1_8f_candidates.py
============================
FASE 1.8F — TAREA 0: SELECCIÓN AUTOMÁTICA DE 3 ATLETAS CANDIDATOS

Elige, de forma reproducible y auditable, 3 atletas de la cohorte 250 Hz
(Grupo B de scaling_readiness, NOT_CONFIGURED, sin anomalías, con E01 y con
cobertura S01-S05) para una prueba controlada de generalización del pipeline.

La selección es ESTRUCTURAL (diversidad técnica del dataset), NO de
rendimiento: no usa velocidad, nº de ejecuciones, edad, grado ni ranking.

Fuente ÚNICA de verdad: las tablas de inventario de FASE 1.8E
(output/athlete_inventory/) — el script NO abre archivos C3D.

Este script es de SOLO LECTURA sobre el dataset: no procesa C3D, no segmenta,
no crea configs, no toca el Data Mart ni el dashboard. Única escritura:
output/scaling_selection/phase_1_8f_candidate_selection.csv

Método (determinista, sin ML ni clustering):
  1. Filtrar el universo elegible (see eligible_candidates).
  2. Construir por atleta un vector de bits estructurales (variantes respecto
     al patrón modal de la cohorte 250 Hz / Grupo B).
  3. Seleccionar por búsqueda exhaustiva sobre combos de 3 atletas el trío
     que maximiza la suma de distancias Hamming por pares (cobertura de
     diversidad estructural); empates resueltos por id de atleta menor.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parents[1]
INV_DIR = ROOT / "output" / "athlete_inventory"
OUT_DIR = ROOT / "output" / "scaling_selection"
OUT_CSV = OUT_DIR / "phase_1_8f_candidate_selection.csv"

# Tablas de inventario consumidas (nombres reales de FASE 1.8E).
INV_FILES = [
    "athlete_inventory.csv",
    "file_inventory.csv",
    "coverage_matrix.csv",
    "marker_availability.csv",
    "derived_variable_inventory.csv",
    "configuration_gap.csv",
    "anomaly_inventory.csv",
    "scaling_readiness.csv",
]

TECHNIQUES = ["S01", "S02", "S03", "S04", "S05"]

# Ya representan los dos primeros casos del pipeline (baseline + 2º atleta).
EXCLUDED_ATHLETES = {"B0367", "B0377"}

# Patrón derivado modal de la cohorte 250 Hz / Grupo B (E01/E02 = 74).
MODAL_DERIVED_E01E02 = "74"
MODAL_POINTS = (191, 382)
MODAL_TRIALS_COUNT = 2

OUTPUT_COLUMNS = [
    "rank_internal",
    "athlete_id",
    "sampling_rate_hz",
    "total_c3d",
    "techniques_present",
    "conditions_present",
    "trials_present",
    "derived_structure",
    "marker_structure_summary",
    "anomaly_status",
    "configuration_status",
    "selection_reason",
    "diversity_role",
]

# Columnas de bits estructurales (variantes vs patrón modal).
BIT_COLUMNS = [
    "b_derived",       # estructura derivada E01/E02 != 74
    "b_e03",           # no posee E03
    "b_e04",           # no posee E04
    "b_extra_tech",    # posee técnica fuera de S01-S05 (p. ej. S06)
    "b_deep_trials",   # bustión trials >= 5
    "b_points",        # rango de puntos != modal (191-382)
]

N_SIGNAL_MARKERS = 7  # RFIN, RTOE, LTOE, RANK, LANK, RHEE, LHEE


def load_inventory() -> dict[str, pd.DataFrame]:
    """Carga las 8 tablas de inventario de FASE 1.8E."""
    return {name: pd.read_csv(INV_DIR / name) for name in INV_FILES}


def _derived_structure_of(inv: dict[str, pd.DataFrame], athlete: str) -> str:
    """Devuelve el patrón de variables derivadas 'E01/E02/E03/E04' del atleta.

    Ejemplos: '74/25' (estándar), '101/101' (solo B0367/B0368), '48|74/25' (B0400).
    """
    der = inv["derived_variable_inventory.csv"]
    rows = der[der["athlete"] == athlete]
    e01e02: set[str] = set()
    e03e04: set[str] = set()
    for _, r in rows.iterrows():
        raw = str(r["n_derived_unique"]).strip()
        if raw.startswith("[") and raw.endswith("]"):
            vals = sorted({int(x) for x in raw[1:-1].split(",") if x.strip()})
        else:
            vals = [int(raw)]
        pat = "|".join(str(v) for v in vals)
        cond = str(r["condition"])
        if cond in ("E01", "E02"):
            e01e02.add(pat)
        elif cond in ("E03", "E04"):
            e03e04.add(pat)
    e01e02_pat = "|".join(sorted(e01e02)) if e01e02 else "n/a"
    e03e04_pat = "|".join(sorted(e03e04)) if e03e04 else "n/a"
    return f"{e01e02_pat}/{e03e04_pat}"


def eligible_candidates(inv: dict[str, pd.DataFrame]):
    """Aplica los filtros del universo de candidatos.

    Devuelve (df_elegibles, counts) con counts = diagnóstico reproducible:
    n_total, n_250, n_200, n_group_c, n_excluded_used, n_not_configured,
    n_anomalies, n_no_e01, n_missing_techniques, n_eligible.
    """
    ai = inv["athlete_inventory.csv"].copy()
    sr = inv["scaling_readiness.csv"]
    gap = inv["configuration_gap.csv"]
    anom = inv["anomaly_inventory.csv"]
    cov = inv["coverage_matrix.csv"]

    # Diagnóstico de universo.
    all_ids = sorted(ai["athlete_id"].dropna().unique().tolist())
    rates = sr.set_index("athlete")["rate_hz"].to_dict()
    n_250 = sum(1 for a in all_ids if rates.get(a) == 250.0)
    n_total = len(all_ids)
    n_200 = n_total - n_250
    n_group_c = int((sr["group"] == "C").sum())
    n_excluded_used = len(EXCLUDED_ATHLETES)

    # Filtros.
    filt_250 = ai["athlete_id"].isin([a for a in all_ids if rates.get(a) == 250.0])
    group_b = sr.set_index("athlete")["group"].to_dict()
    filt_b = ai["athlete_id"].map(lambda a: group_b.get(a) == "B")

    status = gap.set_index("athlete")["configuration_status"].to_dict()
    filt_cfg = ai["athlete_id"].map(lambda a: status.get(a) == "NOT_CONFIGURED")
    n_not_configured = int(filt_cfg.sum())

    anomalous = set(
        anom.loc[anom["anomaly_flags"].fillna("").astype(str).str.strip() != "", "athlete_id"]
    )
    filt_anom = ~ai["athlete_id"].isin(anomalous)
    n_anomalies = len(anomalous)

    filt_e01 = ai["conditions"].astype(str).str.contains(r"\bE01\b", regex=True)
    n_no_e01 = int((~filt_e01).sum())

    # Cobertura S01-S05 vía matrices coverage_matrix (>=1 archivo por técnica).
    cov_by = cov.set_index("athlete_id")
    def _has_s01_s05(aid: str) -> bool:
        if aid not in cov_by.index:
            return False
        row = cov_by.loc[aid]
        for t in TECHNIQUES:
            if not any(row.index.str.startswith(t + "-")):
                return False
            if int(row[row.index.str.startswith(t + "-")].sum()) < 1:
                return False
        return True
    filt_tech = ai["athlete_id"].map(_has_s01_s05)
    n_missing_techniques = int((~filt_tech).sum())

    elig = ai[
        filt_250 & filt_b & filt_cfg & filt_anom & filt_e01 & filt_tech
        & ~ai["athlete_id"].isin(EXCLUDED_ATHLETES)
    ].copy()

    counts = {
        "n_total": n_total,
        "n_250": n_250,
        "n_200": n_200,
        "n_group_c": n_group_c,
        "n_excluded_used": n_excluded_used,
        "n_not_configured": n_not_configured,
        "n_anomalies": n_anomalies,
        "n_no_e01": n_no_e01,
        "n_missing_techniques": n_missing_techniques,
        "n_eligible": len(elig),
    }
    return elig, counts


def _signal_markers_summary(inv: dict[str, pd.DataFrame], athlete: str) -> str:
    mk = inv["marker_availability.csv"]
    rows = mk[mk["athlete_id"] == athlete]
    sig_cols = ["RFIN", "RTOE", "LTOE", "RANK", "LANK", "RHEE", "LHEE"]
    if rows.empty:
        n_avail = 0
    else:
        n_avail = int(rows[sig_cols].any(axis=0).sum())
    return f"signal_markers={n_avail}/{N_SIGNAL_MARKERS}"


def build_candidate_frame(inv: dict[str, pd.DataFrame]):
    """Devuelve (df_elegibles, counts) con los vectores estructurales por atleta."""
    elig, counts = eligible_candidates(inv)
    if elig.empty:
        return elig, counts

    rows = []
    for _, r in elig.iterrows():
        aid = r["athlete_id"]
        derived = _derived_structure_of(inv, aid)
        e01e02_pat = derived.split("/")[0]
        pts = (int(r["n_points_min"]), int(r["n_points_max"]))
        n_trials = len(str(r["trials"]).split(","))
        techs = str(r["techniques"])
        conds = str(r["conditions"])

        cond_list = [c.strip() for c in conds.split(",")]
        b_e03 = "E03" not in cond_list
        b_e04 = "E04" not in cond_list
        extra = sorted({t for t in techs.split(",") if t and t not in TECHNIQUES})
        rows.append({
            "athlete_id": aid,
            "sampling_rate_hz": 250.0,
            "total_c3d": int(r["n_files"]),
            "techniques_present": techs,
            "conditions_present": conds,
            "trials_present": str(r["trials"]),
            "derived_structure": derived,
            "marker_structure_summary": _signal_markers_summary(inv, aid),
            "anomaly_status": "ok",
            "configuration_status": "NOT_CONFIGURED",
            "b_derived": int(e01e02_pat != MODAL_DERIVED_E01E02),
            "b_e03": int(b_e03),
            "b_e04": int(b_e04),
            "b_extra_tech": int(len(extra) > 0),
            "b_deep_trials": int(n_trials >= 5),
            "b_points": int(pts != MODAL_POINTS),
        })
    df = pd.DataFrame(rows)
    return df, counts


def _role_and_reason(row: pd.Series) -> tuple[str, str]:
    """Asigna rol de diversidad estructural y razón de selección (español)."""
    if row["b_derived"]:
        role = "derived_structure_variant"
        reason = (
            "Estructura de variables derivadas atípica en la cohorte 250 Hz "
            f"(E01/E02 = {row['derived_structure'].split('/')[0]} en lugar de 74); "
            "permite probar que el pipeline no depende accidentalmente del "
            "patrón derivado estándar."
        )
    elif row["b_e03"] or row["b_e04"]:
        missing = []
        if row["b_e03"]:
            missing.append("E03")
        if row["b_e04"]:
            missing.append("E04")
        role = "condition_gap_variant"
        reason = (
            f"Cobertura de condiciones parcial (sin {' ni '.join(missing)}); "
            "prueba que el pipeline tolera huecos de condición sin romper el "
            "contrato ni asumir condiciones ausentes."
        )
    elif row["b_extra_tech"]:
        role = "extra_technique_variant"
        reason = (
            "Posee una técnica adicional fuera de S01-S05 además de cubrirlas; "
            "prueba que el pipeline no falla ni sobreinterpreta etiquetas de "
            "técnica no esperadas."
        )
    elif row["b_deep_trials"]:
        role = "trials_depth_variant"
        reason = (
            "Profundidad de trials mayor al estándar T01-T02; prueba que la "
            "segmentación por ejecución escala a repeticiones múltiples."
        )
    elif row["b_points"]:
        role = "point_range_variant"
        reason = (
            "Rango de puntos derivados distinto al estándar; prueba que el "
            "pipeline no depende de una cobertura de marcadores particular."
        )
    else:
        role = "baseline_modal_variant"
        reason = (
            "Firma estructural modal de la cohorte 250 Hz / Grupo B; aporta "
            "un caso de referencia para la prueba controlada."
        )
    return role, reason


def select_candidates(df: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    """Selección exhaustiva del trío que maximiza la diversidad estructural.

    Objetivo interno (NO deportivo): maximizar la suma de distancias Hamming
    por pares sobre los bits estructurales. Empates resueltos por id menor.
    """
    if len(df) < k:
        raise ValueError(f"no hay candidatos suficientes ({len(df)} < {k})")

    idx = list(df.index)
    best_combo: tuple = ()
    best_key: tuple = (None,)
    for combo in itertools.combinations(idx, k):
        sub = df.loc[list(combo)]
        vecs = sub[BIT_COLUMNS].astype(int).to_numpy()
        total = 0
        for i in range(k):
            for j in range(i + 1, k):
                total += int(np.count_nonzero(vecs[i] != vecs[j]))
        key = (-total, tuple(sorted(sub["athlete_id"].tolist())))
        if best_key[0] is None or key < best_key:
            best_key = key
            best_combo = combo

    sel = df.loc[list(best_combo)].copy()
    sel = sel.assign(
        _n_bits=sel[BIT_COLUMNS].astype(int).sum(axis=1)
    ).sort_values(["_n_bits", "athlete_id"], ascending=[False, True])

    roles = [_role_and_reason(r) for _, r in sel.iterrows()]
    sel["diversity_role"] = [r[0] for r in roles]
    sel["selection_reason"] = [r[1] for r in roles]
    sel["rank_internal"] = pd.Series(range(1, len(sel) + 1), index=sel.index)

    return sel[OUTPUT_COLUMNS].reset_index(drop=True)


def run_selection():
    """Pipeline completo: carga, filtra, selecciona y escribe el CSV."""
    inv = load_inventory()
    df, counts = build_candidate_frame(inv)
    sel = select_candidates(df)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sel.to_csv(OUT_CSV, index=False)
    return sel, counts


def main() -> None:
    sel, counts = run_selection()
    print("[1.8F] Selección de 3 atletas candidatos (diversidad estructural)")
    print(f"[1.8F] Atletas totales      : {counts['n_total']}")
    print(f"[1.8F] 250 Hz               : {counts['n_250']}")
    print(f"[1.8F] 200 Hz (excluidos)   : {counts['n_200']}")
    print(f"[1.8F] Grupo C (revisión)   : {counts['n_group_c']}")
    print(f"[1.8F] Excluidos usados     : {counts['n_excluded_used']}")
    print(f"[1.8F] NOT_CONFIGURED       : {counts['n_not_configured']}")
    print(f"[1.8F] Con anomalías (excl.) : {counts['n_anomalies']}")
    print(f"[1.8F] Sin E01 (excl.)      : {counts['n_no_e01']}")
    print(f"[1.8F] Sin S01-S05 (excl.)  : {counts['n_missing_techniques']}")
    print(f"[1.8F] Candidatos elegibles : {counts['n_eligible']}")
    print("[1.8F] Seleccionados:")
    for _, r in sel.iterrows():
        print(f"[1.8F]   rk={int(r['rank_internal'])} {r['athlete_id']} "
              f"[{r['diversity_role']}] {r['selection_reason'][:90]}...")
    print(f"[1.8F] Guardado: {OUT_CSV}")


if __name__ == "__main__":
    main()