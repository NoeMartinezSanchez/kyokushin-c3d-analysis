#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
07_build_athlete_data_mart.py
============================
FASE 1.8C — ATHLETE DATA MART

Construye la capa de datos analítica estable que consumirá el futuro Dashboard
(y posteriormente ML). NO es un motor de procesamiento: es una capa de
CONSOLIDACIÓN que lee outputs ya generados y no relee C3D.

Fuentes:
  output/feature_readiness_sample.csv   (golden path, 18 ejecuciones)
  output/executions_sample.csv          (metadata B0367)
  output/B0377/executions_sample.csv    (metadata B0377)
  config/athletes/<id>.yaml             (joints_side / movement_side)

Salidas:
  output/data_mart/athlete_execution_features.csv
  output/data_mart/data_mart_summary.csv

Reglas:
  - 1 fila = 1 ejecución individual
  - execution_id único y estable
  - conserva features raw SIN normalizar (200/250 Hz intactos)
  - separa "feature calculada" de "estado de comparabilidad"
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SPEC = importlib.util.spec_from_file_location(
    "execution_segmentation_module",
    str(Path(__file__).resolve().parent / "02_execution_segmentation.py"),
)
_X02 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_X02)

ROOT = _X02.ROOT
OUT = _X02.OUT
MART_DIR = OUT / "data_mart"
MART_DIR.mkdir(parents=True, exist_ok=True)

GOLDEN_PATH = {
    "athletes": ["B0367", "B0377"],
    "techniques": ["S02", "S03", "S05"],
    "condition": "E01",
    "trial": "T01",
}

# Versiones (siguen el patrón SEGMENTATION_VERSION del motor)
SEGMENTATION_VERSION = _X02.SEGMENTATION_VERSION  # del motor (ej. "1.8.0")
FEATURE_VERSION = "1.0"      # versión del Feature Dataset (capaz de features)
UNITS_VERSION = "1.0"        # versión de las conversiones/unidades
MART_VERSION = "1.0"         # versión de esta capa Data Mart
SOURCE_DATASET = "feature_readiness_sample"  # confirmado por el usuario

# Estados de comparabilidad (según FASE 1.8B-1). Clases permitidas:
DIRECTLY_COMPARABLE = "DIRECTLY_COMPARABLE"
COMPARABLE_WITH_CAVEAT = "COMPARABLE_WITH_CAVEAT"
REQUIRES_NORMALIZATION = "REQUIRES_NORMALIZATION"
NEEDS_VALIDATION = "NEEDS_VALIDATION"
NOT_AVAILABLE = "NOT_AVAILABLE"

COMPARABILITY = {
    "duration_s": DIRECTLY_COMPARABLE,
    "time_to_peak_s": DIRECTLY_COMPARABLE,  # caveat: relativo al evento, no impacto
    "vmax": REQUIRES_NORMALIZATION,
    "vmean": REQUIRES_NORMALIZATION,
    "amax": REQUIRES_NORMALIZATION,
    # displacement es geométrico (inicial->final): magnitud independiente de Hz;
    # Hz solo influye indirectamente vía la elección de frames start/end.
    "displacement": COMPARABLE_WITH_CAVEAT,
    "path_length": REQUIRES_NORMALIZATION,  # sumatorio por frame: depende de discretización
    "hip_rom": COMPARABLE_WITH_CAVEAT,   # depende de joints_side
    "knee_rom": COMPARABLE_WITH_CAVEAT,
    "ankle_rom": COMPARABLE_WITH_CAVEAT,
    "snr": COMPARABLE_WITH_CAVEAT,
}

FEATURE_COLS = ["duration_s", "time_to_peak_s", "vmax", "vmean", "amax",
                "displacement", "path_length", "hip_rom", "knee_rom",
                "ankle_rom", "snr"]

# --------------------------------------------------------------------------- #
# 1. Carga
# --------------------------------------------------------------------------- #

def load_sources() -> pd.DataFrame:
    fr = pd.read_csv(OUT / "feature_readiness_sample.csv")
    # feature_readiness no trae `repetition`; la extraemos del execution_id (último campo)
    fr["repetition"] = fr["execution_id"].str.split("|").str[-1].astype(int)
    fr["execution_id"] = (fr["athlete_id"] + "|" + fr["technique"] + "|"
                          + fr["condition"] + "|" + fr["trial"] + "|"
                          + fr["repetition"].astype(str).str.zfill(3))
    exec_frames = {}
    for athlete in GOLDEN_PATH["athletes"]:
        p = OUT / "executions_sample.csv" if athlete == "B0367" else OUT / athlete / "executions_sample.csv"
        if not p.exists():
            print(f"  [warn] falta {p.name} para {athlete}")
            continue
        d = pd.read_csv(p)
        d["execution_id"] = (d["athlete_id"] + "|" + d["technique"] + "|"
                             + d["condition"] + "|" + d["trial"] + "|"
                             + d["repetition"].astype(int).astype(str).str.zfill(3))
        exec_frames[athlete] = d.set_index("execution_id")
    return fr, exec_frames


# --------------------------------------------------------------------------- #
# 2. Consolidación
# --------------------------------------------------------------------------- #

def build_mart() -> pd.DataFrame:
    fr, exec_frames = load_sources()
    rows = []
    for _, r in fr.iterrows():
        eid = r["execution_id"]
        exec_df = exec_frames.get(r["athlete_id"])
        meta = exec_df.loc[eid].to_dict() if (exec_df is not None and eid in exec_df.index) else {}

        # movement_side desde JOINTS_SIDE (config por técnica), nunca hardcode
        side = str(_X02.JOINTS_SIDE.get(r["technique"], "R")).upper()
        side = "L" if side == "L" else "R"

        row = {
            # identidad
            "athlete_id": r["athlete_id"],
            "execution_id": eid,
            "technique": r["technique"],
            "condition": r["condition"],
            "trial": r["trial"],
            "repetition": int(r["repetition"]),
            # adquisición
            "sampling_rate_hz": meta.get("sampling_rate_hz", np.nan),
            "primary_signal": meta.get("signal_used", r.get("signal_used", np.nan)),
            "movement_side": side,
            "event_id": meta.get("event_id", np.nan),
        }
        # variables temporales + cinemática + articulaciones + calidad
        for c in FEATURE_COLS:
            row[c] = r.get(c, np.nan)
        row["qc_status"] = r.get("qc_status", "accepted")
        row["quality_flag"] = r.get("quality_flag", "")
        # comparabilidad por feature
        for c in FEATURE_COLS:
            row[f"comparability_{c}"] = COMPARABILITY.get(c, NEEDS_VALIDATION)
        # metadatos
        row["feature_version"] = FEATURE_VERSION
        row["segmentation_version"] = SEGMENTATION_VERSION
        row["units_version"] = UNITS_VERSION
        row["source_dataset"] = SOURCE_DATASET
        row["mart_version"] = MART_VERSION
        rows.append(row)

    mart = pd.DataFrame(rows)
    return mart


# --------------------------------------------------------------------------- #
# 3. Validaciones
# --------------------------------------------------------------------------- #

def validate_mart(mart: pd.DataFrame) -> list[str]:
    issues = []
    if len(mart) != 18:
        issues.append(f"filas={len(mart)} != 18")
    # identidad
    if not mart["execution_id"].is_unique:
        issues.append("execution_id duplicado")
    if not set(mart["athlete_id"]) == set(GOLDEN_PATH["athletes"]):
        issues.append("athlete_id fuera del golden path")
    if not set(mart["technique"]) == set(GOLDEN_PATH["techniques"]):
        issues.append("technique fuera del golden path")
    if not set(mart["condition"]) == {"E01"}:
        issues.append("condition distinta de E01")
    if not set(mart["trial"]) == {"T01"}:
        issues.append("trial distinto de T01")
    # calidad
    if not (mart["qc_status"] == "accepted").all():
        issues.append("qc_status no es accepted en todas")
    if not (mart["quality_flag"] == "OK").all():
        issues.append("quality_flag no es OK en todas")
    for c in FEATURE_COLS:
        if mart[c].isna().any():
            issues.append(f"NaN en feature mínima {c}")
    # unidades/frecuencia
    rates = {a: mart.loc[mart["athlete_id"] == a, "sampling_rate_hz"].unique()
             for a in GOLDEN_PATH["athletes"]}
    if not all(rates["B0367"] == [200.0]):
        issues.append(f"B0367 rate != 200 Hz: {rates['B0367']}")
    if not all(rates["B0377"] == [250.0]):
        issues.append(f"B0377 rate != 250 Hz: {rates['B0377']}")
    # movimiento lado = R en golden path
    if not (mart["movement_side"] == "R").all():
        issues.append("movement_side != R en golden path")
    return issues


# --------------------------------------------------------------------------- #
# 4. Summary
# --------------------------------------------------------------------------- #

def build_summary(mart: pd.DataFrame) -> pd.DataFrame:
    missing = int(mart[FEATURE_COLS].isna().sum().sum())
    summary = {
        "total_executions": int(len(mart)),
        "total_athletes": int(mart["athlete_id"].nunique()),
        "total_techniques": int(mart["technique"].nunique()),
        "total_conditions": int(mart["condition"].nunique()),
        "accepted_executions": int((mart["qc_status"] == "accepted").sum()),
        "review_executions": int((mart["qc_status"] == "review").sum()),
        "rejected_executions": int((mart["qc_status"] == "rejected").sum()),
        "missing_feature_values": missing,
        "sampling_rate_200hz": int((mart["sampling_rate_hz"] == 200.0).sum()),
        "sampling_rate_250hz": int((mart["sampling_rate_hz"] == 250.0).sum()),
        "mart_version": MART_VERSION,
        "source_dataset": SOURCE_DATASET,
    }
    return pd.DataFrame([summary])


# --------------------------------------------------------------------------- #
# MAIN
# --------------------------------------------------------------------------- #

def main():
    print(f"FASE 1.8C — ATHLETE DATA MART")
    print(f"Versiones: segmentation={SEGMENTATION_VERSION}, features={FEATURE_VERSION}, "
          f"units={UNITS_VERSION}, mart={MART_VERSION}")
    print(f"Fuente: {SOURCE_DATASET}")

    mart = build_mart()
    if mart.empty:
        print("[ERROR] Data Mart vacío")
        return

    issues = validate_mart(mart)
    if issues:
        print("\n[VALIDACIÓN] PROBLEMAS ENCONTRADOS:")
        for i in issues:
            print(f"  - {i}")
    else:
        print("\n[VALIDACIÓN] OK: 18 filas, identidad/calidad/unidades/frecuencia coherentes.")

    mart.to_csv(MART_DIR / "athlete_execution_features.csv", index=False)
    print(f"[guardado] {MART_DIR/'athlete_execution_features.csv'} ({len(mart)} filas)")

    summary = build_summary(mart)
    summary.to_csv(MART_DIR / "data_mart_summary.csv", index=False)
    print(f"[guardado] {MART_DIR/'data_mart_summary.csv'}")

    # resumen
    print("\nDistribución por atleta/técnica:")
    print(mart.groupby(["athlete_id", "technique"]).size().to_string())
    print("\nSampling rates:")
    print(mart.groupby("athlete_id")["sampling_rate_hz"].unique().to_string())
    print("\nEjemplo de fila:")
    print(mart.iloc[0].to_string())


if __name__ == "__main__":
    main()