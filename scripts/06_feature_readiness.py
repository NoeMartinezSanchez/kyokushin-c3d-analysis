#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
06_feature_readiness.py
=======================
FASE 1.8B-1 — FEATURE READINESS AUDIT

Objetivo: verificar que las ejecuciones segmentadas pueden transformarse en un
dataset mínimo de features COMPARABLE entre B0367 y B0377.

Alcance (golden path):
  atletas : B0367, B0377
  técnicas: S02, S03, S05
  condición: E01
  trial   : T01

NO resuelve S01, NO añade E02/E03/E04/T02, NO hace ML/DTW/resampling.

Salidas:
  output/feature_readiness_sample.csv
  output/feature_readiness_audit.csv

Reutiliza el motor 02 (carga executions_sample y segmentation_events ya
generados por atleta). NO duplica la lógica de extract_features.
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
MM_IN_M = 1e-3

GOLDEN_TECHNIQUES = ["S02", "S03", "S05"]
GOLDEN_CONDITION = "E01"
GOLDEN_TRIAL = "T01"
ATHLETES = ["B0367", "B0377"]

# features solicitadas y columna de origen (en executions_sample por atleta)
FEATURE_MAP = {
    "duration_s": "duration_s",
    "time_to_peak_s": "time_to_peak_s",
    "vmax": "vmax_m_s",
    "vmean": "vmean_m_s",
    "amax": "amax_m_s2",
    "displacement": "displacement_m",
    "path_length": "path_length_m",
    "snr": "signal_snr",
}

FEATURE_UNITS = {
    "duration_s": "s", "time_to_peak_s": "s",
    "vmax": "m/s", "vmean": "m/s",
    "amax": "m/s^2",
    "displacement": "m", "path_length": "m",
    "hip_rom": "deg", "knee_rom": "deg", "ankle_rom": "deg",
    "snr": "adimensional",
}


def exec_path_for(athlete: str) -> Path:
    if athlete == "B0367":
        return OUT / "executions_sample.csv"
    return OUT / athlete / "executions_sample.csv"


def events_path_for(athlete: str) -> Path:
    if athlete == "B0367":
        return OUT / "segmentation_events.csv"
    return OUT / athlete / "segmentation_events.csv"


def load_executions(athlete: str) -> pd.DataFrame:
    p = exec_path_for(athlete)
    if not p.exists():
        print(f"  [warn] {p.name} no existe para {athlete}")
        return pd.DataFrame()
    return pd.read_csv(p)


def load_events(athlete: str) -> pd.DataFrame:
    p = events_path_for(athlete)
    if not p.exists():
        return pd.DataFrame(columns=["athlete_id", "technique", "condition",
                                     "trial", "event_id", "status", "rejection_reason"])
    return pd.read_csv(p)


def golden_rows(athlete: str) -> pd.DataFrame:
    """Ejecuciones del golden path para un atleta, con qc_status y quality_flag."""
    df = load_executions(athlete)
    if df.empty:
        return df
    mask = (df["technique"].isin(GOLDEN_TECHNIQUES)
            & (df["condition"] == GOLDEN_CONDITION)
            & (df["trial"] == GOLDEN_TRIAL))
    sub = df[mask].copy()
    if sub.empty:
        return sub

    # qc_status desde segmentation_events (por event_id) — decision: status del evento
    ev = load_events(athlete)
    if not ev.empty:
        ev_key = ev.set_index(["technique", "condition", "trial", "event_id"])["status"]
        sub_key = sub.set_index(["technique", "condition", "trial", "event_id"]).index
        sub["qc_status"] = sub.apply(
            lambda r: ev_key.get((r["technique"], r["condition"], r["trial"], r["event_id"]), "accepted"),
            axis=1)
    else:
        sub["qc_status"] = "accepted"

    # quality_flag desde execution_quality (OK/WARN)
    qp = OUT / (athlete if athlete != "B0367" else "") / "execution_quality.csv"
    # B0367 quality está en output/ ejecution_quality.csv
    qpath = OUT / "execution_quality.csv" if athlete == "B0367" else qp
    if qpath.exists():
        q = pd.read_csv(qpath)
        qmap = {(r["technique"], r["condition"], r["trial"], r["repetition"]): r["quality_flag"]
                for _, r in q.iterrows()}
        sub["quality_flag"] = sub.apply(
            lambda r: qmap.get((r["technique"], r["condition"], r["trial"], r["repetition"]), ""),
            axis=1)
    else:
        sub["quality_flag"] = ""

    return sub


def map_rom(df: pd.DataFrame, athlete: str) -> pd.DataFrame:
    """Mapea rom_{joints_side}HipAngles -> hip_rom (y knee/ankle) usando la config."""
    out = df.copy()
    for tech in GOLDEN_TECHNIQUES:
        side = str(_X02.JOINTS_SIDE.get(tech, "R")).upper()
        side = "L" if side == "L" else "R"
        for rom, col in [("hip_rom", f"rom_{side}HipAngles"),
                         ("knee_rom", f"rom_{side}KneeAngles"),
                         ("ankle_rom", f"rom_{side}AnkleAngles")]:
            if col in out.columns:
                out.loc[out["technique"] == tech, rom] = out.loc[
                    out["technique"] == tech, col]
            else:
                out.loc[out["technique"] == tech, rom] = np.nan
    return out


def build_sample() -> pd.DataFrame:
    """Construye feature_readiness_sample.csv."""
    frames = []
    for athlete in ATHLETES:
        df = golden_rows(athlete)
        if df.empty:
            print(f"  [warn] {athlete}: sin filas golden path")
            continue
        _X02.set_active_athlete(athlete)
        df = map_rom(df, athlete)
        # execution_id
        df["execution_id"] = (df["athlete_id"] + "|" + df["technique"] + "|"
                              + df["condition"] + "|" + df["trial"] + "|"
                              + df["repetition"].astype(str).str.zfill(3))
        cols = ["athlete_id", "technique", "condition", "trial", "execution_id"]
        for c in FEATURE_MAP:
            cols.append(FEATURE_MAP[c] if FEATURE_MAP[c] in df.columns else c)
        cols += ["hip_rom", "knee_rom", "ankle_rom", "qc_status", "quality_flag"]
        # filtrar a columnas existentes
        cols = [c for c in cols if c in df.columns]
        frames.append(df[cols].rename(columns={
            "vmax_m_s": "vmax", "vmean_m_s": "vmean", "amax_m_s2": "amax",
            "displacement_m": "displacement", "path_length_m": "path_length",
            "signal_snr": "snr"}))
    sample = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    # rellenar vmax/vmean/amax rename ya aplicado
    return sample


def build_audit() -> pd.DataFrame:
    """Tabla de comparabilidad por feature."""
    b36 = golden_rows("B0367")
    b37 = golden_rows("B0377")
    # para ROM: usar la columna fuente por lado (R en golden path JCoord=R)
    rom_src = {
        "hip_rom": "rom_RHipAngles",
        "knee_rom": "rom_RKneeAngles",
        "ankle_rom": "rom_RAnkleAngles",
    }
    rows = []
    features = ["duration_s", "time_to_peak_s", "vmax", "vmean", "amax",
                "displacement", "path_length", "hip_rom", "knee_rom",
                "ankle_rom", "snr"]
    for f in features:
        # columna de origen en executions (para R*)
        src = FEATURE_MAP.get(f, rom_src.get(f, f))
        av36 = (not b36.empty) and src in b36.columns and b36[src].notna().any()
        av37 = (not b37.empty) and src in b37.columns and b37[src].notna().any()
        rate_sens = f in ("vmax", "vmean", "amax", "path_length")
        side_sens = f in ("hip_rom", "knee_rom", "ankle_rom")
        direct = f in ("duration_s", "time_to_peak_s")
        # displacement es geométrico (inicial->final): la magnitud no depende de Hz,
        # solo la elección de frames start/end (detector). Por eso "caveat", no "rate".
        disp_caveat = f == "displacement"
        rows.append({
            "feature": f,
            "available_B0367": bool(av36),
            "available_B0377": bool(av37),
            "same_definition": True if (av36 and av37) else (None if av36 != av37 else True),
            "same_units": True,
            "units": FEATURE_UNITS.get(f, ""),
            "sampling_rate_sensitive": rate_sens,
            "side_sensitive": side_sens,
            "directly_comparable": direct,
            "notes": ("indep de Hz" if direct else
                      ("depende de joints_side" if side_sens else
                       ("magnitud geométrica; Hz solo indirecta vía frames" if disp_caveat else
                        ("requiere normalizacion temporal 200/250 Hz" if rate_sens else "")))),
        })
    return pd.DataFrame(rows)


def main():
    print("FASE 1.8B-1 — FEATURE READINESS AUDIT")
    print(f"Golden path: {ATHLETES} × {GOLDEN_TECHNIQUES} × {GOLDEN_CONDITION}-{GOLDEN_TRIAL}")

    sample = build_sample()
    if sample.empty:
        print("[ERROR] sin filas para construir la muestra")
        return
    sample.to_csv(OUT / "feature_readiness_sample.csv", index=False)
    print(f"[guardado] {OUT/'feature_readiness_sample.csv'} ({len(sample)} filas)")

    print("\nEjecuciones del golden path:")
    print(sample[["athlete_id", "technique", "condition", "trial", "execution_id",
                  "duration_s", "vmax", "amax", "hip_rom", "knee_rom",
                  "ankle_rom", "snr", "qc_status"]].to_string(index=False))

    audit = build_audit()
    audit.to_csv(OUT / "feature_readiness_audit.csv", index=False)
    print(f"\n[guardado] {OUT/'feature_readiness_audit.csv'}")
    print("\nAuditoría de comparabilidad:")
    print(audit.to_string(index=False))


if __name__ == "__main__":
    main()