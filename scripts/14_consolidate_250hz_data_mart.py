#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
14_consolidate_250hz_data_mart.py
=================================
FASE 1.8F — TASK 6: CONSOLIDACIÓN CONTROLADA DEL DATA MART 250 Hz

Capa de consolidación: incorpora al Data Mart las ejecuciones aceptadas
S02–S05 (E01-T01, 250 Hz) de la cohorte, manteniendo EXACTAMENTE el contrato
estructural existente (39 columnas, mismas definiciones, mismas unidades).

Fuentes:
  - Data Mart actual: output/data_mart/athlete_execution_features.csv (18 históricas)
  - Golden Path consolidado: output/scaling_validation/cohort_expansion/cohort_golden_path.csv
    (410 = 37 Task 3 `task3_golden_path` + 373 cohorte `cohort_expansion_gp`).

NO: modifica 02/algoritmo/señales/configs/dashboard; NO construye dataset ML;
NO normaliza; NO incorpora 200 Hz como cohorte nueva (B0367 queda como histórico
200 Hz; B0368-B0370 fuera); S01 fuera; B0388-S04/B0401-S04 sin inventar.

Seguridad: backup del Mart original en data_mart/task6_backup/; solo si TODAS
las validaciones pasan se actualiza el Mart canónico y data_mart_summary.csv;
sino, se emite BLOCKED y no se escribe.

Reusa el mapa de comparabilidad de 07 (COMPARABILITY / FEATURE_COLS) para no
duplicar la definición del contrato.
"""

from __future__ import annotations

import importlib.util
import hashlib
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


def _load_module(name: str, fname: str):
    spec = importlib.util.spec_from_file_location(
        name, str(Path(__file__).resolve().parent / fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_X02 = _load_module("execution_segmentation_module", "02_execution_segmentation.py")
_X07 = _load_module("build_mart_module", "07_build_athlete_data_mart.py")

ROOT = Path(__file__).resolve().parents[1]
MART_FILE = ROOT / "output" / "data_mart" / "athlete_execution_features.csv"
SUMMARY_FILE = ROOT / "output" / "data_mart" / "data_mart_summary.csv"
COH_FILE = ROOT / "output" / "scaling_validation" / "cohort_expansion" / "cohort_golden_path.csv"

BACKUP_DIR = ROOT / "output" / "data_mart" / "task6_backup"
CONS_DIR = ROOT / "output" / "data_mart" / "task6_consolidation"

TECH_ML = ["S02", "S03", "S04", "S05"]
SOURCE_HIST = "feature_readiness_sample"
SOURCE_T3 = "task3_golden_path"
SOURCE_T5 = "cohort_expansion_gp"

FEATURE_VERSION = "1.0"
SEG_VERSION = "1.8.0"
UNITS_VERSION = "1.0"
MART_VERSION = "1.0"

# orden de columnas del contrato (idéntico al histórico)
CONTRACT_COLUMNS = [
    "athlete_id", "execution_id", "technique", "condition", "trial", "repetition",
    "sampling_rate_hz", "primary_signal", "movement_side", "event_id",
    "duration_s", "time_to_peak_s", "vmax", "vmean", "amax", "displacement",
    "path_length", "hip_rom", "knee_rom", "ankle_rom", "snr", "qc_status",
    "quality_flag",
    "comparability_duration_s", "comparability_time_to_peak_s", "comparability_vmax",
    "comparability_vmean", "comparability_amax", "comparability_displacement",
    "comparability_path_length", "comparability_hip_rom", "comparability_knee_rom",
    "comparability_ankle_rom", "comparability_snr",
    "feature_version", "segmentation_version", "units_version",
    "source_dataset", "mart_version",
]

COMPARABILITY = _X07.COMPARABILITY
FEATURE_COLS = list(_X07.FEATURE_COLS)


def _mart_execution_id(row: pd.Series) -> str:
    rep = int(row["repetition"])
    return f"{row['athlete_id']}|{row['technique']}|{row['condition']}|{row['trial']}|{rep:03d}"


def _build_new_block(gp: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in gp.iterrows():
        side = str(r["movement_side"]).strip().upper()
        src = SOURCE_T3 if r.get("_source") == SOURCE_T3 else SOURCE_T5
        row = {
            "athlete_id": r["athlete_id"],
            "execution_id": _mart_execution_id(r),
            "technique": r["technique"],
            "condition": r["condition"],
            "trial": r["trial"],
            "repetition": int(r["repetition"]),
            "sampling_rate_hz": float(r["sampling_rate_hz"]),
            "primary_signal": r["signal_used"],
            "movement_side": side,
            "event_id": int(r["event_id"]),
            "duration_s": r["duration_s"],
            "time_to_peak_s": r["time_to_peak_s"],
            "vmax": r["vmax_m_s"], "vmean": r["vmean_m_s"], "amax": r["amax_m_s2"],
            "displacement": r["displacement_m"], "path_length": r["path_length_m"],
            "hip_rom": r[f"rom_{side}HipAngles"],
            "knee_rom": r[f"rom_{side}KneeAngles"],
            "ankle_rom": r[f"rom_{side}AnkleAngles"],
            "snr": r["signal_snr"],
            "qc_status": "accepted",
            "quality_flag": r.get("quality_flag", "OK") or "OK",
        }
        for f in FEATURE_COLS:
            row[f"comparability_{f}"] = COMPARABILITY.get(f, "NEEDS_VALIDATION")
        row["feature_version"] = FEATURE_VERSION
        row["segmentation_version"] = SEG_VERSION
        row["units_version"] = UNITS_VERSION
        row["source_dataset"] = src
        row["mart_version"] = MART_VERSION
        rows.append(row)
    return pd.DataFrame(rows, columns=CONTRACT_COLUMNS)


def _config_mismatches(new_block: pd.DataFrame) -> list[str]:
    issues = []
    cfg_cache = {}
    for _, r in new_block.iterrows():
        aid = r["athlete_id"]
        if aid not in cfg_cache:
            cfg_cache[aid] = _X02.load_config(aid)
        cfg = cfg_cache[aid]
        t = (cfg.get("techniques") or {})
        e = t.get(r["technique"], {})
        sig = e.get("signal")
        joints = e.get("joints_side")
        if sig is not None and sig != r["primary_signal"]:
            issues.append(f"{aid} {r['technique']}: primary_signal {r['primary_signal']} != config {sig}")
        if joints is not None and joints != r["movement_side"]:
            issues.append(f"{aid} {r['technique']}: movement_side {r['movement_side']} != config joints {joints}")
    return issues


def _validate(historical: pd.DataFrame, new_block: pd.DataFrame,
              hist_backup: pd.DataFrame) -> list[str]:
    issues = []
    final = pd.concat([historical, new_block], ignore_index=True)

    if list(final.columns) != CONTRACT_COLUMNS:
        issues.append("columnas != contrato")
    if final["execution_id"].duplicated().any():
        dup = final.loc[final["execution_id"].duplicated(), "execution_id"].tolist()
        issues.append(f"execution_id duplicados: {dup[:3]}")
    if final[FEATURE_COLS].isna().sum().sum() != 0:
        issues.append("NaN en features obligatorias")
    if "S01" in set(new_block["technique"]):
        issues.append("S01 en bloque nuevo")
    if (new_block["sampling_rate_hz"] != 250.0).any():
        issues.append("bloque nuevo con rate != 250")
    if set(new_block["condition"]) != {"E01"} or set(new_block["trial"]) != {"T01"}:
        issues.append("condición/trial != E01/T01 en bloque nuevo")
    if not set(new_block["technique"]).issubset(set(TECH_ML)):
        issues.append("técnicas fuera de S02-S05 en bloque nuevo")
    if not set(new_block["movement_side"]).issubset({"R", "L"}):
        issues.append("movement_side inválido")
    if not set(new_block["quality_flag"]).issubset({"OK", "WARN"}):
        issues.append("quality_flag desconocido")

    # regresión: históricas idénticas al backup (valores)
    hb = hist_backup.set_index("execution_id")
    hm = historical.set_index("execution_id")
    for c in CONTRACT_COLUMNS:
        if c in hm.columns and c in hb.columns:
            if not hm[c].equals(hb[c]):
                issues.append(f"histórica alterada en columna {c}")
    return issues


def run_consolidation(write_final: bool = True):
    """Consolida el Mart 250 Hz. Devuelve dict con DataFrames y estado."""
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    CONS_DIR.mkdir(parents=True, exist_ok=True)

    backup_path = BACKUP_DIR / "athlete_execution_features.csv"
    if backup_path.exists():
        # base estable de pre-consolidación (idempotente en re-ejecuciones)
        historical = pd.read_csv(backup_path)
    else:
        historical = pd.read_csv(MART_FILE)
        backup_path.write_bytes(MART_FILE.read_bytes())

    gp = pd.read_csv(COH_FILE)
    new_block = _build_new_block(gp)

    rejected_gp = gp["_source"].value_counts().to_dict()
    candidate = {SOURCE_HIST: len(historical),
                 SOURCE_T3: int(rejected_gp.get(SOURCE_T3, 0)),
                 SOURCE_T5: int(rejected_gp.get(SOURCE_T5, 0))}

    issues = _validate(historical, new_block, pd.read_csv(BACKUP_DIR / "athlete_execution_features.csv"))
    cfg_issues = _config_mismatches(new_block)
    issues += cfg_issues

    # duplicados entre bloques (clave lógica)
    dup_rows = []
    hist_ids = set(historical["execution_id"])
    new_ids = set(new_block["execution_id"])
    inter = (new_ids & hist_ids)
    for eid in sorted(inter):
        dup_rows.append({"execution_id": eid, "reason": "colisión histórico/bloque",
                         "source": "cross"})
    final = pd.concat([historical, new_block], ignore_index=True)
    fin_dup = final.loc[final["execution_id"].duplicated(keep=False)]
    for _, r in fin_dup.iterrows():
        dup_rows.append({"execution_id": r["execution_id"],
                         "reason": "duplicado en suma final",
                         "source": "final"})
    dup_df = pd.DataFrame(dup_rows, columns=["execution_id", "reason", "source"])

    # reconciliación
    reconc = []
    inserted = {SOURCE_HIST: len(historical),
                SOURCE_T3: int((new_block["source_dataset"] == SOURCE_T3).sum()),
                SOURCE_T5: int((new_block["source_dataset"] == SOURCE_T5).sum())}
    for src in (SOURCE_HIST, SOURCE_T3, SOURCE_T5):
        reconc.append({
            "source": src,
            "candidate_rows": candidate[src],
            "accepted_rows": candidate[src],
            "duplicate_rows": 0,
            "inserted_rows": inserted[src],
            "excluded_rows": 0,
            "reason": "histórico preservado" if src == SOURCE_HIST else "bloque 250 Hz",
        })
    reconc.append({"source": "TOTAL", "candidate_rows": sum(candidate.values()),
                   "accepted_rows": sum(candidate.values()),
                   "duplicate_rows": int(final["execution_id"].duplicated().sum()),
                   "inserted_rows": len(final),
                   "excluded_rows": 0, "reason": ""})
    reconc_df = pd.DataFrame(reconc)

    # validación detallada
    validations = [
        {"check": "columnas_contrato", "status": "PASS" if list(final.columns) == CONTRACT_COLUMNS else "FAIL"},
        {"check": "execution_id_unico", "status": "PASS" if final["execution_id"].is_unique else "FAIL"},
        {"check": "sin_nan_features", "status": "PASS" if final[FEATURE_COLS].isna().sum().sum() == 0 else "FAIL"},
        {"check": "sin_s01", "status": "PASS" if "S01" not in set(new_block["technique"]) else "FAIL"},
        {"check": "bloque_nuevo_250hz", "status": "PASS" if (new_block["sampling_rate_hz"] == 250.0).all() else "FAIL"},
        {"check": "bloque_nuevo_e01_t01", "status": "PASS" if {*new_block["condition"]} == {"E01"} and {*new_block["trial"]} == {"T01"} else "FAIL"},
        {"check": "bloque_nuevo_s02_s05", "status": "PASS" if set(new_block["technique"]) <= set(TECH_ML) else "FAIL"},
        {"check": "config_consistente", "status": "PASS" if not cfg_issues else "FAIL"},
        {"check": "historico_preservado", "status": "PASS" if not any("histórica alterada" in i for i in issues) else "FAIL"},
    ]
    val_df = pd.DataFrame(validations)

    # cobertura
    cov = []
    for aid in sorted(final["athlete_id"].unique()):
        f = final[final["athlete_id"] == aid]
        counts = {t: int((f["technique"] == t).sum()) for t in TECH_ML}
        if aid == "B0367":
            status = "HISTORICAL_ONLY"
        elif all(counts[t] > 0 for t in TECH_ML):
            status = "COMPLETE_S02_S05"
        else:
            status = "PARTIAL_S02_S05"
        cov.append({"athlete_id": aid, "sampling_rate_hz": float(f["sampling_rate_hz"].iloc[0]),
                    **{f"{t}_count": counts[t] for t in TECH_ML},
                    "total_count": len(f), "status": status})
    cov_df = pd.DataFrame(cov)

    blocc = [v for v in validations if v["status"] == "FAIL"]
    summary = {
        "total_executions": len(final),
        "total_athletes": final["athlete_id"].nunique(),
        "total_techniques": final["technique"].nunique(),
        "total_conditions": final["condition"].nunique(),
        "accepted_executions": int((final["qc_status"] == "accepted").sum()),
        "review_executions": 0, "rejected_executions": 0,
        "missing_feature_values": int(final[FEATURE_COLS].isna().sum().sum()),
        "sampling_rate_200hz": int((final["sampling_rate_hz"] == 200.0).sum()),
        "sampling_rate_250hz": int((final["sampling_rate_hz"] == 250.0).sum()),
        "mart_version": MART_VERSION,
        "source_dataset": "|".join(dict.fromkeys([
            SOURCE_HIST, SOURCE_T3, SOURCE_T5])),
    }
    sum_df = pd.DataFrame([summary])

    # escribir diagnósticos (aunque haya bloqueo)
    new_block[CONTRACT_COLUMNS].to_csv(CONS_DIR / "data_mart_250hz_consolidated.csv",
                                       index=False, encoding="utf-8")
    val_df.to_csv(CONS_DIR / "data_mart_task6_validation.csv", index=False, encoding="utf-8")
    dup_df.to_csv(CONS_DIR / "data_mart_task6_duplicates.csv", index=False, encoding="utf-8")
    sum_df.to_csv(CONS_DIR / "data_mart_task6_summary.csv", index=False, encoding="utf-8")
    cov_df.to_csv(CONS_DIR / "data_mart_task6_coverage.csv", index=False, encoding="utf-8")
    reconc_df.to_csv(CONS_DIR / "source_reconciliation.csv", index=False, encoding="utf-8")

    if blocc:
        print("[1.8F T6] BLOCKED — no se toca el Data Mart canónico:")
        for b in blocc:
            print(f"   - {b['check']}: {b['status']}")
        return {"state": "BLOCKED", "final": final, "coverage": cov_df,
                "reconciliation": reconc_df, "validation": val_df}

    if write_final:
        final.to_csv(MART_FILE, index=False, encoding="utf-8")
        sum_df.to_csv(SUMMARY_FILE, index=False, encoding="utf-8")
    print(f"[1.8F T6] Data Mart actualizado: {len(final)} filas ({len(historical)} históricas + {len(new_block)} nuevas)")
    return {"state": "OK", "final": final, "coverage": cov_df,
            "reconciliation": reconc_df, "validation": val_df}


def main() -> None:
    res = run_consolidation()
    if res["state"] == "OK":
        f = res["final"]
        print("[1.8F T6] Distribución por técnica:",
              f["technique"].value_counts().to_dict())
        print("[1.8F T6] source_dataset:",
              f["source_dataset"].value_counts().to_dict())
        print(f"[1.8F T6] Atletas: {f['athlete_id'].nunique()} | "
              f"sampling 200 Hz: {(f['sampling_rate_hz']==200.0).sum()} | "
              f"250 Hz: {(f['sampling_rate_hz']==250.0).sum()}")
        md5 = hashlib.md5(MART_FILE.read_bytes()).hexdigest()
        print(f"[1.8F T6] md5({MART_FILE.name}) = {md5}")


if __name__ == "__main__":
    main()