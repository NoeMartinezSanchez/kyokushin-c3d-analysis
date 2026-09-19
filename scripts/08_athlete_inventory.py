#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
08_athlete_inventory.py
=======================
FASE 1.8E — FULL ATHLETE INVENTORY & DATASET HOMOGENEITY AUDIT

Inventario estructural completo del dataset (actualmente 36 atletas / 1411 C3D
descargados en atletas/). Es una capa de AUDITORÍA, NO de procesamiento masivo:
no ejecuta segmentación, no construye Data Mart, no crea configs.

Fuente canónica: atletas/  (se EXCLUYE la copia histórica B0367/ de la raíz,
que es idéntica a atletas/B0367/ — 26 archivos — para no duplicar filas).

Enfoque en DOS PASES:
  Pase 1 (rápido): muestreo de 1-2 archivos por atleta para sampling rate,
                    unidades y estructura básica -> quick_summary.
  Pase 2 (completo): lectura de TODOS los C3D com barra de progreso, logging y
                    persistencia incremental (checkpoint por atleta).

Reutiliza inspect_c3d / parse_any_name / is_derived / discover_athletes de 04.

Salidas en output/athlete_inventory/<*.csv> (11 archivos, ver REPORTE).
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_SPEC = importlib.util.spec_from_file_location(
    "dataset_exploration_module",
    str(Path(__file__).resolve().parent / "01_dataset_exploration.py"),
)
_X01 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_X01)

_SPEC4 = importlib.util.spec_from_file_location(
    "athlete_generalization_module",
    str(Path(__file__).resolve().parent / "04_athlete_generalization_audit.py"),
)
_X04 = importlib.util.module_from_spec(_SPEC4)
_SPEC4.loader.exec_module(_X04)

ROOT = _X01.ROOT
OUT_DIR = ROOT / "output" / "athlete_inventory"
CKPT_DIR = OUT_DIR / "_checkpoint"
for _d in (OUT_DIR, CKPT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# reutilizar funciones de 04 (no duplicar lógica)
parse_any_name = _X04.parse_any_name
is_derived = _X04.is_derived
inspect_c3d = _X04.inspect_c3d

# fuente canónica: atletas/ (excluye B0367/ raíz)
DATA_ROOT = ROOT / "atletas"

# señales primarias que el pipeline usa (para el marcado de disponibilidad)
SIGNAL_MARKERS = ["RFIN", "RTOE", "LTOE", "RANK", "LANK", "RHEE", "LHEE"]

# señales usadas por técnica en B0367 (para marcar gap)
CONFIGURED_ATHLETES = ["B0367", "B0377"]
TECHNIQUES = ["S01", "S02", "S03", "S04", "S05"]
CONDITIONS = ["E01", "E02", "E03", "E04"]
TRIALS = ["T01", "T02"]


def discover_all_athletes() -> list[str]:
    return [d.name for d in sorted(DATA_ROOT.glob("B0*")) if d.is_dir()]


def all_files_for(athlete: str) -> list[Path]:
    return sorted((DATA_ROOT / athlete).rglob("*.c3d"))


# --------------------------------------------------------------------------- #
# PASE 1 — muestreo rápido (sampling rate / unidades / estructura básica)
# --------------------------------------------------------------------------- #

def quick_sample(athletes: list[str], n_sample: int = 2) -> pd.DataFrame:
    rows = []
    for ath in athletes:
        files = all_files_for(ath)
        if not files:
            rows.append({"athlete": ath, "n_files": 0, "sample_rate_hz": np.nan,
                         "units_pos": "", "units_angle": "", "n_points_min": np.nan,
                         "n_points_max": np.nan, "note": "sin archivos"})
            continue
        sample = files[:n_sample]
        rates, upos, uangle = set(), set(), set()
        npts = []
        for fp in sample:
            meta = inspect_c3d(fp)
            rates.add(meta["rate_hz"])
            upos.add(meta["units_pos"])
            uangle.add(meta["units_angle"])
            npts.append(meta["n_points"])
        rows.append({
            "athlete": ath, "n_files": len(files),
            "sample_rate_hz": sorted(rates) if len(rates) == 1 else sorted(rates),
            "units_pos": sorted(upos) if len(upos) == 1 else sorted(upos),
            "units_angle": sorted(uangle) if len(uangle) == 1 else sorted(uangle),
            "n_points_min": min(npts), "n_points_max": max(npts),
            "note": "muestra de %d archivos" % min(len(sample), n_sample),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# PASE 2 — lectura completa con checkpoint por atleta
# --------------------------------------------------------------------------- #

def _ckpt_path(athlete: str) -> Path:
    return CKPT_DIR / f"{athlete}.csv"


def load_checkpoint(athlete: str) -> pd.DataFrame | None:
    p = _ckpt_path(athlete)
    if p.exists():
        return pd.read_csv(p)
    return None


def save_checkpoint(athlete: str, df: pd.DataFrame):
    df.to_csv(_ckpt_path(athlete), index=False)


def scan_file(fp: Path, athlete: str, info: dict, meta: dict) -> dict:
    labels = meta.get("labels", [])
    # marcador disponible si existe sin prefijo o con prefijo del atleta
    def _avail(mk: str) -> bool:
        return mk in labels or f"{athlete}:{mk}" in labels or any(
            l.endswith(f":{mk}") for l in labels)
    signal_avail = {m: _avail(m) for m in SIGNAL_MARKERS}
    return {
        "file_name": fp.name,
        "source_path": str(fp.relative_to(ROOT)),
        "athlete_id": athlete,
        "technique": info["technique"],
        "condition": info["condition"],
        "trial": info["trial"],
        "sampling_rate_hz": meta["rate_hz"],
        "units_position": meta["units_pos"],
        "units_angle": meta["units_angle"],
        "units_force": meta["units_force"],
        "units_moment": meta["units_moment"],
        "units_power": meta["units_power"],
        "number_of_points": meta["n_points"],
        "number_of_frames": meta["n_frames"],
        "duration_s": round(meta["n_frames"] / meta["rate_hz"], 3)
        if meta["rate_hz"] and meta["rate_hz"] == meta["rate_hz"] else np.nan,
        "number_of_derived_variables": meta["n_derived"],
        "subject_ids": "|".join(meta["subjects"]) or "",
        "subject_prefixes": "|".join(meta["prefixes"]) or "",
        "tarcza_marker_count": meta["n_tarcza"],
        "analog_used": meta["analog_used"],
        "signal_avail_RFIN": signal_avail["RFIN"],
        "signal_avail_RTOE": signal_avail["RTOE"],
        "signal_avail_LTOE": signal_avail["LTOE"],
        "signal_avail_RANK": signal_avail["RANK"],
        "signal_avail_LANK": signal_avail["LANK"],
        "signal_avail_RHEE": signal_avail["RHEE"],
        "signal_avail_LHEE": signal_avail["LHEE"],
    }


def inspect_full(fp: Path) -> dict:
    """Abre el C3D UNA sola vez y devuelve metadatos + labels (evita doble lectura)."""
    c = _X01.load_c3d(fp)
    p = c.parameters
    pts = c["data"]["points"]
    labels = list(p["POINT"]["LABELS"]["value"])

    def _u(key):
        return p["POINT"][key]["value"][0] if "POINT" in p and key in p["POINT"] else ""

    rate = float(p["POINT"]["RATE"]["value"][0]) if "RATE" in p["POINT"] else np.nan
    prefixes = _X01.get_prefixes(c)
    names = list(p["SUBJECTS"]["NAMES"]["value"]) if "SUBJECTS" in p and "NAMES" in p["SUBJECTS"] else []
    tarcza = [l for l in labels if "Tarcza" in l]
    return {
        "n_points": int(pts.shape[1]),
        "n_frames": int(pts.shape[2]),
        "rate_hz": rate,
        "units_pos": _u("UNITS"), "units_angle": _u("ANGLE_UNITS"),
        "units_force": _u("FORCE_UNITS"), "units_moment": _u("MOMENT_UNITS"),
        "units_power": _u("POWER_UNITS"),
        "analog_used": int(p["ANALOG"]["USED"]["value"][0]) if "ANALOG" in p and "USED" in p["ANALOG"] else 0,
        "n_derived": sum(1 for l in labels if is_derived(l)),
        "n_tarcza": len(tarcza),
        "subjects": names,
        "prefixes": prefixes,
        "labels": labels,
    }


def scan_files(athletes: list[str], resume: bool = True) -> pd.DataFrame:
    """Escanea todos los archivos con checkpoint por atleta (1 apertura/archivo)."""
    frames = []
    for i, ath in enumerate(athletes):
        files = all_files_for(ath)
        if resume:
            ck = load_checkpoint(ath)
            if ck is not None and len(ck) == len(files):
                frames.append(ck)
                _progress(ath, i, len(athletes), len(ck), "checkpoint")
                continue
        rows = []
        for fp in files:
            info = parse_any_name(fp.name)
            if info["technique"] == "UNKNOWN":
                m = re.search(r"-S(\d{2})-E(\d{2})-T(\d{2})", str(fp))
                if m:
                    info = {"date": info["date"], "athlete": ath,
                            "technique": f"S{m.group(1)}",
                            "condition": f"E{m.group(2)}",
                            "trial": f"T{m.group(3)}"}
            meta = inspect_full(fp)
            rows.append(scan_file(fp, ath, info, meta))
        if rows:
            df_ath = pd.DataFrame(rows)
            save_checkpoint(ath, df_ath)
            frames.append(df_ath)
        _progress(ath, i, len(athletes), len(rows), "escaneado")
        if frames:
            pd.concat(frames, ignore_index=True).to_csv(
                CKPT_DIR / "_incremental.csv", index=False)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _progress(ath, idx, total, n, tag):
    print(f"[{idx + 1}/{total}] {ath}: {n} archivos ({tag})", flush=True)


# --------------------------------------------------------------------------- #
# Deriving derived-variable inventory
# --------------------------------------------------------------------------- #

def compile_marker_inventory(files_df: pd.DataFrame, athletes: list[str]) -> dict:
    """Presencia de markers señal por atleta·técnica."""
    rows = []
    for ath in athletes:
        ck = load_checkpoint(ath)
        if ck is None:
            continue
        for tech in TECHNIQUES:
            sub = ck[ck["technique"] == tech]
            if sub.empty:
                continue
            row = {"athlete": ath, "technique": tech, "n_files": len(sub)}
            for m in SIGNAL_MARKERS:
                col = f"signal_avail_{m}"
                row[m] = bool(sub[col].any()) if col in sub else False
            rows.append(row)
    return pd.DataFrame(rows)


def compile_derived_inventory(athletes: list[str]) -> pd.DataFrame:
    """Nº de variables derivadas por atleta (rango por condición)."""
    rows = []
    for ath in athletes:
        ck = load_checkpoint(ath)
        if ck is None:
            continue
        g = ck.groupby("condition")["number_of_derived_variables"]
        for cond, grp in g:
            rows.append({
                "athlete": ath, "condition": cond,
                "n_derived_min": int(grp.min()), "n_derived_max": int(grp.max()),
                "n_derived_unique": sorted(grp.unique().tolist()),
            })
    return pd.DataFrame(rows)


def compile_role_inventory(athletes: list[str]) -> pd.DataFrame:
    rows = []
    for ath in athletes:
        ck = load_checkpoint(ath)
        if ck is None:
            continue
        for _, r in ck.iterrows():
            prefixes = [x for x in str(r.get("subject_prefixes", "")).split("|") if x]
            role = "single"
            if len(prefixes) >= 2:
                role = "multi_subject"
            subj = str(r.get("subject_ids", "")).split("|")
            rows.append({
                "athlete": ath, "file_name": r["file_name"],
                "technique": r["technique"], "condition": r["condition"],
                "trial": r["trial"], "n_subjects": len([s for s in subj if s]),
                "role": role, "subjects": r.get("subject_ids", ""),
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# ANOMALÍAS
# --------------------------------------------------------------------------- #

def detect_anomalies(files_df: pd.DataFrame) -> pd.DataFrame:
    issues = []
    for _, r in files_df.iterrows():
        flags = []
        if pd.isna(r["sampling_rate_hz"]) or r["sampling_rate_hz"] is None:
            flags.append("rate_nan")
        if r["sampling_rate_hz"] not in (200.0, 250.0):
            flags.append("rate_unexpected")
        for u in ["units_position", "units_angle"]:
            if pd.isna(r[u]) or r[u] == "":
                flags.append(f"units_{u.split('_')[1]}_missing")
        if r["number_of_frames"] is None or int(r["number_of_frames"]) <= 0:
            flags.append("zero_frames")
        if r["number_of_points"] is None or int(r["number_of_points"]) <= 0:
            flags.append("zero_points")
        if r["technique"] == "UNKNOWN":
            flags.append("technique_unknown")
        if r["condition"] == "UNKNOWN":
            flags.append("condition_unknown")
        if r["trial"] == "UNKNOWN":
            flags.append("trial_unknown")
        # duplicado (mismo archivo por atleta)
        issues.append({
            "file_name": r["file_name"], "athlete_id": r["athlete_id"],
            "technique": r["technique"], "condition": r["condition"],
            "trial": r["trial"],
            "anomaly_flags": "|".join(flags) if flags else "",
        })
    df = pd.DataFrame(issues)
    # duplicados: mismo athlete+filename
    dup = df.groupby(["athlete_id", "file_name"])["file_name"].transform("count")
    df["anomaly_flags"] = df["anomaly_flags"] + "|duplicate_file" if (dup > 1).any() else \
        df["anomaly_flags"].astype(str).str.replace("duplicate_file", "", regex=False)
    # re-detectar duplicados de forma directa
    dup_keys = df.groupby(["athlete_id", "file_name"]).size()
    dup_keys = dup_keys[dup_keys > 1].index
    def _mark_dup(row):
        return row["anomaly_flags"] or ""
    df["anomaly_flags"] = df.apply(lambda r: r["anomaly_flags"] if not
        ((r["athlete_id"], r["file_name"]) in set(dup_keys)) else
        (r["anomaly_flags"] + ("|" if r["anomaly_flags"] else "") + "duplicate_file"), axis=1)
    return df


# --------------------------------------------------------------------------- #
# CONFIG GAP
# --------------------------------------------------------------------------- #

def conf_gap_table(athletes: list[str]) -> pd.DataFrame:
    rows = []
    for ath in athletes:
        cfg = False
        signals = {"S01": "NOT_CONFIGURED", "S02": "NOT_CONFIGURED",
                   "S03": "NOT_CONFIGURED", "S04": "NOT_CONFIGURED",
                   "S05": "NOT_CONFIGURED"}
        cfg_path = ROOT / "config" / "athletes" / f"{ath}.yaml"
        if cfg_path.exists():
            import yaml
            with open(cfg_path, encoding="utf-8") as f:
                cfg_data = yaml.safe_load(f) or {}
            cfg = True
            tech = cfg_data.get("techniques", {})
            for t in TECHNIQUES:
                entry = tech.get(t, {})
                if entry:
                    signals[t] = entry.get("signal", "NOT_CONFIGURED") \
                        if isinstance(entry, dict) else "NOT_CONFIGURED"
        rows.append({
            "athlete": ath, "configuration_exists": cfg,
            "S01": signals["S01"], "S02": signals["S02"], "S03": signals["S03"],
            "S04": signals["S04"], "S05": signals["S05"],
            "configuration_status": ("validated" if ath == "B0367"
                                     else "provisional" if ath == "B0377"
                                     else "NOT_CONFIGURED"),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# MAIN
# --------------------------------------------------------------------------- #

def main():
    t0 = time.time()
    print("=" * 72)
    print("FASE 1.8E — INVENTARIO COMPLETO DE ATLETAS Y HOMOGENEIDAD")
    print("=" * 72)
    print(f"Fuente canónica: {DATA_ROOT}")
    print("Se excluye la copia histórica B0367/ de la raíz (duplicado de atletas/B0367/).")

    athletes = discover_all_athletes()
    print(f"\n[Pase 1] Atletas detectados: {len(athletes)}")
    quick = quick_sample(athletes, n_sample=2)
    quick.to_csv(OUT_DIR / "_quick_sample.csv", index=False)

    print("\n[Pase 2] Escaneo completo de C3D (con checkpoint)...")
    files_df = scan_files(athletes, resume=True)
    print(f"Archivos inventariados: {len(files_df)}  (tiempo parcial {time.time() - t0:.1f}s)")
    files_df.to_csv(OUT_DIR / "file_inventory.csv", index=False)

    # ---- agregados ----
    # athlete inventory
    aggr = files_df.groupby("athlete_id").agg(
        n_files=("file_name", "count"),
        techniques=("technique", lambda s: ",".join(sorted(set(s)))),
        conditions=("condition", lambda s: ",".join(sorted(set(s)))),
        trials=("trial", lambda s: ",".join(sorted(set(s)))),
        sampling_rates=("sampling_rate_hz", lambda s: ",".join(sorted({str(x) for x in set(s)}))),
        n_points_min=("number_of_points", "min"), n_points_max=("number_of_points", "max"),
        n_derived_min=("number_of_derived_variables", "min"),
        n_derived_max=("number_of_derived_variables", "max"),
    ).reset_index()
    # flags por atleta
    flags = files_df.groupby("athlete_id")["signal_avail_RTOE"].max()
    aggr["roles"] = files_df.groupby("athlete_id")["subject_prefixes"].apply(
        lambda s: "|".join(sorted({x.strip() for x in s.str.split("|").explode() if x.strip()})))
    aggr = aggr.rename(columns={"athlete_id": "athlete_id"})
    aggr["structural_status"] = "ok"
    aggr.to_csv(OUT_DIR / "athlete_inventory.csv", index=False)

    # coverage matrix (fila=athlete, columna=S0X-E0X-T0X, valor=conteo de archivos)
    cov_key = files_df["technique"] + "-" + files_df["condition"] + "-" + files_df["trial"]
    cov = files_df.assign(key=cov_key).groupby(["athlete_id", "key"]).size().unstack(fill_value=0)
    cov.to_csv(OUT_DIR / "coverage_matrix.csv")

    # sampling summary
    sr = files_df.groupby("sampling_rate_hz").agg(
        file_count=("file_name", "count"),
        athlete_count=("athlete_id", "nunique"),
    ).reset_index()
    sr.to_csv(OUT_DIR / "sampling_rate_summary.csv", index=False)

    # sampling rate group (por atleta)
    srg = files_df.groupby("athlete_id")["sampling_rate_hz"].agg(
        lambda s: sorted({float(x) for x in set(s)})).reset_index()
    srg.columns = ["athlete_id", "rates"]
    srg["rate_group"] = srg["rates"].apply(
        lambda r: ("200_Hz" if r == [200.0] else
                   "250_Hz" if r == [250.0] else
                   "mixed" if len(r) > 1 else "other"))
    srg.to_csv(OUT_DIR / "sampling_rate_group.csv", index=False)

    # marker availability (per athlete·technique)
    mcol = [f"signal_avail_{m}" for m in SIGNAL_MARKERS]
    for m in mcol:
        if m not in files_df.columns:
            files_df[m] = False
    mk = files_df.groupby(["athlete_id", "technique"])[mcol].max().reset_index()
    mk.columns = ["athlete_id", "technique"] + SIGNAL_MARKERS
    mk.to_csv(OUT_DIR / "marker_availability.csv", index=False)

    # derived inventory
    der = compile_derived_inventory(athletes)
    der.to_csv(OUT_DIR / "derived_variable_inventory.csv", index=False)

    # role inventory
    roles = compile_role_inventory(athletes)
    roles.to_csv(OUT_DIR / "role_inventory.csv", index=False)

    # anomalies
    anom = detect_anomalies(files_df)
    anom.to_csv(OUT_DIR / "anomaly_inventory.csv", index=False)

    # config gap
    gap = conf_gap_table(athletes)
    gap.to_csv(OUT_DIR / "configuration_gap.csv", index=False)

    # scaling readiness
    sr_know = files_df.groupby("athlete_id")["sampling_rate_hz"].first()
    grouping = []
    for _, r in gap.iterrows():
        rate = sr_know.get(r["athlete"], np.nan)
        if r["athlete"] == "B0367":
            g, ready = "A", "scaling_ready"          # validado/protegido
        elif r["athlete"] == "B0377":
            g, ready = "B", "scaling_ready"          # provisional (config definida)
        elif rate == 250.0 and r["configuration_status"] == "NOT_CONFIGURED":
            g, ready = "B", "scaling_ready"          # compatible tras config
        else:
            g, ready = "C", "requires_review"        # 200 Hz sin config, requiere revisión
        grouping.append({"athlete": r["athlete"], "rate_hz": rate,
                         "group": g, "readiness": ready})
    read = pd.DataFrame(grouping)
    read.to_csv(OUT_DIR / "scaling_readiness.csv", index=False)

    t1 = time.time()
    print(f"\n[TIEMPO] total inventario: {t1 - t0:.1f} s")
    print("\n[RESUMEN]")
    print(f"  Atletas detectados : {len(athletes)}")
    print(f"  C3D inventariados  : {len(files_df)}")
    print(f"  Grupos de Hz       : {sr.to_dict('records')}")
    print(f"  Archivos con anomalías: {int((anom.anomaly_flags != '').sum())}")
    print(f"  Configurados       : {sum(gap.configuration_exists)} (B0367, B0377)")
    print(f"  No configurados    : {len(gap) - sum(gap.configuration_exists)}")
    print("\n[Guardado] output/athlete_inventory/ (11 CSV)")


if __name__ == "__main__":
    main()