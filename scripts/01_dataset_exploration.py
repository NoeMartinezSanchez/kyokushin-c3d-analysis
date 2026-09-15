#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
01_dataset_exploration.py
=========================
Auditoría exploratoria y técnica del dataset público de motion capture
"Optical motion capture dataset of selected techniques in beginner and advanced
Kyokushin karate athletes" (Szczęsna et al., Scientific Data 2021).

Participante local: B0367.

FASES:
  F1. Inventario del dataset (archivos, técnicas, condiciones, trials, fechas)
  F2. Inspección técnica de C3D representativos (ezc3d)
  F3. Mapeo de markers -> regiones anatómicas (nomenclatura PlugInGait FullBody)
  F4. Exploración de señales + detección reproducible de fases de ejecución
  F5. Comparación entre condiciones (S04: E01 aire / E02 escudo / E04 defensor)
  F6. Repetibilidad T01 vs T02 (DTW)
  F7. Feature engineering (features calculables vs. que requerirían más datos)
  F8. Evaluación de problemas de ML viables (documentado, sin entrenar)
  F9. Diseño del demo "Karate Athlete Performance Intelligence" (documentado)
  F10. Genera salidas -> output/figures, output/tables, output/report_summary.md

NOTAS IMPORTANTES:
  - No modifica los archivos originales (solo lectura).
  - No convierte el dataset a CSV de forma masiva.
  - No entrena modelos: la fase ML es propositiva y se documenta.
  - Proyecto raíz: se resuelve a partir de la ubicación del script.
"""

from __future__ import annotations

import glob
import json
import os
import re
import sys
from pathlib import Path

# Forzar salida a consola con UTF-8 (evita errores de codificación en Windows)
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import matplotlib
matplotlib.use("Agg")  # headless, reproducible

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal as sg

import ezc3d

# ----------------------------------------------------------------------------- #
# 0. Rutas y configuración
# ----------------------------------------------------------------------------- #

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "B0367"
OUT_DIR = ROOT / "output"
FIG_DIR = OUT_DIR / "figures"
TAB_DIR = OUT_DIR / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

FREQ_DEFAULT = 200.0           # Hz (verificado en parámetros C3D)
UNITS_DEFAULT = "mm"


def now_tag() -> str:
    import datetime
    return datetime.datetime.now().strftime("%Y%m%d_%H%M")


def save_fig(fig, name: str):
    path = FIG_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    [fig] {path.name}")


def parse_name(fname: str) -> dict:
    """Parse de la nomenclatura oficial.
    2017-01-31-B0367-S04-E02-T01.c3d -> fecha, participante, técnica, condición, trial
    """
    m = re.match(r"(\d{4}-\d{2}-\d{2})-([A-Z]\d{3,4})-S(\d{2})-E(\d{2})-T(\d{2})", fname)
    if not m:
        return {}
    _, athlete, tech, cond, trial = m.groups()
    return {
        "date": _,
        "athlete": athlete,
        "technique": f"S{tech}",
        "condition": f"E{cond}",
        "trial": f"T{trial}",
    }

TECHNIQUE_NAMES = {
    "S01": "Gyaku-Zuki",
    "S02": "Mae-Geri",
    "S03": "Mawashi-Geri gedan",
    "S04": "Mawashi-Geri jodan",
    "S05": "Ushiro-Mawashi-Geri",
}
CONDITION_NAMES = {
    "E01": "air",
    "E02": "shield",
    "E03": "attacker",
    "E04": "defender",
}
# E03 (attacker) no está presente en el subconjunto local.

DERIVED_PATTERNS = re.compile(
    r"(Angles|Angle|Power|Force|Moment|COM|HJC|KJC|AJC|SJC|EJC|WJC|Progress|Thorax|Head|Neck|Spine|Pelvis|Shoulder|Elbow|Wrist|Waist|CentreOfMass)"
)

CLUSTER_PATTERN = re.compile(r"(O|A|L|P)$")
PIG_ANATOMICAL = re.compile(
    r"^(LFHD|RFHD|LBHD|RBHD|C7|T10|CLAV|STRN|RBAK|LSHO|LUPA|LELB|LFRM|LWRA|LWRB|LFIN|"
    r"RSHO|RUPA|RELB|RFRM|RWRA|RWRB|RFIN|LASI|RASI|LPSI|RPSI|LTHI|LKNE|LTIB|LANK|LHEE|LTOE|"
    r"RTHI|RKNE|RTIB|RANK|RHEE|RTOE)$"
)


def class_point_label(label: str) -> str:
    """Clasifica un point label: anatomical, cluster, tarcza o derived."""
    base = label.split(":")[-1]
    if "Tarcza" in base:
        return "shield"
    if DERIVED_PATTERNS.search(base):
        return "derived"
    if PIG_ANATOMICAL.match(base):
        return "anatomical"
    if CLUSTER_PATTERN.search(base):
        return "cluster"
    return "other"


def load_c3d(path: Path) -> ezc3d.c3d:
    return ezc3d.c3d(str(path))


def all_files() -> list[Path]:
    return sorted(DATA_DIR.glob("**/*.c3d"))


def find_files(tag: str) -> list[Path]:
    """Devuelve archivos cuyo stem termina exactamente en '-<tag>' (tag sin .c3d)."""
    return [p for p in all_files() if p.stem.endswith("-" + tag)]


def get_prefixes(c) -> list[str]:
    """Prefijos de sujeto. LABEL_PREFIXES puede no existir en mono-sujeto."""
    p = c.parameters
    prefixes = []
    if "SUBJECTS" in p and "LABEL_PREFIXES" in p["SUBJECTS"]:
        prefixes = [str(v).rstrip(":") for v in p["SUBJECTS"]["LABEL_PREFIXES"]["value"]]
    elif "SUBJECTS" in p and "NAMES" in p["SUBJECTS"]:
        prefixes = [str(v).rstrip(":") for v in p["SUBJECTS"]["NAMES"]["value"]]
    # si un prefijo es vacío o igual al nombre, es redundante
    return [x for x in prefixes if x and not x.isdigit()]


def athlete_prefix(fp: Path, prefixes: list[str]) -> str:
    """Elige el prefijo del atleta principal (el que aparece en el nombre del archivo)
    y no el de un escudo o defensor."""
    name = fp.name
    for pref in prefixes:
        if pref in name:
            return pref
    for pref in prefixes:
        if "Tarcza" not in pref:
            return pref
    return prefixes[0] if prefixes else ""


# ----------------------------------------------------------------------------- #
# FASE 1 — Inventario del dataset
# ----------------------------------------------------------------------------- #

def fase1_inventario():
    print("\n" + "=" * 72)
    print("FASE 1 — INVENTARIO DEL DATASET")
    print("=" * 72)

    files = all_files()
    rows = []
    for fp in files:
        info = parse_name(fp.stem)
        size = fp.stat().st_size
        rows.append({**info, "path": str(fp), "size_bytes": size})
    df = pd.DataFrame(rows)

    n_files = len(df)
    n_athletes = df["athlete"].nunique()
    n_techniques = df["technique"].nunique()
    n_conditions = df["condition"].nunique()
    n_trials = df["trial"].nunique()
    n_days = df["date"].nunique()
    total_size_gb = df["size_bytes"].sum() / 1e9

    print(f"Total archivos .c3d     : {n_files}")
    print(f"Participantes            : {n_athletes} -> {sorted(df['athlete'].unique())}")
    print(f"Técnicas (S)             : {n_techniques} -> {sorted(df['technique'].unique())}")
    print(f"Condiciones (E)          : {n_conditions} -> {sorted(df['condition'].unique())}")
    print(f"Trials (T)               : {n_trials} -> {sorted(df['trial'].unique())}")
    print(f"Fechas                   : {n_days} -> {sorted(df['date'].unique())}")
    print(f"Tamaño total             : {total_size_gb:.3f} GB")

    print("\nDistribución por técnica x condición x trial:")
    pivot = df.pivot_table(index=["technique", "condition"], columns="trial",
                           values="path", aggfunc="count", fill_value=0)
    print(pivot.to_string())

    print("\nTamaño por archivo (bytes):")
    df_sorted = df.sort_values("size_bytes")
    print(df_sorted[["technique", "condition", "trial", "size_bytes"]].to_string(index=False))

    # OJO posibles huecos
    expected = {
        (s, cond, t)
        for s in ["S01", "S02", "S03", "S04", "S05"]
        for cond in ["E01", "E02", "E03", "E04"]
        for t in ["T01", "T02"]
    }
    present = set(zip(df["technique"], df["condition"], df["trial"]))
    missing = sorted(expected - present)
    print(f"\nTrials faltantes (de la rejilla completa): {len(missing)}")
    for miss in missing:
        print(f"   - {miss[0]} {miss[1]} {miss[2]}  ({TECHNIQUE_NAMES[miss[0]]} / {CONDITION_NAMES[miss[1]]})")

    df.to_csv(TAB_DIR / "f1_inventory.csv", index=False)
    print("\n[guardado] output/tables/f1_inventory.csv")

    # Figura distribución
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    df.groupby("technique").size().reindex(sorted(df["technique"].unique())).plot.bar(
        ax=axes[0], color="#2f6fb3", title="Archivos por técnica")
    df.groupby("condition").size().reindex(sorted(df["condition"].unique())).plot.bar(
        ax=axes[1], color="#e08a2e", title="Archivos por condición")
    df.groupby(["technique", "condition"]).size().unstack().fillna(0).plot.bar(
        ax=axes[2], rot=45, title="Heatmap de cobertura técnica x condición")
    axes[2].legend(title="condición")
    fig.tight_layout()
    save_fig(fig, "f1_distributions.png")

    return df, files


# ----------------------------------------------------------------------------- #
# FASE 2 — Inspección técnica de C3D representativos
# ----------------------------------------------------------------------------- #

def inspect_one(path: str | Path) -> dict:
    c = load_c3d(path)
    p = c.parameters
    pts = c["data"]["points"]
    n_frames = pts.shape[2]
    rate = float(p["POINT"]["RATE"]["value"][0])
    units = p["POINT"]["UNITS"]["value"]
    labels = list(p["POINT"]["LABELS"]["value"])
    duration = n_frames / rate

    nan_frac = {f"{c_as[0]}": float(np.isnan(pts[i]).mean() * 100)
                for i, c_as in enumerate([("X", 0), ("Y", 1), ("Z", 2)])}

    analog_used = p["ANALOG"]["USED"]["value"][0] if "ANALOG" in p and "USED" in p["ANALOG"] else 0

    subjects = p["SUBJECTS"]["NAMES"]["value"] if "SUBJECTS" in p and "NAMES" in p["SUBJECTS"] else []
    prefix = p["SUBJECTS"]["LABEL_PREFIXES"]["value"] if "SUBJECTS" in p and "LABEL_PREFIXES" in p["SUBJECTS"] else []

    manufacturer = (p["MANUFACTURER"]["COMPANY"]["value"][0]
                    if "MANUFACTURER" in p and "COMPANY" in p["MANUFACTURER"] else None)

    events = []
    if "EVENT" in p:
        if "LABELS" in p["EVENT"] and "CONTEXTS" in p["EVENT"]:
            for lab, ctx in zip(p["EVENT"]["LABELS"]["value"], p["EVENT"]["CONTEXTS"]["value"]):
                events.append((lab, ctx))
        elif "LABELS" in p["EVENT"]:
            events = [(lab, "") for lab in p["EVENT"]["LABELS"]["value"]]

    # rango de valores por categoría de punto
    classes = {}
    ranges = {}
    for i, lab in enumerate(labels):
        cls = class_point_label(lab)
        classes.setdefault(cls, 0)
        classes[cls] += 1
        arr = pts[:3, i, :]
        finite = arr[np.isfinite(arr)]
        if finite.size:
            vmin, vmax = float(finite.min()), float(finite.max())
            if cls not in ranges:
                ranges[cls] = {"min": vmin, "max": vmax}
            else:
                ranges[cls]["min"] = min(ranges[cls]["min"], vmin)
                ranges[cls]["max"] = max(ranges[cls]["max"], vmax)
        else:
            ranges[cls] = {"min": np.nan, "max": np.nan}

    return {
        "file": Path(path).name,
        "frames": int(n_frames),
        "rate_hz": rate,
        "duration_s": round(duration, 3),
        "n_points": int(pts.shape[1]),
        "coord_dims": list(pts.shape[0:3]),
        "labels": labels,
        "n_anatomical": classes.get("anatomical", 0),
        "n_cluster": classes.get("cluster", 0),
        "n_shield": classes.get("shield", 0),
        "n_derived": classes.get("derived", 0),
        "n_other": classes.get("other", 0),
        "units": units,
        "analog_used": int(analog_used),
        "subjects": subjects,
        "label_prefixes": prefix,
        "manufacturer": manufacturer,
        "nan_pct_by_axis": nan_frac,
        "range_by_class": ranges,
        "events": events,
    }


def fase2_inspeccion():
    print("\n" + "=" * 72)
    print("FASE 2 — INSPECCIÓN TÉCNICA DE C3D REPRESENTATIVOS")
    print("=" * 72)

    representatives = [
        "S01-E01-T01", "S02-E01-T01", "S04-E01-T01",
        "S04-E02-T01", "S04-E04-T01", "S05-E01-T01",
    ]
    reps = []
    for tag in representatives:
        matches = find_files(tag)
        if len(matches) > 1:
            # preferir el que coincida exactamente con el sufijo
            exact = [m for m in matches if m.stem.endswith("-" + tag)]
            fp = exact[0] if exact else matches[0]
        elif matches:
            fp = matches[0]
        else:
            print(f"  [warn] no encontrado: {tag}")
            continue
        info = inspect_one(fp)
        reps.append(info)
        print(f"\n--- {info['file']} ---")
        print(f"  frames={info['frames']}  rate={info['rate_hz']} Hz  duración={info['duration_s']} s")
        print(f"  n_points={info['n_points']}  anatomical={info['n_anatomical']}  "
              f"cluster={info['n_cluster']}  shield={info['n_shield']}  derived={info['n_derived']}")
        print(f"  unidades={info['units']}  analog_used={info['analog_used']}")
        print(f"  subjects={info['subjects']}  prefixes={info['label_prefixes']}")
        print(f"  manufacturer={info['manufacturer']}  events={info['events']}")
        print(f"  NaN % por eje: {info['nan_pct_by_axis']}")
        for cls, r in info["range_by_class"].items():
            print(f"    rango {cls}: [{r['min']:.1f}, {r['max']:.1f}] {info['units']}")

    df_reps = pd.DataFrame([
        {
            "file": r["file"], "frames": r["frames"], "rate": r["rate_hz"],
            "duration_s": r["duration_s"], "n_points": r["n_points"],
            "n_anatomical": r["n_anatomical"], "n_cluster": r["n_cluster"],
            "n_shield": r["n_shield"], "n_derived": r["n_derived"],
            "analog": r["analog_used"], "units": ",".join(r["units"]),
            "subjects": "|".join(r["subjects"]), "nan_pct": np.mean(list(r["nan_pct_by_axis"].values())),
        }
        for r in reps
    ])
    df_reps.to_csv(TAB_DIR / "f2_representative_inspection.csv", index=False)
    print("\n[guardado] output/tables/f2_representative_inspection.csv")

    return reps


# ----------------------------------------------------------------------------- #
# FASE 3 — Mapeo de markers -> regiones anatómicas
# ----------------------------------------------------------------------------- #

ANATOMICAL_MAP = {
    "LFHD": "Cabeza (frente izq.)", "RFHD": "Cabeza (frente der.)",
    "LBHD": "Cabeza (post. izq.)", "RBHD": "Cabeza (post. der.)",
    "C7": "Columna (C7)", "T10": "Columna (T10)", "CLAV": "Clavícula/esternón",
    "STRN": "Esternón", "RBAK": "Espalda (escápula der.)",
    "LSHO": "Hombro izq.", "RSHO": "Hombro der.",
    "LUPA": "Brazo izq. (húmero)", "RUPA": "Brazo der. (húmero)",
    "LELB": "Codo izq.", "RELB": "Codo der.",
    "LFRM": "Antebrazo izq.", "RFRM": "Antebrazo der.",
    "LWRA": "Muñeca izq. (lado A)", "LWRB": "Muñeca izq. (lado B)",
    "RWRA": "Muñeca der. (lado A)", "RWRB": "Muñeca der. (lado B)",
    "LFIN": "Mano/dedos izq.", "RFIN": "Mano/dedos der.",
    "LASI": "Pelvis (ASIS izq.)", "RASI": "Pelvis (ASIS der.)",
    "LPSI": "Pelvis (PSIS izq.)", "RPSI": "Pelvis (PSIS der.)",
    "LTHI": "Muslo izq.", "RTHI": "Muslo der.",
    "LKNE": "Rodilla izq.", "RKNE": "Rodilla der.",
    "LTIB": "Pierna/tibia izq.", "RTIB": "Pierna/tibia der.",
    "LANK": "Tobillo izq.", "RANK": "Tobillo der.",
    "LHEE": "Talón izq.", "RHEE": "Talón der.",
    "LTOE": "Pie/punta izq.", "RTOE": "Pie/punta der.",
}

CLUSTER_MAP = {
    "PEL": "Cluster pelvis", "FEO/FEA/FEL/FEP": "Cluster muslo izq.",
    "TIO/TIA/TIL/TIP": "Cluster pierna izq.", "FOO/FOA/FOL/FOP": "Cluster pie izq.",
    "TOO/TOA/TOL/TOP": "Cluster punta izq.",
    "HED": "Cluster cabeza", "CLO/CLA/CLL/CLP": "Cluster clavícula izq.",
    "TRX": "Cluster torso", "HUO/HUA/HUL/HUP": "Cluster húmero izq.",
    "RAO/RAA/RAL/RAP": "Cluster radio izq.", "HNO/HNA/HNL/HNP": "Cluster mano izq.",
}


def fase3_mapping():
    print("\n" + "=" * 72)
    print("FASE 3 — MAPEO DE MARKERS A REGIONES ANATÓMICAS")
    print("=" * 72)

    # Tomar las etiquetas de un E01 (sin prefijo) y un E02 (para Tarcza)
    f_e01 = find_files("S04-E01-T01")[0]
    f_e02 = find_files("S04-E02-T01")[0]
    labs_e01 = list(load_c3d(f_e01).parameters["POINT"]["LABELS"]["value"])
    labs_e02 = list(load_c3d(f_e02).parameters["POINT"]["LABELS"]["value"])

    rows = []
    for lab in labs_e01:
        base = lab.split(":")[-1]
        cls = class_point_label(lab)
        if cls == "anatomical":
            region = ANATOMICAL_MAP.get(base, "por determinar")
        elif cls == "cluster":
            # agrupamos por prefijo raíz
            root = re.sub(r"(O|A|L|P)$", "", base)
            region = CLUSTER_MAP.get(root, f"Cluster ({root}*)")
            region = f"{region} (4 marcadores de cluster)"
        elif cls == "derived":
            region = f"Variable derivada: {base}"
        else:
            region = "otro"
        rows.append({"marker": lab, "region_anatomica": region, "clase": cls,
                     "presencia": "global"})
    for lab in labs_e02:
        if "Tarcza" in lab:
            rows.append({"marker": lab, "region_anatomica": "Escudo (Tarcza)",
                         "clase": "shield", "presencia": "solo condición E02"})

    df = pd.DataFrame(rows)
    # puntos clave del enunciado
    print("\nMarcadores segmentados por región (resumen de cabeza/tronco/miembros):")
    for region_key in ["Cabeza", "Pelvis", "Muslo", "Rodilla", "Tobillo", "Pie",
                       "Hombro", "Codo", "Muñeca", "Mano", "Columna", "Escudo"]:
        mask = df["region_anatomica"].str.contains(region_key, na=False)
        labs = df.loc[mask, "marker"].tolist()
        print(f"  {region_key:<12}: {len(labs):>2} -> {labs[:20]}")

    df.to_csv(TAB_DIR / "f3_marker_map.csv", index=False)
    print("\n[guardado] output/tables/f3_marker_map.csv  (tabla completa marker|región|clase)")

    return df


# ----------------------------------------------------------------------------- #
# FASE 4 — Exploración de señales y detección reproducible de fases
# ----------------------------------------------------------------------------- #

def get_time(labels, name, prefix_if_needed="") -> int:
    if prefix_if_needed:
        full = f"{prefix_if_needed}:{name}"
        if full in labels:
            return labels.index(full)
    if name in labels:
        return labels.index(name)
    # variantes
    for i, l in enumerate(labels):
        if l.endswith(":" + name):
            return i
    return -1


def velocity(pts3, axis, rate):
    """pts3: (3, frames) -> velocidad (norma) en mm/s y por componente."""
    v = np.gradient(pts3, axis=axis) * rate
    speed = np.linalg.norm(v, axis=0)  # magnitud 3D
    return v, speed


def smooth(x, window=7) -> np.ndarray:
    if window <= 1:
        return x
    kernel = np.hanning(window)
    kernel /= kernel.sum()
    return np.convolve(x, kernel, mode="same")


def detect_phases(speed: np.ndarray, rate: float, stall_frac: float = 0.10,
                  onset_frac: float = 0.15) -> dict:
    """
    Detección reproducible de fases a partir del perfil de velocidad.

    Método (basado en umbrales relativos, sin arbitrariedad manual):
      1. Baseline = mediana de la velocidad en el primer segundo de archivo
         (el atleta suele estar quieto en la posición inicial).
      2. v_smooth = velocidad suavizada (ventana Hanning ~35 ms).
      3. Evento principal: v_max = argmax(v_smooth).
      4. Onset : primera muestra (antes de v_max) donde v_smooth supera
                 baseline + onset_frac*(v_max - baseline).
      5. Stall : última muestra (antes de v_max) donde v_smooth cae por debajo
                 de baseline + stall_frac*(v_max - baseline)  -> fin de preparación
                 (retroceso/golpe de cámara) e inicio de la ejecución.
      6. Offset: última muestra (después de v_max) donde v_smooth supera
                 baseline + stall_frac*(v_max - baseline)  -> fase final/recuperación.
      7. Fase 1 (preparatoria)  : onset -> stall
         Fase 2 (ejecución)     : stall -> v_max (impacto/máxima velocidad)
         Fase 3 (final/recuperación): v_max -> offset
    """
    n = len(speed)
    baseline = float(np.median(speed[: int(rate)]))
    vs = smooth(speed, window=int(rate * 0.035) | 1)
    vmax = float(np.max(vs))
    imax = int(np.argmax(vs))
    if vmax <= 0:
        return {"baseline": baseline, "vmax": vmax, "onset": 0, "stall": 0,
                "peak": imax, "offset": n - 1,
                "t_onset": 0, "t_stall": 0, "t_peak": imax / rate, "t_offset": (n - 1) / rate,
                "phase_prep": [0, 0], "phase_exec": [0, imax], "phase_final": [imax, n - 1]}
    thr_onset = baseline + onset_frac * (vmax - baseline)
    thr_stall = baseline + stall_frac * (vmax - baseline)

    # onset: antes del pico
    before = vs[:imax]
    idx_onset = np.flatnonzero(before > thr_onset)
    onset = int(idx_onset[0]) if idx_onset.size else 0

    # stall: última vez antes del pico que cae bajo thr_stall
    sub = vs[onset:imax]
    idx_under = np.flatnonzero(sub <= thr_stall)
    stall = int(idx_under[-1]) + onset if idx_under.size else onset

    # offset: después del pico, última vez sobre thr_stall
    after = vs[imax:]
    idx_after = np.flatnonzero(after > thr_stall)
    offset = int(idx_after[-1]) + imax if idx_after.size else n - 1

    return {
        "baseline": baseline, "vmax": vmax, "onset": onset, "stall": stall,
        "peak": imax, "offset": offset,
        "t_onset": onset / rate, "t_stall": stall / rate,
        "t_peak": imax / rate, "t_offset": offset / rate,
        "phase_prep": [onset, stall], "phase_exec": [stall, imax], "phase_final": [imax, offset],
    }


def phase_labels(det: dict) -> pd.DataFrame:
    return pd.DataFrame([det])


def fase4_señales():
    print("\n" + "=" * 72)
    print("FASE 4 — EXPLORACIÓN DE SEÑALES Y DETECCIÓN DE FASES")
    print("=" * 72)

    target_files = ["S04-E01-T01", "S04-E02-T01", "S04-E04-T01"]
    res = []
    for tag in target_files:
        cand = find_files(tag)
        if not cand:
            continue
        fp = cand[0]
        c = load_c3d(fp)
        pts = c["data"]["points"]
        labels = list(c.parameters["POINT"]["LABELS"]["value"])
        rate = float(c.parameters["POINT"]["RATE"]["value"][0])
        prefixes = get_prefixes(c)
        # marker de punto de impacto: para kicks usar la punta del pie del lado de la patada.
        # Para S04 (Mawashi-Geri jodan) en sujeto diestro la patada es derecha (RTOE).
        pref = athlete_prefix(fp, prefixes)
        i_toe = get_time(labels, "RTOE", pref)
        i_ank = get_time(labels, "RANK", pref)
        i_com = get_time(labels, "CentreOfMass", pref)
        i_knee = get_time(labels, "RKneeAngles", pref)

        if i_toe < 0:
            print(f"  [warn] {tag}: RTOE no encontrado -> salto")
            continue

        traj = pts[:3, i_toe, :].astype(float)  # (3, frames) mm
        v, speed = velocity(traj, axis=1, rate=rate)
        acc = np.gradient(speed, 1.0 / rate)

        det = detect_phases(speed, rate)
        res.append({
            "file": Path(fp).name, "marker": "RTOE", "rate": rate,
            "n_frames": traj.shape[1], "duration_s": round(traj.shape[1] / rate, 3),
            **{k: round(det[k], 3) for k in
               ["t_onset", "t_stall", "t_peak", "t_offset"]},
            "vmax_mm_s": round(det["vmax"], 1),
            "nan_frac": round(float(np.isnan(traj).mean() * 100), 2),
        })

        # ---- figura 1: trayectoria X/Y/Z ----
        t = np.arange(traj.shape[1]) / rate
        fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
        for ax, comp, color in zip(axes, ["X", "Y", "Z"], ["#d1495b", "#2f6fb3", "#39a07b"]):
            ax.plot(t, traj[0] if comp == "X" else traj[1] if comp == "Y" else traj[2],
                    color=color, lw=1)
            ax.set_ylabel(f"{comp} [mm]")
        axes[0].set_title(f"{Path(fp).name} — trayectoria RTOE (mm)")
        axes[-1].set_xlabel("tiempo [s]")
        fig.tight_layout()
        save_fig(fig, f"f4_{tag}_xyz.png")

        # ---- figura 2: trayectoria 3D ----
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="3d")
        finite = np.isfinite(traj[0]) & np.isfinite(traj[1]) & np.isfinite(traj[2])
        ax.plot(traj[0, finite], traj[1, finite], traj[2, finite], lw=0.8, color="#2f6fb3")
        ax.scatter(traj[0, det["onset"]], traj[1, det["onset"]], traj[2, det["onset"]],
                   color="#e08a2e", s=70, label="onset")
        ax.scatter(traj[0, det["stall"]], traj[1, det["stall"]], traj[2, det["stall"]],
                   color="#39a07b", s=70, label="stall / inicio ejecución")
        ax.scatter(traj[0, det["peak"]], traj[1, det["peak"]], traj[2, det["peak"]],
                   color="#d1495b", s=70, label="pico velocidad")
        ax.scatter(traj[0, det["offset"]], traj[1, det["offset"]], traj[2, det["offset"]],
                   color="#6c5b7b", s=70, label="offset")
        ax.set_xlabel("X [mm]"); ax.set_ylabel("Y [mm]"); ax.set_zlabel("Z [mm]")
        ax.set_title(f"{Path(fp).name} — trayectoria 3D RTOE")
        ax.legend(loc="upper right", fontsize=8)
        save_fig(fig, f"f4_{tag}_traj3d.png")

        # ---- figura 3: velocidad + fases ----
        fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
        axes[0].plot(t, speed, color="#2f6fb3", lw=1, label="speed RTOE")
        axes[0].plot(t, smooth(speed, window=9), color="#111", lw=1, label="speed suavizada")
        for dl, xpos in [("onset", det["t_onset"]), ("stall", det["t_stall"]),
                         ("peak", det["t_peak"]), ("offset", det["t_offset"])]:
            axes[0].axvline(xpos, color="#d1495b", ls="--", lw=0.8)
            axes[0].annotate(dl, (xpos, axes[0].get_ylim()[1] * 0.9),
                             textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
        axes[0].set_ylabel("velocidad [mm/s]")
        axes[0].set_title(f"{Path(fp).name} — detección de fases (RTOE)")
        axes[1].plot(t, acc, color="#6c5b7b", lw=0.8)
        axes[1].axhline(0, color="k", lw=0.5)
        axes[1].set_ylabel("aceleración [mm/s²]")
        axes[1].set_xlabel("tiempo [s]")
        fig.tight_layout()
        save_fig(fig, f"f4_{tag}_phases.png")

        # ---- figura 4: ángulo de rodilla si existe ----
        if i_knee >= 0:
            knee = pts[:3, i_knee, :]
            fig, ax = plt.subplots(figsize=(12, 3.5))
            for comp, cl in zip(["X", "Y", "Z"], ["#d1495b", "#2f6fb3", "#39a07b"]):
                ax.plot(t, knee[0] if comp == "X" else knee[1] if comp == "Y" else knee[2],
                        color=cl, lw=1, label=comp)
            ax.set_xlabel("tiempo [s]"); ax.set_ylabel("ángulo [rad]")
            ax.set_title(f"{Path(fp).name} — RKneeAngles")
            ax.legend()
            fig.tight_layout()
            save_fig(fig, f"f4_{tag}_knee_angle.png")

        # ---- figura 5: COM si existe ----
        if i_com >= 0:
            com = pts[:3, i_com, :].astype(float)
            com_speed = np.linalg.norm(np.gradient(com, axis=1) * rate, axis=0)
            fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
            for comp, cl in zip(["X", "Y", "Z"], ["#d1495b", "#2f6fb3", "#39a07b"]):
                axes[0].plot(t, com[0] if comp == "X" else com[1] if comp == "Y" else com[2],
                             color=cl, lw=1, label=comp)
            axes[0].set_ylabel("COM [mm]"); axes[0].legend(fontsize=8)
            axes[0].set_title(f"{Path(fp).name} — CentreOfMass")
            axes[1].plot(t, com_speed, color="#6c5b7b", lw=1)
            axes[1].set_ylabel("velocidad COM [mm/s]"); axes[1].set_xlabel("tiempo [s]")
            fig.tight_layout()
            save_fig(fig, f"f4_{tag}_com.png")

    df = pd.DataFrame(res)
    df.to_csv(TAB_DIR / "f4_phase_detection.csv", index=False)
    print("\n[guardado] output/tables/f4_phase_detection.csv")
    return df


# ----------------------------------------------------------------------------- #
# FASE 5 — Comparación entre condiciones (S04: E01 / E02 / E04)
# ----------------------------------------------------------------------------- #

def compute_kinematics(pts3: np.ndarray, rate: float) -> dict:
    """Cinemática básica de una trayectoria."""
    v = np.gradient(pts3, axis=1) * rate
    speed = np.linalg.norm(v, axis=0)
    acc = np.gradient(speed, 1.0 / rate)
    disp = float(np.linalg.norm(pts3[:, -1] - pts3[:, 0]))
    pathlen = float(np.sum(np.linalg.norm(np.diff(pts3, axis=1), axis=0)))
    rom = float(np.max(np.linalg.norm(pts3 - pts3[:, :1], axis=0), axis=0))
    d = {
        "vmax": float(np.nanmax(speed)),
        "vmean": float(np.nanmean(speed)),
        "amax": float(np.nanmax(np.abs(acc))),
        "displacement": disp,
        "path_length": pathlen,
        "rom": rom,
        "time_to_vmax": float(np.nanargmax(speed)) / rate,
    }
    return d


def fase5_condiciones():
    print("\n" + "=" * 72)
    print("FASE 5 — COMPARACIÓN ENTRE CONDICIONES (S04 / Mawashi-Geri jodan)")
    print("=" * 72)
    print("NOTA METODOLÓGICA: en condiciones de oponente (E03/E04) el archivo contiene")
    print("DOS sujetos. En E04 (defender) el atleta del dataset (B0367) DEFIENDE y el")
    print("oponente (B0368) ejecuta la técnica. Analizamos ambos roles por separado.")

    def kin_row(fp, pref, role, cond, trial):
        c = load_c3d(fp)
        pts = c["data"]["points"]
        labels = list(c.parameters["POINT"]["LABELS"]["value"])
        rate = float(c.parameters["POINT"]["RATE"]["value"][0])
        i_toe = get_time(labels, "RTOE", pref)
        i_com = get_time(labels, "CentreOfMass", pref)
        i_knee = get_time(labels, "RKneeAngles", pref)
        if i_toe < 0:
            return None
        traj = pts[:3, i_toe, :].astype(float)
        kin = compute_kinematics(traj, rate)
        det = detect_phases(np.linalg.norm(np.gradient(traj, axis=1) * rate, axis=0), rate)
        kin_units = {
            "vmax": "vmax_mm_s", "vmean": "vmean_mm_s", "amax": "amax_mm_s2",
            "displacement": "displacement_mm", "path_length": "path_length_mm",
            "rom": "rom_mm", "time_to_vmax": "time_to_vmax_s",
        }
        row = {"condition": cond, "trial": trial, "subject": pref, "role": role,
               "file": Path(fp).name, "n_frames": traj.shape[1],
               "duration_s": round(traj.shape[1] / rate, 3),
               **{kin_units[k]: v for k, v in kin.items()},
               "t_onset": det["t_onset"], "t_stall": det["t_stall"],
               "t_peak": det["t_peak"], "t_offset": det["t_offset"],
               "dur_prep_s": det["t_stall"] - det["t_onset"],
               "dur_exec_s": det["t_peak"] - det["t_stall"],
               "dur_final_s": det["t_offset"] - det["t_peak"],
               }
        if i_com >= 0:
            com = pts[:3, i_com, :].astype(float)
            com_speed = np.linalg.norm(np.gradient(com, axis=1) * rate, axis=0)
            row["com_vmax"] = float(np.nanmax(com_speed))
            row["com_rom"] = float(np.nanmax(np.linalg.norm(com - com[:, :1], axis=0)))
        return row

    rows = []
    for cond in ["E01", "E02", "E04"]:
        for trial in ["T01", "T02"]:
            cand = find_files(f"S04-{cond}-{trial}")
            if not cand:
                print(f"  [warn] S04-{cond}-{trial} no presente")
                continue
            fp = cand[0]
            c = load_c3d(fp)
            prefixes = get_prefixes(c)
            athlete = athlete_prefix(fp, prefixes)
            others = [p for p in prefixes if p != athlete]
            # rol del atleta del dataset según condición
            role_athlete = "atacante" if cond in ("E01", "E02") else "defensor"
            r = kin_row(fp, athlete, role_athlete, cond, trial)
            if r:
                rows.append(r)
            # en condiciones de oponente, también medir al oponente (si ejecuta kick)
            for other in others:
                if "Tarcza" in other:
                    continue
                r2 = kin_row(fp, other, "oponente (atacante)", cond, trial)
                if r2:
                    rows.append(r2)

    df = pd.DataFrame(rows)
    df.to_csv(TAB_DIR / "f5_condition_comparison.csv", index=False)

    # Análisis agregado por condición (solo atleta del dataset)
    print("\nAgregado por condición (atleta B0367, media T01/T02) — RTOE:")
    g = df[df["role"].ne("oponente (atacante)")].groupby("condition").mean(numeric_only=True)
    show_cols = ["duration_s", "vmax_mm_s", "vmean_mm_s", "amax_mm_s2", "rom_mm",
                 "path_length_mm", "time_to_vmax_s", "dur_prep_s", "dur_exec_s", "dur_final_s"]
    print(g[show_cols].round(1).to_string())

    # Figura comparativa (solo atleta)
    fig, axes = plt.subplots(2, 3, figsize=(16, 7))
    sub_df = df[df["role"].ne("oponente (atacante)")]
    for cond, color in zip(["E01", "E02", "E04"], ["#2f6fb3", "#e08a2e", "#d1495b"]):
        sub = sub_df[sub_df["condition"] == cond]
        for i, col in enumerate(["vmax_mm_s", "vmean_mm_s", "amax_mm_s2"]):
            ax = axes[0, i]
            ax.bar(sub["trial"], sub[col], color=color, alpha=0.75, label=cond)
            ax.set_title(col); ax.legend(fontsize=7)
        for i, col in enumerate(["duration_s", "rom_mm", "time_to_vmax_s"]):
            ax = axes[1, i]
            ax.bar(sub["trial"], sub[col], color=color, alpha=0.75, label=cond)
            ax.set_title(col); ax.legend(fontsize=7)
    fig.suptitle("S04 Mawashi-Geri jodan — comparación E01/E02/E04 (atleta)", fontsize=13)
    fig.tight_layout()
    save_fig(fig, "f5_condition_comparison.png")

    # Figura de perfiles de velocidad superpuestos (T01)
    fig, ax = plt.subplots(figsize=(12, 5))
    for cond, color in zip(["E01", "E02", "E04"], ["#2f6fb3", "#e08a2e", "#d1495b"]):
        sub = df[(df["condition"] == cond) & (df["trial"] == "T01")]
        if sub.empty:
            continue
        click = next(iter(find_files(f"S04-{cond}-T01")))
        c = load_c3d(click)
        labels = list(c.parameters["POINT"]["LABELS"]["value"])
        rate = float(c.parameters["POINT"]["RATE"]["value"][0])
        prefixes = get_prefixes(c)
        pref = athlete_prefix(fp, prefixes)
        itoe = get_time(labels, "RTOE", pref)
        traj = c["data"]["points"][:3, itoe, :].astype(float)
        v = np.linalg.norm(np.gradient(traj, axis=1) * rate, axis=0)
        ax.plot(np.arange(len(v)) / rate, smooth(v, window=17), color=color, label=cond)
    ax.set_xlabel("tiempo [s]"); ax.set_ylabel("velocidad RTOE [mm/s]")
    ax.set_title("S04 — perfiles de velocidad superpuestos (T01)")
    ax.legend()
    fig.tight_layout()
    save_fig(fig, "f5_condition_speed_overlay.png")

    return df


# ----------------------------------------------------------------------------- #
# FASE 6 — Repetibilidad T01 vs T02 (DTW)
# ----------------------------------------------------------------------------- #

def dtw_dist(a: np.ndarray, b: np.ndarray, radius: int = -1) -> float:
    """Distancia DTW (euclídea) con radio en celdas.
    Si radius <= 0, se usa radio completo (Sakoe-Chiba nulo) para no depender
    de la diferencia de longitud entre las dos ejecuciones."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n, m = len(a), len(b)
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    r = max(radius, 1)
    for i in range(1, n + 1):
        lo = 1 if radius <= 0 else max(1, i - r)
        hi = m if radius <= 0 else min(m, i + r)
        for j in range(lo, hi + 1):
            cost = (a[i - 1] - b[j - 1]) ** 2
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return float(np.sqrt(D[n, m]))


def normalize_(x: np.ndarray) -> np.ndarray:
    return (x - np.nanmean(x)) / (np.nanstd(x) + 1e-12)


def fase6_repetibilidad():
    print("\n" + "=" * 72)
    print("FASE 6 — REPETIBILIDAD T01 vs T02 (DTW)")
    print("=" * 72)

    rows = []
    for tech in ["S01", "S02", "S03", "S04", "S05"]:
        for cond in ["E01", "E02", "E04"]:
            cand = find_files(f"{tech}-{cond}-T01")
            cand2 = find_files(f"{tech}-{cond}-T02")
            if not cand or not cand2:
                continue
            fp1, fp2 = cand[0], cand2[0]
            rates = []
            trajs = []
            for fp in (fp1, fp2):
                c = load_c3d(fp)
                labels = list(c.parameters["POINT"]["LABELS"]["value"])
                rate = float(c.parameters["POINT"]["RATE"]["value"][0])
                prefixes = get_prefixes(c)
                pref = athlete_prefix(fp, prefixes)
                itoe = get_time(labels, "RTOE", pref)
                # si RTOE no disponible (E04 con prefijo B0367, siempre hay)
                traj = c["data"]["points"][:3, itoe, :].astype(float)
                # interpolamos NaN para DTW
                for k in range(3):
                    idx = np.where(np.isnan(traj[k]))
                    if idx[0].size:
                        ok = np.isfinite(traj[k])
                        if ok.sum() > 1:
                            traj[k, idx] = np.interp(idx[0], np.flatnonzero(ok), traj[k, ok])
                rates.append(rate)
                trajs.append(traj)

            if rates[0] != rates[1]:
                print(f"  [warn] {tech}-{cond}: rates distintos {rates}")
            rate = rates[0]
            v1 = np.linalg.norm(np.gradient(trajs[0], axis=1) * rate, axis=0)
            v2 = np.linalg.norm(np.gradient(trajs[1], axis=1) * rate, axis=0)
            v1s, v2s = smooth(v1, 17), smooth(v2, 17)

            dtw_raw = dtw_dist(normalize_(v1s), normalize_(v2s))
            # distancia euclídea re-muestreada a la misma longitud
            m = min(len(v1s), len(v2s))
            euc = float(np.sqrt(np.mean((normalize_(v1s[:m]) - normalize_(v2s[:m])) ** 2)))

            # coeficiente de variación de vmax
            cv_vmax = float(np.std([np.max(v1s), np.max(v2s)]) / np.mean([np.max(v1s), np.max(v2s)]) * 100)
            cv_dur = float(np.std([len(v1), len(v2)]) / np.mean([len(v1), len(v2)]) * 100)

            rows.append({
                "technique": tech, "condition": cond,
                "dtw_normalized": round(dtw_raw, 3),
                "euclidean_norm": round(euc, 3),
                "similarity_dtw": round(1 / (1 + dtw_raw), 3),
                "cv_vmax_pct": round(cv_vmax, 2),
                "cv_duration_pct": round(cv_dur, 2),
                "len_t01": len(v1), "len_t02": len(v2),
            })

    df = pd.DataFrame(rows)
    df.to_csv(TAB_DIR / "f6_repeatability.csv", index=False)
    print(df.round(3).to_string(index=False))

    # figura DTW de ejemplo S04-E01
    cand1 = next(iter(find_files("S04-E01-T01")))
    cand2 = next(iter(find_files("S04-E01-T02")))
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    for ax, fp, col in zip(axes, [cand1, cand2], ["#2f6fb3", "#e08a2e"]):
        c = load_c3d(fp)
        labels = list(c.parameters["POINT"]["LABELS"]["value"])
        rate = float(c.parameters["POINT"]["RATE"]["value"][0])
        prefixes = get_prefixes(c)
        pref = athlete_prefix(fp, prefixes)
        itoe = get_time(labels, "RTOE", pref)
        traj = c["data"]["points"][:3, itoe, :].astype(float)
        v = smooth(np.linalg.norm(np.gradient(traj, axis=1) * rate, axis=0), 17)
        ax.plot(np.arange(len(v)) / rate, v, color=col)
        ax.set_title(Path(fp).name)
        ax.set_ylabel("velocidad RTOE [mm/s]")
    fig.suptitle("S04 / E01 — repetibilidad T01 vs T02")
    fig.tight_layout()
    save_fig(fig, "f6_repeatability_s04_e01.png")

    return df


# ----------------------------------------------------------------------------- #
# FASE 7 — Feature engineering
# ----------------------------------------------------------------------------- #

FEATURES_CALCULABLE = [
    "duración total",
    "tiempo hasta velocidad máxima (pico)",
    "tiempo hasta aceleración máxima",
    "velocidad máxima (endpoint)",
    "velocidad media",
    "aceleración máxima",
    "desplazamiento neto",
    "distancia recorrida (path length)",
    "rango de movimiento (ROM) del endpoint",
    "ROM articular (cadera/rodilla/tobillo/hombro/codo) — ya calculados en C3D",
    "velocidad angular y aceleración angular articulares (derivables)",
    "retraso temporal entre picos de segmentos (por ej. cadera -> rodilla -> pie)",
    "correlación cruzada entre ángulos articulares",
    "desplazamiento del COM",
    "velocidad y aceleración del COM",
    "simetría izquierda/derecha (comparación de marcadores L/R equivalentes)",
    "duración de fases (preparatoria / ejecución / final) — de la detección de fases",
    "potencia articular máxima (momento * velocidad angular) — ya disponibles",
    "integridad/calidad: % de frames con NaN por marcador",
]

FEATURES_NO_CALCULABLE = [
    "fuerzas de reacción del suelo (GRF) — no hay plataformas ni analógicos",
    "momentos reales en el pie (dependen de GRF)",
    "presión plantar distribuida",
    "EMG / activación muscular",
    "frecuencia cardíaca, lactato, VO2, carga fisiológica",
    "acelerometría inercial (IMU) local por segmento",
    "velocidad de impacto real sobre el objetivo (requiere instrumentar al escudo)",
    "precisión/distorsión percibida del golpe (resultado de combate)",
    "fatiga (requiere series prolongadas + fisiológicos)",
    "datos de ojo/inicio de intención (latencias cognitivas)",
]


def fase7_features():
    print("\n" + "=" * 72)
    print("FASE 7 — FEATURE ENGINEERING (propuesta)")
    print("=" * 72)
    df_a = pd.DataFrame({"feature": FEATURES_CALCULABLE, "calculable": True,
                         "fuente": "C3D directo / derivado"})
    df_b = pd.DataFrame({"feature": FEATURES_NO_CALCULABLE, "calculable": False,
                         "fuente": "requiere sensores adicionales"})
    df = pd.concat([df_a, df_b], ignore_index=True)
    df.to_csv(TAB_DIR / "f7_feature_matrix.csv", index=False)
    print(f"Features calculables: {len(FEATURES_CALCULABLE)}")
    print(f"Features que requieren más datos: {len(FEATURES_NO_CALCULABLE)}")
    print("\nCalculables:")
    for x in FEATURES_CALCULABLE:
        print(f"   + {x}")
    print("\nNo calculables con este dataset:")
    for x in FEATURES_NO_CALCULABLE:
        print(f"   - {x}")
    return df


# ----------------------------------------------------------------------------- #
# FASE 8-9 — resumen ML y demo (documentado en el informe final)
# ----------------------------------------------------------------------------- #

def fase89_ml_demo():
    print("\n" + "=" * 72)
    print("FASE 8 — PROBLEMAS DE ML EVALUADOS (sin entrenar)")
    print("=" * 72)
    ml_rows = [
        {
            "problema": "Clasificación de técnica (S01..S05)",
            "objetivo": "técnica (5 clases)",
            "features": "features F7 (temporal, cinemática, articular) por ejecución",
            "modelo": "RandomForest / GradientBoosting / MLP",
            "leakage": "si se mezclan trials del mismo atleta entre train/test",
            "muestra_necesaria": ">= 100 ejecuciones/clase; viable con dataset completo (5 tandas)",
            "viable_con_actual": "parcial — solo B0367 no da generalización, sirve como demo de framework",
        },
        {
            "problema": "Clasificación de condición (air/shield/defender)",
            "objetivo": "E01/E02/E04 (3 clases)",
            "features": "velocidad pico, altura de rodilla, distancia al objetivo (Tarcza), duración",
            "modelo": "SVM / LogisticRegression",
            "leakage": "mismo atleta", "muestra_necesaria": "~ 30-60/condición",
            "viable_con_actual": "sí, conceptualmente sobre B0367",
        },
        {
            "problema": "Clasificación nivel atleta (beginner vs advanced)",
            "objetivo": "nivel (2 clases)",
            "features": "features F7",
            "modelo": "ensemble + GroupKFold / LOSO",
            "leakage": "MUY ALTO: los trials del mismo atleta aparecen en train y test — usar división por participante",
            "muestra_necesaria": ">= 20 atletas por grupo para algo serio; mejor 50+",
            "viable_con_actual": "NO con B0367 (1 atleta) — necesario dataset completo",
        },
        {
            "problema": "Clustering de patrones de movimiento",
            "objetivo": "agrupar ejecuciones similares (sin etiqueta)",
            "features": "features F7 + curvas DTW-k",
            "modelo": "k-means / HDBSCAN sobre embedding (t-SNE)",
            "leakage": "n/a", "muestra_necesaria": ">= 50 ejecuciones",
            "viable_con_actual": "sí, exploratorio sobre B0367",
        },
        {
            "problema": "Anomaly detection (ejecuciones atípicas)",
            "objetivo": "score de anormalidad por ejecución",
            "features": "features F7 / distancias DTW",
            "modelo": "IsolationForest / OneClassSVM",
            "leakage": "n/a", "muestra_necesaria": ">= 30 ejecuciones baseline",
            "viable_con_actual": "sí, sobre B0367 (dentro-atleta)",
        },
        {
            "problema": "Similarity search",
            "objetivo": "recuperar ejecuciones biomecánicamente similares",
            "features": "curvas normalizadas + DTW",
            "modelo": "kNN + DTW / index vectorial",
            "leakage": "n/a", "muestra_necesaria": ">= 100 ejecuciones indexadas",
            "viable_con_actual": "sí, sobre el set local (26 archivos)",
        },
    ]
    df = pd.DataFrame(ml_rows)
    df.to_csv(TAB_DIR / "f8_ml_problems.csv", index=False)
    for _, r in df.iterrows():
        print(f"\n- {r['problema']}:")
        print(f"    objetivo: {r['objetivo']}")
        print(f"    modelo:   {r['modelo']}")
        print(f"    leakage:  {r['leakage']}")
        print(f"    muestra:  {r['muestra_necesaria']}")
        print(f"    viable?   {r['viable_con_actual']}")

    print("\n" + "=" * 72)
    print("FASE 9 — ARQUITECTURA DEL DEMO (ver informe docs/dataset_audit.md)")
    print("=" * 72)

    return df


# ----------------------------------------------------------------------------- #
# MAIN
# ----------------------------------------------------------------------------- #

def main():
    print(f"Proyecto raíz: {ROOT}")
    print(f"Dataset:      {DATA_DIR}")
    print(f"ezc3d:        {ezc3d.__version__}")

    phase_funcs = [
        fase1_inventario,
        fase2_inspeccion,
        fase3_mapping,
        fase4_señales,
        fase5_condiciones,
        fase6_repetibilidad,
        fase7_features,
        fase89_ml_demo,
    ]
    for fn in phase_funcs:
        try:
            fn()
        except Exception as e:
            import traceback
            print(f"[ERROR] {fn.__name__}: {e}")
            traceback.print_exc()

    print("\n" + "=" * 72)
    print("FIN — salidas en output/figures y output/tables")
    print("=" * 72)


if __name__ == "__main__":
    main()