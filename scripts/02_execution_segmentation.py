#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
02_execution_segmentation.py
============================
FASE 1.5 — VALIDACIÓN DEL PIPELINE DE EJECUCIONES INDIVIDUALES

Objetivo: validar experimentalmente que podemos transformar un C3D que contiene
varias repeticiones de una técnica en un dataset tabular donde 1 fila = 1 ejecución.

C3D -> identificación de repeticiones -> segmentación -> features -> 1 fila/ejecución

Reutiliza funciones de 01_dataset_exploration.py (no duplica lógica).

NO modifica C3D, NO entrena modelos, NO construye dashboard.

Salidas:
  output/executions_sample.csv      (1 fila por ejecución)
  output/execution_quality.csv      (control de calidad por ejecución)
  output/phase_validation/          (gráficas de segmentación)
  output/dtw_validation/            (validación DTW T01 vs T02)
  output/condition_comparison/      (comparación E01 vs E02, S04)
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import yaml

# ------- reutilizar funciones de la Fase 1 -------------------------------- #
# 01_dataset_exploration.py no es importable por `import` (nombre con dígito),
# así que lo cargamos con importlib y extraemos los símbolos.
import importlib.util

_SPEC = importlib.util.spec_from_file_location(
    "dataset_exploration_module",
    str(Path(__file__).resolve().parent / "01_dataset_exploration.py"),
)
_X01 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_X01)

ROOT = _X01.ROOT
TECHNIQUE_NAMES = _X01.TECHNIQUE_NAMES
CONDITION_NAMES = _X01.CONDITION_NAMES
load_c3d = _X01.load_c3d
get_prefixes = _X01.get_prefixes
athlete_prefix = _X01.athlete_prefix
get_time = _X01.get_time
smooth = _X01.smooth
dtw_dist = _X01.dtw_dist
normalize_ = _X01.normalize_
parse_name = _X01.parse_name

# ------- atleta activo y resolución de rutas (helper local, NO toca 01) ------ #
# 01_dataset_exploration.py fija DATA_DIR = ROOT/"B0367". Para soportar más
# atletas sin modificar 01, resolvemos aquí la ruta de datos por atleta_id.

ACTIVE_ATHLETE: str = "B0367"
ATHLETE_CFG_DIR = ROOT / "config" / "athletes"


def data_dir_for(athlete_id: str) -> Path:
    """Carpeta de datos de un atleta (B0367 en raíz, otros en atletas/<id>)."""
    if athlete_id == "B0367":
        return ROOT / "B0367"
    return ROOT / "atletas" / athlete_id


def all_files(athlete_id: str | None = None) -> list[Path]:
    ath = athlete_id or ACTIVE_ATHLETE
    return sorted(data_dir_for(ath).glob("**/*.c3d"))


def find_files(tag: str, athlete_id: str | None = None) -> list[Path]:
    """Devuelve archivos cuyo stem termina exactamente en '-<tag>' (tag sin .c3d)."""
    return [p for p in all_files(athlete_id) if p.stem.endswith("-" + tag)]

# forzar UTF-8 en consola
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

OUT = ROOT / "output"
SEG_DIR = OUT / "phase_validation"
DTW_DIR = OUT / "dtw_validation"
CMP_DIR = OUT / "condition_comparison"
for d in (SEG_DIR, DTW_DIR, CMP_DIR):
    d.mkdir(parents=True, exist_ok=True)

MM_IN_M = 1e-3

# --------------------------------------------------------------------------- #
# Configuración centralizada (config/segmentation.yaml)
# --------------------------------------------------------------------------- #

CONFIG_PATH = ROOT / "config" / "segmentation.yaml"
SEGMENTATION_METHOD = "activity_bands"
SEGMENTATION_VERSION = "1.8.0"


def _deep_merge(base: dict, override: dict) -> dict:
    """Merge profundo: override con clave presente gana sobre base."""
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(athlete_id: str | None = None) -> dict:
    """Carga la configuración global y, si existe, la del atleta (merge).

    `athlete_id=None` devuelve solo la config global (comportamiento histórico).
    `athlete_id=B0367` reproduce exactamente el comportamiento actual.
    """
    defaults = {
        "segmentation_method": SEGMENTATION_METHOD,
        "segmentation_version": SEGMENTATION_VERSION,
        "params": {
            "smooth_win": 15, "activity_frac": 0.15,
            "min_dur_s": 0.15, "max_dur_s": 2.5,
            "min_gap_s": 0.30, "min_peak_ratio": 0.45,
            "review_peak_ratio": 0.30,
        },
        "signals": {
            "S01": {"signal": "RFIN", "movement": "punch"},
            "S02": {"signal": "RTOE", "movement": "kick"},
            "S03": {"signal": "RTOE", "movement": "kick"},
            "S04": {"signal": "RTOE", "movement": "kick"},
            "S05": {"signal": "RTOE", "movement": "kick"},
        },
    }
    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        cfg = defaults
        cfg = _deep_merge(defaults, {k: v for k, v in loaded.items()})
        for k in ("params", "signals"):
            cfg.setdefault(k, defaults[k])
        cfg["params"].update(defaults["params"])
        # config específica del atleta (sobreescribe global solo si existe)
        if athlete_id is not None:
            athlete_cfg_path = ATHLETE_CFG_DIR / f"{athlete_id}.yaml"
            if athlete_cfg_path.exists():
                with open(athlete_cfg_path, encoding="utf-8") as f:
                    a_loaded = yaml.safe_load(f) or {}
                cfg = _deep_merge(cfg, a_loaded)
        return cfg
    except Exception as e:
        print(f"  [warn] no se pudo leer config ({e}); usando defaults")
        return defaults


def set_active_athlete(athlete_id: str):
    """Activa un atleta: recarga config, señales y rutas de salida."""
    global ACTIVE_ATHLETE, CFG, _PRIMARY_CFG, PRIMARY_SIGNAL, FALLBACK_SIGNAL, JOINTS_SIDE, OUT_ATHLETE
    ACTIVE_ATHLETE = athlete_id
    CFG = load_config(athlete_id)
    _PRIMARY_CFG = {t: v["signal"] for t, v in (CFG.get("signals") or {}).items()}
    PRIMARY_SIGNAL, FALLBACK_SIGNAL, JOINTS_SIDE = _build_signals(CFG)
    # rutas de salida: B0367 mantiene output/ raíz; otros en output/<athlete>/
    if athlete_id == "B0367":
        OUT_ATHLETE = OUT
    else:
        OUT_ATHLETE = OUT / athlete_id
        OUT_ATHLETE.mkdir(parents=True, exist_ok=True)


def defaults_signal_for(t: str) -> str:
    return {"S01": "RFIN", "S02": "RTOE", "S03": "RTOE",
            "S04": "RTOE", "S05": "RTOE"}.get(t, "RTOE")


def _build_signals(cfg: dict) -> tuple[dict, dict, dict]:
    """Deriva PRIMARY_SIGNAL / FALLBACK_SIGNAL / joints desde la config resuelta."""
    src = (cfg.get("techniques") or cfg.get("signals") or {})
    primary = {}
    for t in ["S01", "S02", "S03", "S04", "S05"]:
        entry = src.get(t, {})
        sig = (entry.get("signal") if isinstance(entry, dict) else str(entry)) or "TBD"
        primary[t] = [sig] if sig != "TBD" else [defaults_signal_for(t)]
    fallback = {
        "S01": ["RTOE", "RHJC", "LKNE"],
        "S02": ["LTOE", "RKNE", "RHJC"],
        "S03": ["LTOE", "RKNE", "RHJC"],
        "S04": ["LTOE", "RKNE", "RHJC"],
        "S05": ["LTOE", "RKNE", "RHJC"],
    }
    joints = {}
    for t in ["S01", "S02", "S03", "S04", "S05"]:
        entry = src.get(t, {})
        joints[t] = (entry.get("joints_side", "R") if isinstance(entry, dict) else "R")
    return primary, fallback, joints


# Config resuelta al importar (baseline B0367). set_active_athlete() la re-resuelve.
CFG = load_config("B0367")
_PRIMARY_CFG = {t: v["signal"] for t, v in CFG["signals"].items()}
PRIMARY_SIGNAL, FALLBACK_SIGNAL, JOINTS_SIDE = _build_signals(CFG)
# ruta de salida para el atleta activo (default = baseline)
OUT_ATHLETE: Path = OUT

# --------------------------------------------------------------------------- #
# 0. Selección de señal candidata por técnica (PRIMER PASO — inspección)
# --------------------------------------------------------------------------- #

# Determinado empíricamente en la inspección previa (ver informe 1.5):
#   - S01 Gyaku-Zuki (puño): endpoint = mano derecha (RFIN/RWRB), no RTOE.
#   - S02..S05 (patadas):    endpoint = pie derecho dominante (RTOE/RANK/RHEE).
# La pierna dominante de B0367 es la derecha en las técnicas de patada.
#
# Señal PRIMARIA por gesto técnico (no solo SNR): para el puño el pie apenas
# se mueve (baseline ~0) y una SNR vmax/baseline enorme sesgaría hacia él.
# Para cada técnica definimos el segmento efectuador y comprobamos su SNR.

# Señales primarias resueltas en set_active_athlete() (config atleta > global).
# El bloque anterior (PRIMARY_SIGNAL/FALLBACK_SIGNAL) fue movido arriba junto a la
# carga de configuración para permitir re-derivarlo por atleta.


def pick_best_signal(fp: Path, prefix: str, technique: str, min_snr: float = 8.0):
    """Elige el marker efectuador de la técnica (primario) con SNR adecuada.

    Devuelve dict con marker, snr, vmax, baseline, index y `primary` (bool).
    """
    c = load_c3d(fp)
    pts = c["data"]["points"]
    labels = list(c.parameters["POINT"]["LABELS"]["value"])
    rate = float(c.parameters["POINT"]["RATE"]["value"][0])

    def snr_of(mk):
        i = get_time(labels, mk, prefix)
        if i < 0:
            return None
        traj = pts[:3, i, :].astype(float)
        v = np.linalg.norm(np.gradient(traj, axis=1) * rate, axis=0)
        bl = np.median(v[: int(rate)])
        return {"marker": mk, "snr": float(np.max(v) / (bl + 1.0)),
                "vmax": float(np.max(v)), "baseline": float(bl), "index": i}

    best = None
    for mk in PRIMARY_SIGNAL.get(technique, ["RTOE"]):
        s = snr_of(mk)
        if s and s["snr"] >= min_snr:
            best = s
            best["primary"] = True
            break
    if best is None:
        for mk in FALLBACK_SIGNAL.get(technique, ["RTOE"]):
            s = snr_of(mk)
            if s and s["snr"] >= min_snr:
                best = s
                best["primary"] = True
                break
    if best is None:
        # último recurso: el que tenga mejor SNR (aunque no sea el efectuador)
        for mk in PRIMARY_SIGNAL.get(technique, ["RTOE"]) + FALLBACK_SIGNAL.get(technique, []):
            s = snr_of(mk)
            if s and (best is None or s["snr"] > best["snr"]):
                best = s
        if best is not None:
            best["primary"] = False
    if best is None:
        return None
    best["suitable"] = best["snr"] >= min_snr
    return best


def get_signal(fp: Path, prefix: str, marker: str) -> tuple[np.ndarray, float, int]:
    """Devuelve la trayectoria (3, frames) del marcador y la velocidad 3D."""
    c = load_c3d(fp)
    pts = c["data"]["points"]
    labels = list(c.parameters["POINT"]["LABELS"]["value"])
    rate = float(c.parameters["POINT"]["RATE"]["value"][0])
    i = get_time(labels, marker, prefix)
    traj = pts[:3, i, :].astype(float)
    v = np.linalg.norm(np.gradient(traj, axis=1) * rate, axis=0)
    return traj, v, rate


# --------------------------------------------------------------------------- #
# 1. Segmentación de repeticiones
# --------------------------------------------------------------------------- #

def segment_repetitions(v: np.ndarray, rate: float,
                        smooth_win: int | None = None,
                        activity_frac: float | None = None,
                        min_dur_s: float | None = None,
                        max_dur_s: float | None = None,
                        min_gap_s: float | None = None,
                        min_peak_ratio: float | None = None,
                        review_peak_ratio: float | None = None,
                        return_events: bool = True
                        ) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Detecta repeticiones completas en el perfil de velocidad de un endpoint.

    Método por BANDAS DE ACTIVIDAD (más robusto que "buscar máximos"):

      1. v_s = smooth(v) — preprocesamiento mínimo (Hanning ~150 ms @200 Hz).
      2. baseline = mediana del primer segundo (atleta en kumite-no-kamae).
      3. nivel de actividad = baseline + activity_frac * (max(v_s) - baseline).
      4. bandas = períodos continuos donde v_s > nivel de actividad.
      5. Dentro de cada banda: peak = argmax local; start/end = límites de la banda.
      6. Fusión: si dos bandas contiguas están separadas por menos de min_gap_s y
         la segunda tiene pico <= que la primera, se consideran la MISMA ejecución
         (pico chico = retorno/recuperación tras el golpe, no repetición nueva).
      7. Clasificación por pico y duración:
           - pico < review_peak_ratio * vmax  -> rejected (activity_too_low)
           - review_ratio <= pico < min_peak_ratio -> review (low_activity_uncertain)
           - pico >= min_peak_ratio           -> accepted (si duración válida)
           - duración fuera de [min_dur, max_dur] -> rejected (dur...)
      8. Devuelve también el registro de eventos (accepted/rejected/review).

    Devuelve (tabla aceptada con start/peak/end, eventos) si return_events=True;
    si False, devuelve (tabla aceptada, señal suavizada) — compatibilidad.
    """
    prm = CFG["params"]
    smooth_win = smooth_win if smooth_win is not None else prm["smooth_win"]
    activity_frac = activity_frac if activity_frac is not None else prm["activity_frac"]
    min_dur_s = min_dur_s if min_dur_s is not None else prm["min_dur_s"]
    max_dur_s = max_dur_s if max_dur_s is not None else prm["max_dur_s"]
    min_gap_s = min_gap_s if min_gap_s is not None else prm["min_gap_s"]
    min_peak_ratio = min_peak_ratio if min_peak_ratio is not None else prm["min_peak_ratio"]
    review_peak_ratio = review_peak_ratio if review_peak_ratio is not None else prm["review_peak_ratio"]

    v_s = smooth(v, window=int(smooth_win) | 1)
    n = len(v_s)
    baseline = float(np.median(v_s[: int(rate)]))
    vmax = float(np.max(v_s))
    if vmax <= baseline:
        empty = pd.DataFrame(columns=["start_frame", "peak_frame", "end_frame"])
        return (empty, empty) if return_events else (empty, v_s)
    level = baseline + activity_frac * (vmax - baseline)

    above = v_s > level
    labels = np.zeros(n, dtype=int)
    cur = 0
    prev = above[0]
    for i in range(n):
        if above[i] and not prev:
            cur += 1
        if above[i]:
            labels[i] = cur
        prev = above[i]

    # bandas brutas (con motivo inicial)
    bands = []
    for b in range(1, cur + 1):
        idx = np.flatnonzero(labels == b)
        if idx.size == 0:
            continue
        start, end = int(idx[0]), int(idx[-1])
        pk = int(np.argmax(v_s[start:end + 1])) + start
        reason = None
        if (end - start) < int(min_dur_s * rate):
            reason = "duration_too_short"
        bands.append({"start_frame": start, "peak_frame": pk, "end_frame": end,
                      "peak_v": float(v_s[pk]), "reason": reason})

    # fusión de bandas contiguas de la misma ejecución (retorno/recuperación)
    merged = []
    for b in bands:
        if merged:
            last = merged[-1]
            gap = b["start_frame"] - last["end_frame"]
            if gap <= int(min_gap_s * rate) and b["peak_v"] <= last["peak_v"]:
                # la banda actual es un sub-pico de la anterior -> secondary_peak
                if b.get("reason") is None:
                    b["reason"] = "secondary_peak"
                b["absorbed_into"] = len(merged) - 1
                merged.append(b)
                last["end_frame"] = b["end_frame"]
                if b["peak_v"] > float(v_s[last["peak_frame"]]):
                    last["peak_frame"] = b["peak_frame"]
                    last["peak_v"] = b["peak_v"]
                    last["reason"] = None  # el pico dominante dejó de ser secundario
                continue
        merged.append(dict(b))

    # --- construir registro de eventos ---
    events = []
    for b in merged:
        ratio = b["peak_v"] / vmax if vmax > 0 else 0.0
        dur_f = b["end_frame"] - b["start_frame"]
        dur = dur_f / rate
        status = "accepted"
        reason = b.get("reason")
        if b.get("absorbed_into") is not None:
            status, reason = "rejected", "secondary_peak"
        elif reason == "duration_too_short":
            status = "rejected"
        elif ratio < review_peak_ratio:
            status, reason = "rejected", "activity_too_low"
        elif ratio < min_peak_ratio:
            status, reason = "review", "low_activity_uncertain"
        elif dur < min_dur_s:
            status, reason = "rejected", "duration_too_short"
        elif dur > max_dur_s:
            status, reason = "rejected", "duration_too_long"
        events.append({
            "start_frame": int(b["start_frame"]), "peak_frame": int(b["peak_frame"]),
            "end_frame": int(b["end_frame"]), "duration": round(dur, 3),
            "peak_v_mm_s": round(b["peak_v"], 1), "peak_ratio": round(ratio, 2),
            "status": status, "rejection_reason": reason or "",
        })

    accepted = [e for e in events if e["status"] == "accepted"]
    rows = [{"start_frame": e["start_frame"], "peak_frame": e["peak_frame"],
             "end_frame": e["end_frame"]} for e in accepted]

    df = pd.DataFrame(rows)
    if not df.empty:
        df["duration_frames"] = df["end_frame"] - df["start_frame"]
    ev = pd.DataFrame(events)
    if ev.empty:
        ev = pd.DataFrame(columns=["start_frame", "peak_frame", "end_frame",
                                   "duration", "peak_v_mm_s", "peak_ratio",
                                   "status", "rejection_reason"])
    if return_events:
        return df, ev
    return df, v_s


# --------------------------------------------------------------------------- #
# 2. Features por ejecución
# --------------------------------------------------------------------------- #

def extract_features(fp: Path, prefix: str, seg: dict, marker: str,
                     rate: float, technique: str) -> dict:
    """Extrae features de UNA ejecución (window start..peak..end)."""
    c = load_c3d(fp)
    pts = c["data"]["points"]
    labels = list(c.parameters["POINT"]["LABELS"]["value"])

    s, p, e = int(seg["start_frame"]), int(seg["peak_frame"]), int(seg["end_frame"])
    dur_s = (e - s) / rate
    t2peak = (p - s) / rate

    # ---- posición del endpoint ----
    i = get_time(labels, marker, prefix)
    traj = pts[:3, i, :].astype(float)
    v_full = np.linalg.norm(np.gradient(traj, axis=1) * rate, axis=0)
    v = v_full[s:e]
    v_s = smooth(v_full, window=15)[s:e]

    vmax = float(np.max(v_s))
    vmean = float(np.mean(v_s))
    acc = np.gradient(v_s, 1.0 / rate)
    amax = float(np.max(np.abs(acc)))
    amean = float(np.mean(np.abs(acc)))

    # desplazamiento neto y longitud de trayectoria (dentro de la ventana)
    seg3 = traj[:, s:e] * MM_IN_M  # mm -> m
    displacement = float(np.linalg.norm(seg3[:, -1] - seg3[:, 0]))
    path_length = float(np.sum(np.linalg.norm(np.diff(seg3, axis=1), axis=0)))
    rom = float(np.max(np.linalg.norm(seg3 - seg3[:, :1], axis=0)))

    # ---- variables articulares (pierna derecha / brazo derecho) ----
    def angle_window(label):
        i = get_time(labels, label, prefix)
        if i < 0:
            return None
        a = pts[:3, i, s:e].astype(float)
        return a

    # lado de articulaciones para features (config atleta > default R)
    side = str(JOINTS_SIDE.get(technique, "R")).upper()
    side = "L" if side == "L" else "R"
    joins = {
        "S01": [f"{side}ShoulderAngles", f"{side}ElbowAngles",
                f"{side}HipAngles", f"{side}KneeAngles"],
        "S02": [f"{side}HipAngles", f"{side}KneeAngles", f"{side}AnkleAngles"],
        "S03": [f"{side}HipAngles", f"{side}KneeAngles", f"{side}AnkleAngles"],
        "S04": [f"{side}HipAngles", f"{side}KneeAngles", f"{side}AnkleAngles"],
        "S05": [f"{side}HipAngles", f"{side}KneeAngles", f"{side}AnkleAngles"],
    }
    # cadena proximal->distal para timing/coordinación (según técnica)
    chain = {
        "S01": [f"{side}HipAngles", f"{side}ElbowAngles"],
        "S02": [f"{side}HipAngles", f"{side}KneeAngles", f"{side}AnkleAngles"],
        "S03": [f"{side}HipAngles", f"{side}KneeAngles", f"{side}AnkleAngles"],
        "S04": [f"{side}HipAngles", f"{side}KneeAngles", f"{side}AnkleAngles"],
        "S05": [f"{side}HipAngles", f"{side}KneeAngles", f"{side}AnkleAngles"],
    }
    feats = {}
    angvel_peak = {}
    for lab in joins.get(technique, []):
        a = angle_window(lab)
        if a is None or np.isnan(a).all():
            feats[f"rom_{lab}"] = np.nan
            feats[f"avg_angvel_{lab}"] = np.nan
            continue
        rom_deg = float(np.nanmax(a, axis=1).max() - np.nanmin(a, axis=1).min())
        # velocidad angular promedio (deg/s) derivada del ángulo medio
        a_mean = np.nanmean(a, axis=0)
        a_mean = smooth(a_mean, window=9)
        angvel = np.gradient(a_mean, 1.0 / rate)
        feats[f"rom_{lab}"] = rom_deg
        feats[f"avg_angvel_{lab}"] = float(np.mean(np.abs(angvel)))
        # tiempo hasta el pico de velocidad angular (relativo al inicio)
        pk_angvel = int(np.argmax(np.abs(angvel)))
        angvel_peak[lab] = pk_angvel / rate

    # ---- coordinación: timing proximal->distal ----
    # delay = tiempo del pico de vel. angular de una articulación distal
    #         menos el de la proximal (0 = sincrónico; >0 = distal después).
    feats["coord_delay_proximal_distal_s"] = np.nan
    if len(angvel_peak) >= 2 and any(l in angvel_peak for l in chain.get(technique, [])):
        labs_chain = [l for l in chain.get(technique, []) if l in angvel_peak]
        if len(labs_chain) >= 2:
            feats["coord_delay_proximal_distal_s"] = round(
                angvel_peak[labs_chain[-1]] - angvel_peak[labs_chain[0]], 3)

    # ---- COM ----
    i_com = get_time(labels, "CentreOfMass", prefix)
    com_vmax = np.nan
    com_rom = np.nan
    if i_com >= 0:
        com = pts[:3, i_com, :].astype(float) * MM_IN_M
        com_v = np.linalg.norm(np.gradient(com, axis=1) * rate, axis=0)
        com_vmax = float(np.max(com_v[s:e]))
        com_rom = float(np.max(np.linalg.norm(com[:, s:e] - com[:, s:s+1], axis=0)))

    return {
        "duration_s": round(dur_s, 3),
        "time_to_peak_s": round(t2peak, 3),
        "vmax_m_s": round(vmax * MM_IN_M, 3),
        "vmean_m_s": round(vmean * MM_IN_M, 3),
        "amax_m_s2": round(amax * MM_IN_M, 2),
        "amean_m_s2": round(amean * MM_IN_M, 2),
        "displacement_m": round(displacement, 3),
        "path_length_m": round(path_length, 3),
        "rom_m": round(rom, 3),
        "com_vmax_m_s": round(com_vmax, 3) if np.isfinite(com_vmax) else np.nan,
        "com_rom_m": round(com_rom, 3) if np.isfinite(com_rom) else np.nan,
        **feats,
    }


# --------------------------------------------------------------------------- #
# 3. Procesamiento de un archivo -> ejecuciones
# --------------------------------------------------------------------------- #

def process_file(fp: Path) -> tuple[list[dict], list[dict], dict]:
    """Devuelve (filas de ejecuciones aceptadas, eventos, resumen del archivo)."""
    c = load_c3d(fp)
    p = c.parameters
    labels = list(p["POINT"]["LABELS"]["value"])
    rate = float(p["POINT"]["RATE"]["value"][0])
    prefixes = get_prefixes(c)
    athlete = athlete_prefix(fp, prefixes)

    units = {
        "position_unit": "mm", "velocity_unit": "mm/s",
        "acceleration_unit": "mm/s2", "angle_unit": "deg",
        "force_unit": "N", "moment_unit": "Nmm", "power_unit": "W",
    }

    # técnica / condición / trial desde el nombre
    info = parse_name(fp.stem)
    tech = info["technique"]; cond = info["condition"]; trial = info["trial"]

    # rol del atleta del dataset según condición
    role = "defensor" if cond == "E04" else "atacante"
    exec_type = "defensive_response" if cond == "E04" else "technique"

    best = pick_best_signal(fp, athlete, tech)
    if best is None:
        return [], [], {"error": "sin señal candidata"}

    traj, v, rate2 = get_signal(fp, athlete, best["marker"])
    segs, events = segment_repetitions(v, rate)

    # eventos -> filas trazables
    event_rows = []
    for i, ev in events.iterrows():
        event_rows.append({
            "source_file": Path(fp).name,
            "athlete_id": athlete,
            "technique": tech,
            "condition": cond,
            "trial": trial,
            "event_id": int(i),
            "start_frame": int(ev["start_frame"]),
            "peak_frame": int(ev["peak_frame"]),
            "end_frame": int(ev["end_frame"]),
            "start_time": round(ev["start_frame"] / rate, 3),
            "peak_time": round(ev["peak_frame"] / rate, 3),
            "end_time": round(ev["end_frame"] / rate, 3),
            "duration": ev["duration"],
            "signal_used": best["marker"],
            "peak_ratio": ev["peak_ratio"],
            "status": ev["status"],
            "rejection_reason": ev["rejection_reason"],
        })

    rows = []
    if not segs.empty:
        # mapa (start, peak, end frame) -> event_id GLOBAL del registro de eventos.
        # (Corrige trazabilidad FASE 1.8B: antes event_id era el índice posicional
        #  de la aceptada, no el índice global, divergiendo de segmentation_events.csv)
        ev_id_map = {}
        for i, ev in events.iterrows():
            ev_id_map[(int(ev["start_frame"]), int(ev["peak_frame"]),
                       int(ev["end_frame"]))] = int(i)
        for k, seg in segs.iterrows():
            feat = extract_features(fp, athlete, seg, best["marker"], rate, tech)
            key = (int(seg["start_frame"]), int(seg["peak_frame"]),
                   int(seg["end_frame"]))
            rows.append({
                "source_file": Path(fp).name,
                "athlete_id": athlete,
                "technique": tech,
                "technique_name": TECHNIQUE_NAMES.get(tech, tech),
                "condition": cond,
                "condition_name": CONDITION_NAMES.get(cond, cond),
                "trial": trial,
                "role": role,
                "execution_type": exec_type,
                "repetition": int(k) + 1,
                "signal_used": best["marker"],
                "signal_snr": round(best["snr"], 1),
                "sampling_rate_hz": rate,
                **units,
                "segmentation_method": SEGMENTATION_METHOD,
                "segmentation_version": SEGMENTATION_VERSION,
                "event_id": ev_id_map.get(key, int(k)),
                "start_frame": int(seg["start_frame"]),
                "peak_frame": int(seg["peak_frame"]),
                "end_frame": int(seg["end_frame"]),
                "start_time_s": round(seg["start_frame"] / rate, 3),
                "peak_time_s": round(seg["peak_frame"] / rate, 3),
                "end_time_s": round(seg["end_frame"] / rate, 3),
                **feat,
            })

    summary = {"file": Path(fp).name, "athlete": athlete, "technique": tech,
               "condition": cond, "trial": trial, "rate": rate,
               "signal_marker": best["marker"], "signal_snr": round(best["snr"], 1),
               "n_rep_detected": len(rows),
               "n_events": len(events),
               "n_accepted": int((events["status"] == "accepted").sum()) if not events.empty else 0,
               "n_frames": traj.shape[1], "duration_s": round(traj.shape[1] / rate, 3)}
    return rows, event_rows, summary


def plot_segmentation(fp: Path, segs: pd.DataFrame, v_s: np.ndarray, rate: float,
                      marker: str, out_dir: Path) -> str:
    """Gráfica de validación con repeticiones marcadas (start/peak/end)."""
    t = np.arange(len(v_s)) / rate
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(t, v_s, color="#2f6fb3", lw=1.2, label=f"velocidad {marker} [mm/s]")
    for _, r in segs.iterrows():
        for x, lbl, col in [(r["start_frame"], "start", "#39a07b"),
                            (r["peak_frame"], "peak", "#d1495b"),
                            (r["end_frame"], "end", "#6c5b7b")]:
            ax.axvline(x / rate, color=col, ls="--", lw=1)
            ax.annotate(f"{lbl} {x / rate:.2f}s", (x / rate, ax.get_ylim()[1] * 0.95),
                        ha="center", fontsize=7, color=col)
        ax.axvspan(r["start_frame"] / rate, r["end_frame"] / rate, alpha=0.08,
                   color="#2f6fb3")
    ax.set_xlabel("tiempo [s]")
    ax.set_ylabel("velocidad [mm/s]")
    ax.set_title(f"{Path(fp).name} — segmentación de ejecuciones (n={len(segs)})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    name = Path(fp).stem + ".png"
    fig.savefig(out_dir / name, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return name


# --------------------------------------------------------------------------- #
# 4. Calidad por ejecución
# --------------------------------------------------------------------------- #

def quality_check(fp: Path, prefix: str, rows: list[dict], rate: float) -> list[dict]:
    c = load_c3d(fp)
    pts = c["data"]["points"]
    labels = list(c.parameters["POINT"]["LABELS"]["value"])
    out = []
    for row in rows:
        notes = []
        flag = "OK"
        dur = row.get("duration_s", 0)
        if dur < 0.1:
            flag = "WARN"; notes.append("duración < 0.1 s")
        if dur > 2.0:
            flag = "WARN"; notes.append("duración > 2.0 s")
        if row.get("vmax_m_s", 0) > 20:
            flag = "WARN"; notes.append("vmax > 20 m/s (revisar)")
        # valores no finitos en features numéricas relevantes
        for fcol in ["vmax_m_s", "amax_m_s2", "displacement_m", "rom_m"]:
            v = row.get(fcol)
            if v is not None and (not np.isfinite(v) if isinstance(v, float) else False):
                flag = "WARN"; notes.append(f"{fcol} no finito")
        s, e = row["start_frame"], row["end_frame"]
        # NaN del endpoint en la ventana
        i = get_time(labels, row.get("signal_used", row.get("signal_marker")), prefix)
        win = pts[:3, i, s:e]
        nan_frac = float(np.isnan(win).mean() * 100)
        if nan_frac > 5:
            flag = "WARN"; notes.append(f"NaN {nan_frac:.1f}% en endpoint")
        out.append({
            "source_file": Path(fp).name,
            "athlete": prefix, "technique": row["technique"],
            "condition": row["condition"], "trial": row["trial"],
            "repetition": row["repetition"],
            "quality_flag": flag,
            "quality_notes": "; ".join(notes) if notes else "ok",
            "nan_endpoint_pct": round(nan_frac, 2),
        })
    return out


# --------------------------------------------------------------------------- #
# 4b. Control de calidad — resumen y control 100 %
# --------------------------------------------------------------------------- #

def qc_summary(rows: list[dict], quality: list[dict], events: pd.DataFrame) -> dict:
    """Resumen: total/valid/review/invalid. Cubre el 100 % de aceptadas."""
    qmap = { (q["condition"], q["trial"], q["repetition"]): q["quality_flag"]
             for q in quality }
    n_total = len(rows)
    n_valid = sum(1 for r in rows if qmap.get((r["condition"], r["trial"], r["repetition"])) == "OK")
    n_review = sum(1 for r in rows if qmap.get((r["condition"], r["trial"], r["repetition"])) == "REVIEW")
    n_invalid = sum(1 for r in rows if qmap.get((r["condition"], r["trial"], r["repetition"])) == "INVALID")
    n_warn = sum(1 for r in rows if qmap.get((r["condition"], r["trial"], r["repetition"])) == "WARN")
    res = {
        "total_executions": n_total,
        "valid_executions": n_valid,
        "warn_executions": n_warn,
        "review_executions": n_review,
        "invalid_executions": n_invalid,
        "coverage_pct": round(100.0 * n_total / max(n_total, 1), 1),
    }
    if not events.empty:
        res["events_total"] = int(len(events))
        res["events_accepted"] = int((events["status"] == "accepted").sum())
        res["events_rejected"] = int((events["status"] == "rejected").sum())
        res["events_review"] = int((events["status"] == "review").sum())
    return res


# --------------------------------------------------------------------------- #
# 4c. Separación de features por nivel conceptual
# --------------------------------------------------------------------------- #

FEATURE_GROUPS = {
    "metadata": ["source_file", "athlete_id", "technique", "technique_name",
                 "condition", "condition_name", "trial", "role",
                 "execution_type", "repetition", "signal_used", "signal_snr",
                 "sampling_rate_hz", "position_unit", "velocity_unit",
                 "acceleration_unit", "segmentation_method",
                 "segmentation_version", "event_id"],
    "segmentation": ["start_frame", "peak_frame", "end_frame",
                     "start_time_s", "peak_time_s", "end_time_s", "duration_s",
                     "time_to_peak_s"],
    "temporal": ["duration_s", "time_to_peak_s"],
    "kinematic": ["vmax_m_s", "vmean_m_s", "amax_m_s2", "amean_m_s2",
                  "displacement_m", "path_length_m", "rom_m"],
    "joint": ["rom_RShoulderAngles", "avg_angvel_RShoulderAngles",
              "rom_RElbowAngles", "avg_angvel_RElbowAngles",
              "rom_RHipAngles", "avg_angvel_RHipAngles",
              "rom_LKneeAngles", "avg_angvel_LKneeAngles",
              "rom_RKneeAngles", "avg_angvel_RKneeAngles",
              "rom_RAnkleAngles", "avg_angvel_RAnkleAngles"],
    "com": ["com_vmax_m_s", "com_rom_m"],
    "coordination": ["coord_delay_proximal_distal_s"],
}


def write_feature_sample(df_exec: pd.DataFrame, out_dir: Path | None = None) -> pd.DataFrame:
    """Escribe execution_features_sample.csv con columnas agrupadas por nivel."""
    out_dir = out_dir or OUT_ATHLETE
    cols = []
    for group in ["metadata", "segmentation", "kinematic", "joint", "com",
                  "coordination"]:
        for c in FEATURE_GROUPS.get(group, []):
            if c in df_exec.columns and c not in cols:
                cols.append(c)
    # añadir cualquier columna restante no categorizada al final
    for c in df_exec.columns:
        if c not in cols:
            cols.append(c)
    df = df_exec[cols]
    df.to_csv(out_dir / "execution_features_sample.csv", index=False)
    return df


# --------------------------------------------------------------------------- #
# 5. DTW — repeticiones T01 vs T02
# --------------------------------------------------------------------------- #

def dtw_alignment_path(a, b, radius=-1):
    """Devuelve el camino de alineación DTW (listas de índices)."""
    a = np.asarray(a, float); b = np.asarray(b, float)
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
    # backtrack
    i, j = n, m
    path = [(i - 1, j - 1)]
    while i > 1 or j > 1:
        cand = []
        if i > 1: cand.append((i - 1, j, D[i - 1, j]))
        if j > 1: cand.append((i, j - 1, D[i, j - 1]))
        if i > 1 and j > 1: cand.append((i - 1, j - 1, D[i - 1, j - 1]))
        i, j, _ = min(cand, key=lambda x: x[2])
        path.append((i - 1, j - 1))
    path.reverse()
    return np.array(path)


def validate_dtw():
    print("\n===== VALIDACIÓN DTW (repeticiones T01 vs T02) =====")
    fn_pairs = [
        ("S02-E01", "T01", "T02"),
        ("S04-E01", "T01", "T02"),
        ("S04-E02", "T01", "T02"),
    ]
    results = []
    for base, t1, t2 in fn_pairs:
        f1 = find_files(f"{base}-{t1}")
        f2 = find_files(f"{base}-{t2}")
        if not f1 or not f2:
            print(f"  skip {base}: faltan trials")
            continue
        f1, f2 = f1[0], f2[0]
        c1 = load_c3d(f1); c2 = load_c3d(f2)
        l1 = list(c1.parameters["POINT"]["LABELS"]["value"])
        rate1 = float(c1.parameters["POINT"]["RATE"]["value"][0])
        pref1 = athlete_prefix(f1, get_prefixes(c1))
        best1 = pick_best_signal(f1, pref1, base.split("-")[0])

        _, v1, _ = get_signal(f1, pref1, best1["marker"])
        segs1, vs1 = segment_repetitions(v1, rate1, return_events=False)

        v2, rate2, pref2 = None, rate1, pref1
        # misma técnica, elegir mejor señal de f2
        best2 = pick_best_signal(f2, pref2, base.split("-")[0])
        _, v2, _ = get_signal(f2, pref2, best2["marker"])
        segs2, vs2 = segment_repetitions(v2, rate2, return_events=False)

        n = min(len(segs1), len(segs2))
        for k in range(n):
            s1, e1 = segs1.iloc[k]["start_frame"], segs1.iloc[k]["end_frame"]
            s2, e2 = segs2.iloc[k]["start_frame"], segs2.iloc[k]["end_frame"]
            a = normalize_(vs1[s1:e1])
            b = normalize_(vs2[s2:e2])
            dtw = dtw_dist(a, b)
            m = min(len(a), len(b))
            euc = float(np.sqrt(np.mean((normalize_(a[:m]) - normalize_(b[:m])) ** 2)))
            results.append({
                "base": base, "repetition": k + 1,
                "dtw_dist": round(dtw, 3), "euclid_dist": round(euc, 3),
                "similarity": round(1 / (1 + dtw), 3),
                "len_a": len(a), "len_b": len(b),
                "a_start_s": round(s1 / rate1, 2), "b_start_s": round(s2 / rate2, 2),
            })

            # figura de alineamiento (una por par en S02-E01)
            if base == "S02-E01" and k == 0:
                fig, axes = plt.subplots(3, 1, figsize=(11, 8))
                axes[0].plot(a, color="#2f6fb3", lw=1, label="T01")
                axes[0].set_title(f"{base}: ejecución {k+1} — DTW dist={dtw:.2f}")
                axes[0].legend(); axes[0].set_ylabel("vel. norm.")
                axes[1].plot(b, color="#e08a2e", lw=1, label="T02")
                axes[1].legend(); axes[1].set_ylabel("vel. norm.")
                path = dtw_alignment_path(a, b)
                axes[2].plot(path[:, 0], path[:, 1], color="#6c5b7b", lw=0.5)
                axes[2].set_xlabel("T01 (frames)"); axes[2].set_ylabel("T02 (frames)")
                axes[2].set_title("Camino de alineamiento DTW")
                fig.tight_layout()
                fig.savefig(DTW_DIR / f"dtw_{base}_rep{k+1}.png", dpi=130)
                plt.close(fig)

    df = pd.DataFrame(results)
    df.to_csv(OUT / "dtw_validation_results.csv", index=False)
    print(df.round(3).to_string(index=False))
    return df


# --------------------------------------------------------------------------- #
# 6. Comparación E01 vs E02 (S04)
# --------------------------------------------------------------------------- #

def compare_air_shield():
    print("\n===== COMPARACIÓN E01 (aire) vs E02 (escudo) — S04 =====")
    # usar primeras repeticiones de T01 de cada condición
    recs = []
    for cond, col in [("E01", "#2f6fb3"), ("E02", "#e08a2e")]:
        fp = find_files(f"S04-{cond}-T01")[0]
        c = load_c3d(fp)
        rate = float(c.parameters["POINT"]["RATE"]["value"][0])
        pref = athlete_prefix(fp, get_prefixes(c))
        best = pick_best_signal(fp, pref, "S04")
        _, v, _ = get_signal(fp, pref, best["marker"])
        segs, vs = segment_repetitions(v, rate, return_events=False)
        for _, seg in segs.iterrows():
            feat = extract_features(fp, pref, seg, best["marker"], rate, "S04")
            recs.append({"condition": cond, "repetition": int(seg.name) + 1,
                         **feat})
    df = pd.DataFrame(recs)
    df.to_csv(OUT / "condition_s04_air_shield.csv", index=False)
    cols = ["duration_s", "time_to_peak_s", "vmax_m_s", "vmean_m_s", "amax_m_s2",
            "displacement_m", "path_length_m", "rom_m",
            "rom_RHipAngles", "rom_RKneeAngles", "rom_RAnkleAngles"]
    print(df.groupby("condition")[cols].mean().round(3).to_string())

    # perfil de velocidad de las primeras repeticiones
    fig, ax = plt.subplots(figsize=(11, 5))
    for cond, color in [("E01", "#2f6fb3"), ("E02", "#e08a2e")]:
        fp = find_files(f"S04-{cond}-T01")[0]
        c = load_c3d(fp)
        rate = float(c.parameters["POINT"]["RATE"]["value"][0])
        pref = athlete_prefix(fp, get_prefixes(c))
        best = pick_best_signal(fp, pref, "S04")
        _, v, _ = get_signal(fp, pref, best["marker"])
        segs, vs = segment_repetitions(v, rate, return_events=False)
        if segs.empty:
            continue
        r0 = segs.iloc[0]
        s, e = r0["start_frame"], r0["end_frame"]
        t = np.arange(len(vs[s:e])) / rate
        ax.plot(t, vs[s:e] * MM_IN_M, color=color, lw=1.3, label=f"{cond}")
    ax.set_xlabel("tiempo desde start [s]")
    ax.set_ylabel("velocidad RTOE [m/s]")
    ax.set_title("S04 Mawashi-Geri jodan — E01 (aire) vs E02 (escudo), repetición 1")
    ax.legend()
    fig.tight_layout()
    fig.savefig(CMP_DIR / "s04_air_vs_shield_speed.png", dpi=130)
    plt.close(fig)
    return df


# --------------------------------------------------------------------------- #
# 7. E04 — análisis defensor (B0367) vs atacante (B0368)
# --------------------------------------------------------------------------- #

def analyze_e04():
    print("\n===== ANÁLISIS E04 (defensor vs atacante) — S04 =====")
    fp = find_files("S04-E04-T01")[0]
    c = load_c3d(fp)
    pts = c["data"]["points"]
    labels = list(c.parameters["POINT"]["LABELS"]["value"])
    rate = float(c.parameters["POINT"]["RATE"]["value"][0])
    prefixes = get_prefixes(c)  # ['B0367','B0368']
    pref_def = next(x for x in prefixes if x == "B0367")
    pref_atk = next(x for x in prefixes if x == "B0368")

    # marcadores pélvicos para centro de cada sujeto (metros)
    def center(sub):
        ids = [get_time(labels, m, sub) for m in ["LASI", "RASI", "LPSI", "RPSI"]]
        ids = [i for i in ids if i >= 0]
        arr = np.nanmean(pts[:3, ids, :], axis=1)  # (3, frames) mm
        return arr * MM_IN_M  # m

    c_def = center(pref_def)
    c_atk = center(pref_atk)
    # distancia y velocidad relativa (metros)
    rel = c_atk - c_def  # vector del defensor al atacante
    dist = np.linalg.norm(rel, axis=0)
    vel_rel = np.linalg.norm(np.gradient(rel, axis=1) * rate, axis=0)

    # pie del atacante para ver el impacto
    i_toe_atk = get_time(labels, "RTOE", pref_atk)
    toe_atk = pts[:3, i_toe_atk, :].astype(float) * MM_IN_M
    dist_toe_def = np.linalg.norm(toe_atk - c_def, axis=0)  # m

    # segmentación del atacante (quien ejecuta la patada)
    best = pick_best_signal(fp, pref_atk, "S04")
    _, v_atk, _ = get_signal(fp, pref_atk, best["marker"])
    segs, vs = segment_repetitions(v_atk, rate, return_events=False)

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    t = np.arange(len(v_atk)) / rate
    axes[0].plot(t, v_atk * MM_IN_M, color="#d1495b", lw=1)
    for _, r in segs.iterrows():
        axes[0].axvline(r["start_frame"] / rate, color="g", ls="--", lw=.7)
        axes[0].axvline(r["peak_frame"] / rate, color="r", ls="--", lw=.7)
        axes[0].axvline(r["end_frame"] / rate, color="purple", ls="--", lw=.7)
    axes[0].set_ylabel("vel atacante RTOE [m/s]")
    axes[0].set_title("S04-E04-T01: velocidad del atacante (B0368) con ejecuciones")
    axes[1].plot(t, c_def[2], color="#2f6fb3", lw=1, label="defensor Z")
    axes[1].plot(t, c_atk[2], color="#e08a2e", lw=1, label="atacante Z")
    axes[1].set_ylabel("altura centro pélvico [m]")
    axes[1].legend(fontsize=8)
    axes[2].plot(t, dist, color="#39a07b", lw=1)
    axes[2].set_ylabel("distancia entre sujetos [m]")
    axes[3].plot(t, vel_rel, color="#6c5b7b", lw=1)
    axes[3].set_ylabel("velocidad relativa [m/s]")
    axes[3].set_xlabel("tiempo [s]")
    fig.tight_layout()
    fig.savefig(SEG_DIR / "e04_defender_vs_attacker.png", dpi=130)
    plt.close(fig)

    # tabla resumen de distancias por ejecución
    print(f"Sujetos: defensor={pref_def} ({len([l for l in labels if l.startswith(pref_def+':')])} pts), "
          f"atacante={pref_atk} ({len([l for l in labels if l.startswith(pref_atk+':')])} pts)")
    print(f"Variables B0367: {len([l for l in labels if l.startswith('B0367:') and any(k in l for k in ['Angles','Power','Force','Moment','COM'])])} derivadas")
    print(f"Variables B0368: {len([l for l in labels if l.startswith('B0368:') and any(k in l for k in ['Angles','Power','Force','Moment','COM'])])} derivadas")
    print(f"Ejecuciones detectadas (atacante): {len(segs)}")
    dtw_meta = []
    for rep_k, r in segs.iterrows():
        s, p, e = r["start_frame"], r["peak_frame"], r["end_frame"]
        dtw_meta.append({
            "repetition": int(rep_k) + 1,
            "min_dist_m": round(float(np.min(dist[s:e])), 3),
            "distance_at_peak_m": round(float(dist[p]), 3),
            "max_rel_vel_m_s": round(float(np.max(vel_rel[s:e])), 3),
            "toe_atk_min_def_m": round(float(np.min(dist_toe_def[s:e])), 3),
        })
    df = pd.DataFrame(dtw_meta)
    df.to_csv(OUT / "e04_defender_attacker_distances.csv", index=False)
    print(df.round(3).to_string(index=False))
    return df


# --------------------------------------------------------------------------- #
# 8. Verificación de unidades/frecuencia directamente desde los C3D
# --------------------------------------------------------------------------- #

def verify_units_from_c3d(athlete_id: str | None = None,
                          out_dir: Path | None = None) -> pd.DataFrame:
    """Lee los parámetros de unidades y frecuencia de C3D representativos.

    Para el baseline usa los tags históricos; para otros atletas muestrea un
    conjunto por técnica (E01-T01) para confirmar homogeneidad de unidades/Hz.
    """
    athlete_id = athlete_id or ACTIVE_ATHLETE
    out_dir = out_dir or OUT_ATHLETE
    if athlete_id == "B0367":
        tags = ["S01-E01-T01", "S02-E01-T01", "S04-E01-T01",
                "S04-E02-T01", "S04-E04-T01"]
    else:
        tags = [f"S{tech:02d}-E01-T01" for tech in range(1, 6)]
    rows = []
    for tag in tags:
        fp = find_files(tag, athlete_id)
        fp = fp[0] if fp else None
        if fp is None:
            continue
        c = load_c3d(fp)
        p = c.parameters
        g = "POINT"
        rows.append({
            "athlete_id": athlete_id,
            "source_file": Path(fp).name,
            "sampling_rate_hz": float(p[g]["RATE"]["value"][0]) if "RATE" in p[g] else np.nan,
            "position_unit": str(p[g]["UNITS"]["value"][0]) if "UNITS" in p[g] else "",
            "angle_unit": str(p[g]["ANGLE_UNITS"]["value"][0]) if "ANGLE_UNITS" in p[g] else "",
            "force_unit": str(p[g]["FORCE_UNITS"]["value"][0]) if "FORCE_UNITS" in p[g] else "",
            "moment_unit": str(p[g]["MOMENT_UNITS"]["value"][0]) if "MOMENT_UNITS" in p[g] else "",
            "power_unit": str(p[g]["POWER_UNITS"]["value"][0]) if "POWER_UNITS" in p[g] else "",
            "analog_used": int(p["ANALOG"]["USED"]["value"][0]) if "ANALOG" in p and "USED" in p["ANALOG"] else 0,
            "fp_used": int(p["FORCE_PLATFORM"]["USED"]["value"][0]) if "FORCE_PLATFORM" in p and "USED" in p["FORCE_PLATFORM"] else 0,
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "units_verified.csv", index=False)
    return df


# --------------------------------------------------------------------------- #
# MAIN
# --------------------------------------------------------------------------- #

def main(athlete_id: str = "B0367"):
    """Pipeline de segmentación configurado por atleta.

    Para B0367 reproduce el baseline histórico (salidas en output/ raíz).
    Para otros atletas escribe en output/<athlete_id>/ y ejecuta solo el flujo
    de segmentación (las validaciones DTW/condición/E04 son específicas del
    baseline y también requieren config propio).
    """
    set_active_athlete(athlete_id)
    print(f"Proyecto raíz: {ROOT}")
    print(f"Atleta:        {ACTIVE_ATHLETE}")
    print(f"Config:        {CONFIG_PATH}")
    print(f"Config atleta: {ATHLETE_CFG_DIR / f'{ACTIVE_ATHLETE}.yaml'}")
    print(f"Método:        {SEGMENTATION_METHOD} v{SEGMENTATION_VERSION}")
    print(f"Salidas:       {OUT_ATHLETE}")

    # lista controlada de trials (por atleta: baseline usa la histórica)
    if ACTIVE_ATHLETE == "B0367":
        selected = ["S01-E01-T01", "S02-E01-T01", "S03-E01-T01",
                    "S04-E01-T01", "S04-E02-T01", "S04-E04-T01",
                    "S05-E01-T01", "S05-E02-T01"]
    else:
        selected = [f"S{tech:02d}-E01-T01" for tech in range(1, 6)]

    all_rows = []
    all_events = []
    all_summaries = []
    all_quality = []

    for tag in selected:
        fp = find_files(tag, ACTIVE_ATHLETE)
        fp = fp[0] if fp else None
        if fp is None:
            print(f"[falta] {tag} no presente — documentar")
            continue
        print(f"\n--- Procesando {Path(fp).name} ---")
        c = load_c3d(fp)
        rate = float(c.parameters["POINT"]["RATE"]["value"][0])
        prefixes = get_prefixes(c)
        athlete = athlete_prefix(fp, prefixes)

        best = pick_best_signal(fp, athlete, tag.split("-")[0])
        print(f"  señal seleccionada: {best['marker']}  SNR={best['snr']:.1f} "
              f"adecuada={best['suitable']}")

        rows, events, summary = process_file(fp)
        print(f"  repeticiones detectadas: {summary['n_rep_detected']}  "
              f"(eventos={summary['n_events']}, aceptados={summary['n_accepted']})")

        # gráfica de validación con la señal suavizada
        seg_dir = OUT_ATHLETE / "phase_validation"
        seg_dir.mkdir(parents=True, exist_ok=True)
        _, v, _ = get_signal(fp, athlete, best["marker"])
        segs, v_s = segment_repetitions(v, rate, return_events=False)
        plot_segmentation(fp, segs, v_s, rate, best["marker"], seg_dir)
        quality = quality_check(fp, athlete, rows, rate)

        all_rows += rows
        all_events += events
        all_summaries.append(summary)
        all_quality += quality

    # tablas (por atleta)
    df_exec = pd.DataFrame(all_rows)
    df_exec.to_csv(OUT_ATHLETE / "executions_sample.csv", index=False)
    pd.DataFrame(all_summaries).to_csv(OUT_ATHLETE / "execution_file_summary.csv", index=False)
    pd.DataFrame(all_quality).to_csv(OUT_ATHLETE / "execution_quality.csv", index=False)
    if all_events:
        df_events = pd.DataFrame(all_events)
        df_events.to_csv(OUT_ATHLETE / "segmentation_events.csv", index=False)
    else:
        df_events = pd.DataFrame()
    # features separadas por nivel conceptual + resumen QC
    write_feature_sample(df_exec, OUT_ATHLETE)
    qc = qc_summary(all_rows, all_quality, df_events)
    pd.DataFrame([qc]).to_csv(OUT_ATHLETE / "qc_summary.csv", index=False)
    print(f"\n[guardado] {OUT_ATHLETE/'executions_sample.csv'} ({len(df_exec)} filas)")
    print(f"[guardado] {OUT_ATHLETE/'execution_file_summary.csv'}")
    print(f"[guardado] {OUT_ATHLETE/'execution_quality.csv'} ({len(all_quality)} filas)")
    print(f"[guardado] {OUT_ATHLETE/'execution_features_sample.csv'}")
    print(f"[guardado] {OUT_ATHLETE/'qc_summary.csv'}")
    if not df_events.empty:
        print(f"[guardado] {OUT_ATHLETE/'segmentation_events.csv'} ({len(df_events)} filas)")

    # verificación de unidades/frecuencia desde los C3D
    unit_df = verify_units_from_c3d(ACTIVE_ATHLETE, OUT_ATHLETE)
    print(f"\n[guardado] {OUT_ATHLETE/'units_verified.csv'}")
    print("\nUnidades y frecuencia verificadas DESDE los C3D:")
    print(unit_df.to_string(index=False))
    if not unit_df.empty:
        print("\nDiscrepancia documentada: paper=250 Hz, C3D=" +
              f"{float(unit_df['sampling_rate_hz'].iloc[0])} Hz")

    # resumen de eventos
    if not df_events.empty:
        print("\nResumen de eventos por status:")
        print(df_events.groupby("status").size().to_string())
        print("\nRazones de rechazo/review:")
        sub = df_events[df_events["status"] != "accepted"]
        if not sub.empty:
            print(sub.groupby(["status", "rejection_reason"]).size().to_string())

    # validaciones específicas del baseline (requieren config/trials propios)
    if ACTIVE_ATHLETE == "B0367":
        res_dtw = validate_dtw()
        res_cmp = compare_air_shield()
        res_e04 = analyze_e04()


if __name__ == "__main__":
    import sys as _sys
    _ath = "B0367"
    if len(_sys.argv) > 1:
        _ath = _sys.argv[1]
    main(_ath)