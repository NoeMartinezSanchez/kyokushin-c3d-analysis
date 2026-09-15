#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
03_signal_validation.py
=======================
FASE 1.6.1 — VALIDACIÓN DE SEÑALES DE SEGMENTACIÓN PARA S03 y S05

Determina experimentalmente la mejor señal (marker) para representar la
ejecución de:
  S03 = Mawashi-Geri gedan
  S05 = Ushiro-Mawashi-Geri
para B0367.

NO entrena, NO modifica C3D, NO procesa otros atletas. Solo evalúa candidatos
y produce figuras + una tabla multicriterio reproducible.

Salidas:
  output/signal_validation/signal_validation_S03_*.png
  output/signal_validation/signal_validation_S05_*.png
  output/signal_validation/signal_validation_candidates.csv

Reutiliza funciones de 01 (get_time, smooth, etc.) y 02 (segment_repetitions).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

_SPEC = importlib.util.spec_from_file_location(
    "dataset_exploration_module",
    str(Path(__file__).resolve().parent / "01_dataset_exploration.py"),
)
_X01 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_X01)

_SPEC2 = importlib.util.spec_from_file_location(
    "execution_segmentation_module",
    str(Path(__file__).resolve().parent / "02_execution_segmentation.py"),
)
_X02 = importlib.util.module_from_spec(_SPEC2)
_SPEC2.loader.exec_module(_X02)

ROOT = _X01.ROOT
OUT = ROOT / "output"
SIG_DIR = OUT / "signal_validation"
SIG_DIR.mkdir(parents=True, exist_ok=True)

CANDIDATES = ["RTOE", "RANK", "RHEE", "RKNE", "RTHI", "RHJC"]
MIN_DIST_S = 0.4
PEAK_THR_FRAC = 0.30


def evaluate_candidate(fp: Path, prefix: str, marker: str, technique: str) -> dict:
    """Métricas multicriterio de un candidato en un archivo."""
    c = _X01.load_c3d(fp)
    rate = float(c.parameters["POINT"]["RATE"]["value"][0])
    labels = list(c.parameters["POINT"]["LABELS"]["value"])
    i = _X01.get_time(labels, marker, prefix)
    if i < 0:
        return None
    pts = c["data"]["points"]
    traj = pts[:3, i, :].astype(float)
    v = np.linalg.norm(np.gradient(traj, axis=1) * rate, axis=0)
    vs = _X01.smooth(v, window=15)
    baseline = float(np.median(vs[: int(rate)]))
    vmax = float(np.max(vs))
    vmean = float(np.mean(vs))
    if vmax <= 0:
        return None
    amax = float(np.max(np.abs(np.gradient(vs, 1.0 / rate))))
    snr = vmax / (baseline + 1.0)

    # picos candidatos (por encima del 30% del rango)
    thr = baseline + PEAK_THR_FRAC * (vmax - baseline)
    pk, _ = find_peaks(vs, height=thr, distance=int(MIN_DIST_S * rate))
    n_peaks = len(pk)

    # segmentación real con el metodo del pipeline
    segs, events = _X02.segment_repetitions(v, rate)
    n_accepted = len(segs)
    n_rejected = int((events["status"] == "rejected").sum()) if not events.empty else 0
    n_review = int((events["status"] == "review").sum()) if not events.empty else 0

    # amplitud de movimiento (ROM del endpoint en la ventana completa)
    rom = float(np.max(np.linalg.norm(traj - traj[:, :1], axis=0))) * 1e-3  # m

    # separación temporal entre picos (mediana), en segundos
    if n_peaks >= 2:
        sep = float(np.median(np.diff(pk) / rate))
    else:
        sep = np.nan

    return {
        "technique": technique,
        "marker": marker,
        "condition": Path(fp).name.split("-E")[1].split("-")[0],
        "trial": Path(fp).name.split("-T")[1].replace(".c3d", ""),
        "source_file": Path(fp).name,
        "n_frames": int(traj.shape[1]),
        "sampling_rate_hz": rate,
        "activity_range_mm": round(vmax, 1),
        "vmax_mm_s": round(vmax, 1),
        "vmean_mm_s": round(vmean, 1),
        "amax_mm_s2": round(amax, 1),
        "snr": round(snr, 1),
        "baseline_mm_s": round(baseline, 1),
        "rom_m": round(rom, 3),
        "n_peaks_candidate": n_peaks,
        "median_separation_s": round(sep, 3) if np.isfinite(sep) else np.nan,
        "n_accepted": n_accepted,
        "n_rejected": n_rejected,
        "n_review": n_review,
    }


def plot_candidates(technique: str, trial_files: list[Path], marker_list: list[str],
                    out_name: str):
    """Figura comparativa de candidatos para una técnica/trial."""
    n_mk = len(marker_list)
    fig, axes = plt.subplots(n_mk, 1, figsize=(13, 3.0 * n_mk), sharex=True)
    if n_mk == 1:
        axes = [axes]
    for ax, mk in zip(axes, marker_list):
        fp = trial_files[0]
        c = _X01.load_c3d(fp)
        rate = float(c.parameters["POINT"]["RATE"]["value"][0])
        prefix = _X01.athlete_prefix(fp, _X01.get_prefixes(c))
        labels = list(c.parameters["POINT"]["LABELS"]["value"])
        i = _X01.get_time(labels, mk, prefix)
        if i < 0:
            ax.set_title(f"{mk}: MISSING"); ax.set_ylim(0, 1); continue
        traj = c["data"]["points"][:3, i, :].astype(float)
        v = np.linalg.norm(np.gradient(traj, axis=1) * rate, axis=0)
        vs = _X01.smooth(v, window=15)
        t = np.arange(len(vs)) / rate
        baseline = np.median(vs[: int(rate)])
        vmax = np.max(vs)
        thr = baseline + PEAK_THR_FRAC * (vmax - baseline)
        pk, _ = find_peaks(vs, height=thr, distance=int(MIN_DIST_S * rate))
        segs, _ = _X02.segment_repetitions(v, rate)
        ax.plot(t, vs * 1e-3, color="#2f6fb3", lw=1.1)
        ax.axhline(thr * 1e-3, color="grey", ls=":", lw=1)
        ax.plot(pk / rate, vs[pk] * 1e-3, "o", color="#d1495b", ms=3)
        for _, r in segs.iterrows():
            ax.axvspan(r["start_frame"] / rate, r["end_frame"] / rate,
                       color="#39a07b", alpha=0.10)
        ax.set_ylabel(f"{mk}\n[m/s]")
        ax.set_title(f"{mk}: vmax={vmax*1e-3:.1f} m/s  SNR={vmax/(baseline+1):.0f}  "
                     f"picos={len(pk)}  aceptadas={len(segs)}", fontsize=9)
    axes[-1].set_xlabel("tiempo [s]")
    fig.suptitle(f"{out_name} — comparación de señales candidatas", fontsize=12)
    fig.tight_layout()
    fig.savefig(SIG_DIR / f"{out_name}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    print(f"Salidas -> {SIG_DIR}")
    all_rows = []

    for technique, trials in [("S03", ["E01-T01", "E01-T02", "E02-T01", "E02-T02"]),
                              ("S05", ["E01-T01", "E01-T02", "E02-T01"])]:
        print(f"\n===== {technique} =====")
        for tag in trials:
            fp = _X01.find_files(f"{technique}-{tag}")
            if not fp:
                print(f"  [warn] {technique}-{tag} no presente")
                continue
            fp = fp[0]
            c = _X01.load_c3d(fp)
            prefix = _X01.athlete_prefix(fp, _X01.get_prefixes(c))
            print(f"  {tag}: {fp.name}")
            for mk in CANDIDATES:
                row = evaluate_candidate(fp, prefix, mk, technique)
                if row:
                    all_rows.append(row)

    df = pd.DataFrame(all_rows)
    df.to_csv(SIG_DIR / "signal_validation_candidates.csv", index=False)
    print(f"\n[guardado] {SIG_DIR/'signal_validation_candidates.csv'} ({len(df)} filas)")

    # figuras comparativas (primer trial E01 de cada técnica)
    for technique, tag, markers in [
        ("S03", "E01-T01", ["RTOE", "RANK", "RHEE"]),
        ("S03", "E02-T01", ["RTOE", "RANK", "RHEE"]),
        ("S05", "E01-T01", ["RTOE", "RANK", "RHEE"]),
        ("S05", "E02-T01", ["RTOE", "RANK", "RHEE"]),
    ]:
        fp = _X01.find_files(f"{technique}-{tag}")
        if fp:
            plot_candidates(technique, [fp[0]], markers,
                            f"signal_validation_{technique}_{tag}")

    # tabla de decisión resumida
    print("\n===== RESUMEN POR CANDIDATO (media por técnica, todos los trials) =====")
    g = df.groupby(["technique", "marker"]).agg(
        n=("source_file", "count"),
        vmax_mm_s=("vmax_mm_s", "mean"),
        snr=("snr", "mean"),
        baseline=("baseline_mm_s", "mean"),
        n_peaks=("n_peaks_candidate", "mean"),
        n_accepted=("n_accepted", "mean"),
        n_rejected=("n_rejected", "mean"),
        n_review=("n_review", "mean"),
        sep=("median_separation_s", "mean"),
    ).round(2)
    print(g.to_string())


if __name__ == "__main__":
    main()