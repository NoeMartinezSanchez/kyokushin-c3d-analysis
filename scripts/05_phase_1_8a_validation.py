#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
05_phase_1_8a_validation.py
===========================
FASE 1.8A — VALIDACIÓN DE LA CONFIGURACIÓN POR ATLETA

Genera las salidas de validación de la fase:
  output/phase_1_8a/b0367_regression.csv          (baseline no roto)
  output/phase_1_8a/s01_b0377_threshold_validation.csv (investigación S01)
  output/phase_1_8a/s04_b0377_validation.csv      (LTOE vs RTOE S04)
  output/phase_1_8a/b0377_config_validation.csv   (prueba controlada B0377)
  output/phase_1_8a/figures/*.png

Reutiliza el motor 02 (ya parametrizado por atleta).
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

_SPEC1 = importlib.util.spec_from_file_location('x01', str(Path(__file__).resolve().parent / "01_dataset_exploration.py"))
_X01 = importlib.util.module_from_spec(_SPEC1); _SPEC1.loader.exec_module(_X01)
_SPEC2 = importlib.util.spec_from_file_location('x02', str(Path(__file__).resolve().parent / "02_execution_segmentation.py"))
_X02 = importlib.util.module_from_spec(_SPEC2); _SPEC2.loader.exec_module(_X02)

ROOT = _X01.ROOT
P1_8A = ROOT / "output" / "phase_1_8a"
FIG = P1_8A / "figures"
P1_8A.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)


def b0367_regression():
    """Compara el baseline B0367 regenerado contra el histórico esperado (26/26)."""
    # regenerar el baseline con el motor parametrizado
    _X02.set_active_athlete("B0367")
    _X02.main("B0367")
    df = pd.read_csv(_X02.OUT_ATHLETE / "executions_sample.csv")
    qc = pd.read_csv(_X02.OUT_ATHLETE / "qc_summary.csv").iloc[0]
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "athlete": r["athlete_id"], "technique": r["technique"],
            "condition": r["condition"], "trial": r["trial"],
            "repetition": r["repetition"],
            "start_time_s": r["start_time_s"], "peak_time_s": r["peak_time_s"],
            "end_time_s": r["end_time_s"],
            "duration_s": r["duration_s"],
            "signal": r["signal_used"],
            "status": "accepted",
        })
    reg = pd.DataFrame(rows)
    reg.to_csv(P1_8A / "b0367_regression.csv", index=False)
    expected = 26
    actual = len(df)
    print(f"\n[baseline] b0367_regression: {actual} ejecuciones (esperado {expected}) -> "
          f"{'OK' if actual == expected else 'DESVIACIÓN'}")
    print(f"[baseline] QC: total={qc['total_executions']} valid={qc['valid_executions']} "
          f"coverage={qc['coverage_pct']}%")
    return reg


def s01_threshold_validation():
    """Investiga el umbral de S01 en B0377 (baseline alto) vs B0367."""
    rows = []
    for athlete, trials in [("B0377", ["E01-T01", "E01-T02"]),
                            ("B0367", ["E01-T01"])]:
        for tr in trials:
            tag = f"S01-{tr}"
            fp = _X02.find_files(tag, athlete)
            if not fp:
                continue
            fp = fp[0]
            c = _X01.load_c3d(fp)
            rate = float(c.parameters["POINT"]["RATE"]["value"][0])
            pref = _X01.athlete_prefix(fp, _X01.get_prefixes(c))
            best = _X02.pick_best_signal(fp, pref, "S01")
            _, v, _ = _X02.get_signal(fp, pref, best["marker"])
            vs = _X01.smooth(v, 15)
            first_s = vs[: int(rate)]
            med_all = float(np.median(vs))
            mad = float(np.median(np.abs(vs - med_all)))
            thr = med_all + 3 * mad
            pk, _ = find_peaks(vs, height=thr, distance=int(rate * 0.5))
            segs, events = _X02.segment_repetitions(v, rate)
            rows.append({
                "athlete": athlete, "tag": tag,
                "signal": best["marker"],
                "vmax_mm_s": round(float(np.max(vs)), 1),
                "baseline_1st_sec_median": round(float(np.median(first_s)), 1),
                "signal_p50": round(med_all, 1),
                "signal_MAD": round(mad, 1),
                "threshold_med_plus_3MAD": round(thr, 1),
                "n_peaks_robust": len(pk),
                "n_accepted_pipeline": len(segs),
                "n_review_pipeline": int((events["status"] == "review").sum()) if not events.empty else 0,
                "conclusion": "REQUIERE_VALIDACION" if athlete == "B0377" else "baseline_ok",
            })
    df = pd.DataFrame(rows)
    df.to_csv(P1_8A / "s01_b0377_threshold_validation.csv", index=False)
    print("\n[S01] resultado: ")
    print(df[["athlete", "tag", "baseline_1st_sec_median", "signal_p50",
              "signal_MAD", "threshold_med_plus_3MAD", "n_peaks_robust",
              "n_accepted_pipeline", "conclusion"]].to_string(index=False))
    # figura comparativa
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=False)
    for ax, (ath, tr) in zip(axes, [("B0377", "E01-T01"), ("B0367", "E01-T01")]):
        fp = _X02.find_files(f"S01-{tr}", ath)
        fp = fp[0] if fp else None
        if fp is None:
            continue
        c = _X01.load_c3d(fp)
        rate = float(c.parameters["POINT"]["RATE"]["value"][0])
        pref = _X01.athlete_prefix(fp, _X01.get_prefixes(c))
        best = _X02.pick_best_signal(fp, pref, "S01")
        _, v, _ = _X02.get_signal(fp, pref, best["marker"])
        vs = _X01.smooth(v, 15)
        t = np.arange(len(vs)) / rate
        ax.plot(t, vs * 1e-3, lw=1)
        med_all = np.median(vs); mad = np.median(np.abs(vs - med_all))
        ax.axhline((med_all + 3 * mad) * 1e-3, color="r", ls="--", lw=1, label="med+3MAD")
        ax.set_title(f"{ath} S01 ({tr}) RFIN [m/s]")
        ax.legend(fontsize=8)
    axes[-1].set_xlabel("tiempo [s]")
    fig.tight_layout()
    fig.savefig(FIG / "s01_threshold_comparison.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("[S01] figura s01_threshold_comparison.png")
    return df


def s04_validation():
    """Compara RTOE vs LTOE en S04 de B0377 (lateralidad izquierda)."""
    rows = []
    fp = _X02.find_files("S04-E01-T01", "B0377")[0]
    c = _X01.load_c3d(fp)
    rate = float(c.parameters["POINT"]["RATE"]["value"][0])
    pref = _X01.athlete_prefix(fp, _X01.get_prefixes(c))
    for mk in ["RTOE", "LTOE", "RANK", "LANK", "RHEE", "LHEE"]:
        i = _X01.get_time(list(c.parameters["POINT"]["LABELS"]["value"]), mk, pref)
        if i < 0:
            rows.append({"marker": mk, "available": "NO"})
            continue
        traj = c["data"]["points"][:3, i, :].astype(float)
        v = np.linalg.norm(np.gradient(traj, axis=1) * rate, axis=0)
        segs, events = _X02.segment_repetitions(v, rate)
        vs = _X01.smooth(v, 15)
        bl = float(np.median(vs[: int(rate)]))
        rows.append({
            "marker": mk, "available": "YES",
            "baseline_mm_s": round(bl, 1),
            "vmax_mm_s": round(float(np.max(vs)), 1),
            "snr": round(float(np.max(vs) / (bl + 1)), 1),
            "n_accepted": len(segs),
            "n_rejected": int((events["status"] == "rejected").sum()) if not events.empty else 0,
            "n_review": int((events["status"] == "review").sum()) if not events.empty else 0,
        })
    df = pd.DataFrame(rows)
    df.to_csv(P1_8A / "s04_b0377_validation.csv", index=False)
    print("\n[S04] LTOE vs RTOE en B0377:")
    print(df.to_string(index=False))
    # figura comparativa
    fig, axes = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
    for ax, mk, col in zip(axes, ["LTOE", "RTOE"], ["#2f6fb3", "#d1495b"]):
        i = _X01.get_time(list(c.parameters["POINT"]["LABELS"]["value"]), mk, pref)
        tr = c["data"]["points"][:3, i, :].astype(float)
        v = np.linalg.norm(np.gradient(tr, axis=1) * rate, axis=0)
        vs = _X01.smooth(v, 15)
        segs, _ = _X02.segment_repetitions(v, rate)
        t = np.arange(len(vs)) / rate
        ax.plot(t, vs * 1e-3, color=col, lw=1)
        ax.set_title(f"S04 B0377 {mk} [m/s] — {len(segs)} aceptadas")
    axes[-1].set_xlabel("tiempo [s]")
    fig.tight_layout()
    fig.savefig(FIG / "s04_laterality_comparison.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("[S04] figura s04_laterality_comparison.png")
    return df


def b0377_config_validation():
    """Prueba controlada del pipeline sobre B0377 con su config."""
    _X02.set_active_athlete("B0377")
    df = pd.read_csv(_X02.OUT_ATHLETE / "executions_sample.csv")
    qc = pd.read_csv(_X02.OUT_ATHLETE / "qc_summary.csv").iloc[0]
    df.to_csv(P1_8A / "b0377_config_validation.csv", index=False)
    print("\n[B0377] config_validation:")
    print("ejecuciones por técnica/condición:")
    print(df.groupby(["technique", "condition", "trial"]).size().to_string())
    print(f"total={qc['total_executions']} valid={qc['valid_executions']} "
          f"coverage={qc['coverage_pct']}%")
    return df


def main():
    print("FASE 1.8A — VALIDACIÓN DE CONFIGURACIÓN POR ATLETA")
    b0367_regression()
    s01_threshold_validation()
    s04_validation()
    b0377_config_validation()
    print("\n=== FASE 1.8A VALIDACIÓN COMPLETA ===")


if __name__ == "__main__":
    main()