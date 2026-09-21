#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
10_phase_1_8f_task3_golden_path.py
==================================
FASE 1.8F — TAREA 3: GOLDEN PATH E01-T01 × S01-S05 (B0400 / B0371 / B0380)

Ejecuta por PRIMERA VEZ el pipeline de segmentación sobre los tres atletas
seleccionados, usando EXCLUSIVAMENTE las configuraciones de la Tarea 2.

Es un experimento controlado de VALIDACIÓN DEL PIPELINE, NO de optimización:
  - NO modifica thresholds, algoritmo, señales ni configs.
  - NO hace tuning/ML/normalización/DTW.
  - NO toca el Data Mart ni los históricos B0367/B0377.
  - NO procesa E02/E03/E04, T02, otros trials ni otros atletas.
Única escritura: output/scaling_validation/.

Reutiliza el pipeline existente (no duplica el algoritmo):
  - 02.set_active_athlete / find_files / process_file / quality_check /
    qc_summary / plot_segmentation / extract_features.
  - Registra y verifica `signal_used`; si la señal del pipeline desvía de la
    config (primary snr<8), se DOCUMENTA como anomalía, no se corrige.
"""

from __future__ import annotations

import importlib.util
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


def _load_module(name: str, fname: str):
    spec = importlib.util.spec_from_file_location(
        name, str(Path(__file__).resolve().parent / fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_X01 = _load_module("dataset_exploration_module", "01_dataset_exploration.py")
_X02 = _load_module("execution_segmentation_module", "02_execution_segmentation.py")

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "output" / "scaling_validation"
FIG_DIR = OUT_DIR / "figures"

ATHLETES = ["B0400", "B0371", "B0380"]
TECHNIQUES = ["S01", "S02", "S03", "S04", "S05"]
CONDITION = "E01"
TRIAL = "T01"

# Frecuencia esperada de la cohorte 250 Hz (se reporta la real del archivo).
EXPECTED_RATE = 250.0

SUMMARY_COLUMNS = [
    "athlete_id", "technique", "condition", "trial", "source_file",
    "sampling_rate_hz", "signal_used", "movement_side", "signal_snr",
    "candidate_count", "accepted_count", "rejected_count", "review_count",
    "qc_ok", "qc_warn", "qc_review", "qc_invalid",
    "accepted_mean_duration_s", "accepted_mean_time_to_peak_s",
    "accepted_mean_vmax_m_s", "accepted_mean_vmean_m_s",
    "accepted_mean_amax_m_s2", "accepted_mean_displacement_m",
    "accepted_mean_path_length_m", "accepted_mean_hip_rom_deg",
    "accepted_mean_knee_rom_deg", "accepted_mean_ankle_rom_deg",
    "accepted_mean_signal_snr", "validation_status",
]


def _cell_tag(tech: str) -> str:
    return f"{tech}-{CONDITION}-{TRIAL}"


def _quality_map(quality: list[dict]) -> dict:
    out = {}
    for q in quality:
        key = (q["source_file"], q["condition"], q["trial"], q["repetition"])
        out[key] = (q["quality_flag"], q["quality_notes"], q["nan_endpoint_pct"])
    return out


def _rom_cols(tech: str) -> tuple[str, str, str]:
    side = str(_X02.JOINTS_SIDE.get(tech, "R")).upper()
    return (f"rom_{side}HipAngles", f"rom_{side}KneeAngles",
            f"rom_{side}AnkleAngles")


def _status_of(tech: str, cell: dict, accepted: pd.DataFrame) -> str:
    """Estado descriptivo de validación (solo informe/tabla nueva, no esquema)."""
    if cell["source_file"] == "":
        return "INSUFFICIENT_EVIDENCE"
    if cell["candidate_count"] == 0 or cell["accepted_count"] == 0:
        return "INSUFFICIENT_EVIDENCE"
    flagged = (cell["qc_warn"] + cell["qc_review"] + cell["qc_invalid"]) > 0
    cfg_thresholds = (_X02.CFG.get("techniques") or {})
    needs_val = (cfg_thresholds.get(tech, {}).get("thresholds") or {}).get(
        "status") == "NEEDS_VALIDATION"
    if flagged or needs_val:
        return "REVIEW_REQUIRED"
    return "OBSERVED_OK"


def _process_cell(aid: str, tech: str) -> dict:
    """Procesa UNA celda atleta×técnica y devuelve todo lo necesario.

    Devuelve: cell (resumen), rows (ejecuciones aceptadas), event_rows,
    quality (QC por ejecución), fp, rate, prefix, signal_used.
    """
    tag = _cell_tag(tech)
    fp = _X02.find_files(tag, aid)
    fp = fp[0] if fp else None
    base_cell = {"athlete_id": aid, "technique": tech, "condition": CONDITION,
                 "trial": TRIAL}
    if fp is None:
        cell = {**base_cell, "source_file": "", "sampling_rate_hz": np.nan,
                "signal_used": "", "movement_side": "", "signal_snr": np.nan,
                "candidate_count": 0, "accepted_count": 0, "rejected_count": 0,
                "review_count": 0, "qc_ok": 0, "qc_warn": 0, "qc_review": 0,
                "qc_invalid": 0, "validation_status": "INSUFFICIENT_EVIDENCE"}
        return {"cell": cell, "rows": [], "event_rows": [], "quality": [],
                "fp": None, "rate": np.nan, "prefix": "", "signal_used": ""}

    c = _X01.load_c3d(fp)
    rate = float(c.parameters["POINT"]["RATE"]["value"][0])
    prefixes = _X01.get_prefixes(c)
    prefix = _X01.athlete_prefix(fp, prefixes)

    rows, event_rows, summary = _X02.process_file(fp)
    # identidad robusta del atleta (E01 mono-sujeto; no depende del prefijo)
    for r in rows:
        r["athlete_id"] = aid
    for e in event_rows:
        e["athlete_id"] = aid

    quality = _X02.quality_check(fp, prefix, rows, rate)
    for q in quality:
        q["athlete_id"] = aid

    # merge QC en filas aceptadas
    qmap = _quality_map(quality)
    for r in rows:
        key = (r["source_file"], r["condition"], r["trial"], r["repetition"])
        flag, notes, nanpct = qmap.get(key, ("OK", "", 0.0))
        r["quality_flag"] = flag
        r["quality_notes"] = notes
        r["nan_endpoint_pct"] = nanpct
        r["execution_id"] = f"{Path(r['source_file']).stem}:e{int(r['event_id'])}"
        r["movement_side"] = str(_X02.JOINTS_SIDE.get(tech, "R")).upper()

    events = pd.DataFrame(event_rows)
    accepted = pd.DataFrame(rows)

    n_cand = int(len(events))
    n_acc = int((events["status"] == "accepted").sum()) if not events.empty else 0
    n_rej = int((events["status"] == "rejected").sum()) if not events.empty else 0
    n_rev = int((events["status"] == "review").sum()) if not events.empty else 0
    qcounts = {"qc_ok": 0, "qc_warn": 0, "qc_review": 0, "qc_invalid": 0}
    for q in quality:
        key = "qc_" + ("ok" if q["quality_flag"] == "OK" else q["quality_flag"].lower())
        qcounts[key] = qcounts.get(key, 0) + 1
    qc = _X02.qc_summary(rows, quality, events)

    signal_used = str(summary.get("signal_marker", ""))
    cell = {**base_cell, "source_file": Path(fp).name,
            "sampling_rate_hz": rate,
            "signal_used": signal_used,
            "movement_side": str(_X02.JOINTS_SIDE.get(tech, "R")).upper(),
            "signal_snr": round(summary.get("signal_snr", np.nan), 1),
            "candidate_count": n_cand, "accepted_count": n_acc,
            "rejected_count": n_rej, "review_count": n_rev,
            "qc_ok": qcounts.get("qc_ok", 0), "qc_warn": qcounts.get("qc_warn", 0),
            "qc_review": qcounts.get("qc_review", 0),
            "qc_invalid": qcounts.get("qc_invalid", 0)}

    # medias descriptivas de features OBSERVADAS (no recalculadas fuera del pipeline)
    for col, src in [
        ("accepted_mean_duration_s", "duration_s"),
        ("accepted_mean_time_to_peak_s", "time_to_peak_s"),
        ("accepted_mean_vmax_m_s", "vmax_m_s"),
        ("accepted_mean_vmean_m_s", "vmean_m_s"),
        ("accepted_mean_amax_m_s2", "amax_m_s2"),
        ("accepted_mean_displacement_m", "displacement_m"),
        ("accepted_mean_path_length_m", "path_length_m"),
        ("accepted_mean_signal_snr", "signal_snr"),
    ]:
        cell[col] = round(float(accepted[src].mean()), 3) if not accepted.empty \
            and src in accepted.columns else np.nan
    hip, knee, ank = _rom_cols(tech)
    for col, src in [("accepted_mean_hip_rom_deg", hip),
                     ("accepted_mean_knee_rom_deg", knee),
                     ("accepted_mean_ankle_rom_deg", ank)]:
        cell[col] = round(float(accepted[src].mean()), 2) if not accepted.empty \
            and src in accepted.columns else np.nan

    cell["validation_status"] = _status_of(tech, cell, accepted)
    return {"cell": cell, "rows": rows, "event_rows": event_rows,
            "quality": quality, "fp": fp, "rate": rate, "prefix": prefix,
            "signal_used": signal_used}


def run_golden_path():
    """Procesa el Golden Path y escribe CSVs + figuras en output/scaling_validation/."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    all_rows, all_events, all_quality, all_cells = [], [], [], []

    for aid in ATHLETES:
        _X02.set_active_athlete(aid)
        for tech in TECHNIQUES:
            res = _process_cell(aid, tech)
            all_cells.append(res["cell"])
            all_rows += res["rows"]
            all_events += res["event_rows"]
            all_quality += res["quality"]
            if res["fp"] is None:
                continue
            # figura de auditoría de segmentación (reutiliza 02.plot_segmentation)
            _, v, _ = _X02.get_signal(res["fp"], res["prefix"], res["signal_used"])
            segs, v_s = _X02.segment_repetitions(v, res["rate"], return_events=False)
            _X02.plot_segmentation(res["fp"], segs, v_s, res["rate"],
                                   res["signal_used"], FIG_DIR)

    gp = pd.DataFrame(all_rows)
    ev = pd.DataFrame(all_events)
    ql = pd.DataFrame(all_quality)
    cells = pd.DataFrame([{k: v for k, v in c.items() if k != "qc_summary"}
                          for c in all_cells], columns=SUMMARY_COLUMNS)

    gp.to_csv(OUT_DIR / "phase_1_8f_task3_golden_path.csv", index=False)
    ev.to_csv(OUT_DIR / "phase_1_8f_task3_events.csv", index=False)
    ql.to_csv(OUT_DIR / "phase_1_8f_task3_execution_quality.csv", index=False)
    cells.to_csv(OUT_DIR / "phase_1_8f_task3_summary.csv", index=False)

    # figuras resumen (neutras)
    _plot_status_by_cell(cells)
    _plot_duration(gp)

    return gp, ev, ql, cells


def _plot_status_by_cell(cells: pd.DataFrame) -> None:
    labs = [f"{r.athlete_id[:4]}-{r.technique}" for r in cells.itertuples()]
    x = np.arange(len(cells))
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.bar(x - 0.25, cells["accepted_count"], width=0.25, label="accepted", color="#3a7ca5")
    ax.bar(x, cells["review_count"], width=0.25, label="review", color="#e0a83a")
    ax.bar(x + 0.25, cells["rejected_count"], width=0.25, label="rejected", color="#c2554f")
    ax.set_xticks(x)
    ax.set_xticklabels(labs, rotation=90, fontsize=7)
    ax.set_ylabel("nº ejecuciones (eventos)")
    ax.set_title("Fase 1.8F Task 3 — eventos por atleta×técnica (E01-T01)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "gp_status_by_cell.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def _plot_duration(gp: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.5))
    if not gp.empty and "duration_s" in gp.columns:
        data = [gp[gp["athlete_id"] == aid]["duration_s"].dropna().to_list()
                for aid in ATHLETES]
        ax.boxplot(data, tick_labels=[aid for aid in ATHLETES], patch_artist=True,
                   boxprops=dict(facecolor="#dbe7f2"), medianprops=dict(color="#c0392b"))
        ax.set_ylabel("duración de ejecución [s]")
        ax.set_title("Fase 1.8F Task 3 — duración por ejecución aceptada (E01-T01)")
    else:
        ax.text(0.5, 0.5, "sin ejecuciones aceptadas", ha="center")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "gp_duration_boxplot.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    gp, ev, ql, cells = run_golden_path()
    _X02.set_active_athlete("B0367")  # restaurar estado por defecto del módulo
    print("[1.8F T3] Golden Path E01-T01 × S01-S05")
    print(f"[1.8F T3] Filas ejecución    : {len(gp)}  (aceptadas, 1 fila = 1 ejecución)")
    print(f"[1.8F T3] Eventos totales    : {len(ev)}")
    print(f"[1.8F T3] QC                 : {len(ql)} filas")
    print(f"[1.8F T3] Celdas (3×5)       : {len(cells)}")
    cols = ["athlete_id", "technique", "signal_used", "movement_side",
            "sampling_rate_hz", "candidate_count", "accepted_count",
            "rejected_count", "review_count", "validation_status"]
    print(cells[cols].to_string(index=False))
    print(f"[1.8F T3] Guardado en {OUT_DIR}")


if __name__ == "__main__":
    main()