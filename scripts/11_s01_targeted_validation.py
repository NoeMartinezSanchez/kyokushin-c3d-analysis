#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
11_s01_targeted_validation.py
=============================
FASE 1.8F — TAREA 4: VALIDACIÓN DIRIGIDA DE S01 (pre-ML)

Investiga la inconsistencia de representación de S01 (Gyaku-Zuki) entre
atletas, SIN cambiar el algoritmo ni las configuraciones:

  B0367 S01 -> RFIN (histórico, 200 Hz)
  B0377 S01 -> RFIN (NEEDS_VALIDATION)
  B0400 S01 -> RFIN (funciona)
  B0371 S01 -> RFIN configurado, pipeline usó RTOE (gate SNR)
  B0380 S01 -> RFIN configurado, pipeline usó RTOE (gate SNR)

Universo: B0367·B0377·B0400·B0371·B0380 × S01 × E01-T01 × {RFIN, LFIN, RTOE}.

Metodología: capa de AUDITORÍA (read-only) que reutiliza funciones existentes:
  - 09_signal_laterality_audit._metrics_of / _events_of / _velocity
    (que a su vez usan 02.get_signal / 02.segment_repetitions / 01.smooth).
  - 02.load_config para la señal configurada y el estado NEEDS_VALIDATION.
Regla de idoneidad existente (04/fase4): snr>=8 y baseline<100 mm/s y
accepted>=3 -> segmentable. No se inventan umbrales nuevos.

NO modifica: 02, segmentation.yaml, config/athletes/*, Data Mart, dashboard.
Única escritura: output/scaling_validation/s01_validation/.
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
_X09 = _load_module("signal_laterality_audit_module", "09_signal_laterality_audit.py")

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "output" / "scaling_validation" / "s01_validation"
FIG_DIR = OUT_DIR / "figures"

TECHNIQUE = "S01"
CONDITION = "E01"
TRIAL = "T01"
SIGNALS = ["RFIN", "LFIN", "RTOE"]

# Orden: 2 referencias + 3 nuevos (plan Task 4).
ATHLETES = ["B0367", "B0377", "B0400", "B0371", "B0380"]
EXPECTED_RATE = {"B0367": 200.0, "B0377": 250.0, "B0400": 250.0,
                 "B0371": 250.0, "B0380": 250.0}

# Regla de idoneidad existente (misma que 04/fase4 y 09).
MIN_SNR = 8.0
MAX_BASELINE = 100.0
MIN_ACCEPTED = 3

COMPARE_COLUMNS = [
    "athlete_id", "technique", "condition", "trial", "signal",
    "baseline", "mad", "vmax", "snr", "threshold",
    "candidate_count", "accepted_count", "rejected_count", "review_count",
    "mean_duration_s", "candidate_separation_s", "signal_status",
]

ASSESS_COLUMNS = [
    "athlete_id", "configured_signal", "observed_best_signal",
    "rfin_status", "lfin_status", "rtoe_status",
    "fallback_detected", "fallback_interpretation",
    "evidence_level", "recommendation",
]


def _signal_status(baseline: float, mad: float, vmax: float, snr: float,
                   acc: int, cx: float) -> str:
    """Estado por señal (determinista, sin inventar umbrales numéricos).

    - segmentable      : cumple la regla existente (snr, baseline, accepted).
    - marginal         : algo positivo pero la regla no se cumple completa.
    - not_segmentable  : evidencia clara de no separabilidad.
    - missing          : marcador ausente en el C3D.
    """
    if not np.isfinite(vmax) or vmax <= 0:
        return "not_segmentable"
    if snr >= MIN_SNR and baseline < MAX_BASELINE and acc >= MIN_ACCEPTED:
        return "segmentable"
    if snr >= MIN_SNR or acc >= 1:
        return "marginal"
    return "not_segmentable"


def _opponent_problem(baseline: float, mad: float, sep: float) -> list[str]:
    flags = []
    if baseline >= MAX_BASELINE:
        flags.append(f"baseline_alto({baseline:.0f})")
    if np.isfinite(mad) and baseline > 0 and (mad / baseline) > 0.35:
        flags.append(f"mad_dispersa)ratio={mad / baseline:.2f})")
    if np.isfinite(sep) and sep < 0.6:
        flags.append(f"sep_corta({sep:.2f}s)")
    return flags


def _evidence_and_recommendation(rows: dict) -> tuple[str, str]:
    """Clasifica la evidencia y la recomendación por atleta (determinista)."""
    cfg = rows["configured"]
    r = rows["sigs"]["RFIN"]
    l = rows["sigs"]["LFIN"]
    t = rows["sigs"]["RTOE"]

    flags = []
    if r["status"] == "segmentable":
        return "VALIDATED", ""
    # si la señal configurada no es segmenteable (y no es LFIN por elección)
    lfin_alt = (l["status"] == "segmentable")
    rtoe_alt = (t["status"] == "segmentable")
    if r["status"] == "not_segmentable" and l["status"] == "not_segmentable":
        evidence = "SIGNAL_PROBLEM"
        flags.append("RFIN_NOT_VALIDATED")
        if rows["fallback_detected"] and not rtoe_alt:
            flags.append("RTOE_FALLBACK_NOT_VALIDATED")
    elif r["status"] == "marginal":
        evidence = "PROMISING_BUT_INCOMPLETE"
        flags.append("RFIN_NOT_VALIDATED")
    elif lfin_alt or rtoe_alt:
        evidence = "REPRESENTATION_PROBLEM"
        flags.append("RFIN_NOT_VALIDATED")
    else:
        evidence = "INSUFFICIENT_EVIDENCE"
    # señal alternativa viable -> la evidencia es de representación/validación
    if not (lfin_alt or rtoe_alt) and r["status"] in ("marginal", "not_segmentable"):
        if np.isfinite(r["baseline"]) and r["baseline"] < MAX_BASELINE \
                and np.isfinite(r["snr"]) and r["snr"] >= 4:
            flags.append("PIPELINE_GATE_LIMITATION")
    if rows["fallback"] and not rtoe_alt:
        flags.append("RTOE_FALLBACK_NOT_VALIDATED")
    return evidence, ";".join(dict.fromkeys(flags))


def _signal_cell(aid: str, rows: dict) -> list[dict]:
    out = []
    for mk in SIGNALS:
        row = rows["sigs"][mk]
        status = row["status"] if row["present"] else "missing"
        out.append({
            "athlete_id": aid, "technique": TECHNIQUE,
            "condition": CONDITION, "trial": TRIAL, "signal": mk,
            "baseline": round(row["baseline"], 1) if np.isfinite(row["baseline"]) else np.nan,
            "mad": round(row["mad"], 1) if np.isfinite(row["mad"]) else np.nan,
            "vmax": round(row["vmax"], 1) if np.isfinite(row["vmax"]) else np.nan,
            "snr": round(row["snr"], 1) if np.isfinite(row["snr"]) else np.nan,
            "threshold": round(row["threshold"], 1) if np.isfinite(row["threshold"]) else np.nan,
            "candidate_count": row["candidate"],
            "accepted_count": row["accepted"],
            "rejected_count": row["rejected"],
            "review_count": row["review"],
            "mean_duration_s": row["mean_dur"] if np.isfinite(row["mean_dur"]) else np.nan,
            "candidate_separation_s": row["sep"] if np.isfinite(row["sep"]) else np.nan,
            "signal_status": status,
        })
    return out


def _proc(athlete: str) -> dict:
    """Procesa un atleta: métricas por señal + assessments."""
    fp = _X09._file_for(athlete, TECHNIQUE)
    cfg = _X02.load_config(athlete)
    techniques = cfg.get("techniques") or cfg.get("signals") or {}
    t_cfg = techniques.get(TECHNIQUE, {})
    configured = str(t_cfg.get("signal", ""))
    needs_val = (t_cfg.get("thresholds") or {}).get("status") == "NEEDS_VALIDATION"

    if fp is None:
        sigs = {mk: {"present": False, "baseline": np.nan, "mad": np.nan,
                     "vmax": np.nan, "snr": np.nan, "threshold": np.nan,
                     "candidate": 0, "accepted": 0, "rejected": 0, "review": 0,
                     "mean_dur": np.nan, "sep": np.nan, "status": "missing"}
                for mk in SIGNALS}
        return {"athlete": athlete, "configured": configured,
                "needs_val": needs_val, "sigs": sigs, "events": {},
                "fallback": False, "golden_used": ""}

    c = _X01.load_c3d(fp)
    labels = list(c.parameters["POINT"]["LABELS"]["value"])
    rate = float(c.parameters["POINT"]["RATE"]["value"][0])
    prefixes = _X01.get_prefixes(c)
    prefix = _X01.athlete_prefix(fp, prefixes)

    sigs = {}
    events = {}
    for mk in SIGNALS:
        i = _X01.get_time(labels, mk, prefix)
        empty = {"present": False, "baseline": np.nan, "mad": np.nan,
                 "vmax": np.nan, "snr": np.nan, "threshold": np.nan,
                 "candidate": 0, "accepted": 0, "rejected": 0, "review": 0,
                 "mean_dur": np.nan, "sep": np.nan, "status": "missing"}
        if i < 0:
            sigs[mk] = empty
            events[mk] = pd.DataFrame()
            continue
        v, r = _X09._velocity(fp, prefix, mk)
        m = _X09._metrics_of(v, r)
        ev, acc, rej, rev, sep = _X09._events_of(v, r)
        mean_dur = float(ev["duration"].mean()) if not ev.empty else np.nan
        status = _signal_status(m["baseline_median"], m["baseline_mad"],
                                m["vmax"], m["snr"], acc, sep)
        sigs[mk] = {
            "present": True, "baseline": m["baseline_median"],
            "mad": m["baseline_mad"], "vmax": m["vmax"], "snr": m["snr"],
            "threshold": m["threshold"], "candidate": int(len(ev)),
            "accepted": acc, "rejected": rej, "review": rev,
            "mean_dur": mean_dur, "sep": sep, "status": status,
        }
        events[mk] = ev

    # señal usada en el Golden Path (si el atleta fue procesado en Task 3)
    golden_used = ""
    gp_sum = ROOT / "output" / "scaling_validation" / "phase_1_8f_task3_summary.csv"
    if gp_sum.exists():
        g = pd.read_csv(gp_sum)
        hit = g[(g["athlete_id"] == athlete) & (g["technique"] == TECHNIQUE)]
        if not hit.empty:
            golden_used = str(hit.iloc[0]["signal_used"])
    fallback = bool(golden_used) and golden_used != configured

    return {"athlete": athlete, "configured": configured,
            "needs_val": needs_val, "sigs": sigs, "events": events,
            "fallback": fallback, "golden_used": golden_used}


def _observed_best(res: dict) -> str:
    """Señal observada en el Golden Path, o la configurada si no hubo GP.

    Técnicamente: lo que el pipeline usó (o lo que la config indica).
    """
    if res["golden_used"]:
        return res["golden_used"]
    return res["configured"] or "n/a"


def _fallback_interpretation(athlete: str, rows: dict) -> str:
    t = rows["sigs"]["RTOE"]
    if not rows["fallback"]:
        return "no se detectó fallback (señal configurada en uso o sin golden path)"
    bits = [f"{athlete}-S01 golden path usó RTOE en lugar de {rows['configured']}"]
    if np.isfinite(t["snr"]):
        bits.append(f"snr={t['snr']:.1f}")
    if np.isfinite(t["baseline"]):
        bits.append(f"baseline={t['baseline']:.0f}")
    bits.append(f"acc={t['accepted']}")
    if np.isfinite(t["mean_dur"]):
        bits.append(f"dur={t['mean_dur']:.2f}s")
    return "; ".join(bits)


def run_s01_validation():
    """Genera los CSV y figuras. Devuelve (comparison, assessment)."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    all_rows = []
    assessments = []
    processed = {}
    for aid in ATHLETES:
        res = _proc(aid)
        processed[aid] = res
        all_rows += _signal_cell(aid, res)
        evidence, rec = _evidence_and_recommendation(res)
        obs_best = _observed_best(res)
        assessments.append({
            "athlete_id": aid,
            "configured_signal": res["configured"] or "n/a",
            "observed_best_signal": obs_best,
            "rfin_status": res["sigs"]["RFIN"]["status"],
            "lfin_status": res["sigs"]["LFIN"]["status"],
            "rtoe_status": res["sigs"]["RTOE"]["status"],
            "fallback_detected": res["fallback"],
            "fallback_interpretation": _fallback_interpretation(aid, res),
            "evidence_level": evidence,
            "recommendation": rec,
        })

    comparison = pd.DataFrame(all_rows, columns=COMPARE_COLUMNS)
    assessment = pd.DataFrame(assessments, columns=ASSESS_COLUMNS)

    # recomendación global determinista (opciones A-D del plan)
    new250 = [a for a in ATHLETES if a in ("B0400", "B0371", "B0380")]
    cfg_fail = [a for a in new250
                if processed[a]["sigs"].get(processed[a]["configured"], {}).get("status")
                in ("not_segmentable", "marginal", "missing")]
    alt_ok = any(processed[a]["sigs"][mk]["status"] == "segmentable"
                 for a in cfg_fail for mk in (mk for mk in SIGNALS
                                              if mk != processed[a]["configured"]))
    if not cfg_fail:
        overall = "A"
    elif alt_ok:
        overall = "C"
    else:
        overall = "B"

    comparison.to_csv(OUT_DIR / "s01_signal_comparison.csv", index=False,
                      encoding="utf-8")
    assessment.to_csv(OUT_DIR / "s01_athlete_assessment.csv", index=False,
                      encoding="utf-8")

    # figuras
    for aid in ATHLETES:
        _plot_athlete(aid, processed[aid], FIG_DIR)
    _plot_compare_snr(comparison, FIG_DIR)
    _plot_compare_baseline(comparison, FIG_DIR)
    _plot_compare_candidates(comparison, FIG_DIR)

    return comparison, assessment, overall


# --------------------------------------------------------------------------- #
# Figuras
# --------------------------------------------------------------------------- #

def _plot_athlete(aid: str, res: dict, fig_dir: Path) -> None:
    fp = _X09._file_for(aid, TECHNIQUE)
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5), sharex=True)
    if fp is None:
        for ax in axes:
            ax.set_title("sin archivo E01-T01")
        fig.suptitle(f"S01 {aid} — sin datos")
        fig.tight_layout()
        fig.savefig(fig_dir / f"s01_{aid}.png", dpi=130, bbox_inches="tight")
        plt.close(fig)
        return
    c = _X01.load_c3d(fp)
    labels = list(c.parameters["POINT"]["LABELS"]["value"])
    rate = float(c.parameters["POINT"]["RATE"]["value"][0])
    prefixes = _X01.get_prefixes(c)
    prefix = _X01.athlete_prefix(fp, prefixes)
    idx = {mk: _X01.get_time(labels, mk, prefix) for mk in SIGNALS}

    for j, mk in enumerate(SIGNALS):
        ax = axes[j]
        s = res["sigs"][mk]
        ev = res["events"].get(mk)
        if not s["present"] or idx.get(mk) is None or idx[mk] < 0:
            ax.set_title(f"{mk} — marcador ausente")
            continue
        pts = c["data"]["points"]
        traj = pts[:3, idx[mk], :].astype(float)
        v = np.linalg.norm(np.gradient(traj, axis=1) * rate, axis=0)
        v_s = _X01.smooth(v, window=int(_X02.CFG["params"]["smooth_win"]) | 1)
        t = np.arange(len(v_s)) / rate
        ax.plot(t, v_s * 1e-3, lw=1.0, color="#2f6fb3")
        ax.axhline(s["threshold"] * 1e-3, color="#c0392b", ls=":", lw=1.0,
                   label=f"threshold {s['threshold'] / 1e3:.2f}")
        ax.axhline(s["baseline"] * 1e-3, color="#27ae60", ls="--", lw=1.0,
                   label=f"baseline {s['baseline'] / 1e3:.2f}")
        if not ev.empty:
            for _, r in ev.iterrows():
                ax.axvspan(r["start_frame"] / rate, r["end_frame"] / rate,
                           alpha=0.08, color="#2f6fb3")
                ax.plot([r["peak_frame"] / rate], [v_s[r["peak_frame"]] * 1e-3],
                        "x", color="#d1495b", ms=6)
        ax.set_title(f"{mk}: {s['status']} (snr {s['snr']:.1f}, acc {s['accepted']})")
        ax.set_xlabel("tiempo [s]")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6, loc="upper right")
    axes[0].set_ylabel("velocidad [m/s]")
    fig.suptitle(f"S01 {aid} (E01-T01) — RFIN/LFIN/RTOE, config={res['configured'] or 'n/a'}"
                 f"{'· fallback a RTOE en pipeline' if res['fallback'] else ''}")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(fig_dir / f"s01_{aid}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def _plot_compare_snr(comp: pd.DataFrame, fig_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 4.5))
    x = np.arange(len(ATHLETES))
    w = 0.25
    for j, mk in enumerate(SIGNALS):
        vals = [comp[(comp["athlete_id"] == a) & (comp["signal"] == mk)]["snr"].iloc[0]
                if not comp[(comp["athlete_id"] == a) & (comp["signal"] == mk)].empty else np.nan
                for a in ATHLETES]
        ax.bar(x + (j - 1) * w, vals, width=w, label=mk)
    ax.set_xticks(x); ax.set_xticklabels(ATHLETES)
    ax.set_ylabel("SNR")
    ax.set_title("S01 — SNR por señal y atleta (E01-T01)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "s01_compare_snr.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def _plot_compare_baseline(comp: pd.DataFrame, fig_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 4.5))
    x = np.arange(len(ATHLETES))
    w = 0.25
    for j, mk in enumerate(SIGNALS):
        vals = [comp[(comp["athlete_id"] == a) & (comp["signal"] == mk)]["baseline"].iloc[0]
                if not comp[(comp["athlete_id"] == a) & (comp["signal"] == mk)].empty else np.nan
                for a in ATHLETES]
        er = [comp[(comp["athlete_id"] == a) & (comp["signal"] == mk)]["mad"].iloc[0]
              if not comp[(comp["athlete_id"] == a) & (comp["signal"] == mk)].empty else np.nan
              for a in ATHLETES]
        ax.bar(x + (j - 1) * w, vals, width=w, label=mk, yerr=er, capsize=2)
    ax.axhline(100, color="#c0392b", ls="--", lw=1, label="baseline 100 mm/s (regla)")
    ax.set_xticks(x); ax.set_xticklabels(ATHLETES)
    ax.set_ylabel("baseline [mm/s] (± MAD)")
    ax.set_title("S01 — baseline por señal y atleta (E01-T01)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "s01_compare_baseline.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def _plot_compare_candidates(comp: pd.DataFrame, fig_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 4.5))
    x = np.arange(len(ATHLETES))
    w = 0.25
    for j, mk in enumerate(SIGNALS):
        sub = comp[(comp["signal"] == mk)].set_index("athlete_id")
        accs = [int(sub.loc[a, "accepted_count"]) if a in sub.index else 0 for a in ATHLETES]
        ax.bar(x + (j - 1) * w, accs, width=w, label=f"{mk} aceptadas")
    ax.set_xticks(x); ax.set_xticklabels(ATHLETES)
    ax.set_ylabel("nº aceptadas")
    ax.set_title("S01 — ejecuciones aceptadas por señal y atleta (E01-T01)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "s01_compare_candidates.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    comp, ass, overall = run_s01_validation()
    print("[1.8F T4] Validación dirigida de S01 (E01-T01 × RFIN/LFIN/RTOE)")
    cols = ["athlete_id", "configured_signal", "observed_best_signal",
            "rfin_status", "lfin_status", "rtoe_status",
            "fallback_detected", "evidence_level"]
    print(ass[cols].to_string(index=False))
    print(f"[1.8F T4] Recomendación global: Opción {overall}")
    print(f"[1.8F T4] Guardado en {OUT_DIR}")


if __name__ == "__main__":
    main()