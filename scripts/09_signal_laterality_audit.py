#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
09_signal_laterality_audit.py
=============================
FASE 1.8F — TAREA 1: AUDITORÍA DE SEÑAL Y LATERALIDAD (B0400 / B0371 / B0380)

Auditoría controlada para determinar, CON EVIDENCIA de los C3D, qué
marker/señal y qué lateralidad son más apropiados para segmentar las cinco
técnicas. La decisión es MULTICRITERIO y determinista (no "mayor vmax/SNR").

Alcance EXACTO: B0400, B0371, B0380 × S01-S05 × E01-T01 × 250 Hz.
NO analiza E02/E03/E04, T02 ni S06.

Esta tarea es SOLO AUDITORÍA:
  - NO crea/edita config/athletes/*.yaml
  - NO procesa la cohorte completa
  - NO toca el Data Mart ni el dashboard
  - NO ejecuta ML, NO normaliza frecuencias, NO cambia el algoritmo
Única escritura: output/scaling_selection/ (2 CSV + figuras + log).

Método (reutiliza lógica existente, no la duplica):
  - 02.get_signal (velocidad 3D), 02.segment_repetitions (eventos
    accepted/rejected/review), 02.smooth de 01 (mismo preprocesado).
  - 04.lateral_pair_scores (comparación bilateral L/R por par anatómico,
    idéntico al usado en FASE 1.7 para B0377-S04).
  - Regla de idoneidad existente (04/fase4_signals):
    ok = snr >= 8.0 and baseline < 100 (mm/s) and accepted_count >= 3.
  - Umbral reportado = nivel de actividad del pipeline:
    threshold = baseline + activity_frac * (vmax - baseline).
  - baseline_mad = MAD del primer segundo (complemento diagnóstico robusto
    para detectar baselines altos tipo S01); NO sustituye al pipeline.
"""

from __future__ import annotations

import importlib.util
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation

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
_X04 = _load_module("athlete_generalization_module", "04_athlete_generalization_audit.py")

ROOT = Path(__file__).resolve().parents[1]
INV_FILE = ROOT / "output" / "athlete_inventory" / "file_inventory.csv"
OUT_DIR = ROOT / "output" / "scaling_selection"
FIG_DIR = OUT_DIR / "signal_audit"
AUDIT_CSV = OUT_DIR / "phase_1_8f_signal_audit.csv"
REC_CSV = OUT_DIR / "phase_1_8f_signal_recommendations.csv"

ATHLETES = ["B0400", "B0371", "B0380"]
TECHNIQUES = ["S01", "S02", "S03", "S04", "S05"]
CONDITION = "E01"
TRIAL = "T01"

CANDIDATE_SIGNALS = {
    "S01": ["RFIN", "LFIN"],
    "S02": ["RTOE", "LTOE", "RANK", "LANK", "RHEE", "LHEE"],
    "S03": ["RTOE", "LTOE", "RANK", "LANK", "RHEE", "LHEE"],
    "S04": ["RTOE", "LTOE", "RANK", "LANK", "RHEE", "LHEE"],
    "S05": ["RTOE", "LTOE", "RANK", "LANK", "RHEE", "LHEE"],
}

# Pairs anatómicos para comparación bilateral (igual que FASE 1.7).
PAIRS = {
    "S01": {"finger": ("LFIN", "RFIN")},
    "S02": {"toe": ("LTOE", "RTOE"), "ankle": ("LANK", "RANK"),
            "heel": ("LHEE", "RHEE")},
    "S03": {"toe": ("LTOE", "RTOE"), "ankle": ("LANK", "RANK"),
            "heel": ("LHEE", "RHEE")},
    "S04": {"toe": ("LTOE", "RTOE"), "ankle": ("LANK", "RANK"),
            "heel": ("LHEE", "RHEE")},
    "S05": {"toe": ("LTOE", "RTOE"), "ankle": ("LANK", "RANK"),
            "heel": ("LHEE", "RHEE")},
}

# Parámetros del pipeline (config activa, no normalizada a 250 Hz a propósito).
P = _X02.CFG["params"]
SMOOTH_WIN = int(P["smooth_win"]) | 1
ACTIVITY_FRAC = float(P["activity_frac"])

# Regla de idoneidad existente (04/fase4_signals).
MIN_SNR = 8.0
MAX_BASELINE = 100.0
MIN_ACCEPTED = 3

# Umbrales de lado (mismos que FASE 1.7).
RATIO_R = 1.3
RATIO_L = 0.77

AUDIT_COLUMNS = [
    "athlete_id", "technique", "condition", "trial", "signal", "side",
    "baseline_median", "baseline_mad", "threshold", "vmax", "snr",
    "candidate_count", "accepted_count", "rejected_count", "review_count",
    "candidate_separation_s", "quality_status", "suitability_status", "notes",
]

REC_COLUMNS = [
    "athlete_id", "technique", "recommended_signal", "recommended_side",
    "recommendation_status", "evidence_summary", "alternative_signal",
    "alternative_side", "alternative_reason",
]


def _file_for(athlete: str, technique: str) -> Path | None:
    """Ruta real del C3D atleta×técnica×E01×T01 desde file_inventory.csv."""
    inv = pd.read_csv(INV_FILE)
    hit = inv[
        (inv["athlete_id"] == athlete)
        & (inv["technique"] == technique)
        & (inv["condition"] == CONDITION)
        & (inv["trial"] == TRIAL)
    ]
    if hit.empty:
        return None
    return ROOT / str(hit.iloc[0]["source_path"])


def _side_of(marker: str) -> str:
    return "R" if marker.startswith("R") else "L"


def _velocity(fp: Path, prefix: str, marker: str) -> tuple[np.ndarray, float]:
    """Velocidad 3D del marcador + rate (reutiliza 02.get_signal)."""
    _, v, rate = _X02.get_signal(fp, prefix, marker)
    return v.astype(float), float(rate)


def _metrics_of(v: np.ndarray, rate: float) -> dict:
    """Métricas por señal sobre la señal suavizada (igual preprocesado al pipeline)."""
    v_s = _X01.smooth(v, window=SMOOTH_WIN)
    n_first = max(1, int(round(rate)))
    base_slice = v_s[:n_first]
    baseline = float(np.median(base_slice))
    mad = float(median_abs_deviation(base_slice, scale=1.4826))
    vmax = float(np.max(v_s))
    snr = float(vmax / (baseline + 1.0))
    threshold = baseline + ACTIVITY_FRAC * (vmax - baseline)
    return {"v_s": v_s, "baseline_median": baseline, "baseline_mad": mad,
            "vmax": vmax, "snr": snr, "threshold": threshold}


def _events_of(v: np.ndarray, rate: float) -> tuple[pd.DataFrame, int, int, int, float]:
    """Eventos del pipeline (reutiliza 02.segment_repetitions)."""
    _, ev = _X02.segment_repetitions(v, rate, return_events=True)
    if ev.empty:
        return ev, 0, 0, 0, np.nan
    acc = int((ev["status"] == "accepted").sum())
    rej = int((ev["status"] == "rejected").sum())
    rev = int((ev["status"] == "review").sum())
    peaks = np.sort(ev["peak_frame"].to_numpy())
    sep = float(np.median(np.diff(peaks)) / rate) if len(peaks) >= 2 else np.nan
    return ev, acc, rej, rev, sep


def audit_cells() -> list[dict]:
    """Computa una fila por atleta × técnica × señal candidata."""
    rows = []
    for ath in ATHLETES:
        for tech in TECHNIQUES:
            fp = _file_for(ath, tech)
            if fp is None:
                for mk in CANDIDATE_SIGNALS[tech]:
                    rows.append({
                        "athlete_id": ath, "technique": tech,
                        "condition": CONDITION, "trial": TRIAL,
                        "signal": mk, "side": _side_of(mk),
                        "quality_status": "file_missing",
                        "suitability_status": "n/a",
                        "notes": "archivo E01-T01 no encontrado en file_inventory",
                    })
                continue
            c = _X01.load_c3d(fp)
            labels = list(c.parameters["POINT"]["LABELS"]["value"])
            rate = float(c.parameters["POINT"]["RATE"]["value"][0])
            prefixes = _X01.get_prefixes(c)
            prefix = _X01.athlete_prefix(fp, prefixes)
            for mk in CANDIDATE_SIGNALS[tech]:
                i = _X01.get_time(labels, mk, prefix)
                base = {
                    "athlete_id": ath, "technique": tech,
                    "condition": CONDITION, "trial": TRIAL,
                    "signal": mk, "side": _side_of(mk),
                }
                if i < 0:
                    rows.append({**base, "quality_status": "marker_missing",
                                 "suitability_status": "n/a",
                                 "notes": f"{mk} no existe en el C3D"})
                    continue
                v, r = _velocity(fp, prefix, mk)
                m = _metrics_of(v, r)
                ev, acc, rej, rev, sep = _events_of(v, r)
                qstatus = ("ok" if m["vmax"] > 0 and np.isfinite(m["vmax"])
                           else "no_activity")
                suitable = (m["snr"] >= MIN_SNR and m["baseline_median"] < MAX_BASELINE
                            and acc >= MIN_ACCEPTED) if qstatus == "ok" else False
                note_bits = []
                if m["baseline_median"] >= MAX_BASELINE:
                    note_bits.append(f"baseline alto ({m['baseline_median']:.0f} mm/s)")
                if rev > 0:
                    note_bits.append(f"{rev} en review")
                if acc == 0 and (rej + rev) > 0:
                    note_bits.append("sin ejecuciones aceptadas")
                rows.append({
                    **base,
                    "baseline_median": round(m["baseline_median"], 1),
                    "baseline_mad": round(m["baseline_mad"], 1),
                    "threshold": round(m["threshold"], 1),
                    "vmax": round(m["vmax"], 1),
                    "snr": round(m["snr"], 1),
                    "candidate_count": acc + rej + rev,
                    "accepted_count": acc,
                    "rejected_count": rej,
                    "review_count": rev,
                    "candidate_separation_s": round(sep, 3) if np.isfinite(sep) else np.nan,
                    "quality_status": qstatus,
                    "suitability_status": "suitable" if suitable else "not_suitable",
                    "notes": "; ".join(note_bits),
                })
    return rows


def _side_verdict(ratio: float) -> str | None:
    if not np.isfinite(ratio):
        return None
    return "R" if ratio > RATIO_R else ("L" if ratio < RATIO_L else None)


def _lateral_evidence(ath: str, tech: str) -> tuple[str | None, str]:
    """Comparación bilateral (reutiliza 04.lateral_pair_scores).

    Devuelve (lado, evidencia}: 'R'/'L'/None -> evidencia resumida.
    Para patadas se exige coherencia en >=2 de 3 pares anatómicos.
    """
    fp = _file_for(ath, tech)
    if fp is None:
        return None, "sin archivo E01-T01"
    c = _X01.load_c3d(fp)
    rate = float(c.parameters["POINT"]["RATE"]["value"][0])
    prefixes = _X01.get_prefixes(c)
    prefix = _X01.athlete_prefix(fp, prefixes)
    scores = _X04.lateral_pair_scores(fp, prefix, rate, PAIRS[tech])
    bits = []
    for lbl, s in scores.items():
        if s["left_avail"] and s["right_avail"]:
            bits.append(f"{lbl} R/L vmax {s['r_vmax']/1000:.2f}/{s['l_vmax']/1000:.2f} "
                        f"m/s ratio={s['ratio']:.2f}")
    evidence = "; ".join(bits) if bits else "sin pares disponibles"
    if tech == "S01":
        s = scores.get("finger", {})
        if s.get("left_avail") and s.get("right_avail"):
            return _side_verdict(s["ratio"]), evidence
        if s.get("left_avail"):
            return "L", evidence + " (solo LFIN disponible)"
        if s.get("right_avail"):
            return "R", evidence + " (solo RFIN disponible)"
        return None, evidence
    verdicts = []
    for lbl in ("toe", "ankle", "heel"):
        s = scores.get(lbl, {})
        if s.get("left_avail") and s.get("right_avail"):
            v = _side_verdict(s["ratio"])
            if v:
                verdicts.append((lbl, v))
    counts = Counter(v for _, v in verdicts)
    side = "R" if counts.get("R", 0) >= 2 else ("L" if counts.get("L", 0) >= 2 else None)
    if side is None:
        toe = scores.get("toe", {})
        side = _side_verdict(toe.get("ratio", np.nan))
    if side is None:
        # sin dominancia clara: no forzar; se documenta como AMBIGUOUS
        pass
    return side, evidence


def _audit_row(sub: pd.DataFrame, mk: str) -> pd.Series | None:
    row = sub[(sub["signal"] == mk) & (sub["quality_status"] == "ok")]
    return row.iloc[0] if not row.empty else None


def _evidence_summary(row: pd.Series, lateral: str) -> str:
    return (
        f"{row['signal']}: vmax {row['vmax']:.0f} mm/s, snr {row['snr']:.1f}, "
        f"baseline {row['baseline_median']:.0f}, acc {row['accepted_count']}"
        f"/{row['candidate_count']}, sep {row['candidate_separation_s']:.2f}s "
        f"| lateral: {lateral}"
    )


def _kick_out(row: pd.Series, lateral: str, status: str) -> dict:
    side = _side_of(row["signal"])
    alt = ("RANK" if side == "R" else "LANK")
    return {
        "recommended_signal": row["signal"], "recommended_side": side,
        "recommendation_status": status,
        "evidence_summary": _evidence_summary(row, lateral),
        "alternative_signal": alt, "alternative_side": side,
        "alternative_reason": "respaldo del mismo lado si el toe pierde rastreo",
    }


def _recommendations(audit: pd.DataFrame) -> pd.DataFrame:
    """Recomendación determinista por atleta × técnica (multicriterio).

    Prioriza la aptitud REAL de segmentación (snr/baseline/accepted sobre las
    señales suavizadas y segmentadas por el pipeline), complementada por la
    comparación bilateral de 04.lateral_pair_scores. No decide por vmax/SNR
    a secas.
    """
    out = []
    for ath in ATHLETES:
        for tech in TECHNIQUES:
            sub = audit[(audit["athlete_id"] == ath) & (audit["technique"] == tech)]
            base = {"athlete_id": ath, "technique": tech}
            if (sub["quality_status"] == "file_missing").any():
                out.append({**base, "recommended_signal": "", "recommended_side": "",
                            "recommendation_status": "INSUFFICIENT_DATA",
                            "evidence_summary": "archivo E01-T01 ausente",
                            "alternative_signal": "", "alternative_side": "",
                            "alternative_reason": ""})
                continue
            lateral_side, lateral = _lateral_evidence(ath, tech)

            if tech == "S01":
                cands = [_audit_row(sub, mk) for mk in ("RFIN", "LFIN")]
                cands = [r for r in cands if r is not None]
                if not cands:
                    out.append({**base, "recommended_signal": "", "recommended_side": "",
                                "recommendation_status": "INSUFFICIENT_DATA",
                                "evidence_summary": "RFIN/LFIN ausentes",
                                "alternative_signal": "", "alternative_side": "",
                                "alternative_reason": ""})
                    continue
                chosen = max(cands, key=lambda r: (r["suitability_status"] == "suitable",
                                                   r["snr"], -r["baseline_median"]))
                side = _side_of(chosen["signal"])
                alt = "LFIN" if side == "R" else "RFIN"
                status = ("RECOMMENDED" if chosen["suitability_status"] == "suitable"
                          else "NEEDS_VALIDATION")
                out.append({
                    **base, "recommended_signal": chosen["signal"],
                    "recommended_side": side, "recommendation_status": status,
                    "evidence_summary": _evidence_summary(chosen, lateral),
                    "alternative_signal": alt,
                    "alternative_side": "L" if side == "R" else "R",
                    "alternative_reason": "verificación lateral contralateral del puño",
                })
                continue

            # PATADAS (S02-S05): aptitud de segmentación por lado + lateralidad
            def best_for(side: str):
                for part in ("TOE", "ANK", "HEE"):
                    r = _audit_row(sub, side + part)
                    if r is not None and r["suitability_status"] == "suitable":
                        return r
                return None

            r_suit, l_suit = best_for("R"), best_for("L")
            if r_suit is not None and l_suit is None:
                out.append({**base, **_kick_out(r_suit, lateral, "RECOMMENDED")})
            elif l_suit is not None and r_suit is None:
                out.append({**base, **_kick_out(l_suit, lateral, "RECOMMENDED")})
            elif r_suit is not None and l_suit is not None:
                chosen = r_suit if r_suit["snr"] >= l_suit["snr"] else l_suit
                out.append({**base, **_kick_out(chosen, lateral, "RECOMMENDED")})
            elif lateral_side is None:
                out.append({**base, "recommended_signal": "", "recommended_side": "",
                            "recommendation_status": "AMBIGUOUS",
                            "evidence_summary": f"ninguna señal usable y lateralidad simétrica: {lateral}",
                            "alternative_signal": "", "alternative_side": "",
                            "alternative_reason": "requiere más trials/condiciones para decantar lateralidad"})
            else:
                row = _audit_row(sub, lateral_side + "TOE")
                if row is None:
                    out.append({**base, "recommended_signal": "", "recommended_side": lateral_side,
                                "recommendation_status": "NEEDS_VALIDATION",
                                "evidence_summary": f"{lateral_side}TOE sin datos: {lateral}",
                                "alternative_signal": "", "alternative_side": "",
                                "alternative_reason": ""})
                else:
                    out.append({**base, **_kick_out(row, lateral, "NEEDS_VALIDATION")})
    return pd.DataFrame(out, columns=REC_COLUMNS)


# --------------------------------------------------------------------------- #
# Figuras (1 por técnica, 3 subplots de atleta)
# --------------------------------------------------------------------------- #

def _plot_technique(tech: str, audit: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5), sharey=True)
    for j, ath in enumerate(ATHLETES):
        ax = axes[j]
        fp = _file_for(ath, tech)
        if fp is None:
            ax.set_title(f"{ath} — sin archivo")
            continue
        c = _X01.load_c3d(fp)
        labels = list(c.parameters["POINT"]["LABELS"]["value"])
        rate = float(c.parameters["POINT"]["RATE"]["value"][0])
        prefixes = _X01.get_prefixes(c)
        prefix = _X01.athlete_prefix(fp, prefixes)
        sub = audit[(audit["athlete_id"] == ath) & (audit["technique"] == tech)
                    & (audit["quality_status"] == "ok")]
        if tech == "S01":
            markers = ["RFIN", "LFIN"]
        else:
            rec = sub[sub["suitability_status"] == "suitable"]
            base = "RTOE"
            if not rec.empty:
                base = rec.iloc[0]["signal"]
            side = _side_of(base)
            markers = (["RTOE", "LTOE", "RANK", "LHEE"] if side == "R"
                       else ["LTOE", "RTOE", "LANK", "RHEE"])
        style = {"RFIN": "#c0392b", "LFIN": "#2c6fbb",
                 "RTOE": "#c0392b", "LTOE": "#2c6fbb",
                 "RANK": "#e67e22", "LANK": "#27ae60",
                 "RHEE": "#8e44ad", "LHEE": "#16a085"}
        for mk in markers:
            v, r = _velocity(fp, prefix, mk)
            v_s = _X01.smooth(v, window=SMOOTH_WIN)
            m = _metrics_of(v, r)
            ax.plot(v_s * 1e-3, lw=1.1, label=f"{mk} (vmax {m['vmax']*1e-3:.1f} m/s)",
                    color=style[mk])
            ev, _, _, _, _ = _events_of(v, r)
            hrow = sub[sub["signal"] == mk]
            if not hrow.empty:
                thr = hrow.iloc[0]["threshold"]
                ax.axhline(thr * 1e-3, color=style[mk], ls=":", lw=1.0, alpha=0.6)
            if not ev.empty:
                ax.plot(ev["peak_frame"], v_s[ev["peak_frame"].to_numpy()] * 1e-3,
                        "x", color=style[mk], ms=5, mew=1.5)
        ax.set_title(f"{ath} ({tech})")
        ax.set_xlabel("frame")
        if j == 0:
            ax.set_ylabel("velocidad (m/s)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="upper right")
    fig.suptitle(f"Fase 1.8F — auditoría de señal: {tech} (E01-T01, x = pico candidato)")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = FIG_DIR / f"audit_{tech}.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# Ejecución
# --------------------------------------------------------------------------- #

def run_audit():
    """Genera tabla audit, recomendaciones y figuras. Devuelve (audit, recs)."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    audit = pd.DataFrame(audit_cells(), columns=AUDIT_COLUMNS)
    recs = _recommendations(audit)
    audit.to_csv(AUDIT_CSV, index=False)
    recs.to_csv(REC_CSV, index=False)
    for tech in TECHNIQUES:
        _plot_technique(tech, audit)
    return audit, recs


def main() -> None:
    audit, recs = run_audit()
    print("[1.8F T1] Auditoría de señal y lateralidad")
    print(f"[1.8F T1] Filas audit     : {len(audit)} (3 atletas × S01-S05 × señales candidatas)")
    print(f"[1.8F T1] Filas recomend. : {len(recs)} (3 × 5)")
    print("[1.8F T1] Recomendaciones:")
    for _, r in recs.iterrows():
        flag = "OK " if r["recommendation_status"] == "RECOMMENDED" else "   "
        print(f"  {r['athlete_id']} {r['technique']}: [{r['recommendation_status']}] "
              f"{r['recommended_signal'] or '-'} ({r['recommended_side'] or '-'})")
    print(f"[1.8F T1] Guardado: {AUDIT_CSV}")
    print(f"[1.8F T1] Guardado: {REC_CSV}")
    print(f"[1.8F T1] Figuras : {FIG_DIR} ({len(list(FIG_DIR.glob('audit_*.png')))} PNG)")


if __name__ == "__main__":
    main()