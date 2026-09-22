#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
13_phase_1_8f_cohort_expansion.py
=================================
FASE 1.8F — TASK 5: EXPANSIÓN CONTROLADA DE LA COHORTE 250 Hz (S02–S05)

Lleva el pipeline validado hacia la cohorte 250 Hz completa antes del dataset
ML, con el flujo controlado ya probado:

  auditoría de señales/lateralidad (E01-T01 × S02-S05)
  → recomendaciones por atleta×técnica
  → creación de configs justificadas (config/athletes/<id>.yaml)
  → Golden Path S02-S05 (reutiliza 10)
  → estados finales por atleta

REGLAS CLAVE:
  - NO modifica 02, segmentation.yaml, Data Mart, dashboard ni históricos.
  - NO reprocesa B0400/B0371/B0380 (consolida su Golden Path de Task 3,
    sin recalcular C3D) ni B0377 (EXISTING_PARTIAL_DATA: 9 filas en Mart).
  - B0367 = 200HZ_REFERENCE (fuera de la cohorte 250 Hz).
  - S01: presente en config con thresholds.status=NEEDS_VALIDATION pero NO
    se procesa en Golden Path ni se usa para resultados ML.
  - Configs solo tras evidencia; técnicas sin evidencia -> NEEDS_VALIDATION
    (no forzar). Regla existente (snr>=8 ∧ baseline<100 ∧ accepted>=3).
  - NO normalización, NO ML, NO balanceo.

Reuso: importa 09 (_metrics_of/_events_of/_velocity/_lateral_evidence/_file_for
/_side_of), 11 (_signal_status) y 10 (_process_cell) vía spec_from_file_location.
No duplica el algoritmo de segmentación.

Uso:
  .venv\\Scripts\\python scripts\\13_phase_1_8f_cohort_expansion.py           # 29 atletas
  .venv\\Scripts\\python scripts\\13_phase_1_8f_cohort_expansion.py --limit 2 # muestra (tests)
"""

from __future__ import annotations

import argparse
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
_X10 = _load_module("task3_golden_path_module", "10_phase_1_8f_task3_golden_path.py")
_X11 = _load_module("s01_validation_module", "11_s01_targeted_validation.py")

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "output" / "scaling_validation" / "cohort_expansion"
FIG_DIR = OUT_DIR / "figures"
CFG_DIR = ROOT / "config" / "athletes"
INV_DIR = ROOT / "output" / "athlete_inventory"

TECH_ML = ["S02", "S03", "S04", "S05"]
KICK_SIGNALS = ["RTOE", "LTOE", "RANK", "LANK", "RHEE", "LHEE"]
CONDITION = "E01"
TRIAL = "T01"
EXPECTED_RATE = 250.0

SIDE_STR = {"R": "right", "L": "left"}
MOVE = {"S01": "punch"}

# Atletas ya tratados (no reprocesar).
EXISTING_GP = ("B0400", "B0371", "B0380")
PARTIAL_MART = "B0377"

STATUS_COLUMNS = [
    "athlete_id", "sampling_rate_hz", "total_c3d", "techniques_available",
    "conditions_available", "trials_available", "E01_available", "T01_available",
    "config_status", "signal_audit_status", "golden_path_status",
    "readiness_tier", "final_status",
]

REC_COLUMNS = [
    "athlete_id", "technique", "recommended_signal", "movement_side",
    "evidence_status", "baseline", "snr", "accepted_count",
    "rejected_count", "review_count", "recommendation_reason",
]

AUDIT_COLUMNS = [
    "athlete_id", "technique", "condition", "trial", "signal", "side",
    "baseline", "mad", "vmax", "snr", "threshold",
    "candidate_count", "accepted_count", "rejected_count", "review_count",
    "mean_duration_s", "candidate_separation_s", "signal_status",
]


# --------------------------------------------------------------------------- #
# Universo
# --------------------------------------------------------------------------- #

def _cohort_250() -> list[str]:
    sr = pd.read_csv(INV_DIR / "scaling_readiness.csv")
    ids = sorted(sr.loc[sr["rate_hz"] == 250.0, "athlete"].tolist())
    return ids


def _inventory_meta(athlete: str) -> dict:
    ai = pd.read_csv(INV_DIR / "athlete_inventory.csv")
    row = ai[ai["athlete_id"] == athlete]
    meta = {"total_c3d": 0, "techniques": "", "conditions": "", "trials": ""}
    if not row.empty:
        r = row.iloc[0]
        meta = {"total_c3d": int(r["n_files"]), "techniques": str(r["techniques"]),
                "conditions": str(r["conditions"]), "trials": str(r["trials"])}
    fi = pd.read_csv(INV_DIR / "file_inventory.csv")
    sub = fi[fi["athlete_id"] == athlete]
    sr = pd.read_csv(INV_DIR / "scaling_readiness.csv")
    rate_row = sr[sr["athlete"] == athlete]
    rate_observed = float(rate_row["rate_hz"].iloc[0]) if not rate_row.empty else np.nan
    return {**meta, "E01": bool((sub["condition"] == "E01").any()),
            "T01": bool((sub["trial"] == "T01").any()),
            "rate_observed": rate_observed}


# --------------------------------------------------------------------------- #
# Auditoría de señales (pendientes)
# --------------------------------------------------------------------------- #

def _audit_pending_cell(aid: str, tech: str) -> list[dict]:
    fp = _X09._file_for(aid, tech)
    base = {"athlete_id": aid, "technique": tech, "condition": CONDITION,
            "trial": TRIAL}
    if fp is None:
        return [{**base, "signal": mk, "side": _X09._side_of(mk),
                 "baseline": np.nan, "mad": np.nan, "vmax": np.nan, "snr": np.nan,
                 "threshold": np.nan, "candidate_count": 0, "accepted_count": 0,
                 "rejected_count": 0, "review_count": 0, "mean_duration_s": np.nan,
                 "candidate_separation_s": np.nan, "signal_status": "file_missing"}
                for mk in KICK_SIGNALS]

    c = _X01.load_c3d(fp)
    labels = list(c.parameters["POINT"]["LABELS"]["value"])
    rate = float(c.parameters["POINT"]["RATE"]["value"][0])
    prefixes = _X01.get_prefixes(c)
    prefix = _X01.athlete_prefix(fp, prefixes)
    out = []
    for mk in KICK_SIGNALS:
        i = _X01.get_time(labels, mk, prefix)
        row = {**base, "signal": mk, "side": _X09._side_of(mk)}
        if i < 0:
            out.append({**row, "baseline": np.nan, "mad": np.nan, "vmax": np.nan,
                        "snr": np.nan, "threshold": np.nan, "candidate_count": 0,
                        "accepted_count": 0, "rejected_count": 0, "review_count": 0,
                        "mean_duration_s": np.nan, "candidate_separation_s": np.nan,
                        "signal_status": "missing"})
            continue
        v, r = _X09._velocity(fp, prefix, mk)
        m = _X09._metrics_of(v, r)
        ev, acc, rej, rev, sep = _X09._events_of(v, r)
        mean_dur = float(ev["duration"].mean()) if not ev.empty else np.nan
        status = _X11._signal_status(m["baseline_median"], m["baseline_mad"],
                                     m["vmax"], m["snr"], acc, sep)
        out.append({
            **row,
            "baseline": round(m["baseline_median"], 1),
            "mad": round(m["baseline_mad"], 1),
            "vmax": round(m["vmax"], 1),
            "snr": round(m["snr"], 1),
            "threshold": round(m["threshold"], 1),
            "candidate_count": int(len(ev)),
            "accepted_count": acc, "rejected_count": rej, "review_count": rev,
            "mean_duration_s": round(mean_dur, 3) if np.isfinite(mean_dur) else np.nan,
            "candidate_separation_s": round(sep, 3) if np.isfinite(sep) else np.nan,
            "signal_status": status,
        })
    return out


# --------------------------------------------------------------------------- #
# Recomendación por técnica (mismo criterio que Task 1, sin inventar umbrales)
# --------------------------------------------------------------------------- #

def _recommend_cell(aid: str, tech: str, audit_rows: pd.DataFrame) -> dict:
    cells = audit_rows[(audit_rows["athlete_id"] == aid)
                       & (audit_rows["technique"] == tech)]
    base = {"athlete_id": aid, "technique": tech}
    if cells.empty or (cells["signal_status"] == "file_missing").all():
        return {**base, "recommended_signal": "", "movement_side": "",
                "evidence_status": "INSUFFICIENT_DATA",
                "baseline": np.nan, "snr": np.nan, "accepted_count": 0,
                "rejected_count": 0, "review_count": 0,
                "recommendation_reason": "sin archivo E01-T01"}
    lateral_side, _ = _X09._lateral_evidence(aid, tech)

    def segm(part: str, side: str):
        r = cells[(cells["signal"] == side + part) & (cells["signal_status"] == "segmentable")]
        if not r.empty:
            return r.iloc[0]
        return None

    def best_for(side: str, want_segmentable: bool):
        for part in ("TOE", "ANK", "HEE"):
            sub = cells[(cells["signal"] == side + part)]
            if sub.empty or sub.iloc[0]["signal_status"] == "missing":
                continue
            if want_segmentable and sub.iloc[0]["signal_status"] == "segmentable":
                return sub.iloc[0]
            if (not want_segmentable) and sub.iloc[0]["signal_status"] == "marginal":
                return sub.iloc[0]
        return None

    r_s = best_for("R", True)
    l_s = best_for("L", True)
    chosen = None
    if r_s is not None and l_s is None:
        chosen = r_s
    elif l_s is not None and r_s is None:
        chosen = l_s
    elif r_s is not None and l_s is not None:
        chosen = r_s if r_s["snr"] >= l_s["snr"] else l_s
    else:
        # ninguna segmentable: ¿marginal? (mismo enfoque Task 1)
        r_m = best_for("R", False)
        l_m = best_for("L", False)
        if r_m is not None or l_m is not None:
            cm = r_m if l_m is None else (l_m if r_m is None
                                          else (r_m if r_m["snr"] >= l_m["snr"] else l_m))
            return {**base, "recommended_signal": cm["signal"],
                    "movement_side": cm["side"],
                    "evidence_status": "NEEDS_VALIDATION",
                    "baseline": cm["baseline"], "snr": cm["snr"],
                    "accepted_count": cm["accepted_count"],
                    "rejected_count": cm["rejected_count"],
                    "review_count": cm["review_count"],
                    "recommendation_reason": "señal marginal: no alcanza regla snr/baseline/accepted"}
        return {**base, "recommended_signal": "", "movement_side": "",
                "evidence_status": "INSUFFICIENT_DATA",
                "baseline": np.nan, "snr": np.nan, "accepted_count": 0,
                "rejected_count": 0, "review_count": 0,
                "recommendation_reason": "ninguna señal segmentable"}
    return {**base, "recommended_signal": chosen["signal"],
            "movement_side": chosen["side"],
            "evidence_status": "RECOMMENDED",
            "baseline": chosen["baseline"], "snr": chosen["snr"],
            "accepted_count": chosen["accepted_count"],
            "rejected_count": chosen["rejected_count"],
            "review_count": chosen["review_count"],
            "recommendation_reason": "cumple regla snr>=8, baseline<100, accepted>=3"}


def _athlete_classification(recomms: pd.DataFrame, audit_df: pd.DataFrame,
                           aid: str) -> str:
    sub = recomms[recomms["athlete_id"] == aid]
    n_rec = int((sub["evidence_status"] == "RECOMMENDED").sum())
    n_missing = int((sub["evidence_status"] == "INSUFFICIENT_DATA").sum())
    if n_rec == len(sub) and n_rec > 0:
        return "READY_FOR_CONFIG"
    if n_rec > 0:
        return "PARTIAL_CONFIG"
    if n_missing == len(sub):
        return "INSUFFICIENT_DATA"
    return "NEEDS_VALIDATION"


# --------------------------------------------------------------------------- #
# Config YAML (solo tras evidencia)
# --------------------------------------------------------------------------- #

def _config_yaml(aid: str, recomms: pd.DataFrame) -> str | None:
    sub = recomms[(recomms["athlete_id"] == aid)
                  & (recomms["evidence_status"] == "RECOMMENDED")]
    if sub.empty:
        return None
    lines = [
        f"# Configuración por atleta {aid} (FASE 1.8F, Task 5 — cohorte 250 Hz).",
        "# Auditoría de señales/lateralidad E01-T01 × S02-S05; config provisional.",
        "# S01 = RFIN con thresholds.status=NEEDS_VALIDATION (NO se procesa en Golden Path).",
        "",
        f"athlete_id: {aid}",
        "",
        "metadata:",
        f"  data_dir: atletas/{aid}",
        "  sampling_rate: 250.0",
        "  units: mm",
        "",
        "techniques:",
        "  S01:",
        "    signal: RFIN",
        "    laterality: right",
        "    joints_side: R",
        "    movement: punch",
        "    thresholds:",
        "      status: NEEDS_VALIDATION",
        "      notes: 'S01 fuera de la expansión ML (Task 1-4: representación inconsistente).'",
    ]
    for _, r in sub.iterrows():
        tech = r["technique"]
        side = r["movement_side"]
        sig = r["recommended_signal"]
        lines += [
            f"  {tech}:",
            f"    signal: {sig}",
            f"    laterality: {SIDE_STR.get(side, 'right')}",
            f"    joints_side: {side}",
            f"    movement: kick",
        ]
    lines += [
        "",
        "validation:",
        "  status: provisional",
        f"  notes: 'Config auditoría Task 5: técnicas {', '.join(sorted(sub['technique']))}. Sin procesamiento completo aún.'",
        "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Golden Path (nuevos) + consolidación (3 existentes)
# --------------------------------------------------------------------------- #

def _gp_new(aid: str, recomms: pd.DataFrame) -> tuple[list, list, list]:
    # activar la config del atleta en el módulo 02 QUE USA 10 (instancias
    # separadas por spec_from_file_location) para que pick_best_signal use su señal
    _X10._X02.set_active_athlete(aid)
    sub = recomms[(recomms["athlete_id"] == aid)
                  & (recomms["evidence_status"] == "RECOMMENDED")]
    rows, events, quality = [], [], []
    for _, r in sub.iterrows():
        res = _X10._process_cell(aid, r["technique"])
        for row in res["rows"]:
            row["_source"] = "cohort_expansion_gp"
            rows.append(row)
        for e in res["event_rows"]:
            e["_source"] = "cohort_expansion_gp"
            events.append(e)
        for q in res["quality"]:
            q["_source"] = "cohort_expansion_gp"
            quality.append(q)
    return rows, events, quality


def _gp_existing_3() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = ROOT / "output" / "scaling_validation"
    gp = pd.read_csv(base / "phase_1_8f_task3_golden_path.csv")
    ev = pd.read_csv(base / "phase_1_8f_task3_events.csv")
    ql = pd.read_csv(base / "phase_1_8f_task3_execution_quality.csv")
    m = gp["technique"].isin(TECH_ML)
    g = gp.loc[m].copy()
    g["_source"] = "task3_golden_path"
    e = ev[ev["technique"].isin(TECH_ML)].copy()
    e["_source"] = "task3_golden_path"
    q = ql[ql["technique"].isin(TECH_ML)].copy()
    q["_source"] = "task3_golden_path"
    return g, e, q


# --------------------------------------------------------------------------- #
# Ejecución principal
# --------------------------------------------------------------------------- #

def run_cohort_expansion(limit: int | None = None, write: bool = True):
    cohort = _cohort_250()
    cfg_ids = {p.stem for p in CFG_DIR.glob("*.yaml")}
    pending = [a for a in cohort if a not in EXISTING_GP and a != PARTIAL_MART]
    if limit:
        pending = pending[:limit]

    status_rows, audit_rows, rec_rows = [], [], []

    # 1) Pendientes: auditoría + recomendaciones
    rec_dfs = []
    for aid in pending:
        arows = []
        for tech in TECH_ML:
            arows += _audit_pending_cell(aid, tech)
        audit_rows += arows
        ad = pd.DataFrame(arows, columns=AUDIT_COLUMNS)
        for tech in TECH_ML:
            rec_rows.append(_recommend_cell(aid, tech, ad))
        rec_dfs.append(ad)

    audit_df = pd.DataFrame(audit_rows, columns=AUDIT_COLUMNS)
    rec_df = pd.DataFrame(rec_rows, columns=REC_COLUMNS)

    # 2) Estados pendientes
    for aid in pending:
        grade = _athlete_classification(rec_df, audit_df, aid)
        status_rows.append(_status_row(aid, grade, has_config=False,
                                       audited=True, gp_done=False))

    # 3) Configs nuevas (solo READY_FOR_CONFIG / PARTIAL_CONFIG con >=1 rec)
    new_cfg_ids = set()
    if write:
        for aid in pending:
            yml = _config_yaml(aid, rec_df)
            if yml is not None:
                (CFG_DIR / f"{aid}.yaml").write_text(yml, encoding="utf-8")
                new_cfg_ids.add(aid)

    # 4) Golden Path nuevos (para quienes tengan >=1 RECOMMENDED)
    gp_rows, ev_rows, ql_rows = [], [], []
    gp_new_ids = set()
    for aid in pending:
        if (rec_df["athlete_id"] == aid).any() and \
           (rec_df[(rec_df["athlete_id"] == aid)]["evidence_status"] == "RECOMMENDED").any():
            rows, events, quality = _gp_new(aid, rec_df)
            if rows:
                gp_new_ids.add(aid)
            gp_rows += rows; ev_rows += events; ql_rows += quality

    # 5) Consolidación de los 3 existentes (sin reprocesar)
    g_ex, e_ex, q_ex = _gp_existing_3()

    gp_all = pd.concat([pd.DataFrame(gp_rows), g_ex], ignore_index=True)
    ev_all = pd.concat([pd.DataFrame(ev_rows), e_ex], ignore_index=True)
    ql_all = pd.concat([pd.DataFrame(ql_rows), q_ex], ignore_index=True)

    # 6) Estados de los ya tratados
    for a in EXISTING_GP:
        status_rows.append(_status_row(a, "EXISTING_GOLDEN_PATH",
                                       has_config=True, audited=True, gp_done=True))
    status_rows.append(_status_row(PARTIAL_MART, "EXISTING_PARTIAL_DATA",
                                   has_config=True, audited=False, gp_done=True))

    status_df = pd.DataFrame(status_rows, columns=STATUS_COLUMNS)
    # actualizar config/gp para pendientes ya procesados
    for idx, r in status_df.iterrows():
        a = r["athlete_id"]
        if a in new_cfg_ids:
            status_df.loc[idx, "config_status"] = "provisional"
        if a in gp_new_ids:
            status_df.loc[idx, "golden_path_status"] = "done"
            status_df.loc[idx, "final_status"] = (
                "READY_FOR_CONFIG" if r["final_status"] == "READY_FOR_CONFIG"
                else r["final_status"])

    result = {"status": status_df, "audit": audit_df, "recs": rec_df,
              "gp": gp_all, "events": ev_all, "quality": ql_all,
              "new_cfg_ids": sorted(new_cfg_ids), "gp_new_ids": sorted(gp_new_ids)}

    if write:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        status_df.to_csv(OUT_DIR / "cohort_250hz_status.csv", index=False,
                         encoding="utf-8")
        audit_df.to_csv(OUT_DIR / "cohort_signal_audit.csv", index=False,
                        encoding="utf-8")
        rec_df.to_csv(OUT_DIR / "cohort_signal_recommendations.csv", index=False,
                      encoding="utf-8")
        gp_all.to_csv(OUT_DIR / "cohort_golden_path.csv", index=False,
                      encoding="utf-8")
        ev_all.to_csv(OUT_DIR / "cohort_events.csv", index=False, encoding="utf-8")
        ql_all.to_csv(OUT_DIR / "cohort_execution_quality.csv", index=False,
                      encoding="utf-8")
        _plot_figures(status_df, rec_df, gp_all, audit_df)
    return result


def _status_row(athlete: str, final_status: str, has_config: bool,
                audited: bool, gp_done: bool) -> dict:
    meta = _inventory_meta(athlete)
    cfg_status = "provisional" if has_config else "none"
    if athlete == PARTIAL_MART:
        cfg_status = "provisional (mart partial)"
    audit_status = ("existing(1.8F-T1)" if athlete in EXISTING_GP
                    else ("not_audited" if not audited else "done"))
    if athlete == PARTIAL_MART:
        audit_status = "not_audited"
    gp_status = ("task3_golden_path" if athlete in EXISTING_GP
                 else ("mart_partial" if athlete == PARTIAL_MART else "none"))
    return {
        "athlete_id": athlete, "sampling_rate_hz": 250.0,
        "total_c3d": meta["total_c3d"],
        "techniques_available": meta["techniques"],
        "conditions_available": meta["conditions"],
        "trials_available": meta["trials"],
        "E01_available": meta["E01"], "T01_available": meta["T01"],
        "config_status": cfg_status, "signal_audit_status": audit_status,
        "golden_path_status": gp_status,
        "readiness_tier": "C" if final_status in ("READY_FOR_CONFIG",
                                                  "PARTIAL_CONFIG",
                                                  "EXISTING_GOLDEN_PATH")
                          else ("B" if final_status in ("NEEDS_VALIDATION",
                                                        "INSUFFICIENT_DATA")
                                else "A"),
        "final_status": final_status,
    }


# --------------------------------------------------------------------------- #
# Figuras (resumen)
# --------------------------------------------------------------------------- #

def _plot_figures(status: pd.DataFrame, recs: pd.DataFrame,
                  gp: pd.DataFrame, audit: pd.DataFrame) -> None:
    import matplotlib.colors as mcolors

    # 1) cobertura por atleta
    fig, ax = plt.subplots(figsize=(14, 4.5))
    n_tech = [len([t for t in r["techniques_available"].split(",") if t])
              if r["techniques_available"] else 0 for _, r in status.iterrows()]
    colors = ["#27ae60" if s == "READY_FOR_CONFIG" else "#e0a83a"
              if s == "PARTIAL_CONFIG" else "#8e44ad"
              if s == "EXISTING_GOLDEN_PATH" else "#95a5a6"
              for s in status["final_status"]]
    ax.bar(status["athlete_id"], n_tech, color=colors)
    ax.set_xticks(range(len(status)))
    ax.set_xticklabels(status["athlete_id"], rotation=90, fontsize=6)
    ax.set_ylabel("técnicas disponibles (inventario)")
    ax.set_title(f"Cobertura cohorte 250 Hz — status ({len(status)} atletas)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_coverage.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # 2) señal recomendada por atleta×técnica (anotada)
    fig, ax = plt.subplots(figsize=(14, max(4, len(recs["athlete_id"].unique())) * 0.35))
    aths = sorted(recs["athlete_id"].unique())
    grid = recs.pivot(index="athlete_id", columns="technique",
                      values="recommended_signal").reindex(aths)
    ax.imshow([[1 if pd.notna(v) else 0 for v in row] for row in grid.values],
              cmap="Blues", aspect="auto")
    for i, a in enumerate(aths):
        for j, t in enumerate(TECH_ML):
            v = grid.loc[a, t] if t in grid.columns else np.nan
            ax.text(j, i, "" if pd.isna(v) else f"{v}({recs[(recs['athlete_id']==a)&(recs['technique']==t)]['movement_side'].iloc[0]})",
                    ha="center", va="center", fontsize=6)
    ax.set_xticks(range(len(TECH_ML))); ax.set_xticklabels(TECH_ML)
    ax.set_yticks(range(len(aths))); ax.set_yticklabels(aths, fontsize=7)
    ax.set_title("Señal recomendada (S02-S05, E01-T01)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_recommended_signal.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # 3) lateralidad
    fig, ax = plt.subplots(figsize=(14, max(4, len(aths)) * 0.35))
    grid2 = recs.pivot(index="athlete_id", columns="technique",
                       values="movement_side").reindex(aths)
    ax.imshow([[0 if pd.isna(v) else (1 if v == "R" else 2) for v in row]
               for row in grid2.values],
              cmap=mcolors.ListedColormap(["#ffffff", "#3a7ca5", "#e0a83a"]),
              aspect="auto")
    for i, a in enumerate(aths):
        for j, t in enumerate(TECH_ML):
            v = grid2.loc[a, t] if t in grid2.columns else np.nan
            ax.text(j, i, "" if pd.isna(v) else str(v), ha="center", va="center",
                    fontsize=7, color="#fff")
    ax.set_xticks(range(len(TECH_ML))); ax.set_xticklabels(TECH_ML)
    ax.set_yticks(range(len(aths))); ax.set_yticklabels(aths, fontsize=7)
    ax.set_title("Lateralidad (movement_side, S02-S05)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_laterality.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # 4) resumen procesado (accepted/rejected/review) por atleta×técnica
    evs = pd.read_csv(OUT_DIR / "cohort_events.csv")
    if not evs.empty and "status" in evs.columns:
        col = evs.pivot_table(index=["athlete_id", "technique"], columns="status",
                              values="event_id", aggfunc="count",
                              fill_value=0).reset_index()
        if "athlete_id" not in col.columns:
            col = col.rename(columns={"index": "athlete_id"})
        for k in ("accepted", "rejected", "review"):
            if k not in col.columns:
                col[k] = 0
        lbl = [f"{r.athlete_id}{r.technique}" for _, r in col.iterrows()]
        fig, ax = plt.subplots(figsize=(14, 4.5))
        x = np.arange(len(col))
        ax.bar(x - 0.25, col["accepted"], 0.25, label="accepted", color="#3a7ca5")
        ax.bar(x, col["review"], 0.25, label="review", color="#e0a83a")
        ax.bar(x + 0.25, col["rejected"], 0.25, label="rejected", color="#c2554f")
        ax.set_xticks(x); ax.set_xticklabels(lbl, rotation=90, fontsize=6)
        ax.set_ylabel("nº eventos")
        ax.set_title("Eventos golden path por atleta (S02-S05)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIG_DIR / "fig_processed.png", dpi=130, bbox_inches="tight")
        plt.close(fig)


# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="procesar solo los primeros N atletas pendientes")
    args = ap.parse_args()
    res = run_cohort_expansion(limit=args.limit, write=True)
    st = res["status"]
    print("[1.8F T5] Expansión cohorte 250 Hz (S02-S05, E01-T01)")
    print("[1.8F T5] Estado por atleta:")
    print(st[["athlete_id", "final_status", "config_status",
              "golden_path_status"]].to_string(index=False))
    print(f"[1.8F T5] Nuevas configs : {len(res['new_cfg_ids'])} -> "
          f"{', '.join(res['new_cfg_ids']) or '-'}")
    if not res["gp"].empty:
        gp = res["gp"]
        print(f"[1.8F T5] Golden Path    : {len(res['gp_new_ids'])} atletas nuevos / "
              f"{len(gp)} ejecuciones aceptadas (S02-S05)")
        ev = res["events"]
        print("[1.8F T5] Eventos        :",
              ev.groupby("status").size().to_dict() if not ev.empty else "-")
    print(f"[1.8F T5] Guardado en {OUT_DIR}")
    _X10._X02.set_active_athlete("B0367")  # restaurar estado por defecto del 02 usado por 10
    for aid in res["new_cfg_ids"]:
        _X02.load_config(aid)  # sanity: config recién creada parseada


if __name__ == "__main__":
    main()