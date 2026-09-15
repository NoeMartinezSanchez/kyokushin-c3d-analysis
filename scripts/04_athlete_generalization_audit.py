#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
04_athlete_generalization_audit.py
==================================
FASE 1.7 — AUDITORÍA COMPARATIVA DE GENERALIZACIÓN A UN SEGUNDO ATLETA

Determina qué partes del pipeline desarrollado con B0367 funcionan sin cambios
sobre B0377 y cuáles dependen específicamente de B0367.

NO entrena ML, NO modifica scripts/02, NO modifica AGENTS.md, NO cambia
config globalmente. Todo el output se escribe en output/athlete_generalization/.

Salidas:
  output/athlete_generalization/athlete_inventory.csv
  output/athlete_generalization/marker_comparison.csv
  output/athlete_generalization/signal_comparison.csv
  output/athlete_generalization/lateralality_analysis.csv
  output/athlete_generalization/pipeline_compatibility.csv
  + figuras

Reutiliza funciones de 01 (inventario/lectura) y 02 (segmentación).
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
OUT = ROOT / "output" / "athlete_generalization"
OUT.mkdir(parents=True, exist_ok=True)

# atleta 1 = baseline B0367 (en carpeta raíz)
BASELINE_DIR = ROOT / "B0367"
# atleta 2 = nuevo (en atletas/<id>)
NEW_DIR = ROOT / "atletas"

derived_kw = ["Angles", "Angle", "Power", "Force", "Moment", "COM", "HJC", "KJC",
              "AJC", "SJC", "EJC", "WJC", "Waist", "Thorax", "Spine", "Pelvis",
              "AbsAnkle"]

def is_derived(lbl: str) -> bool:
    base = lbl.split(":")[-1]
    return any(k in base for k in derived_kw)

def discover_athletes() -> list[str]:
    """Descubre atletas en B0367/ y atletas/*/ (vía rutas del motor 02)."""
    athletes = ["B0367"]  # baseline siempre
    for d in sorted(_X02.data_dir_for("B0377").parent.glob("B0*")):
        if d.is_dir() and d.name not in athletes:
            athletes.append(d.name)
    return athletes

def all_files_for(athlete: str) -> list[Path]:
    """Resuelve la ruta de datos con el mismo helper del motor (FASE 1.8A)."""
    if athlete in ("BASELINE", "B0367"):
        return sorted(_X02.data_dir_for("B0367").rglob("*.c3d"))
    return sorted(_X02.data_dir_for(athlete).rglob("*.c3d"))

def parse_any_name(fname: str) -> dict:
    """Parse de nomenclatura; devuelve dict con campos o claves UNKNOWN."""
    m = re.match(r"(\d{4}-\d{2}-\d{2})-([A-Z]\d{3,4})-S(\d{2})-E(\d{2})-T(\d{2})", fname)
    if not m:
        return {"date": "UNKNOWN", "athlete": "UNKNOWN", "technique": "UNKNOWN",
                "condition": "UNKNOWN", "trial": "UNKNOWN"}
    _, a, s, e, t = m.groups()
    return {"date": _, "athlete": a, "technique": f"S{s}", "condition": f"E{e}",
            "trial": f"T{t}"}

def inspect_c3d(fp: Path) -> dict:
    c = _X01.load_c3d(fp)
    p = c.parameters
    pts = c["data"]["points"]
    labels = list(p["POINT"]["LABELS"]["value"])
    rate = float(p["POINT"]["RATE"]["value"][0]) if "RATE" in p["POINT"] else np.nan
    units = p["POINT"]["UNITS"]["value"][0] if "UNITS" in p["POINT"] else ""
    angle_u = p["POINT"]["ANGLE_UNITS"]["value"][0] if "ANGLE_UNITS" in p["POINT"] else ""
    force_u = p["POINT"]["FORCE_UNITS"]["value"][0] if "FORCE_UNITS" in p["POINT"] else ""
    moment_u = p["POINT"]["MOMENT_UNITS"]["value"][0] if "MOMENT_UNITS" in p["POINT"] else ""
    power_u = p["POINT"]["POWER_UNITS"]["value"][0] if "POWER_UNITS" in p["POINT"] else ""
    analog = int(p["ANALOG"]["USED"]["value"][0]) if "ANALOG" in p and "USED" in p["ANALOG"] else 0
    prefixes = _X01.get_prefixes(c)
    names = list(p["SUBJECTS"]["NAMES"]["value"]) if "SUBJECTS" in p and "NAMES" in p["SUBJECTS"] else []
    tarcza = [l for l in labels if "Tarcza" in l]
    n_anat = sum(1 for l in labels
                 if not is_derived(l) and "Tarcza" not in l and ":" not in l)
    n_deriv = sum(1 for l in labels if is_derived(l))
    n_prefixed = len([l for l in labels if ":" in l])
    return {
        "n_points": int(pts.shape[1]),
        "n_frames": int(pts.shape[2]),
        "rate_hz": rate,
        "units_pos": units, "units_angle": angle_u,
        "units_force": force_u, "units_moment": moment_u, "units_power": power_u,
        "analog_used": analog,
        "n_anatomical": n_anat, "n_derived": n_deriv,
        "n_tarcza": len(tarcza), "n_prefixed": n_prefixed,
        "subjects": names, "prefixes": prefixes,
    }


# --------------------------------------------------------------------------- #
# FASE 1 — inventario
# --------------------------------------------------------------------------- #

def fase1_inventory():
    print("=" * 72)
    print("FASE 1 — INVENTARIO DE NUEVO ATLETA (B0377)")
    print("=" * 72)
    rows = []
    for athlete in discover_athletes():
        for fp in all_files_for(athlete):
            info = parse_any_name(fp.name)
            meta = inspect_c3d(fp)
            rows.append({"athlete_id": info["athlete"],
                         "filename": fp.name,
                         "technique": info["technique"],
                         "condition": info["condition"],
                         "trial": info["trial"],
                         "source_path": str(fp), **meta})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "athlete_inventory.csv", index=False)
    print(f"[guardado] {OUT/'athlete_inventory.csv'} ({len(df)} filas)")
    b0377 = df[df["athlete_id"] == "B0377"]
    print(f"\nB0377: {len(b0377)} archivos")
    if not b0377.empty:
        print("frecuencias detectadas:", sorted(b0377["rate_hz"].unique()))
        print("técnicas:", sorted(b0377["technique"].unique()))
        print("condiciones:", sorted(b0377["condition"].unique()))
        print("trials:", sorted(b0377["trial"].unique()))
        print("sujetos vistos:",
              sorted({s for sub in b0377["subjects"] for s in sub}))
        print("n_points: min", int(b0377["n_points"].min()), "max", int(b0377["n_points"].max()))
    return df


# --------------------------------------------------------------------------- #
# FASE 2 — comparación estructural
# --------------------------------------------------------------------------- #

def _summary_athlete(df: pd.DataFrame, athlete: str) -> dict:
    sub = df[df["athlete_id"] == athlete]
    if sub.empty:
        return {}
    rates = sorted(sub["rate_hz"].dropna().unique())
    return {
        "n_files": int(len(sub)),
        "techniques": sorted(sub["technique"].unique()),
        "conditions": sorted(sub["condition"].unique()),
        "trials": sorted(sub["trial"].unique()),
        "rate_hz": rates,
        "units_pos": list(sub["units_pos"].dropna().unique()),
        "n_points_range": (int(sub["n_points"].min()), int(sub["n_points"].max())),
        "n_frames_range": (int(sub["n_frames"].min()), int(sub["n_frames"].max())),
        "analog_used": list(sub["analog_used"].unique()),
    }


def fase2_structural():
    print("\n" + "=" * 72)
    print("FASE 2 — COMPARACIÓN ESTRUCTURAL B0367 vs B0377")
    print("=" * 72)
    df = pd.read_csv(OUT / "athlete_inventory.csv")
    a367 = _summary_athlete(df, "B0367")
    a377 = _summary_athlete(df, "B0377")

    # comparación de markers usando un E01 de cada uno
    f367 = _X01.find_files("S04-E01-T01")[0]
    files377 = [p for p in all_files_for("B0377") if "S04-E01-T01" in p.name]
    f377 = files377[0] if files377 else None

    set367 = set(list(_X01.load_c3d(f367).parameters["POINT"]["LABELS"]["value"]))
    set377 = set(list(_X01.load_c3d(f377).parameters["POINT"]["LABELS"]["value"])) if f377 else set()
    common = set367 & set377
    only367 = set367 - set377
    only377 = set377 - set367

    # clase de markers
    rows = []
    for lbl in sorted(set367 | set377):
        cls367 = _X01.class_point_label(lbl) if lbl in set367 else "ausente"
        cls377 = _X01.class_point_label(lbl) if lbl in set377 else "ausente"
        rows.append({"attribute": "marker", "name": lbl,
                     "B0367": cls367, "B0377": cls377,
                     "compatible": "yes" if (lbl in common and cls367 == cls377)
                     else "no", "notes": ""})
    # otros atributos
    attr = [
        ("sampling_rate_hz", a367.get("rate_hz"), a377.get("rate_hz")),
        ("units_position", a367.get("units_pos"), a377.get("units_pos")),
        ("analog_used", a367.get("analog_used"), a377.get("analog_used")),
        ("n_files", a367.get("n_files"), a377.get("n_files")),
        ("techniques", a367.get("techniques"), a377.get("techniques")),
        ("conditions", a367.get("conditions"), a377.get("conditions")),
        ("trials", a367.get("trials"), a377.get("trials")),
        ("n_points_range", a367.get("n_points_range"), a377.get("n_points_range")),
        ("n_frames_range", a367.get("n_frames_range"), a377.get("n_frames_range")),
        ("plugin_gait_derived", "si (~103)", "si (~72)"),
        ("tarcza", "si (6 en E02)", "si (4 en E02)"),
        ("subjects_multi", "si (E04: 2)", "si (E03/E04: 2)"),
    ]
    for name, v367, v377 in attr:
        comp = "yes" if v367 == v377 else "no"
        rows.append({"attribute": name, "name": name,
                     "B0367": str(v367), "B0377": str(v377),
                     "compatible": comp, "notes": ""})
    df2 = pd.DataFrame(rows)
    df2.to_csv(OUT / "marker_comparison.csv", index=False)
    print(f"[guardado] {OUT/'marker_comparison.csv'} ({len(df2)} filas)")
    print(f"\nMarcadores comunes: {len(common)}, solo B0367: {len(only367)}, solo B0377: {len(only377)}")
    print("\nAtributos estructurales compatibles (yes):",
          int((df2['compatible'] == 'yes').sum()), "de", len(df2))
    return df2


# --------------------------------------------------------------------------- #
# FASE 3 — lateralidad (sin asumir derecha)
# --------------------------------------------------------------------------- #

def lateral_pair_scores(fp: Path, prefix: str, rate: float, pairs: dict) -> dict:
    """Compara L vs R por pareja; devuelve scores de actividad/vmax/snr."""
    c = _X01.load_c3d(fp)
    labels = list(c.parameters["POINT"]["LABELS"]["value"])
    pts = c["data"]["points"]
    out = {}
    for lbl, (l, r) in pairs.items():
        il, ir = _X01.get_time(labels, l, prefix), _X01.get_time(labels, r, prefix)
        if il < 0 or ir < 0:
            out[lbl] = {"left_avail": il >= 0, "right_avail": ir >= 0,
                        "l_vmax": np.nan, "r_vmax": np.nan, "l_snr": np.nan,
                        "r_snr": np.nan, "ratio": np.nan}
            continue
        def stats(i):
            tr = pts[:3, i, :].astype(float)
            v = np.linalg.norm(np.gradient(tr, axis=1) * rate, axis=0)
            bl = np.median(v[:int(rate)])
            return float(np.max(v)), float(np.max(v) / (bl + 1))
        l_v, l_s = stats(il); r_v, r_s = stats(ir)
        out[lbl] = {"left_avail": True, "right_avail": True,
                    "l_vmax": l_v, "r_vmax": r_v, "l_snr": l_s, "r_snr": r_s,
                    "ratio": r_v / (l_v + 1e-9)}
    return out


def fase3_laterality():
    print("\n" + "=" * 72)
    print("FASE 3 — LATERALIDAD (B0377, sin asumir derecha)")
    print("=" * 72)
    pairs = {
        "toe": ("LTOE", "RTOE"), "ankle": ("LANK", "RANK"),
        "heel": ("LHEE", "RHEE"), "knee": ("LKNE", "RKNE"),
        "thigh": ("LTHI", "RTHI"), "hipjc": ("LHJC", "RHJC"),
    }
    rows = []
    # analizar sobre E01 T01 de cada tecnica
    for tech in ["S01", "S02", "S03", "S04", "S05"]:
        cands = [p for p in all_files_for("B0377") if f"-{tech}-E01-T01.c3d" in p.name]
        if not cands:
            continue
        fp = cands[0]
        c = _X01.load_c3d(fp)
        rate = float(c.parameters["POINT"]["RATE"]["value"][0])
        prefix = _X01.athlete_prefix(fp, _X01.get_prefixes(c))
        scores = lateral_pair_scores(fp, prefix, rate, pairs)
        for lbl, s in scores.items():
            rows.append({"technique": tech, "pair": lbl, **s})

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "lateralality_analysis.csv", index=False)
    print(f"[guardado] {OUT/'lateralality_analysis.csv'} ({len(df)} filas)")

    # evidencia agregada por par (media sobre técnicas)
    g = df.groupby("pair")[["l_vmax", "r_vmax", "l_snr", "r_snr", "ratio"]].mean().round(1)
    print("\nMedia de R/L vmax por par (ratio > 1 => derecha más activa):")
    print(g.to_string())

    # decisión: criterio combinado con umbral
    med_ratio = g["ratio"].median()
    print(f"\nMediana del ratio R/L (macro): {med_ratio:.2f}")
    final = "RIGHT" if med_ratio > 1.3 else ("LEFT" if med_ratio < 0.77 else "UNKNOWN")
    print(f"Lateralidad inferida (umbral combinado): {final}")

    # lateralidad POR TÉCNICA (toe ratio) — hallazgo de parametrización
    print("\nLateralidad por técnica (ratio R/L en toe):")
    per_tech = {}
    for tech in df["technique"].unique():
        sub = df[(df["technique"] == tech) & (df["pair"] == "toe")]
        if sub.empty:
            continue
        r = sub.iloc[0]
        per_tech[tech] = r["ratio"]
        verdict = "R" if r["ratio"] > 1.3 else ("L" if r["ratio"] < 0.77 else "?")
        print(f"  {tech}: LTOE={r['l_vmax']:.0f} RTOE={r['r_vmax']:.0f} ratio={r['ratio']:.2f} "
              f"-> pierna {'derecha' if verdict=='R' else 'izquierda' if verdict=='L' else 'indefinida'}")

    # figura de lateralidad por técnica
    fig, ax = plt.subplots(figsize=(9, 4))
    techs = list(per_tech.keys())
    ratios = [per_tech[t] for t in techs]
    colors = ["#2f6fb3" if r >= 1 else "#d1495b" for r in ratios]
    ax.bar(techs, ratios, color=colors)
    ax.axhline(1.0, color="grey", ls="--", lw=1)
    ax.axhline(0.77, color="grey", ls=":", lw=1)
    ax.axhline(1.3, color="grey", ls=":", lw=1)
    ax.set_ylabel("ratio vmax R/L (toe)")
    ax.set_title("B0377 — lateralidad por técnica (ratio R/L)")
    fig.tight_layout()
    fig.savefig(OUT / "lateralality_by_technique.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("\n[figura] lateralality_by_technique.png")
    return df, final


# --------------------------------------------------------------------------- #
# FASE 4 — compatibilidad de señales
# --------------------------------------------------------------------------- #

def fase4_signals():
    print("\n" + "=" * 72)
    print("FASE 4 — COMPATIBILIDAD DE SEÑALES (señales B0367 en B0377)")
    print("=" * 72)
    cfg = _X02.CFG["signals"]
    rows = []
    plots_data = []
    for tech in ["S01", "S02", "S03", "S04", "S05"]:
        sig_b367 = cfg[tech]["signal"]  # señal definida para B0367
        cands = [p for p in all_files_for("B0377") if f"-{tech}-E01-T01.c3d" in p.name]
        if not cands:
            rows.append({"technique": tech, "signal_b367": sig_b367,
                         "signal_available": "NO FILE",
                         "status": "requires_parametrization", "notes": "sin archivo E01"})
            continue
        fp = cands[0]
        c = _X01.load_c3d(fp)
        rate = float(c.parameters["POINT"]["RATE"]["value"][0])
        prefix = _X01.athlete_prefix(fp, _X01.get_prefixes(c))
        labels = list(c.parameters["POINT"]["LABELS"]["value"])
        i = _X01.get_time(labels, sig_b367, prefix)
        if i < 0:
            rows.append({"technique": tech, "signal_b367": sig_b367,
                         "signal_available": "NO",
                         "status": "requires_parametrization",
                         "notes": f"{sig_b367} no existe en B0377"})
            continue
        traj = c["data"]["points"][:3, i, :].astype(float)
        v = np.linalg.norm(np.gradient(traj, axis=1) * rate, axis=0)
        vs = _X01.smooth(v, window=15)
        baseline = float(np.median(vs[:int(rate)]))
        vmax = float(np.max(vs))
        snr = vmax / (baseline + 1)
        segs, events = _X02.segment_repetitions(v, rate)
        n_acc = len(segs)
        n_rej = int((events["status"] == "rejected").sum()) if not events.empty else 0
        n_rev = int((events["status"] == "review").sum()) if not events.empty else 0
        ok = (snr >= 8.0) and (baseline < 100) and (n_acc >= 3)
        rows.append({
            "technique": tech, "signal_b367": sig_b367,
            "signal_available": "YES",
            "baseline_mm_s": round(baseline, 1),
            "vmax_mm_s": round(vmax, 1), "snr": round(snr, 1),
            "n_accepted": n_acc, "n_rejected": n_rej, "n_review": n_rev,
            "status": "compatible" if ok else "requires_parametrization",
            "notes": "" if ok else "revisar umbrales/parámetros",
        })
        plots_data.append((tech, sig_b367, vs, rate))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "signal_comparison.csv", index=False)
    print(f"[guardado] {OUT/'signal_comparison.csv'} ({len(df)} filas)")
    print(df[["technique", "signal_b367", "signal_available", "baseline_mm_s",
              "vmax_mm_s", "snr", "n_accepted", "status"]].to_string(index=False))

    # figura: perfiles de velocidad de la señal B0367 en B0377
    fig, axes = plt.subplots(5, 1, figsize=(12, 12), sharex=False)
    for ax, (tech, sig, vs, rate) in zip(axes, plots_data[:5]):
        t = np.arange(len(vs)) / rate
        ax.plot(t, vs * 1e-3, lw=1)
        ax.set_title(f"{tech} señal B0367={sig} en B0377")
        ax.set_ylabel("m/s")
    axes[-1].set_xlabel("tiempo [s]")
    fig.tight_layout()
    fig.savefig(OUT / "signal_comparison_b0377.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("[figura] signal_comparison_b0377.png")
    return df


# --------------------------------------------------------------------------- #
# FASE 5 — prueba controlada del pipeline actual
# --------------------------------------------------------------------------- #

def fase5_pipeline_run():
    print("\n" + "=" * 72)
    print("FASE 5 — PRUEBA DEL PIPELINE ACTUAL SOBRE B0377 (sin modificar)")
    print("=" * 72)
    # simular el loop del pipeline sobre E01-T01 de cada técnica (lista representativa)
    selected_b377 = [f"S{tech:02d}-E01-T01" for tech in range(1, 6)]
    rows = []
    for tag in selected_b377:
        cands = [p for p in all_files_for("B0377") if p.name.endswith(f"-{tag}.c3d")]
        if not cands:
            rows.append({"tag": tag, "status": "no_file"})
            continue
        fp = cands[0]
        try:
            c = _X01.load_c3d(fp)
            rate = float(c.parameters["POINT"]["RATE"]["value"][0])
            prefix = _X01.athlete_prefix(fp, _X01.get_prefixes(c))
            tech = tag.split("-")[0]
            best = _X02.pick_best_signal(fp, prefix, tech)
            if best is None:
                rows.append({"tag": tag, "status": "no_signal", "marker": "None",
                             "n_accepted": 0})
                continue
            _, v, _ = _X02.get_signal(fp, prefix, best["marker"])
            segs, events = _X02.segment_repetitions(v, rate)
            rows.append({
                "tag": tag, "status": "processed",
                "marker": best["marker"], "snr": round(best["snr"], 1),
                "n_accepted": len(segs),
                "n_rejected": int((events["status"] == "rejected").sum()) if not events.empty else 0,
                "n_review": int((events["status"] == "review").sum()) if not events.empty else 0,
                "rate_hz": rate,
            })
        except Exception as e:
            rows.append({"tag": tag, "status": f"error: {e}", "marker": "",
                         "n_accepted": 0})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "pipeline_b0377_e01_run.csv", index=False)
    print(df.to_string(index=False))
    return df


# --------------------------------------------------------------------------- #
# FASE 6 — matriz de compatibilidad de componentes
# --------------------------------------------------------------------------- #

def fase6_matrix():
    print("\n" + "=" * 72)
    print("FASE 6 — MATRIZ COMPONENTE | B0367 | B0377 | GENERALIZABLE")
    print("=" * 72)
    rows = [
        {"component": "C3D reader (ezc3d)", "B0367": "ok", "B0377": "ok",
         "class": "A", "action": "sin cambios"},
        {"component": "units (mm/deg/N/Nmm/W)", "B0367": "mm,deg", "B0377": "mm,deg",
         "class": "A", "action": "sin cambios (iguales)"},
        {"component": "sampling rate", "B0367": "200 Hz", "B0377": "250 Hz",
         "class": "B", "action": "parametrizar rate por archivo (ya se lee del C3D)"},
        {"component": "marker mapping (nomenclatura)", "B0367": "PlugInGait", "B0377": "PlugInGait",
         "class": "A", "action": "sin cambios"},
        {"component": "derived variables", "B0367": "103 (E01/E02)", "B0377": "72 (E01/E02)",
         "class": "B", "action": "verificar disponibilidad por archivo"},
        {"component": "lateralidad", "B0367": "derecha (R*)", "B0377": "por determinar",
         "class": "B", "action": "analizar por atleta; no asumir"},
        {"component": "signal selection", "B0367": "RFIN/RTOE hardcoded en config",
         "B0377": "a evaluar", "class": "C", "action": "parametrizar por atleta+técnica"},
        {"component": "execution segmentation (bandas)", "B0367": "ok", "B0377": "a probar",
         "class": "B", "action": "probar umbrales por atleta"},
        {"component": "QC", "B0367": "26/26", "B0377": "a probar",
         "class": "A", "action": "sin cambios"},
        {"component": "feature extraction", "B0367": "ok", "B0377": "a probar",
         "class": "B", "action": "depende de markers/ángulos disponibles"},
        {"component": "E04 multiples sujetos", "B0367": "defensor B0367 vs atacante B0368",
         "B0377": "E04: B0377 vs B0376; E03: B0378 vs B0377",
         "class": "B", "action": "rol cambia por condición E03/E04; parametrizar"},
        {"component": "Tarcza", "B0367": "6 markers", "B0377": "4 markers",
         "class": "B", "action": "parametrizar nº marcadores escudo"},
        {"component": "DTW (faithfulness)", "B0367": "probado", "B0377": "a probar",
         "class": "B", "action": "requiere misma condición/técnica/trial"},
    ]
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "pipeline_compatibility.csv", index=False)
    print(df[["component", "B0367", "B0377", "class"]].to_string(index=False))
    return df


def main():
    print(f"Baseline: {BASELINE_DIR}")
    print(f"Nuevo atleta: {NEW_DIR}")
    athletes = discover_athletes()
    print(f"Atletas descubiertos: {athletes}")
    inv = fase1_inventory()
    mc = fase2_structural()
    lat_df, lat_final = fase3_laterality()
    sig = fase4_signals()
    run = fase5_pipeline_run()
    mat = fase6_matrix()
    print("\n=== AUDITORÍA COMPLETA — FIN ===")


if __name__ == "__main__":
    main()