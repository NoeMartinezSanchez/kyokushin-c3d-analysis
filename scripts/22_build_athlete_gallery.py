#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
22_build_athlete_gallery.py
===========================
Genera la GALERÍA de wireframes 3D (GIF) POR ATLETA × TÉCNICA (S02–S05) en
output/gallery/athlete_<id>_<S0X>.gif, usando ejecuciones reales (E01-T01).

Reutiliza scripts/17 (grid fijo, trayectoria del marcador efectuador en rojo,
puntos en los marcadores). Cada C3D se lee UNA sola vez (escena en caché).

Dos pasadas lógicas:
  1) bbox GLOBAL del lote (grid estándar común a todas las imágenes)
  2) render de cada GIF con esos límites fijos (desde la escena cacheada).

Endpoint por celda: señal configurada en config/athletes/<id>.yaml
(techniques.<S0X>.signal); fallback al toe (R/L) si no está disponible.

Celdas SIN ejecución aceptada en v0 (p. ej. B0377/B0388/B0401-S04) se omiten;
el dashboard mostrará placeholder (solo GIF exacto).

Solo LEE C3D y escribe en output/gallery/. Uso:
  .venv\\Scripts\\python scripts\\22_build_athlete_gallery.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import yaml

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parents[1]
V0_FILE = ROOT / "output" / "ml_dataset_v0" / "ml_dataset_v0.csv"
FI_FILE = ROOT / "output" / "athlete_inventory" / "file_inventory.csv"
GALLERY = ROOT / "output" / "gallery"
MANIFEST = GALLERY / "manifest.csv"
CFG_DIR = ROOT / "config" / "athletes"

TECH = ["S02", "S03", "S04", "S05"]

_spec = importlib.util.spec_from_file_location(
    "illustration17", str(ROOT / "scripts" / "17_make_mawashi_illustration.py"))
_ILL = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ILL)


def cell_c3d(athlete: str, tech: str) -> Path | None:
    fi = pd.read_csv(FI_FILE)
    f = fi[(fi["athlete_id"] == athlete) & (fi["technique"] == tech)
           & (fi["condition"] == "E01") & (fi["trial"] == "T01")]
    if f.empty:
        return None
    return ROOT / str(f.iloc[0]["source_path"])


def config_signal(athlete: str, tech: str) -> str:
    p = CFG_DIR / f"{athlete}.yaml"
    if p.exists():
        cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
        t = cfg.get("techniques") or {}
        sig = t.get(tech, {}).get("signal")
        if sig:
            return str(sig)
    return "RTOE" if tech in TECH else "RFIN"


def cells() -> list[dict]:
    v0 = pd.read_csv(V0_FILE)
    out = []
    for athlete in sorted(v0["athlete_id"].unique()):
        for tech in TECH:
            has = ((v0["athlete_id"] == athlete) & (v0["technique"] == tech)).any()
            if not has:
                continue
            c3d = cell_c3d(athlete, tech)
            if c3d is None or not c3d.exists():
                continue
            out.append({"athlete": str(athlete), "technique": tech, "c3d": c3d,
                        "endpoint": config_signal(str(athlete), tech)})
    return out


def resolve(cell: dict):
    """Resuelve endpoint usable y cachea la escena (1 sola apertura del C3D)."""
    cands = []
    ep = cell["endpoint"]
    cands.append(ep)
    side = "L" if str(ep).startswith("L") else "R"
    for c in (f"{side}TOE", "RTOE", "LTOE"):
        if c != ep:
            cands.append(c)
    for cand in cands:
        try:
            sc = _ILL.scene(cell["c3d"], cand)
            return cand, sc
        except RuntimeError:
            continue
    return None


def main() -> None:
    GALLERY.mkdir(parents=True, exist_ok=True)
    cs = cells()
    print(f"[22] celdas con ejecución: {len(cs)}")

    resolved = []
    for cell in cs:
        r = resolve(cell)
        if r is not None:
            resolved.append((cell, r[0], r[1]))
    print(f"[22] celdas resueltas: {len(resolved)}")
    if not resolved:
        raise RuntimeError("sin celdas válidas")

    # ---- pasada 1: grid global del lote ----
    xs0s, xs1s, ys0s, ys1s, zs0s, zs1s = [], [], [], [], [], []
    for _, _, sc in resolved:
        ex = _ILL.extent_of(sc)
        xs0s.append(ex[0]); xs1s.append(ex[1])
        ys0s.append(ex[2]); ys1s.append(ex[3])
        zs0s.append(ex[4]); zs1s.append(ex[5])
    global_limits = (min(xs0s), max(xs1s), min(ys0s), max(ys1s),
                     min(zs0s), max(zs1s))
    print(f"[22] grid global: {[round(v, 2) for v in global_limits]}")

    # ---- pasada 2: render (escena cacheada, sin reabrir C3D) ----
    rows = []
    for cell, endpoint, sc in resolved:
        out = GALLERY / f"athlete_{cell['athlete']}_{cell['technique']}.gif"
        _ILL.make_wireframe_gif(cell["c3d"], out, endpoint,
                                axes_limits=global_limits, scene_dict=sc)
        rows.append({"athlete": cell["athlete"], "technique": cell["technique"],
                     "gif": out.name, "exists": True})
    pd.DataFrame(rows).to_csv(MANIFEST, index=False, encoding="utf-8")

    total = sum(p.stat().st_size for p in GALLERY.glob("*.gif")) / 1e6
    print(f"[22] GIF generados: {len(resolved)}")
    print(f"[22] tamaño galería: {total:.1f} MB | manifest: {MANIFEST}")


if __name__ == "__main__":
    main()