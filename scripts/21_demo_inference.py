#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
21_demo_inference.py — CLI de demostración (Task 9).

Ejecuta inferencia sobre una ejecución REAL del ML Dataset v0 usando el
modelo demostrativo de la capa inference (Random Forest de Task 8B).

Uso:
  .venv\\Scripts\\python scripts\\21_demo_inference.py --execution-id <id>
  .venv\\Scripts\\python scripts\\21_demo_inference.py --technique S04   # primera real de la técnica

Salida orientada a demo (no técnica-exhaustiva).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from inference import predict_execution, model_info  # noqa: E402
from inference.schemas import FEATURES, InferenceError  # noqa: E402

V0_FILE = ROOT / "output" / "ml_dataset_v0" / "ml_dataset_v0.csv"

BAR = "------------------------------------------------------------"


def _pick_row(execution_id: str | None, technique: str | None) -> pd.Series:
    v0 = pd.read_csv(V0_FILE)
    if execution_id:
        hits = v0[v0["execution_id"] == execution_id]
        if hits.empty:
            raise SystemExit(f"execution_id no encontrado: {execution_id}")
        return hits.iloc[0]
    if technique:
        hits = v0[v0["technique"] == technique]
        if hits.empty:
            raise SystemExit(f"no hay ejecuciones para {technique}")
        return hits.iloc[0]
    raise SystemExit("proporciona --execution-id o --technique")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--execution-id", default=None)
    ap.add_argument("--technique", default=None, choices=["S02", "S03", "S04", "S05"])
    args = ap.parse_args()

    row = _pick_row(args.execution_id, args.technique)
    feats = {f: float(row[f]) for f in FEATURES}

    try:
        res = predict_execution(feats)
    except InferenceError as e:
        print(f"[inference] ERROR controlado: {e}")
        raise SystemExit(2)

    info = model_info()
    print(BAR)
    print("KARATE PERFORMANCE INTELLIGENCE")
    print(BAR)
    print(f"Execution    : {row['execution_id']}")
    print(f"Atleta       : {row['athlete_id']}")
    print(f"Técnica real : {row['technique']}")
    print()
    print(f"Predicted technique : {res['predicted_technique']} "
          f"({res['technique_name']})")
    print("Probabilities:")
    for c in ("S02", "S03", "S04", "S05"):
        print(f"  {c}: {res['probabilities'][c]:.3f}")
    print(f"Confidence : {res['confidence'] * 100:.1f} %")
    print()
    print("Biomechanical profile:")
    print(f"  Máx. velocidad       : {res['features']['vmax']:.2f} m/s")
    print(f"  Máx. aceleración     : {res['features']['amax']:.1f} m/s^2")
    print(f"  ROM cadera / rodilla / tobillo : "
          f"{res['features']['hip_rom']:.1f} / "
          f"{res['features']['knee_rom']:.1f} / {res['features']['ankle_rom']:.1f} grados")
    print(f"  Duración             : {res['features']['duration_s']:.3f} s")
    v = info.get("model_version", "n/d")
    print(BAR)
    print(f"(modelo demo: {v} — confidence es la probabilidad del clasificador, "
          "no un pronóstico de éxito deportivo)")
    print(BAR)


if __name__ == "__main__":
    main()