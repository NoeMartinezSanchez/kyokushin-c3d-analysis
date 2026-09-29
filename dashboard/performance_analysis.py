# -*- coding: utf-8 -*-
"""
performance_analysis.py — Capa ANÁLISIS del Athlete Performance Dashboard
(Task 11, demo).

Comparación describa de una ejecución frente al PERFIL DE REFERENCIA
(descriptivo) de su misma técnica, y comparación entre dos ejecuciones.
Sin rankings, sin scores, sin causalidad y sin feature importance como consejo.

La referencia representa la distribución OBSERVADA en el dataset del demo;
no es un estándar normativo ni óptimo.
"""

from __future__ import annotations

import pandas as pd

from inference.schemas import FEATURES  # noqa: E402
from performance_ui import FEATURE_LABELS  # noqa: E402


def reference_profile(df: pd.DataFrame) -> dict[str, dict[str, dict]]:
    """Perfil de referencia por técnica: median / Q1 / Q3 por feature.

    Devuelve {technique: {feature: {"median":.., "q1":.., "q3":..}}}.
    """
    out = {}
    for tech, g in df.groupby("technique"):
        p = {}
        for f in FEATURES:
            s = g[f].dropna()
            p[f] = {"median": float(s.median()), "q1": float(s.quantile(.25)),
                    "q3": float(s.quantile(.75))}
        out[str(tech)] = p
    return out


def iqr_status(value: float, q1: float, q3: float) -> str:
    """Estado descriptivo respecto al rango central observado (Q1-Q3)."""
    if value > q3:
        return "ABOVE_REFERENCE_RANGE"
    if value < q1:
        return "BELOW_REFERENCE_RANGE"
    return "WITHIN_REFERENCE_RANGE"


def _pct(value: float, ref: float):
    """Diferencia porcentual respecto a la mediana; None si no interpretable."""
    if ref == 0:
        return None
    return 100.0 * (value - ref) / ref


def compare_to_reference(features: dict, profile: dict) -> list[dict]:
    """Compara una ejecución contra el perfil de referencia de su técnica."""
    rows = []
    for f in FEATURES:
        ref = profile[f]
        val = float(features[f])
        diff_abs = val - ref["median"]
        rows.append({
            "feature": f, "label": FEATURE_LABELS.get(f, f),
            "athlete": val, "reference_median": ref["median"],
            "reference_q1": ref["q1"], "reference_q3": ref["q3"],
            "difference_abs": diff_abs,
            "difference_pct": _pct(val, ref["median"]),
            "status": iqr_status(val, ref["q1"], ref["q3"]),
        })
    return rows


def compare_executions(a: dict, b: dict) -> list[dict]:
    """Diferencias entre dos ejecuciones (misma técnica/atleta). Abs + %(rel a a)."""
    rows = []
    for f in FEATURES:
        av, bv = float(a[f]), float(b[f])
        rows.append({
            "feature": f, "label": FEATURE_LABELS.get(f, f),
            "execution_a": av, "execution_b": bv,
            "difference_abs": bv - av,
            "difference_pct": _pct(bv, av),
        })
    return rows


def observed_differences(comparison: list[dict], max_items: int = 4) -> list[dict]:
    """Selecciona diferencias descriptivamente relevantes (sin score).

    Prioriza features fuera del rango central, ordenadas por |%| (o |abs|).
    """
    outside = [r for r in comparison if r["status"] != "WITHIN_REFERENCE_RANGE"]
    key = lambda r: (abs(r["difference_pct"]) if r["difference_pct"] is not None else 0.0,  # noqa: E731
                     abs(r["difference_abs"]))
    return sorted(outside, key=key, reverse=True)[:max_items]


def coach_insights(outside: list[dict]) -> list[dict]:
    """Sugerencias descriptivas para el entrenador (observación -> revisión).

    Sin causalidad y sin usar feature importance como consejo.
    """
    insights = []
    for r in outside:
        label = r["label"].lower()
        if r["difference_pct"] is not None:
            obs = (f"{r['label']} se observa {_status_es(r['status'])} la mediana "
                   f"de referencia ({abs(r['difference_pct']):.0f} %).")
        else:
            obs = (f"{r['label']} se observa {_status_es(r['status'])} la mediana "
                   f"de referencia (diferencia absoluta {abs(r['difference_abs']):.2f}).")
        review = ("Posible área de observación: revisar " + label +
                  " durante el movimiento (inspección).")
        insights.append({"observation": obs, "coach_review": review})
    return insights


def _status_es(status: str) -> str:
    return {
        "ABOVE_REFERENCE_RANGE": "por encima de",
        "BELOW_REFERENCE_RANGE": "por debajo de",
        "WITHIN_REFERENCE_RANGE": "dentro del rango de",
    }[status]