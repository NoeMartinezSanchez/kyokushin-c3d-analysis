# -*- coding: utf-8 -*-
"""
components.py — Helpers de presentación del dashboard (Fase 1.8D).

- Mapa de nombres técnicos (Data Mart) -> etiquetas legibles en español.
- Helpers de gráficos con Plotly (headless, sin render en consola).
- Funciones para secciones de comparabilidad / cobertura.

Reglas:
  - La UI es en español; los nombres técnicos originales se conservan como
    tooltip/metadato (Step 6 del prompt).
  - No se interpreta "mayor = mejor"; lenguaje descriptivo/neutral.
"""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.express as px

# nombre_técnico -> (etiqueta_es, unidad)
LABELS = {
    "execution_id": ("ID de ejecución", ""),
    "duration_s": ("Duración de la ejecución", "s"),
    "time_to_peak_s": ("Tiempo hasta el pico", "s"),
    "vmax": ("Velocidad máxima", "m/s"),
    "vmean": ("Velocidad media", "m/s"),
    "amax": ("Aceleración máxima", "m/s²"),
    "displacement": ("Desplazamiento del endpoint", "m"),
    "path_length": ("Longitud de la trayectoria", "m"),
    "hip_rom": ("ROM de cadera (Hip ROM)", "deg"),
    "knee_rom": ("ROM de rodilla (Knee ROM)", "deg"),
    "ankle_rom": ("ROM de tobillo (Ankle ROM)", "deg"),
    "snr": ("Relación señal-ruido (SNR)", ""),
    "sampling_rate_hz": ("Frecuencia de muestreo", "Hz"),
    "primary_signal": ("Señal primaria", ""),
    "movement_side": ("Lado del movimiento", ""),
    "qc_status": ("Estado QC", ""),
    "quality_flag": ("Bandera de calidad", ""),
}

# Grupos funcionales para la vista de técnica/análisis
TEMPORAL_METRICS = ["duration_s", "time_to_peak_s"]
KINEMATIC_METRICS = ["vmax", "vmean", "amax", "displacement", "path_length"]
JOINT_METRICS = ["hip_rom", "knee_rom", "ankle_rom"]

# Metadatos de comparabilidad que el dashboard muestra (desde el Data Mart)
COMPARABILITY_LABELS = {
    "DIRECTLY_COMPARABLE": "Comparable directamente",
    "COMPARABLE_WITH_CAVEAT": "Comparable con reservas",
    "REQUIRES_NORMALIZATION": "Requiere normalización temporal",
    "NEEDS_VALIDATION": "Requiere validación",
    "NOT_AVAILABLE": "No disponible",
}

TECHNIQUE_NAMES = {
    "S01": "Gyaku-Zuki",
    "S02": "Mae-Geri",
    "S03": "Mawashi-Geri gedan",
    "S04": "Mawashi-Geri jodan",
    "S05": "Ushiro-Mawashi-Geri",
}

CONDITION_NAMES = {
    "E01": "Aire (air)",
    "E02": "Escudo (shield)",
    "E03": "Atacante (attacker)",
    "E04": "Defensor (defender)",
}


def label(metric: str) -> str:
    """Etiqueta legible en español + unidad (conservando nombre técnico como título largo)."""
    if metric in LABELS:
        es, unit = LABELS[metric]
        return f"{es} [{unit}]" if unit else es
    return metric


def full_label(metric: str) -> str:
    """Etiqueta técnica completa para tooltips/metadato."""
    return f"{metric} — {label(metric)}"


def metric_units(metric: str) -> str:
    return LABELS.get(metric, ("", ""))[1]


# --------------------------------------------------------------------------- #
# Gráficos
# --------------------------------------------------------------------------- #

def box_and_points(df, y: str, x: str = "repetition") -> go.Figure:
    """Caja + puntos por ejecución para una métrica."""
    fig = go.Figure()
    fig.add_trace(go.Box(
        y=df[y].astype(float), name="Distribución",
        boxpoints="all", jitter=0.3, pointpos=-1.8,
        marker=dict(color="#2f6fb3", size=6),
        line=dict(color="#1d4e89")))
    fig.update_layout(
        title=None,
        xaxis_title="Repeticiones (distribución)",
        yaxis_title=f"{label(y)}",
        template="plotly_white",
        height=320,
        margin=dict(l=40, r=20, t=30, b=40),
    )
    return fig


def rep_scatter(df, y: str, title: str = "") -> go.Figure:
    """Puntos por repetición (x = repetition)."""
    fig = px.scatter(
        df, x="repetition", y=df[y].astype(float),
        title=title or label(y),
        labels={"repetition": "Repetición", y: label(y)},
        color_discrete_sequence=["#e08a2e"],
    )
    fig.update_traces(marker=dict(size=10))
    fig.update_layout(template="plotly_white", height=320)
    return fig


def comparison_bar(df_a, df_b, metric: str, athlete_a: str, athlete_b: str) -> go.Figure:
    """Barras comparativas A vs B para una métrica (misma técnica/condición/trial)."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[f"{athlete_a} (media)"], y=[df_a[metric].astype(float).mean()],
        name=athlete_a, marker_color="#2f6fb3"))
    fig.add_trace(go.Bar(
        x=[f"{athlete_b} (media)"], y=[df_b[metric].astype(float).mean()],
        name=athlete_b, marker_color="#d1495b"))
    # puntos individuales
    for ath, dfx, colr in [(athlete_a, df_a, "#2f6fb3"), (athlete_b, df_b, "#d1495b")]:
        fig.add_trace(go.Scatter(
            x=[f"{ath} (media)"] * len(dfx), y=dfx[metric].astype(float),
            mode="markers", name=f"{ath} reps", marker=dict(color=colr, size=6, opacity=0.6)))
    fig.update_layout(
        title=full_label(metric),
        yaxis_title=label(metric),
        template="plotly_white", height=320,
        barmode="group",
    )
    return fig