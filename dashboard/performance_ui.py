# -*- coding: utf-8 -*-
"""
performance_ui.py — Helpers de PRESENTACIÓN del Athlete Performance Dashboard
(Task 10). Formato y agrupación visual; sin lógica de ML y sin streamlit
(criptestables sin runtime de UI).
"""

from __future__ import annotations

# Agrupación visual de las features (castellano) con sus unidades.
GROUPS = [
    ("Ejecución", [
        ("duration_s", "Duración", "s"),
        ("time_to_peak_s", "Tiempo hasta el pico", "s"),
    ]),
    ("Velocidad y aceleración", [
        ("vmax", "Velocidad máxima", "m/s"),
        ("vmean", "Velocidad media", "m/s"),
        ("amax", "Aceleración máxima", "m/s²"),
    ]),
    ("Movimiento", [
        ("displacement", "Desplazamiento", "m"),
        ("path_length", "Longitud de trayecto", "m"),
    ]),
    ("Rango de movimiento", [
        ("hip_rom", "ROM cadera", "°"),
        ("knee_rom", "ROM rodilla", "°"),
        ("ankle_rom", "ROM tobillo", "°"),
    ]),
]

FEATURE_UNITS = {f: unit for _, items in GROUPS for f, _label, unit in items}
FEATURE_LABELS = {f: label for _, items in GROUPS for f, label, _u in items}

# Variables que el modelo "ve" (para la sección explicativa).
MODEL_INPUT_NAMES = {
    "duration_s": "Duración",
    "time_to_peak_s": "Tiempo hasta el pico",
    "vmax": "Velocidad máxima",
    "vmean": "Velocidad media",
    "amax": "Aceleración máxima",
    "displacement": "Desplazamiento",
    "path_length": "Longitud de trayecto",
    "hip_rom": "ROM cadera",
    "knee_rom": "ROM rodilla",
    "ankle_rom": "ROM tobillo",
}


def format_metric(value: float, unit: str) -> str:
    """Una métrica legible: valor + unidad."""
    if unit == "°":
        return f"{value:.1f}°"
    if unit == "m/s":
        return f"{value:.2f} m/s"
    if unit == "m/s²":
        return f"{value:.1f} m/s²"
    if unit in ("m",):
        return f"{value:.3f} m"
    return f"{value:.3f} s" if unit == "s" else f"{value:.3f} {unit}"


def prob_bar(value: float, width: int = 10) -> str:
    """Barra horizontal de texto para una probabilidad (0..1)."""
    filled = int(round(max(0.0, min(1.0, value)) * width))
    return "█" * filled + "░" * (width - filled)


def metric_card_html(label: str, value_str: str) -> str:
    """Tarjeta simple (HTML) para una métrica del perfil."""
    return f'<div style="border:1px solid #dfe6ee;border-radius:8px;' \
           f'padding:8px 10px;min-height:64px;">' \
           f'<div style="font-size:.8rem;color:#4a5a6a;">{label}</div>' \
           f'<div style="font-size:1.15rem;font-weight:600;color:#1f4e79;">' \
           f'{value_str}</div></div>'


def demo_badge_html(text: str = "DEMO") -> str:
    return (f'<span style="display:inline-block;background:#e0a83a;'
            f'color:#fff;border-radius:999px;padding:.15rem .7rem;'
            f'font-size:.8rem;font-weight:700;margin-left:.6rem;">{text}</span>')


def hero_title_html(title: str, subtitle: str) -> str:
    return (f'<div style="font-size:2.2rem;font-weight:800;color:#1f4e79;">'
            f'{title}</div>'
            f'<div style="font-size:1.05rem;color:#4a5a6a;">{subtitle}</div>')


def flow_bar_html(items: list[str], active: int | None = None) -> str:
    """Barra de flujo A–D: 'EJECUCIÓN › ANÁLISIS › COMPARACIÓN › ENTRENADOR'."""
    parts = []
    for i, label in enumerate(items):
        style = ("color:#1f4e79;font-weight:700;" if i == active
                 else "color:#4a5a6a;")
        parts.append(f'<span style="{style}">{label}</span>')
        if i < len(items) - 1:
            parts.append('<span style="color:#94a6b8;"> › </span>')
    return f'<div style="font-size:1rem;letter-spacing:.02em;">{"".join(parts)}</div>'


def mini_bar(value: float, scale: float, width: int = 12) -> str:
    """Barra horizontal de proporción (valor/scale) para comparación."""
    if scale <= 0:
        scale = 1e-9
    filled = int(round(max(0.0, min(1.0, value / scale)) * width))
    return "█" * filled + "░" * (width - filled)


def future_card_html(title: str) -> str:
    return (f'<div style="border:1px dashed #b0becc;border-radius:12px;'
            f'padding:1rem 1.2rem;background:#f8fafc;">'
            f'<div style="font-size:1.05rem;font-weight:700;color:#1f4e79;">'
            f'{title}</div>'
            f'<div style="font-size:.95rem;color:#4a5a6a;">Seguimiento '
            f'longitudinal — <b>futura función</b>. Con sesiones registradas a '
            f'lo largo del tiempo, esta vista permitirá observar la evolución '
            f'de las variables biomecánicas y comparar ejecuciones entre '
            f'sesiones.</div></div>')