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

# Iconos (pequeños) para acompañar cada métrica en las tarjetas de comparación.
FEATURE_ICONS = {
    "duration_s": "⏱️",
    "time_to_peak_s": "🎯",
    "vmax": "⚡",
    "vmean": "🌊",
    "amax": "🚀",
    "displacement": "📏",
    "path_length": "🛤️",
    "hip_rom": "🦵",
    "knee_rom": "🦿",
    "ankle_rom": "🦶",
}

# Chips suaves por estado (sin connotación bueno/malo).
_STATUS_CHIP = {
    "ABOVE_REFERENCE_RANGE": ("por encima", "#dbe7f2", "#1f4e79"),
    "BELOW_REFERENCE_RANGE": ("por debajo", "#f3e3c3", "#7a5b12"),
    "WITHIN_REFERENCE_RANGE": ("dentro de", "#d9e7d9", "#2c6e49"),
}


def comparison_card_html(item: dict, unit: str) -> str:
    """Tarjeta de comparación: icono + label + chip + valores + barra única."""
    icon = FEATURE_ICONS.get(item["feature"], "•")
    label = item["label"]
    chip_txt, chip_bg, chip_fg = _STATUS_CHIP.get(
        item["status"], (item["status"], "#e2e8f0", "#334155"))
    atl = item["athlete"]
    med = item["reference_median"]
    q1, q3 = item["reference_q1"], item["reference_q3"]
    fmt = lambda v: format_metric(v, unit)  # noqa: E731
    scale = max(atl, q3, 1e-9)
    atl_pct = max(0.0, min(100.0, atl / scale * 100))
    q1_pct = max(0.0, min(100.0, q1 / scale * 100))
    q3_pct = max(0.0, min(100.0, q3 / scale * 100))
    med_pct = max(0.0, min(100.0, med / scale * 100))
    return (
        f'<div style="border:1px solid #dfe6ee;border-radius:12px;'
        f'padding:.7rem .9rem;background:#ffffff;margin-bottom:.6rem;">'
        f'<div style="display:flex;align-items:center;gap:.4rem;flex-wrap:wrap;">'
        f'<span style="font-size:1.1rem;">{icon}</span>'
        f'<span style="font-weight:700;color:#1f4e79;">{label}</span>'
        f'<span style="margin-left:auto;background:{chip_bg};color:{chip_fg};'
        f'border-radius:999px;padding:.1rem .6rem;font-size:.75rem;'
        f'font-weight:600;">{chip_txt}</span></div>'
        f'<div style="font-size:1.25rem;font-weight:700;color:#1f4e79;'
        f'margin:.15rem 0 .1rem;">{fmt(atl)}</div>'
        f'<div style="font-size:.82rem;color:#4a5a6a;">Ref. mediana '
        f'{fmt(med)} · Rango {fmt(q1)} – {fmt(q3)}</div>'
        f'<div style="position:relative;height:14px;background:#eef2f7;'
        f'border-radius:7px;margin-top:.45rem;">'
        f'<div style="position:absolute;top:0;bottom:0;left:{q1_pct:.1f}%;'
        f'width:{max(0.0, q3_pct - q1_pct):.1f}%;background:#cfe0f0;"></div>'
        f'<div style="position:absolute;top:0;bottom:0;left:0;'
        f'width:{atl_pct:.1f}%;background:#e0a83a;border-radius:7px;"></div>'
        f'<div style="position:absolute;top:-2px;bottom:-2px;'
        f'left:{med_pct:.1f}%;width:2px;background:#1f4e79;"></div>'
        f'</div></div>'
    )

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


FLOW_COLORS = {"A": "#1f4e79", "B": "#2e6fb3", "C": "#c98a1b", "D": "#5b8db8"}


def flow_bar_html(items: list[str]) -> str:
    """Barra de flujo A–D: recuadros de colores + flecha corta y ancha (➜).

    Los pasos ya incluyen su letra ("A · EJECUCIÓN", ...); el recuadro toma el
    color de su letra. La flecha apunta al siguiente paso.
    """
    parts = []
    for i, label in enumerate(items):
        letter = label[:1]
        color = FLOW_COLORS.get(letter, "#5b8db8")
        parts.append(
            f'<span style="display:inline-block;background:{color};'
            f'color:#fff;border-radius:12px;padding:.45rem 1rem;'
            f'font-weight:700;font-size:1.02rem;line-height:1.15;">{label}</span>')
        if i < len(items) - 1:
            parts.append(
                '<span style="display:inline-block;color:#94a6b8;'
                'font-size:1.6rem;font-weight:800;padding:0 .05rem;'
                'vertical-align:middle;">➜</span>')
    return f'<div style="display:flex;align-items:center;gap:.45rem;' \
           f'flex-wrap:wrap;margin:.5rem 0;">{"".join(parts)}</div>'


def data_uri(path) -> str:
    """Convierte un archivo (jpg/png/gif) en un data-URI base64."""
    import base64
    ext = str(path)[str(path).rfind(".") + 1:].lower()
    mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
            "gif": "image/gif"}.get(ext, "image/png")
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


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