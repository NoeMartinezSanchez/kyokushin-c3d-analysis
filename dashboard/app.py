# -*- coding: utf-8 -*-
"""
app.py — Sports Performance Intelligence — Dashboard MVP (Fase 1.8D).

Fuente ÚNICA de datos: output/data_mart/athlete_execution_features.csv.

Reglas:
  - No lee C3D.
  - No recalcula features ni ejecuta el pipeline de segmentación.
  - No hace ML, no rankea, no genera scores.
  - No modifica el Data Mart.
  - Lenguaje descriptivo y neutral (interfaz en español).

Lanzamiento:  streamlit run dashboard/app.py
"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import (  # noqa: E402
    load_data_mart, validate_data_mart, comparability_status,
    filter_by, available_options,
)
from components import (  # noqa: E402
    LABELS, COMPARABILITY_LABELS, TECHNIQUE_NAMES, CONDITION_NAMES,
    label, full_label, box_and_points, rep_scatter, comparison_bar,
    KINEMATIC_METRICS, TEMPORAL_METRICS, JOINT_METRICS,
)

# Raíz del proyecto y carpeta de imágenes del subconjunto local (raíz /images).
ROOT_DIR = Path(__file__).resolve().parent.parent
IMAGES_DIR = ROOT_DIR / "images"

# --------------------------------------------------------------------------- #
# Carga (caché de runtime de Streamlit; NO toca el CSV)
# --------------------------------------------------------------------------- #

@st.cache_data(show_spinner=False)
def _load() -> pd.DataFrame:
    df = load_data_mart()
    # verificación de contrato
    issues = validate_data_mart(df)
    if issues:
        st.error("Problemas de contrato del Data Mart: " + "; ".join(issues))
    return df


def main():
    st.set_page_config(page_title="Sports Performance Intelligence",
                       page_icon="🥋", layout="wide")

    _inject_css()

    df = _load()

    tab_welcome, tab_dash = st.tabs(["Bienvenida · Proyecto",
                                     "Dashboard · Análisis de datos"])

    # ------------------------------------------------- TAB 1 - Bienvenida
    with tab_welcome:
        _view_welcome()

    # ------------------------------------------------- TAB 2 - Dashboard
    with tab_dash:
        athletes = available_options(df, "athlete_id")
        techniques = available_options(df, "technique")
        conditions = available_options(df, "condition")
        trials = available_options(df, "trial")

        c = st.columns(4)
        sel_athlete = c[0].selectbox("Atleta", athletes, key="f_atleta")
        sel_technique = c[1].selectbox("Técnica", techniques, key="f_tecnica")
        sel_condition = c[2].selectbox("Condición", conditions, key="f_condicion")
        sel_trial = c[3].selectbox("Trial", trials, key="f_trial")

        df_ath = filter_by(df, athlete_id=sel_athlete)
        df_main = filter_by(df_ath, technique=sel_technique,
                            condition=sel_condition, trial=sel_trial)

        # ------------------------------------------------------------ VIEW 1
        st.header("1 · Descripción del atleta (Athlete Overview)")
        _view_overview(df, df_ath, sel_athlete)

        # ---------------------------------------------------- VIEW 2 · Técnica
        st.header("2 · Análisis de técnica / ejecución")
        _view_technique(df_main, sel_technique)

        # ----------------------------------------------------- VIEW 3 · Consist
        st.header("3 · Consistencia entre repeticiones")
        _view_consistency(df_main)

        # -------------------------------------------------- VIEW 4 · Comparación
        st.header("4 · Comparación entre atletas")
        _view_comparison(df, sel_athlete, sel_technique, sel_condition, sel_trial)

        # ---------------------------- DATA & COMPARABILITY / COVERAGE
        st.header("5 · Datos, comparabilidad y cobertura")
        _view_meta(df)


def _inject_css():
    st.markdown("""
    <style>
    .block-container {padding-top: 2.2rem;}
    h1, h2, h3 {color: #1f4e79;}
    .hero-title {font-size: 2.1rem; color: #1f4e79; font-weight: 700;}
    .hero-sub {color: #4a5a6a; font-size: 1.05rem;}
    .hero-card {background: #f2f6fb; border-left: 5px solid #2f6fb3;
                padding: 0.9rem 1.1rem; border-radius: 8px;}
    .tech-caption {text-align: center; font-weight: 600; color: #1f4e79;}
    </style>
    """, unsafe_allow_html=True)


def _view_welcome():
    """Pestaña de bienvenida (presentación para la federación).

    Contenido informativo + imágenes del subconjunto local. No evalúa
    rendimiento; lenguaje descriptivo.
    """
    st.markdown('<div class="hero-title">Sports Performance Intelligence</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Análisis biomecánico de karate '
                'Kyokushin — de la captura de movimiento al Data Mart.</div>',
                unsafe_allow_html=True)

    st.markdown((
        '<div class="hero-card">Este proyecto transforma datos públicos de '
        'captura óptica de movimiento (motion capture, formato C3D) de '
        'técnicas de karate en características biomecánicas comparables a '
        'nivel de ejecución. El pipeline cubre: detección de repeticiones → '
        'segmentación → características → representación por atleta y técnica '
        '→ visualización (este panel) y, en etapas siguientes, modelos de '
        'aprendizaje automático. Los resultados son descriptivos y se presentan '
        'sin rankings ni puntuaciones.</div>'),
        unsafe_allow_html=True)

    # Sistema de captura + atleta con marcadores
    st.subheader("Captura de movimiento")
    col_a, col_b = st.columns(2)
    with col_a:
        _show_image_or_pending(
            "camera_system",
            "Sistema de cámaras de motion capture (Vicon).")
    with col_b:
        _show_image_or_pending(
            "athlete_markers",
            "Atleta con marcadores reflectantes (modelo PlugInGait).")

    # Técnicas
    st.subheader("Técnicas analizadas (S01–S05)")
    st.caption("Imagen ilustrativa por técnica.")
    tech = [
        ("technique_S01_gyaku_zuki", "S01 · Gyaku-Zuki"),
        ("technique_S02_mae_geri", "S02 · Mae-Geri"),
        ("technique_S03_mawashi_gedan", "S03 · Mawashi-Geri gedan"),
        ("technique_S04_mawashi_jodan", "S04 · Mawashi-Geri jodan"),
        ("technique_S05_ushiro_mawashi", "S05 · Ushiro-Mawashi-Geri"),
    ]
    cols = st.columns(5)
    for col, (name, cap) in zip(cols, tech):
        with col:
            _show_image_or_pending(name, cap)

    # Inventario del subconjunto local
    st.subheader("Dataset local — inventario del subconjunto")
    st.caption("Inventario de la Fase 1.8E (output/athlete_inventory/); "
               "información descriptiva del subconjunto analizado.")
    sz67 = _folder_size_gb(ROOT_DIR / "B0367") or "~0.212 GB"
    sz77 = _folder_size_gb(ROOT_DIR / "atletas" / "B0377") or "—"
    st.markdown(f"""| Atributo | B0367 | B0377 |
|---|---|---|
| Archivos C3D | 26 | 39 |
| Fecha | 2017-01-31 | 2017-02-20 |
| Técnicas | S01–S05 | S01–S05 |
| Condiciones | E01, E02, E04 | E01, E02, **E03**, E04 |
| Trials | T01, T02 | T01, T02 |
| Frecuencia real | **200 Hz** | **250 Hz** |
| Tamaño | {sz67} | {sz77} |""")

    # Qué hay dentro de cada C3D
    st.subheader("Qué hay dentro de cada C3D")
    st.caption("Estructura interna de los archivos C3D del subconjunto "
               "(formato de motion capture). Información descriptiva.")
    st.markdown(
        "Por punto, el C3D guarda **filas X, Y, Z + residual**, en **mm**. "
        "Una grabación **E01** trae **218 puntos**; la **E02**, **224** "
        "(218 + 6 marcadores del escudo `Tarcza1–Tarcza6`):\n\n"
        "- **39 marcadores anatómicos PlugInGait** (cabeza, columna/clavícula/"
        "tórax, hombros, brazos/antebrazos, codos, muñecas, manos, pelvis, "
        "muslos, rodillas, tibias, tobillos, talones, dedos de los pies).\n"
        "- **76 marcadores de cluster** (grupos de 4 por segmento: pelvis, "
        "fémur, tibia, pie, punta, cabeza, clavícula, torso, húmero, radio, "
        "mano).\n"
        "- **103 variables biomecánicas derivadas** ya calculadas por el "
        "pipeline Vicon/PlugInGait (no hay que derivarlas nosotros):\n"
        "  - **30 ángulos articulares** (LHipAngles, LKneeAngles, "
        "LElbowAngles, LSpineAngles, RHeadAngles, ...).\n"
        "  - **16 potencias** (LHipPower, LKneePower, LAnklePower, "
        "LShoulderPower, ...).\n"
        "  - **16 fuerzas + 16 momentos** (estimaciones del modelo, NO "
        "medidas de plataformas).\n"
        "  - **15 centros de masa** (CentreOfMass, PelvisCOM, LeftFemurCOM, "
        "HeadCOM, ...).\n"
        "  - **~10 centros articulares** (LHJC, RHJC, LKJC, RKJC, LAJC, "
        "RAJC, ...)."
    )

    # Ilustración: wireframe + velocidades de pierna
    st.subheader("Ilustración: Mawashi-Geri jodan")
    st.caption("Modelo de alambres de la ejecución con la trayectoria del pie "
               "en rojo, y velocidades de los puntos de la pierna "
               "(B0400 · S04 · E01-T01).")
    c1, c2 = st.columns(2)
    with c1:
        _show_image_or_pending(
            "s04_mawashi_wireframe",
            "Wireframe de la ejecución — la línea roja marca la trayectoria del pie.")
    with c2:
        _show_image_or_pending(
            "s04_mawashi_leg_velocities",
            "Velocidad de los puntos de la pierna durante la patada.")


# --------------------------------------------------------------------------- #
# VIEW 1 — Overview
# --------------------------------------------------------------------------- #

def _view_overview(df_all, df_ath, athlete: str):
    if df_ath.empty:
        st.warning("Sin ejecuciones para este atleta.")
        return
    cols = st.columns(5)
    rate = df_ath["sampling_rate_hz"].iloc[0]
    qc = df_ath["qc_status"].iloc[0]
    qflag = df_ath["quality_flag"].iloc[0]
    techs = ", ".join(sorted(df_ath["technique"].unique()))
    conds = ", ".join(sorted(df_ath["condition"].unique()))
    metrics = [
        ("Atleta", f"{athlete} ({int(df_ath['sampling_rate_hz'].iloc[0])} Hz)"),
        ("Ejecuciones", str(len(df_ath))),
        ("Técnicas", techs),
        ("Condiciones", conds),
        ("Estado QC", f"{qc} · {qflag}"),
    ]
    for col, (k, v) in zip(cols, metrics):
        col.metric(k, v)
    st.caption("Descripción observada. No se genera ningún índice o evaluación del atleta.")


# --------------------------------------------------------------------------- #
# VIEW 2 — Technique / Execution
# --------------------------------------------------------------------------- #

def _view_technique(df_main, technique: str):
    if df_main.empty:
        st.warning("Sin ejecuciones para esta selección.")
        return
    st.subheader(f"{TECHNIQUE_NAMES.get(technique, technique)} — repeticiones")
    st.caption("Valores observados por repetición (datos del Data Mart). "
               "Ninguna métrica se interpreta como calidad.")

    all_metrics = TEMPORAL_METRICS + KINEMATIC_METRICS + JOINT_METRICS + ["snr"]

    # tarjetas compactas
    for m in all_metrics:
        col1, col2 = st.columns([1, 3])
        v = df_main[m].astype(float)
        with col1:
            st.metric(full_label(m), f"{v.mean():.2f} {metric_unit(m)}"
                      if not v.isna().all() else "n/d")
        with col2:
            st.plotly_chart(box_and_points(df_main, m), width='stretch')
        st.caption("nombre técnico: `%s`" % m)


# --------------------------------------------------------------------------- #
# VIEW 3 — Consistency
# --------------------------------------------------------------------------- #

_CONSISTENCY_METRICS = ["duration_s", "time_to_peak_s", "displacement",
                        "hip_rom", "knee_rom", "ankle_rom"]

def _view_consistency(df_main):
    if df_main.empty:
        st.warning("Sin ejecuciones para esta selección.")
        return
    st.caption("Estadísticos descriptivos de variabilidad entre repeticiones. "
               "Variabilidad no se interpreta como mejor/peor rendimiento.")
    rows = []
    for m in _CONSISTENCY_METRICS:
        v = df_main[m].astype(float)
        sd = v.std(ddof=1)
        mean = v.mean()
        rows.append({
            "Métrica": full_label(m),
            "Media": round(mean, 4),
            "Mediana": round(v.median(), 4),
            "Desv. estándar": round(sd, 4),
            "CV (%)": round((sd / mean * 100), 2) if mean and mean != 0 else None,
            "n": int(v.count()),
        })
    st.table(pd.DataFrame(rows))
    st.caption("CV = coeficiente de variación (descriptivo).")


# --------------------------------------------------------------------------- #
# VIEW 4 — Athlete Comparison
# --------------------------------------------------------------------------- #

_COMPARABILITY_GROUPS = {
    "DIRECTLY_COMPARABLE": ["duration_s", "time_to_peak_s"],
    "COMPARABLE_WITH_CAVEAT": ["displacement", "hip_rom", "knee_rom",
                               "ankle_rom", "snr"],
    "REQUIRES_NORMALIZATION": ["vmax", "vmean", "amax", "path_length"],
}

def _view_comparison(df, sel_athlete, sel_technique, sel_condition, sel_trial):
    st.caption("Comparación solo con los estados de comparabilidad definidos en "
               "el Data Mart. No se declara un atleta mejor/peor.")
    other = [a for a in available_options(df, "athlete_id") if a != sel_athlete]
    if not other:
        st.info("No hay otro atleta para comparar.")
        return
    other_athlete = st.selectbox("Comparar contra", other, key="cmp_ath")
    df_a = filter_by(df, athlete_id=sel_athlete, technique=sel_technique,
                     condition=sel_condition, trial=sel_trial)
    df_b = filter_by(df, athlete_id=other_athlete, technique=sel_technique,
                     condition=sel_condition, trial=sel_trial)
    if df_a.empty or df_b.empty:
        st.warning("No hay ejecuciones comparables para ambos atletas en esta selección.")
        return

    for group, metrics in _COMPARABILITY_GROUPS.items():
        with st.expander(f"Grupo: {COMPARABILITY_LABELS.get(group, group)}", expanded=True):
            st.caption(_group_caveat(group))
            cols = st.columns(2)
            for i, m in enumerate(metrics):
                with cols[i % 2]:
                    st.plotly_chart(comparison_bar(df_a, df_b, m,
                                                   sel_athlete, other_athlete),
                                    width='stretch')
                    if group == "REQUIRES_NORMALIZATION":
                        st.warning("Exploratorio solo — la normalización temporal "
                                   "(200 vs 250 Hz) está pendiente.")


def _group_caveat(group: str) -> str:
    if group == "DIRECTLY_COMPARABLE":
        return "Comparación permitida dentro del alcance actual del Data Mart."
    if group == "COMPARABLE_WITH_CAVEAT":
        return ("Comparación visible con reserva: el estado depende de `movement_side` "
                "(golden path) o de la elección de frames (displacement).")
    if group == "REQUIRES_NORMALIZATION":
        return ("Exploratorio: la comparación formal requiere normalización temporal "
                "(200 vs 250 Hz), aún pendiente.")
    return ""


# --------------------------------------------------------------------------- #
# VIEW 5 — Data & Comparability / Coverage
# --------------------------------------------------------------------------- #

def _view_meta(df):
    st.subheader("Estados de comparabilidad")
    for status, label_es in COMPARABILITY_LABELS.items():
        st.markdown(f"- **{status}** — {label_es}")

    st.subheader("Cobertura actual del prototipo")
    st.write({
        "Atletas": int(df["athlete_id"].nunique()),
        "Técnicas": int(df["technique"].nunique()),
        "Condiciones": int(df["condition"].nunique()),
        "Trials": int(df["trial"].nunique()),
        "Ejecuciones": int(len(df)),
    })
    st.warning(
        "Este es un prototipo sobre el golden-path del Data Mart, no el dataset "
        "completo de karate. S01 y S04 no forman parte de las métricas validadas "
        "actuales; B0377 mantiene validación pendiente en S01/S04; la normalización "
        "temporal 200/250 Hz aún no se ha completado."
    )
    st.caption(f"Versiones del Data Mart: {df['mart_version'].iloc[0]} · "
               f"features {df['feature_version'].iloc[0]} · "
               f"segmentación {df['segmentation_version'].iloc[0]} · "
               f"unidades {df['units_version'].iloc[0]} · "
               f"fuente {df['source_dataset'].iloc[0]}")


_IMAGE_NAMES = [
    "camera_system", "athlete_markers",
    "technique_S01_gyaku_zuki", "technique_S02_mae_geri",
    "technique_S03_mawashi_gedan", "technique_S04_mawashi_jodan",
    "technique_S05_ushiro_mawashi",
]


def _image_for(name: str):
    """Autodetección png/jpg/jpeg/gif dentro de /images (raíz del proyecto)."""
    for ext in (".png", ".jpg", ".jpeg", ".gif"):
        p = IMAGES_DIR / f"{name}{ext}"
        if p.exists():
            return p
    return None


def _folder_size_gb(folder: Path):
    """Tamaño en disco (GB) de una carpeta, recursivo; None si no existe."""
    if not folder.is_dir():
        return None
    total = sum(f.stat().st_size for f in folder.rglob("*") if f.is_file())
    return f"{total / 1e9:.3f} GB"


def _show_image_or_pending(name: str, caption: str):
    p = _image_for(name)
    if p is not None:
        st.image(str(p), caption=caption, use_container_width=True)
    else:
        st.info(f"Imagen pendiente — colócala en `images/{name}` "
                f"(`.png` o `.jpg`).")


def metric_unit(m: str) -> str:
    return LABELS.get(m, ("", ""))[1]


if __name__ == "__main__":
    main()