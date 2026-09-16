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

    # HEADER
    st.title("Sports Performance Intelligence")
    st.caption("Karate Biomechanics — Pipeline C3D → Data Mart (prototipo)")

    df = _load()

    # SIDEBAR
    st.sidebar.header("Controles")
    athletes = available_options(df, "athlete_id")
    techniques = available_options(df, "technique")
    conditions = available_options(df, "condition")
    trials = available_options(df, "trial")

    sel_athlete = st.sidebar.selectbox("Atleta", athletes)
    sel_technique = st.sidebar.selectbox("Técnica", techniques)
    sel_condition = st.sidebar.selectbox("Condición", conditions)
    sel_trial = st.sidebar.selectbox("Trial", trials)

    df_ath = filter_by(df, athlete_id=sel_athlete)
    df_main = filter_by(df_ath, technique=sel_technique,
                        condition=sel_condition, trial=sel_trial)

    # ------------------------------------------------------------------ VIEW 1
    st.header("1 · Descripción del atleta (Athlete Overview)")
    _view_overview(df, df_ath, sel_athlete)

    # ------------------------------------------------------ VIEW 2 · Técnica
    st.header("2 · Análisis de técnica / ejecución")
    _view_technique(df_main, sel_technique)

    # ------------------------------------------------------- VIEW 3 · Consist
    st.header("3 · Consistencia entre repeticiones")
    _view_consistency(df_main)

    # -------------------------------------------------- VIEW 4 · Comparación
    st.header("4 · Comparación entre atletas")
    _view_comparison(df, sel_athlete, sel_technique, sel_condition, sel_trial)

    # ---------------------------------------- DATA & COMPARABILITY / COVERAGE
    st.header("5 · Datos, comparabilidad y cobertura")
    _view_meta(df)


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


def metric_unit(m: str) -> str:
    return LABELS.get(m, ("", ""))[1]


if __name__ == "__main__":
    main()