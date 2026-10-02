# -*- coding: utf-8 -*-
"""
app_performance.py — Athlete Performance Dashboard (Tasks 10-12, demo).

Producto de demostración para la federación, orientado a tablet y con
narrativa de producto:

  MOVIMIENTO → ANÁLISIS BIOMECÁNICO → PREDICCIÓN → PERFIL → COMPARACIÓN
  → OBSERVACIONES PARA EL ENTRENADOR → (futuro) SEGUIMIENTO LONGITUDINAL

Solo PRESENTATION: la DATA/INFERENCE/ANALYSIS viven en performance_data.py,
inference/ y performance_analysis.py (no se tocan). La única entrada al modelo
es `inference.predict_execution`; NO hay sklearn, NO lectura de C3D, NO ML.

Lanzamiento:
    streamlit run dashboard/app_performance.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference import predict_execution, model_info  # noqa: E402
from inference.schemas import CLASSES, InferenceError  # noqa: E402
from performance_data import (available_athletes, default_execution,  # noqa: E402
                              executions_for, execution_row, features_of, load_v0,
                              gallery_gif_for, gallery_thumbnail,
                              technique_image_path)
from performance_ui import (GROUPS, MODEL_INPUT_NAMES, FEATURE_UNITS,  # noqa: E402
                            format_metric, metric_card_html, prob_bar, mini_bar,
                            demo_badge_html, hero_title_html, flow_bar_html,
                            future_card_html, data_uri, comparison_card_html)
from performance_analysis import (reference_profile, compare_to_reference,  # noqa: E402
                                  compare_executions, observed_differences,
                                  coach_insights)

st.set_page_config(page_title="KARATE PERFORMANCE INTELLIGENCE",
                   page_icon="🥋", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.2rem;}
h1 {color: #1f4e79;}
[data-testid="stSelectbox"] > div > div {min-height: 3.1rem;}
.hero-card {border:2px solid #1f4e79;border-radius:14px;padding:1.2rem 1.4rem;
            background:#f2f6fb;}
@keyframes flash {0%{opacity:.30} 50%{opacity:1} 100%{opacity:.30}}
.flash-img {animation: flash 1s ease-in-out 3; border-radius:12px;}
</style>
""", unsafe_allow_html=True)


def _status_text(status: str) -> str:
    return {
        "ABOVE_REFERENCE_RANGE": "por encima del rango observado",
        "BELOW_REFERENCE_RANGE": "por debajo del rango observado",
        "WITHIN_REFERENCE_RANGE": "dentro del rango observado",
    }.get(status, status)


def render_hero(res: dict, ref_tech: str, photo=None) -> None:
    """Tarjeta dominante: técnica detectada + confianza (+ foto de técnica)."""
    bar = prob_bar(res["confidence"])
    st.markdown("### A · Ejecución — ¿qué ocurrió en este movimiento?")
    col_card, col_photo = st.columns([2, 1])
    with col_card:
        st.markdown(
            f"<div class='hero-card'>"
            f"<div style='font-size:.95rem;color:#4a5a6a;'>TÉCNICA DETECTADA</div>"
            f"<div style='font-size:3.2rem;font-weight:800;color:#1f4e79;'>"
            f"{res['predicted_technique']}</div>"
            f"<div style='font-size:1.25rem;color:#4a5a6a;'>{res['technique_name']}</div>"
            f"<div style='font-size:1.8rem;font-weight:700;color:#2e6fb3;'>"
            f"{res['confidence'] * 100:.1f} % confianza</div>"
            f"<div style='font-family:monospace;font-size:1rem;color:#1f4e79;'>"
            f"{bar}</div></div>",
            unsafe_allow_html=True)
        st.caption(f"El modelo identifica esta ejecución como "
                   f"{res['predicted_technique']}.")
        if res["predicted_technique"] == ref_tech:
            st.success("La predicción **coincide** con la técnica observada.")
        else:
            st.info("La predicción **difiere** de la técnica observada "
                    "(resultado válido del demo).")
    if photo is not None:
        with col_photo:
            st.markdown(
                f"<img src=\"{data_uri(photo)}\" class=\"flash-img\" "
                f"style=\"width:100%;\"/>", unsafe_allow_html=True)
            st.caption(f"Técnica observada: {ref_tech}")


def render_probabilities(res: dict) -> None:
    st.markdown("### Probabilidades (secundario)")
    for c in CLASSES:
        v = res["probabilities"][c]
        st.markdown(f"{c} `{prob_bar(v)}` {v * 100:5.1f} %")
    st.caption("Identificación probabilística de la técnica (clasificador).")


def render_profile(res: dict) -> None:
    st.markdown("### Perfil biomecánico de la ejecución")
    cols = st.columns(4)
    for idx, (group_title, items) in enumerate(GROUPS):
        with cols[idx % 4]:
            st.markdown(f"**{group_title}**")
            for f, label, unit in items:
                st.markdown(metric_card_html(label, format_metric(res["features"][f], unit)),
                            unsafe_allow_html=True)


def _render_comparison(comp: list) -> None:
    key_features = ["vmax", "vmean", "amax", "duration_s", "hip_rom",
                    "knee_rom", "ankle_rom"]
    by_feat = {c["feature"]: c for c in comp}
    for group_title, items in GROUPS:
        feats = [f for f, _, _ in items
                 if f in by_feat and f in key_features]
        if not feats:
            continue
        st.markdown(f"**{group_title}**")
        cards = [(by_feat[f], FEATURE_UNITS.get(f, "")) for f in feats]
        for i in range(0, len(cards), 2):
            cols = st.columns(2)
            for j in range(2):
                if i + j < len(cards):
                    item, unit = cards[i + j]
                    with cols[j]:
                        st.markdown(comparison_card_html(item, unit),
                                    unsafe_allow_html=True)


def render_observed(outside: list) -> None:
    if not outside:
        st.caption("Sin diferencias destacadas respecto al rango observado.")
        return
    st.markdown("### Diferencias observadas")
    for o in outside:
        if o["difference_pct"] is not None:
            st.markdown(f"- **{o['label']}** "
                        f"({o['difference_pct']:+.0f} % vs. mediana observada)")
        else:
            st.markdown(f"- **{o['label']}** "
                        f"(diferencia absoluta {abs(o['difference_abs']):.2f})")


def render_coach(insights: list) -> None:
    if not insights:
        return
    st.markdown("### Observaciones para el entrenador")
    for it in insights:
        st.markdown(f"- **Observación:** {it['observation']}")
        st.caption(f"**Posible área de observación:** {it['coach_review']}")


def render_execution_compare(feats: dict, sel_exec: str, row: pd.Series):
    with st.expander("Comparar con otra ejecución (misma técnica)"):
        same = [e for e in executions_for(str(row["athlete_id"]))
                if e != sel_exec and str(execution_row(e)["technique"]) ==
                str(row["technique"])]
        if not same:
            st.caption("No hay otra ejecución de la misma técnica para este atleta.")
            return
        other = st.selectbox("Otra ejecución", same)
        diffs = compare_executions(feats, features_of(execution_row(other)))
        st.markdown(f"**{sel_exec}** vs **{other}**")
        st.markdown("| Variable | Actual | Otra ejecución | Diferencia |")
        st.markdown("|---|---|---|---|")
        for d in diffs:
            av, bv = d["execution_a"], d["execution_b"]
            pct = (f" ({d['difference_pct']:+.0f} %)"
                   if d["difference_pct"] is not None else "")
            st.markdown(f"| {d['label']} | {av:.3f} | {bv:.3f} | "
                        f"{d['difference_abs']:+.3f}{pct} |")
        st.caption("Se observan diferencias descriptivas entre las dos "
                   "ejecuciones; no implica mejora ni empeoramiento.")


def render_details(res: dict, df: pd.DataFrame, ref_tech: str) -> None:
    info = model_info()
    with st.expander("Detalles del modelo"):
        st.markdown("**Qué ve el modelo** — la predicción se genera a partir "
                    "de características biomecánicas de ejecución extraídas "
                    "del movimiento.")
        st.markdown(" · ".join(MODEL_INPUT_NAMES.values()))
        st.markdown("**Modelo:** Random Forest (baseline demo)")
        st.markdown(f"**Versión del artefacto:** {info.get('model_version', 'n/d')}")
        st.markdown(f"**Versión de inferencia:** {info.get('inference_version', 'n/d')}")
        st.markdown(f"**Features:** {len(info.get('features', []))}"
                    f" · **Clases:** {', '.join(info.get('classes', []))}")
        st.markdown("**Dataset:** ML Dataset v0 (419 ejecuciones) · "
                    f"técnica de referencia {ref_tech}")
        st.markdown("**Validación:** GroupKFold por atleta (resultados "
                    "descriptivos, no rendimiento deportivo)")


def main():
    df = load_v0()

    # ---- Header / identidad ----
    st.markdown(hero_title_html(
        "KARATE PERFORMANCE INTELLIGENCE",
        "Análisis biomecánico asistido por inteligencia artificial"
        + demo_badge_html()), unsafe_allow_html=True)
    st.markdown(flow_bar_html(
        ["A · EJECUCIÓN", "B · ANÁLISIS", "C · COMPARACIÓN", "D · ENTRENADOR"]),
        unsafe_allow_html=True)
    st.markdown("---")

    # ---- Selector compacto (info izquierda + GIF derecha) ----
    athletes = available_athletes()
    col_sel, col_gif = st.columns([1, 1])
    with col_sel:
        sel_athlete = st.selectbox("Atleta", athletes, key="ath")
        execs = executions_for(sel_athlete)
        default = default_execution() if sel_athlete else None
        if default not in execs:
            default = execs[0] if execs else None
        sel_exec = st.selectbox("Ejecución", execs,
                                index=execs.index(default) if default in execs else 0,
                                format_func=lambda e: e, key="exec")
        row = execution_row(sel_exec)
        ref_tech = str(row["technique"])
        im1, im2 = st.columns(2)
        im1.metric("Técnica observada", ref_tech)
        im2.metric("Condición", "E01 · T01")
        st.caption(f"Ejecución: {row['execution_id']}")
    with col_gif:
        gif = gallery_gif_for(str(row["athlete_id"]), ref_tech)
        if gif is not None:
            play = st.checkbox("▶ Reproducir animación", value=False)
            if play:
                st.image(str(gif), width=300,
                         caption="Wireframe de la ejecución (animado)")
            else:
                thumb = gallery_thumbnail(str(row["athlete_id"]), ref_tech)
                st.image(thumb, width=300, caption="Wireframe de la ejecución")
            st.caption("Vista estática por defecto; activa la animación para "
                       "ver el movimiento bajo demanda.")
        else:
            st.info("Sin wireframe disponible para este atleta/técnica.")

    feats = features_of(row)
    try:
        res = predict_execution(feats)
    except InferenceError as e:
        st.error(f"Error de inferencia: {e}")
        return

    # ---- A · EJECUCIÓN (hero) ----
    render_hero(res, ref_tech,
                photo=technique_image_path(ref_tech) if ref_tech else None)

    # ---- B · ANÁLISIS ----
    st.markdown("---")
    st.markdown("### B · Análisis — ¿qué observa el modelo?")
    col_prob, col_prof = st.columns([1, 2])
    with col_prob:
        render_probabilities(res)
    with col_prof:
        render_profile(res)

    # ---- C · COMPARACIÓN ----
    st.markdown("---")
    st.markdown("### C · Comparación — ¿cómo se compara con la referencia?")
    profiles = reference_profile(df)
    ref = profiles.get(ref_tech, {})
    comp = compare_to_reference(res["features"], ref) if ref else []
    st.caption(f"Perfil observado en el dataset del demo para la técnica "
               f"{ref_tech} (mediana y rango central Q1–Q3). No es un estándar "
               f"normativo ni óptimo.")
    _render_comparison(comp)
    render_observed(observed_differences(comp))
    render_execution_compare(feats, sel_exec, row)

    # ---- D · ENTRENADOR ----
    st.markdown("---")
    st.markdown("### D · Entrenador — ¿qué aspectos pueden revisarse?")
    render_coach(coach_insights(observed_differences(comp)))
    st.markdown(future_card_html("EVOLUCIÓN DEL ATLETA"), unsafe_allow_html=True)

    # ---- Detalles + disclaimer ----
    st.markdown("---")
    render_details(res, df, ref_tech)
    st.caption("**Demo tecnológica.** Las predicciones y comparaciones son "
               "descriptivas y no constituyen diagnóstico médico, evaluación "
               "definitiva de técnica ni predicción de rendimiento "
               "competitivo.")


if __name__ == "__main__":
    main()