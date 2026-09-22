# -*- coding: utf-8 -*-
"""
app_ml.py — Vista ML del Dashboard (Task 7B).

Página SEPARADA de presentación de resultados del experimento ML baseline
(output/ml_results/). NO entrena, NO ejecuta CV, NO recalcula features, NO lee
C3D y NO modifica archivos: consume exclusivamente artefactos ya calculados.

Lanzamiento:  streamlit run dashboard/app_ml.py
Ver también:  streamlit run dashboard/app.py  (dashboard clásico, intacto)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import load_ml_results, load_ml_config  # noqa: E402

CLASSES = ["S02", "S03", "S04", "S05"]
KEY_METRICS = ["accuracy", "balanced_accuracy", "f1_macro", "f1_weighted"]


def main():
    st.set_page_config(page_title="ML — Clasificación de técnicas",
                       page_icon="🥋", layout="wide")
    st.title("ML · Clasificación de técnicas (baseline)")
    st.caption("TASK 7B — Resultados del experimento `TASK7B_BASELINE_001` "
               "(validación agrupada por atleta).")

    ml = load_ml_results()
    cfg = load_ml_config()
    if not ml:
        st.warning("No hay resultados de ML todavía. Ejecuta: "
                   "`.venv\\Scripts\\python scripts\\16_task7b_baseline_ml.py`")
        return

    # ------------------------------------------------------------------ info
    st.subheader("Información metodológica")
    inv = cfg.get("population", {})
    st.markdown(
        "- **Dataset:** ML Dataset v0 — población 250 Hz / E01 / T01 / S02–S05\n"
        f"- **Observaciones:** {inv.get('rows','n/d')} · "
        f"**atletas:** {inv.get('athletes','n/d')} · "
        f"**técnicas:** {', '.join(inv.get('techniques', []))}\n"
        f"- **Unidad:** 1 ejecución · **grupo de validación:** {cfg.get('group','n/d')}\n"
        f"- **Método:** {cfg.get('cv',{}).get('name','n/d')}, "
        f"{cfg.get('cv',{}).get('n_splits','n/d')} folds · "
        f"**modelo:** {cfg.get('model',{}).get('name','n/d')} "
        f"(`{cfg.get('model',{}).get('solver','')}`, default multinomial)\n"
        f"- **Preprocesamiento:** {cfg.get('scaler',{}).get('name','n/d')} "
        f"ajustado dentro de cada fold (sin leakage)\n"
        f"- **Experimentos:** A = con SNR · B = sin SNR (mismos folds)\n"
        "- Lenguaje descriptivo: las métricas describen el clasificador, "
        "no califican a atletas ni técnicas.")

    # ------------------------------------------------------------------ view
    bc = ml.get("baseline_comparison")
    tab_over, tab_cm, tab_cls, tab_feat, tab_fold, tab_err, tab_ath = \
        st.tabs(["Overview", "Matriz de confusión", "Métricas por técnica",
                 "Features", "Folds", "Errores", "Atletas"])

    exp = st.selectbox("Experimento", ["A", "B"], key="exp_sel",
                       help="A = con SNR · B = sin SNR")

    # Overview
    with tab_over:
        if bc is not None:
            sub = bc[bc["experiment"] == exp]
            cols = st.columns(len(KEY_METRICS))
            for col, m in zip(cols, KEY_METRICS):
                row = sub[sub["metric"] == m].iloc[0]
                col.metric(m, f"{row['mean']:.3f} ± {row['std']:.3f}",
                           help=f"min {row['min']:.3f} · max {row['max']:.3f}")
            st.caption("Media ± desviación sobre 5 folds (GroupKFold por atleta).")

    # Confusion
    with tab_cm:
        cm = ml.get("confusion_matrix")
        if cm is not None:
            sub = cm[cm["experiment"] == exp]
            mat = _matrix(sub, "count")
            st.plotly_chart(_heatmap(mat, "Matriz de confusión (conteos)"),
                            width='stretch')
            norm = _matrix(sub, "normalized_recall")
            st.plotly_chart(_heatmap(norm, "Matriz normalizada por clase real",
                                     vmax=1.0), width='stretch')
            st.caption("Real → Eje y · Predicho → Eje x. Solo patrones observados.")

    # Metrics by class
    with tab_cls:
        mcl = ml.get("metrics_by_class")
        if mcl is not None:
            sub = mcl[mcl["experiment"] == exp]
            fig = px.bar(sub, x="technique", y=["precision", "recall", "f1"],
                         barmode="group",
                         title="Métricas por técnica (OOF pooled) — diagnóstico")
            fig.update_layout(yaxis_range=[0, 1])
            st.plotly_chart(fig, width='stretch')
            st.dataframe(sub)

    # Features
    with tab_feat:
        cs = ml.get("feature_coefficient_summary")
        if cs is not None:
            sub = cs[cs["experiment"] == exp]
            fig = go.Figure(go.Bar(
                x=sub["mean_abs_coefficient"], y=sub["feature"],
                orientation="h", error_x=dict(type="data",
                                              array=sub["std_abs_coefficient"]),
                name="|coef| medio"))
            fig.update_layout(title="Coeficientes de Logistic Regression "
                                    "(|abs| medio ± desviación entre folds)",
                              height=420, yaxis={'autorange': 'reversed'})
            st.plotly_chart(fig, width='stretch')
            st.warning("Los coeficientes corresponden a variables escaladas y "
                       "describen asociación dentro del modelo; no implican "
                       "causalidad biomecánica.")

    # Folds
    with tab_fold:
        fm = ml.get("fold_metrics")
        if fm is not None:
            sub = fm[fm["experiment"] == exp]
            st.dataframe(sub[["fold", "n_train", "n_validation",
                              "n_train_athletes", "n_validation_athletes",
                              "accuracy", "balanced_accuracy", "f1_macro"]])

    # Errors
    with tab_err:
        er = ml.get("classification_errors")
        if er is not None:
            sub = er[er["experiment"] == exp]
            c1, c2, c3 = st.columns(3)
            t_true = c1.multiselect("Técnica real", CLASSES, default=CLASSES)
            t_pred = c2.multiselect("Técnica predicha", CLASSES, default=CLASSES)
            athletes = sorted(sub["athlete_id"].dropna().unique())
            a_sel = c3.multiselect("Atleta", athletes, default=[])
            f = sub[(sub["true_technique"].isin(t_true))
                    & (sub["predicted_technique"].isin(t_pred))]
            if a_sel:
                f = f[f["athlete_id"].isin(a_sel)]
            st.write(f"Errores: **{len(f)}** de {len(sub)} predicciones "
                     f"({exp})")
            st.dataframe(f[["execution_id", "athlete_id", "true_technique",
                            "predicted_technique", "fold",
                            "probability_predicted"]])

    # Athletes
    with tab_ath:
        ma = ml.get("metrics_by_athlete")
        if ma is not None:
            sub = ma[ma["experiment"] == exp]
            st.caption("Variabilidad de generalización por atleta (OOF). "
                       "No se ordena ni califica.")
            st.dataframe(sub[["athlete_id", "total", "correct", "incorrect",
                              "accuracy", "techniques_present"]])


def _matrix(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    mat = pd.DataFrame(0.0, index=CLASSES, columns=CLASSES)
    for _, r in df.iterrows():
        mat.loc[r["true"], r["predicted"]] = r[value_col]
    return mat


def _heatmap(mat, title: str, vmax=None):
    fig = px.imshow(mat.values, x=CLASSES, y=CLASSES,
                    labels=dict(x="Predicho", y="Real", color=""),
                    color_continuous_scale="Blues", zmin=0,
                    zmax=vmax if vmax is not None else None,
                    text_auto=".2f" if vmax else True)
    fig.update_layout(title=title, width=520, height=440)
    return fig


if __name__ == "__main__":
    main()