# -*- coding: utf-8 -*-
"""
performance_data.py — Capa de DATOS del Athlete Performance Dashboard (Task 10).

Carga la lista de ejecuciones desde ML Dataset v0 y extrae las 10 features
para la inferencia. NO contiene lógica de ML: la única entrada al modelo es
`inference.predict_execution`.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

from inference.schemas import FEATURES  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
V0_FILE = ROOT / "output" / "ml_dataset_v0" / "ml_dataset_v0.csv"


@lru_cache(maxsize=1)
def load_v0() -> pd.DataFrame:
    """ML Dataset v0 (419 ejecuciones). Sola lectura, inmutable."""
    return pd.read_csv(V0_FILE)


def available_athletes() -> list[str]:
    """Atletas disponibles, ordenados."""
    return sorted(load_v0()["athlete_id"].unique().tolist())


def executions_for(athlete: str) -> list[str]:
    """Execution ids reales del atleta."""
    df = load_v0()
    return sorted(df.loc[df["athlete_id"] == athlete, "execution_id"].tolist())


def all_execution_ids() -> list[str]:
    return sorted(load_v0()["execution_id"].tolist())


def execution_row(execution_id: str) -> pd.Series:
    """Fila real de la ejecución (con técnica de referencia)."""
    df = load_v0()
    hits = df[df["execution_id"] == execution_id]
    if hits.empty:
        raise KeyError(f"execution_id no existe en el dataset: {execution_id}")
    return hits.iloc[0]


def features_of(row: pd.Series) -> dict:
    """Extrae exactamente las 10 features del contrato de inferencia."""
    return {f: float(row[f]) for f in FEATURES}


def default_execution() -> str:
    """Ejecución inicial: una ejecución real S04 (la usada en el demo de Task 9)."""
    df = load_v0()
    hits = df[df["technique"] == "S04"]
    if hits.empty:
        raise KeyError("no hay ejecuciones S04 en el dataset")
    return str(hits.iloc[0]["execution_id"])


# --------------------------------------------------------------------------- #
# Galería de wireframes (output/gallery/) y fotos de técnica (images/)
# --------------------------------------------------------------------------- #

GALLERY_DIR = ROOT / "output" / "gallery"
IMAGES_DIR = ROOT / "images"


def gallery_gif_for(athlete: str, technique: str) -> Path | None:
    """GIF wireframe del atleta×técnica (solo GIF exacto; None si no existe)."""
    p = GALLERY_DIR / f"athlete_{athlete}_{technique}.gif"
    return p if p.exists() else None


@lru_cache(maxsize=256)
def gallery_thumbnail(athlete: str, technique: str):
    """Primer frame del GIF como imagen estática (PIL), cacheado (bajo consumo)."""
    from PIL import Image
    p = gallery_gif_for(athlete, technique)
    if p is None:
        return None
    with Image.open(p) as im:
        return im.convert("RGB").copy()


TECHNIQUE_IMAGE_BASE = {
    "S01": "technique_S01_gyaku_zuki",
    "S02": "technique_S02_mae_geri",
    "S03": "technique_S03_mawashi_gedan",
    "S04": "technique_S04_mawashi_jodan",
    "S05": "technique_S05_ushiro_mawashi",
}


def technique_image_path(technique: str) -> Path | None:
    """Foto ilustrativa de la técnica: images/<base técnico>.{jpg,png}."""
    base = TECHNIQUE_IMAGE_BASE.get(technique, f"technique_{technique}")
    for ext in (".jpg", ".png"):
        p = IMAGES_DIR / f"{base}{ext}"
        if p.exists():
            return p
    return None