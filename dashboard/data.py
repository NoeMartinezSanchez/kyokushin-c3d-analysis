# -*- coding: utf-8 -*-
"""
data.py — Capa de datos del dashboard (Fase 1.8D).

ÚNICA fuente de datos: output/data_mart/athlete_execution_features.csv.

Responsabilidades:
  1. Leer el Data Mart (sin modificar el archivo).
  2. Validar esquema (columnas esperadas, no vacío, filas).
  3. Parseo numérico correcto de campos numéricos.
  4. Exponer los metadatos de comparabilidad YA presentes en el Data Mart
     (no se recalcula comparabilidad aquí).
  5. Proveer filtros por atleta/técnica/condición/trial.

REGLAS DEL PROYECTO:
  - Este módulo NO accede a C3D (no importa ezc3d).
  - NO recalcula features; solo consume el Data Mart.
  - NO modifica output/data_mart/athlete_execution_features.csv.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_MART_PATH = ROOT / "output" / "data_mart" / "athlete_execution_features.csv"

# Columnas mínimas que el dashboard consume (subconjunto del contrato).
REQUIRED_COLUMNS = [
    "athlete_id", "execution_id", "technique", "condition", "trial", "repetition",
    "sampling_rate_hz", "primary_signal", "movement_side", "event_id",
    "duration_s", "time_to_peak_s",
    "vmax", "vmean", "amax", "displacement", "path_length",
    "hip_rom", "knee_rom", "ankle_rom",
    "snr", "qc_status", "quality_flag",
    "comparability_duration_s", "comparability_time_to_peak_s",
    "comparability_vmax", "comparability_vmean", "comparability_amax",
    "comparability_displacement", "comparability_path_length",
    "comparability_hip_rom", "comparability_knee_rom", "comparability_ankle_rom",
    "comparability_snr",
]

NUMERIC_COLUMNS = [
    "duration_s", "time_to_peak_s", "vmax", "vmean", "amax", "displacement",
    "path_length", "hip_rom", "knee_rom", "ankle_rom", "snr",
]


def load_data_mart() -> pd.DataFrame:
    """Lee el Data Mart y valida el contrato. NO modifica el archivo."""
    df = pd.read_csv(DATA_MART_PATH)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Data Mart sin columnas requeridas: {missing}")
    if df.empty:
        raise ValueError("Data Mart vacío")
    for c in NUMERIC_COLUMNS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def validate_data_mart(df: pd.DataFrame) -> list[str]:
    """Devuelve lista de problemas (vacía si está todo OK)."""
    issues = []
    if df.empty:
        issues.append("Data Mart vacío")
    if not df["execution_id"].is_unique:
        issues.append("execution_id duplicado")
    for c in REQUIRED_COLUMNS:
        if c not in df.columns:
            issues.append(f"falta columna {c}")
    return issues


def comparability_status(df: pd.DataFrame) -> list[str]:
    """Estados de comparabilidad únicos presentes en el Data Mart (no recalculados)."""
    cols = [c for c in df.columns if c.startswith("comparability_")]
    unique = sorted({str(v) for c in cols for v in df[c].dropna().unique()})
    return unique


def filter_by(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Filtro seguro por columnas (atleta/técnica/condición/trial)."""
    out = df.copy()
    for col, val in kwargs.items():
        if col not in out.columns:
            continue
        if val is None:
            continue
        out = out[out[col] == val]
    return out


def available_options(df: pd.DataFrame, col: str) -> list:
    return sorted(df[col].dropna().unique().tolist())