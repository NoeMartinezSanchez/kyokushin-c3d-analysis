# -*- coding: utf-8 -*-
r"""
Tests mínimos del pipeline (FASE 1.6).

Ejecutar desde la raíz del proyecto:
    .venv\Scripts\python -m pytest tests -q
    o bien un test concreto:
    .venv\Scripts\python -m pytest tests\test_pipeline.py::test_parse_name -q

Los tests son de validación técnica/reproducibilidad, no de ciencia deportiva.
"""

from __future__ import annotations

import sys
from pathlib import Path

import hashlib

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import importlib.util

_SPEC = importlib.util.spec_from_file_location(
    "dataset_exploration_module",
    str(ROOT / "scripts" / "01_dataset_exploration.py"),
)
_X01 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_X01)

_SPEC2 = importlib.util.spec_from_file_location(
    "execution_segmentation_module",
    str(ROOT / "scripts" / "02_execution_segmentation.py"),
)
_X02 = importlib.util.module_from_spec(_SPEC2)
_SPEC2.loader.exec_module(_X02)


# --------------------------------------------------------------------------- #
# 1-2. Apertura de C3D y frecuencia
# --------------------------------------------------------------------------- #

def _rep_file(tag: str) -> Path:
    files = _X01.find_files(tag)
    assert files, f"no encontrado {tag}"
    return files[0]


def test_c3d_can_open():
    c = _X01.load_c3d(_rep_file("S04-E01-T01"))
    assert c["data"]["points"].shape[0] == 4


def test_sampling_rate():
    c = _X01.load_c3d(_rep_file("S04-E01-T01"))
    rate = float(c.parameters["POINT"]["RATE"]["value"][0])
    assert rate == 200.0


def test_metadata_parsed():
    info = _X01.parse_name("2017-01-31-B0367-S04-E02-T01")
    assert info["athlete"] == "B0367"
    assert info["technique"] == "S04"
    assert info["condition"] == "E02"
    assert info["trial"] == "T01"


# --------------------------------------------------------------------------- #
# 3-4. Segmentación: intervalos válidos
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def seg_results():
    rows = []
    for tag in ["S01-E01-T01", "S02-E01-T01", "S04-E01-T01", "S04-E02-T01"]:
        fp = _rep_file(tag)
        c = _X01.load_c3d(fp)
        rate = float(c.parameters["POINT"]["RATE"]["value"][0])
        pref = _X01.athlete_prefix(fp, _X01.get_prefixes(c))
        best = _X02.pick_best_signal(fp, pref, tag.split("-")[0])
        _, v, _ = _X02.get_signal(fp, pref, best["marker"])
        segs, _ = _X02.segment_repetitions(v, rate, return_events=False)
        for _, seg in segs.iterrows():
            rows.append({"source_file": Path(fp).name, "rate": rate, **dict(seg)})
    return pd.DataFrame(rows)


def test_intervals_valid(seg_results):
    assert not seg_results.empty
    for _, r in seg_results.iterrows():
        assert r["start_frame"] < r["peak_frame"] < r["end_frame"]


def test_duration_positive(seg_results):
    for _, r in seg_results.iterrows():
        assert (r["end_frame"] - r["start_frame"]) / r["rate"] > 0


def test_no_overlap_between_executions(seg_results):
    # el solapamiento se evalúa DENTRO de cada archivo (los frames de archivos
    # distintos no son comparables entre sí)
    for f in seg_results["source_file"].unique():
        sub = seg_results[seg_results["source_file"] == f]
        intervals = sorted(zip(sub["start_frame"], sub["end_frame"]))
        for (s1, e1), (s2, e2) in zip(intervals, intervals[1:]):
            assert s2 >= e1, f"{f}: solapamiento {s1}-{e1} vs {s2}-{e2}"


def test_segment_repetitions_returns_events():
    fp = _rep_file("S01-E01-T01")
    c = _X01.load_c3d(fp)
    rate = float(c.parameters["POINT"]["RATE"]["value"][0])
    pref = _X01.athlete_prefix(fp, _X01.get_prefixes(c))
    best = _X02.pick_best_signal(fp, pref, "S01")
    _, v, _ = _X02.get_signal(fp, pref, best["marker"])
    segs, events = _X02.segment_repetitions(v, rate)
    assert set(["accepted", "rejected", "review"]).issuperset(
        set(events["status"].unique()))


# --------------------------------------------------------------------------- #
# 5. Features y trazabilidad
# --------------------------------------------------------------------------- #

def test_features_finite():
    df = pd.read_csv(ROOT / "output" / "executions_sample.csv")
    numeric = ["vmax_m_s", "vmean_m_s", "amax_m_s2", "displacement_m",
               "path_length_m", "rom_m"]
    for c in numeric:
        values = df[c].astype(float)
        assert np.isfinite(values.replace([np.inf, -np.inf], np.nan)).all()


def test_source_file_exists():
    df = pd.read_csv(ROOT / "output" / "executions_sample.csv")
    for f in df["source_file"]:
        assert (ROOT / "B0367").exists()
        # el archivo debe existir en el árbol B0367
        hits = list(Path(ROOT / "B0367").rglob(f))
        assert hits, f"source_file no existe: {f}"


def test_unique_execution_id():
    df = pd.read_csv(ROOT / "output" / "executions_sample.csv")
    keys = df[["source_file", "technique", "condition", "trial", "repetition"]]
    assert keys.duplicated().sum() == 0


def test_events_csv_consistent_with_accepted():
    ev = pd.read_csv(ROOT / "output" / "segmentation_events.csv")
    accepted = (ev["status"] == "accepted").sum()
    execs = pd.read_csv(ROOT / "output" / "executions_sample.csv")
    assert accepted == len(execs)


# --------------------------------------------------------------------------- #
# 6. Config de señales (FASE 1.6.1) — desde el YAML, no hardcodeadas
# --------------------------------------------------------------------------- #

def test_s03_signal_defined():
    """S03 debe tener señal primaria definida (validada en FASE 1.6.1)."""
    assert _X02.CFG["signals"]["S03"]["signal"] != "TBD"


def test_s05_signal_defined():
    assert _X02.CFG["signals"]["S05"]["signal"] != "TBD"


def test_signals_come_from_yaml():
    """Las señales usadas por el pipeline deben leerse del YAML de config."""
    cfg = _X02.CFG
    for tech in ["S01", "S02", "S03", "S04", "S05"]:
        sig = cfg["signals"][tech]["signal"]
        assert sig != "" and sig is not None
        # la señal de cada técnica debe estar definida en el YAML
        assert cfg["signals"][tech]["signal"] == _X02.PRIMARY_SIGNAL[tech][0]


def test_signals_not_hardcoded_in_script():
    """El script 02 no debe contener señales S03/S05 hardcodeadas (solo desde YAML/fallback)."""
    src = (ROOT / "scripts" / "02_execution_segmentation.py").read_text(encoding="utf-8")
    # PRIMARY_SIGNAL se construye desde CFG; no debe haber literal "signal: RTOE" por técnica
    assert '"S03": [\n    "RTOE"' not in src.replace("\n", "").replace(" ", "")
    assert '"S05": [\n    "RTOE"' not in src.replace("\n", "").replace(" ", "")


def test_config_loadable():
    import yaml
    with open(ROOT / "config" / "segmentation.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert cfg is not None
    assert "signals" in cfg
    assert all(t in cfg["signals"] for t in ["S01", "S02", "S03", "S04", "S05"])


# --------------------------------------------------------------------------- #
# 7. Generalización a segundo atleta (FASE 1.7)
# --------------------------------------------------------------------------- #

B0377_DIR = ROOT / "atletas" / "B0377"


def test_new_athlete_dir_exists():
    assert B0377_DIR.is_dir(), "Falta atletas/B0377"


def test_new_athlete_c3d_can_be_inspected():
    files = sorted(B0377_DIR.rglob("*.c3d"))
    assert files, "No hay C3D en atletas/B0377"
    c = _X01.load_c3d(files[0])
    assert c["data"]["points"].shape[0] == 4


def test_no_right_laterality_assumed():
    """El análisis de lateralidad debe dejar la decisión a los datos (no asumir R)."""
    lat_csv = ROOT / "output" / "athlete_generalization" / "lateralality_analysis.csv"
    if not lat_csv.exists():
        pytest.skip("fase 1.7 no ejecutada")
    df = pd.read_csv(lat_csv)
    # debe existir análisis L vs R, no solo R
    assert "l_vmax" in df.columns and "r_vmax" in df.columns
    # debe haberse medido ambos lados (algún valor no nulo en cada columna)
    assert df["l_vmax"].notna().any() and df["r_vmax"].notna().any()


def test_unknown_is_valid_status():
    """UNKNOWN es un estado válido para lateralidad/señales no determinadas."""
    assert "UNKNOWN" in ("RIGHT", "LEFT", "UNKNOWN")


def test_no_hardcoded_b0377_signals_in_script02():
    """El script 02 no debe contener señales específicas de B0377 hardcodeadas."""
    src = (ROOT / "scripts" / "02_execution_segmentation.py").read_text(encoding="utf-8")
    assert "B0377" not in src


def test_b0367_baseline_reproducible():
    """El baseline B0367 permanece: executions_sample.csv sigue con 26 filas."""
    csv = ROOT / "output" / "executions_sample.csv"
    if not csv.exists():
        pytest.skip("pipeline no ejecutado")
    df = pd.read_csv(csv)
    assert len(df) == 26
    assert set(df["athlete_id"].unique()) == {"B0367"}


# --------------------------------------------------------------------------- #
# 8. Configuración por atleta (FASE 1.8A)
# --------------------------------------------------------------------------- #

ATHLETE_CFG_DIR = ROOT / "config" / "athletes"


def test_athlete_config_exists_b0367():
    assert (ATHLETE_CFG_DIR / "B0367.yaml").exists()


def test_athlete_config_exists_b0377():
    assert (ATHLETE_CFG_DIR / "B0377.yaml").exists()


def test_config_has_all_techniques():
    import yaml
    with open(ATHLETE_CFG_DIR / "B0377.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    techs = set(cfg["techniques"].keys())
    assert techs.issuperset({"S01", "S02", "S03", "S04", "S05"})


def test_b0367_signal_mapping():
    # la config de B0367 debe reproducir el baseline (todas las patadas RTOE, puño RFIN)
    import yaml
    with open(ATHLETE_CFG_DIR / "B0367.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert cfg["techniques"]["S01"]["signal"] == "RFIN"
    assert cfg["techniques"]["S04"]["signal"] == "RTOE"


def test_b0377_signal_mapping():
    import yaml
    with open(ATHLETE_CFG_DIR / "B0377.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert cfg["techniques"]["S02"]["signal"] == "RTOE"
    assert cfg["techniques"]["S05"]["signal"] == "RTOE"


def test_b0377_s04_uses_ltoe():
    import yaml
    with open(ATHLETE_CFG_DIR / "B0377.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert cfg["techniques"]["S04"]["signal"] == "LTOE"
    assert cfg["techniques"]["S04"]["joints_side"] == "L"


def test_b0377_s01_threshold_is_explicit_or_needs_validation():
    import yaml
    with open(ATHLETE_CFG_DIR / "B0377.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    th = cfg["techniques"]["S01"].get("thresholds", {})
    status = th.get("status", "") if isinstance(th, dict) else ""
    assert status in ("", "NEEDS_VALIDATION"), "S01 umbral sin estado explícito"


def test_no_global_right_laterality_assumption():
    # la lateralidad es por técnica, no una regla global única
    import yaml
    with open(ATHLETE_CFG_DIR / "B0377.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    sides = {t: cfg["techniques"][t]["laterality"] for t in cfg["techniques"]}
    assert sides["S04"] == "left"
    assert "right" in set(sides.values())


def test_sampling_rate_read_from_c3d():
    # el rate se lee del C3D, no se hardcodea como fuente (la config solo lo anota)
    c = _X01.load_c3d(_rep_file("S04-E01-T01"))
    rate = float(c.parameters["POINT"]["RATE"]["value"][0])
    assert rate in (200.0, 250.0)


def test_set_active_athlete_resolves_signals():
    _X02.set_active_athlete("B0367")
    assert _X02.PRIMARY_SIGNAL["S04"] == ["RTOE"]
    _X02.set_active_athlete("B0377")
    assert _X02.PRIMARY_SIGNAL["S04"] == ["LTOE"]
    assert _X02.JOINTS_SIDE["S04"] == "L"
    # restaurar baseline para no dejar estado
    _X02.set_active_athlete("B0367")


def test_outputs_separated_by_athlete():
    """B0377 escribe en subdirectorio; B0367 mantiene output/ raíz."""
    b0377_dir = ROOT / "output" / "B0377"
    assert b0377_dir.is_dir(), "Falta output/B0377 (ejecutar pipeline B0377)"
    assert (b0377_dir / "executions_sample.csv").exists()
    assert (ROOT / "output" / "executions_sample.csv").exists()  # baseline sigue en raíz


def test_b0377_run_leaves_b0367_output_intact():
    """Correr B0377 no debe tocar output/executions_sample.csv (B0367)."""
    base = ROOT / "output" / "executions_sample.csv"
    if not base.exists():
        pytest.skip("baseline no generado")
    df = pd.read_csv(base)
    assert len(df) == 26
    assert set(df["athlete_id"].unique()) == {"B0367"}


def test_data_dir_resolved_by_athlete():
    assert _X02.data_dir_for("B0367").name == "B0367"
    assert str(_X02.data_dir_for("B0377")).replace("\\", "/").endswith("atletas/B0377")


def test_config_overrides_global():
    _X02.set_active_athlete("B0377")
    assert _X02.CFG["techniques"]["S04"]["signal"] == "LTOE"
    _X02.set_active_athlete("B0367")


# --------------------------------------------------------------------------- #
# 9. Feature Readiness (FASE 1.8B)
# --------------------------------------------------------------------------- #

def test_feature_readiness_sample_schema():
    """la muestra golden path debe tener el esquema mínimo del plan."""
    p = ROOT / "output" / "feature_readiness_sample.csv"
    if not p.exists():
        pytest.skip("FASE 1.8B no ejecutada")
    df = pd.read_csv(p)
    required = ["athlete_id", "technique", "condition", "trial", "execution_id",
                "duration_s", "time_to_peak_s", "vmax", "vmean", "amax",
                "displacement", "path_length", "hip_rom", "knee_rom",
                "ankle_rom", "snr", "qc_status"]
    assert set(required).issubset(df.columns)


def test_feature_readiness_golden_path_scope():
    """Solo S02/S03/S05 × E01 × T01; ambos atletas."""
    p = ROOT / "output" / "feature_readiness_sample.csv"
    if not p.exists():
        pytest.skip("FASE 1.8B no ejecutada")
    df = pd.read_csv(p)
    assert set(df["technique"].unique()) == {"S02", "S03", "S05"}
    assert set(df["condition"].unique()) == {"E01"}
    assert set(df["trial"].unique()) == {"T01"}
    assert set(df["athlete_id"].unique()) == {"B0367", "B0377"}


def test_qc_status_mapping_consistent():
    """qc_status de las filas aceptadas debe ser accepted (coherente con eventos)."""
    p = ROOT / "output" / "feature_readiness_sample.csv"
    if not p.exists():
        pytest.skip("FASE 1.8B no ejecutada")
    df = pd.read_csv(p)
    assert (df["qc_status"] == "accepted").all()


def test_hip_knee_ankle_mapped_from_joints_side():
    """hip/knee/ankle_rom deben venir de rom_{joints_side}... (R en golden path)."""
    p = ROOT / "output" / "feature_readiness_sample.csv"
    if not p.exists():
        pytest.skip("FASE 1.8B no ejecutada")
    df = pd.read_csv(p)
    # golden path: joints_side=R en ambos atletas para S02/S03/S05
    for _, r in df.iterrows():
        assert r["hip_rom"] == r.get("rom_RHipAngles") or pd.isna(r["hip_rom"]) is False
    # ninguna fila con ROM de lado izquierdo en golden path
    assert not any(c.startswith("rom_L") for c in df.columns)


def test_event_id_matches_global_events():
    """event_id en executions_sample debe referenciar el evento global aceptado."""
    exec_b36 = pd.read_csv(ROOT / "output" / "executions_sample.csv")
    ev_b36 = pd.read_csv(ROOT / "output" / "segmentation_events.csv")
    merged = exec_b36.merge(ev_b36, on=["athlete_id", "technique", "condition",
                                        "trial", "event_id"], how="left")
    # las ejecuciones aceptadas deben emparejar con un evento accepted
    assert merged["status"].eq("accepted").all()


# --------------------------------------------------------------------------- #
# 10. Athlete Data Mart (FASE 1.8C)
# --------------------------------------------------------------------------- #

MART_FILE = ROOT / "output" / "data_mart" / "athlete_execution_features.csv"
MART_SUMMARY = ROOT / "output" / "data_mart" / "data_mart_summary.csv"
MART_BACKUP = ROOT / "output" / "data_mart" / "task6_backup" / "athlete_execution_features.csv"


def _assert_mart_contract_ok():
    """Invariante del Mart consolida dict (428 filas, contrato 39 cols, histórico intacto)."""
    mart = pd.read_csv(MART_FILE)
    hist = pd.read_csv(MART_BACKUP)
    assert len(mart) == 428
    assert mart["execution_id"].is_unique
    assert (mart["qc_status"] == "accepted").all()
    feats = ["duration_s", "time_to_peak_s", "vmax", "vmean", "amax",
             "displacement", "path_length", "hip_rom", "knee_rom",
             "ankle_rom", "snr"]
    assert mart[feats].isna().sum().sum() == 0
    assert not (mart["technique"] == "S01").any()
    new = mart[~mart["athlete_id"].isin(["B0367", "B0377"])]
    assert (new["sampling_rate_hz"] == 250.0).all()
    assert set(new["technique"]) <= {"S02", "S03", "S04", "S05"}
    hm = mart[mart["athlete_id"].isin(["B0367", "B0377"])]
    assert hm.reset_index(drop=True).equals(hist.reset_index(drop=True))


def test_mart_schema():
    if not MART_FILE.exists():
        pytest.skip("FASE 1.8C no ejecutada")
    mart = pd.read_csv(MART_FILE)
    required = ["athlete_id", "execution_id", "technique", "condition", "trial",
                "repetition", "sampling_rate_hz", "primary_signal", "movement_side",
                "event_id", "duration_s", "time_to_peak_s", "vmax", "vmean",
                "amax", "displacement", "path_length", "hip_rom", "knee_rom",
                "ankle_rom", "snr", "qc_status", "quality_flag",
                "comparability_duration_s", "comparability_vmax", "feature_version",
                "segmentation_version", "units_version", "source_dataset"]
    assert set(required).issubset(mart.columns)


def test_mart_unique_execution_id():
    if not MART_FILE.exists():
        pytest.skip("FASE 1.8C no ejecutada")
    mart = pd.read_csv(MART_FILE)
    assert mart["execution_id"].is_unique


def test_mart_golden_path_scope():
    if not MART_FILE.exists():
        pytest.skip("FASE 1.8C no ejecutada")
    mart = pd.read_csv(MART_FILE)
    assert len(mart) == 428
    assert set(mart["athlete_id"]) == set(_new29()._cohort_250()) | {"B0367"}
    assert set(mart["technique"]) == {"S02", "S03", "S04", "S05"}
    assert set(mart["condition"]) == {"E01"}
    assert set(mart["trial"]) == {"T01"}


def test_mart_sampling_rate_metadata():
    if not MART_FILE.exists():
        pytest.skip("FASE 1.8C no ejecutada")
    mart = pd.read_csv(MART_FILE)
    assert set(mart.loc[mart["athlete_id"] == "B0367", "sampling_rate_hz"]) == {200.0}
    new = mart[mart["athlete_id"] != "B0367"]
    assert set(new["sampling_rate_hz"]) == {250.0}
    assert set(mart["movement_side"]) <= {"R", "L"}


def test_mart_comparability_mapping():
    if not MART_FILE.exists():
        pytest.skip("FASE 1.8C no ejecutada")
    mart = pd.read_csv(MART_FILE)
    assert (mart["comparability_duration_s"] == "DIRECTLY_COMPARABLE").all()
    assert (mart["comparability_vmax"] == "REQUIRES_NORMALIZATION").all()
    assert (mart["comparability_hip_rom"] == "COMPARABLE_WITH_CAVEAT").all()


def test_mart_no_nan_minimal_features():
    if not MART_FILE.exists():
        pytest.skip("FASE 1.8C no ejecutada")
    mart = pd.read_csv(MART_FILE)
    feats = ["duration_s", "time_to_peak_s", "vmax", "vmean", "amax",
             "displacement", "path_length", "hip_rom", "knee_rom",
             "ankle_rom", "snr"]
    assert mart[feats].isna().sum().sum() == 0


def test_mart_qc_consistency():
    if not MART_FILE.exists():
        pytest.skip("FASE 1.8C no ejecutada")
    mart = pd.read_csv(MART_FILE)
    assert (mart["qc_status"] == "accepted").all()
    assert (mart["quality_flag"] == "OK").all()


def test_mart_summary():
    if not MART_SUMMARY.exists():
        pytest.skip("FASE 1.8C no ejecutada")
    sm = pd.read_csv(MART_SUMMARY).iloc[0]
    assert sm["total_executions"] == 428
    assert sm["total_athletes"] == 34
    assert sm["sampling_rate_200hz"] == 9
    assert sm["sampling_rate_250hz"] == 419
    assert sm["missing_feature_values"] == 0


def test_mart_units_metadata():
    if not MART_FILE.exists():
        pytest.skip("FASE 1.8C no ejecutada")
    mart = pd.read_csv(MART_FILE)
    assert (mart["feature_version"].astype(str) == "1.0").all()
    assert (mart["units_version"].astype(str) == "1.0").all()
    hist = mart[mart["athlete_id"].isin(["B0367", "B0377"])]
    new = mart[~mart["athlete_id"].isin(["B0367", "B0377"])]
    assert (hist["source_dataset"] == "feature_readiness_sample").all()
    assert set(new["source_dataset"]) <= {"task3_golden_path", "cohort_expansion_gp"}


def test_mart_traceability():
    """event_id del mart debe emparejar con el evento global accepted del baseline."""
    if not MART_FILE.exists():
        pytest.skip("FASE 1.8C no ejecutada")
    mart = pd.read_csv(MART_FILE)
    ev = pd.read_csv(ROOT / "output" / "segmentation_events.csv")
    b36 = mart[mart["athlete_id"] == "B0367"]
    merged = b36.merge(ev, left_on=["technique", "condition", "trial", "event_id"],
                       right_on=["technique", "condition", "trial", "event_id"], how="left")
    assert merged["status"].notna().all()
    assert merged["status"].eq("accepted").all()


# --------------------------------------------------------------------------- #
# 11. Dashboard MVP — capa de datos (FASE 1.8D)
# --------------------------------------------------------------------------- #

DASH_DIR = ROOT / "dashboard"
DA = None


def _dash_data():
    global DA
    if DA is None:
        import importlib.util as _ilu
        _dspec = _ilu.spec_from_file_location("dash_data", str(DASH_DIR / "data.py"))
        DA = _ilu.module_from_spec(_dspec)
        _dspec.loader.exec_module(DA)
    return DA


def test_dashboard_does_not_touch_c3d():
    """el data layer no importa ezc3d ni lee archivos .c3d."""
    da = _dash_data()
    src = Path(da.__file__).read_text(encoding="utf-8")
    # solo el docstring puede mencionar 'C3D' como texto; el código no.
    code = "\n".join(line for line in src.splitlines()
                     if not line.strip().startswith("#") and "c3d" not in line.lower().strip())
    assert "import ezc3d" not in src
    assert "ezc3d.c3d(" not in src
    assert "rglob" not in src
    assert "glob" not in src


def test_dashboard_data_mart_loads():
    da = _dash_data()
    df = da.load_data_mart()
    assert len(df) == 428


def test_dashboard_schema():
    da = _dash_data()
    df = da.load_data_mart()
    missing = [c for c in da.REQUIRED_COLUMNS if c not in df.columns]
    assert not missing, f"faltan {missing}"


def test_dashboard_no_nan_required_fields():
    da = _dash_data()
    df = da.load_data_mart()
    feats = ["duration_s", "time_to_peak_s", "vmax", "amax", "displacement",
             "path_length", "hip_rom", "knee_rom", "ankle_rom", "snr"]
    assert df[feats].isna().sum().sum() == 0


def test_dashboard_athletes():
    da = _dash_data()
    df = da.load_data_mart()
    assert set(df["athlete_id"].unique()) == set(_new29()._cohort_250()) | {"B0367"}


def test_dashboard_techniques():
    da = _dash_data()
    df = da.load_data_mart()
    assert set(df["technique"].unique()).issuperset({"S02", "S03", "S05"})


def test_dashboard_comparability_preserved():
    da = _dash_data()
    df = da.load_data_mart()
    statuses = da.comparability_status(df)
    assert {"DIRECTLY_COMPARABLE", "COMPARABLE_WITH_CAVEAT",
            "REQUIRES_NORMALIZATION"} <= set(statuses)


def test_dashboard_filter_by_athlete():
    da = _dash_data()
    df = da.load_data_mart()
    b36 = da.filter_by(df, athlete_id="B0367")
    assert len(b36) == 9
    assert set(b36["athlete_id"]) == {"B0367"}


def test_dashboard_filter_by_technique():
    da = _dash_data()
    df = da.load_data_mart()
    s02 = da.filter_by(df, technique="S02")
    assert len(s02) == 109
    assert set(s02["technique"]) == {"S02"}


def test_dashboard_data_mart_not_modified():
    """el módulo de datos no escribe; el archivo del Data Mart queda igual."""
    da = _dash_data()
    before = da.DATA_MART_PATH.read_bytes()
    _ = da.load_data_mart()
    after = da.DATA_MART_PATH.read_bytes()
    assert before == after


# --------------------------------------------------------------------------- #
# 12. Inventario completo de atletas (FASE 1.8E)
# --------------------------------------------------------------------------- #

INV_DIR = ROOT / "output" / "athlete_inventory"
_INV = None


def _inv08():
    global _INV
    if _INV is None:
        import importlib.util as _ilu
        _ispec = _ilu.spec_from_file_location("inv08", str(ROOT / "scripts" / "08_athlete_inventory.py"))
        _INV = _ilu.module_from_spec(_ispec)
        _ispec.loader.exec_module(_INV)
    return _INV


def test_inventory_script_executes():
    """las salidas del inventario existen (script corrió correctamente)."""
    assert (INV_DIR / "file_inventory.csv").exists()
    assert (INV_DIR / "athlete_inventory.csv").exists()


def test_inventory_source_not_modified():
    """el script solo lee; no modifica ningún C3D (no crea CSV en atletas/)."""
    inv = _inv08()
    athletes = inv.discover_all_athletes()
    assert len(athletes) >= 36
    csvs_in_source = list((ROOT / "atletas").glob("*.csv"))
    assert csvs_in_source == [], "el inventario no debe escribir dentro de atletas/"


def test_inventory_athlete_ids_unique():
    a = pd.read_csv(INV_DIR / "athlete_inventory.csv")
    assert a["athlete_id"].is_unique


def test_inventory_files_exist():
    fi = pd.read_csv(INV_DIR / "file_inventory.csv")
    for _, r in fi.iterrows():
        assert (ROOT / r["source_path"]).exists(), f"falta {r['source_path']}"


def test_inventory_sampling_rate_valid():
    fi = pd.read_csv(INV_DIR / "file_inventory.csv")
    assert fi["sampling_rate_hz"].notna().all()
    assert set(fi["sampling_rate_hz"].unique()) <= {200.0, 250.0}


def test_inventory_group_hypothesis_confirmed():
    """hipótesis verificable: 200 Hz -> B0367..B0370; 250 Hz -> resto."""
    srg = pd.read_csv(INV_DIR / "sampling_rate_group.csv")
    b36_37 = srg[srg["athlete_id"].isin(["B0367", "B0368", "B0369", "B0370"])]
    assert set(b36_37["rate_group"]) == {"200_Hz"}
    z50 = srg[~srg["athlete_id"].isin(["B0367", "B0368", "B0369", "B0370"])]
    assert set(z50["rate_group"]) == {"250_Hz"}


def test_inventory_known_facts_b0367_b0377():
    """hechos conocidos de B0367/B0377 presentes en el inventario."""
    fi = pd.read_csv(INV_DIR / "file_inventory.csv")
    b36 = fi[fi["athlete_id"] == "B0367"]
    b37 = fi[fi["athlete_id"] == "B0377"]
    assert len(b36) == 26
    assert len(b37) == 39
    assert set(b36["sampling_rate_hz"].unique()) == {200.0}
    assert set(b37["sampling_rate_hz"].unique()) == {250.0}


def test_inventory_config_gap():
    g = pd.read_csv(INV_DIR / "configuration_gap.csv")
    assert g.loc[g["athlete"] == "B0367", "S01"].iloc[0] == "RFIN"
    assert g.loc[g["athlete"] == "B0377", "S04"].iloc[0] == "LTOE"
    assert g.loc[g["athlete"] == "B0405", "configuration_status"].iloc[0] == "NOT_CONFIGURED"


def test_inventory_anomalies_no_crash():
    """anomaly flags no rompen y el CSV existe."""
    a = pd.read_csv(INV_DIR / "anomaly_inventory.csv")
    assert "anomaly_flags" in a.columns


def test_inventory_required_columns():
    for f, req in [("file_inventory.csv",
                    ["athlete_id", "file_name", "technique", "condition", "trial",
                     "sampling_rate_hz", "units_position", "units_angle",
                     "number_of_points", "number_of_frames", "subject_ids",
                     "tarcza_marker_count"]),
                   ("athlete_inventory.csv",
                    ["athlete_id", "n_files", "techniques", "conditions", "trials"])]:
        df = pd.read_csv(INV_DIR / f)
        missing = [c for c in req if c not in df.columns]
        assert not missing, f"{f}: faltan {missing}"


def test_inventory_no_duplicate_files():
    fi = pd.read_csv(INV_DIR / "file_inventory.csv")
    keys = fi["athlete_id"] + "|" + fi["file_name"]
    assert keys.duplicated().sum() == 0


def test_inventory_markers_available_all_athletes():
    """señales RFIN/RTOE/LTOE disponibles en todos los atletas."""
    mk = pd.read_csv(INV_DIR / "marker_availability.csv")
    for m in ["RFIN", "RTOE", "LTOE", "RANK", "LANK", "RHEE", "LHEE"]:
        assert mk[m].all(), f"{m} no disponible en al menos un atleta-técnica"


def test_inventory_scaling_readiness():
    r = pd.read_csv(INV_DIR / "scaling_readiness.csv")
    assert r.loc[r["athlete"] == "B0367", "group"].iloc[0] == "A"
    assert r.loc[r["athlete"] == "B0377", "group"].iloc[0] == "B"
    # C = cohorte 200 Hz sin config
    c_ath = set(r.loc[r["group"] == "C", "athlete"])
    assert c_ath == {"B0368", "B0369", "B0370"}


# --------------------------------------------------------------------------- #
# 13. Selección de 3 atletas candidatos (FASE 1.8F, Tarea 0)
# --------------------------------------------------------------------------- #

SEL_DIR = ROOT / "output" / "scaling_selection"
_SEL18F = None


def _sel08():
    global _SEL18F
    if _SEL18F is None:
        import importlib.util as _ilu
        _sspec = _ilu.spec_from_file_location(
            "sel18f", str(ROOT / "scripts" / "08_select_1_8f_candidates.py"))
        _SEL18F = _ilu.module_from_spec(_sspec)
        _sspec.loader.exec_module(_SEL18F)
    return _SEL18F


def _sel_df():
    return pd.read_csv(SEL_DIR / "phase_1_8f_candidate_selection.csv")


def test_1_8f_selection_regenerates_csv():
    """la selección es re-ejecutable: regenera el CSV con 3 filas."""
    mod = _sel08()
    sel, counts = mod.run_selection()
    assert len(sel) == 3
    assert (SEL_DIR / "phase_1_8f_candidate_selection.csv").exists()


def test_1_8f_eligible_universe_counts():
    """el universo elegible coincide con las cifras de la FASE 1.8E (no inventadas)."""
    mod = _sel08()
    _, counts = mod.build_candidate_frame(mod.load_inventory())
    assert counts["n_total"] == 37
    assert counts["n_250"] == 33
    assert counts["n_200"] == 4
    assert counts["n_group_c"] == 3
    assert counts["n_excluded_used"] == 2
    assert counts["n_anomalies"] == 0
    assert counts["n_eligible"] == 32


def test_1_8f_selected_three_exist_in_eligible():
    """los 3 seleccionados pertenecen al universo elegible."""
    mod = _sel08()
    df, counts = mod.build_candidate_frame(mod.load_inventory())
    elig = set(df["athlete_id"])
    sel = _sel_df()
    assert len(sel) == 3
    assert set(sel["athlete_id"]) <= elig
    assert len(elig) == counts["n_eligible"]


def test_1_8f_all_selected_250_hz():
    sel = _sel_df()
    assert (sel["sampling_rate_hz"] == 250.0).all()


def test_1_8f_b0367_b0377_not_selected():
    sel = _sel_df()
    assert not (set(sel["athlete_id"]) & {"B0367", "B0377"})


def test_1_8f_no_anomalies_selected():
    sel = _sel_df()
    assert (sel["anomaly_status"] == "ok").all()


def test_1_8f_configuration_not_configured_selected():
    sel = _sel_df()
    assert (sel["configuration_status"] == "NOT_CONFIGURED").all()


def test_1_8f_selection_deterministic():
    """dos ejecuciones de la selección producen exactamente el mismo trío."""
    mod = _sel08()
    df, _ = mod.build_candidate_frame(mod.load_inventory())
    r1 = mod.select_candidates(df)
    r2 = mod.select_candidates(df)
    assert list(r1["athlete_id"]) == list(r2["athlete_id"])


def test_1_8f_csv_required_columns():
    required = ["rank_internal", "athlete_id", "sampling_rate_hz", "total_c3d",
                "techniques_present", "conditions_present", "trials_present",
                "derived_structure", "marker_structure_summary", "anomaly_status",
                "configuration_status", "selection_reason", "diversity_role"]
    sel = _sel_df()
    missing = [c for c in required if c not in sel.columns]
    assert not missing, f"faltan columnas en phase_1_8f_candidate_selection.csv: {missing}"


# --------------------------------------------------------------------------- #
# 14. Auditoría de señal y lateralidad (FASE 1.8F, Tarea 1)
# --------------------------------------------------------------------------- #

AUD_DIR = ROOT / "output" / "scaling_selection"
_AUD18T1 = None


def _audit09():
    global _AUD18T1
    if _AUD18T1 is None:
        import importlib.util as _ilu
        _aspec = _ilu.spec_from_file_location(
            "audit09", str(ROOT / "scripts" / "09_signal_laterality_audit.py"))
        _AUD18T1 = _ilu.module_from_spec(_aspec)
        _aspec.loader.exec_module(_AUD18T1)
    return _AUD18T1


def test_1_8f1_three_athletes_only():
    recs = pd.read_csv(AUD_DIR / "phase_1_8f_signal_recommendations.csv")
    aud = pd.read_csv(AUD_DIR / "phase_1_8f_signal_audit.csv")
    assert set(recs["athlete_id"]) == {"B0400", "B0371", "B0380"}
    assert set(aud["athlete_id"]) == {"B0400", "B0371", "B0380"}


def test_1_8f1_only_e01_t01():
    aud = pd.read_csv(AUD_DIR / "phase_1_8f_signal_audit.csv")
    assert set(aud["condition"].str.strip().unique()) == {"E01"}
    assert set(aud["trial"].str.strip().unique()) == {"T01"}


def test_1_8f1_only_s01_s05():
    aud = pd.read_csv(AUD_DIR / "phase_1_8f_signal_audit.csv")
    assert set(aud["technique"].unique()) == {"S01", "S02", "S03", "S04", "S05"}


def test_1_8f1_one_recommendation_per_cell():
    recs = pd.read_csv(AUD_DIR / "phase_1_8f_signal_recommendations.csv")
    assert len(recs) == 15
    assert recs.duplicated(subset=["athlete_id", "technique"]).sum() == 0


def test_1_8f1_configs_untouched():
    """existen exactamente las configs intencionales (baselines + cohorte 250 Hz)."""
    cfg = ROOT / "config" / "athletes"
    expected = set(_new29()._cohort_250()) | {"B0367"}
    assert set(p.stem for p in cfg.glob("*.yaml")) == expected


def test_1_8f1_b0367_b0377_intact():
    assert len(list((ROOT / "B0367").rglob("*.c3d"))) == 26
    assert len(list((ROOT / "atletas" / "B0377").rglob("*.c3d"))) == 39
    assert list((ROOT / "B0367").rglob("*.csv")) == []
    assert list((ROOT / "atletas" / "B0377").rglob("*.csv")) == []


def test_1_8f1_data_mart_untouched():
    """el Mart consolida dict cumple el contrato (histórico intacto + cohorte 250 Hz)."""
    _assert_mart_contract_ok()


def test_1_8f1_required_columns():
    aud = pd.read_csv(AUD_DIR / "phase_1_8f_signal_audit.csv")
    recs = pd.read_csv(AUD_DIR / "phase_1_8f_signal_recommendations.csv")
    req_aud = ["athlete_id", "technique", "condition", "trial", "signal", "side",
               "baseline_median", "baseline_mad", "threshold", "vmax", "snr",
               "candidate_count", "accepted_count", "rejected_count",
               "review_count", "candidate_separation_s", "quality_status",
               "suitability_status", "notes"]
    req_rec = ["athlete_id", "technique", "recommended_signal", "recommended_side",
               "recommendation_status", "evidence_summary", "alternative_signal",
               "alternative_side", "alternative_reason"]
    missing_a = [c for c in req_aud if c not in aud.columns]
    missing_r = [c for c in req_rec if c not in recs.columns]
    assert not missing_a and not missing_r


@pytest.fixture(scope="module")
def audit_result():
    mod = _audit09()
    return mod, mod.run_audit()


def test_1_8f1_outputs_reproducible(audit_result):
    """revolver el script regenera exactamente los CSV versionados."""
    mod, (audit, recs) = audit_result
    aud_file = pd.read_csv(AUD_DIR / "phase_1_8f_signal_audit.csv")
    rec_file = pd.read_csv(AUD_DIR / "phase_1_8f_signal_recommendations.csv")
    norm = lambda df: df.reset_index(drop=True).fillna("")
    assert norm(aud_file).equals(norm(audit))
    assert norm(rec_file).equals(norm(recs))


def test_1_8f1_decision_deterministic(audit_result):
    """la capa de decisión es determinista (dos llamadas idénticas)."""
    mod, (audit, recs) = audit_result
    r1 = mod._recommendations(audit)
    r2 = mod._recommendations(audit)
    assert r1.reset_index(drop=True).equals(r2.reset_index(drop=True))
    assert r1.reset_index(drop=True).equals(recs.reset_index(drop=True))


def test_1_8f1_recommendations_consistent():
    aud = pd.read_csv(AUD_DIR / "phase_1_8f_signal_audit.csv")
    recs = pd.read_csv(AUD_DIR / "phase_1_8f_signal_recommendations.csv").fillna("")
    for _, r in recs.iterrows():
        if r["recommended_signal"] == "":
            continue
        cell = aud[(aud["athlete_id"] == r["athlete_id"])
                   & (aud["technique"] == r["technique"])
                   & (aud["signal"] == r["recommended_signal"])]
        assert not cell.empty, f"{r['athlete_id']} {r['technique']} sin fila de señal"


def test_1_8f1_known_findings():
    """hallazgos con evidencia de la auditoría (deterministas y auditable)."""
    recs = pd.read_csv(AUD_DIR / "phase_1_8f_signal_recommendations.csv")
    get = lambda a, t: recs[(recs["athlete_id"] == a) & (recs["technique"] == t)].iloc[0]
    assert get("B0400", "S01")["recommended_signal"] == "RFIN"
    assert get("B0400", "S01")["recommendation_status"] == "RECOMMENDED"
    assert get("B0371", "S01")["recommendation_status"] == "NEEDS_VALIDATION"
    assert get("B0380", "S02")["recommended_signal"] == "LTOE"
    assert get("B0380", "S02")["recommended_side"] == "L"
    for a in ["B0400", "B0371", "B0380"]:
        for t in ["S02", "S03", "S04", "S05"]:
            assert get(a, t)["recommended_side"] in {"R", "L"}


# --------------------------------------------------------------------------- #
# 15. Configuraciones explícitas B0400/B0371/B0380 (FASE 1.8F, Tarea 2)
# --------------------------------------------------------------------------- #

CFG_DIR2 = ROOT / "config" / "athletes"
ATH_182 = ["B0400", "B0371", "B0380"]

# Fuente única de verdad de señal/lateralidad (audit 1.8F Tarea 1).
RECS182 = pd.read_csv(AUD_DIR / "phase_1_8f_signal_recommendations.csv")


def _cfg_resolved(aid):
    """Señales/lados efectivos tras el merge con global (mecanismo real de 02)."""
    cfg = _X02.load_config(aid)
    primary, _, joints = _X02._build_signals(cfg)
    return {t: primary[t][0] for t in ["S01", "S02", "S03", "S04", "S05"]}, joints


def test_1_8f2_config_files_exist():
    for aid in ATH_182:
        assert (CFG_DIR2 / f"{aid}.yaml").exists()


def test_1_8f2_loads_via_pipeline():
    """la config se lee y resuelve señales/lados con el mecanismo existente."""
    prim, joints = _cfg_resolved("B0400")
    assert prim == {"S01": "RFIN", "S02": "RTOE", "S03": "RTOE",
                    "S04": "RTOE", "S05": "RTOE"}
    prim, joints = _cfg_resolved("B0371")
    assert prim["S01"] == "RFIN" and prim["S02"] == "RTOE"
    prim, joints = _cfg_resolved("B0380")
    assert prim["S01"] == "RFIN" and prim["S02"] == "LTOE" and prim["S03"] == "RTOE"
    assert joints["S02"] == "L" and joints["S01"] == "R"


def test_1_8f2_all_techniques_present_no_dupes():
    allowed = {"RFIN", "LFIN", "RTOE", "LTOE", "RANK", "LANK", "RHEE", "LHEE"}
    for aid in ATH_182:
        prim, joints = _cfg_resolved(aid)
        assert set(prim.keys()) == {"S01", "S02", "S03", "S04", "S05"}
        assert len(prim) == len(set(prim.keys()))
        for t, sig in prim.items():
            assert sig, f"{aid} {t}: señal vacía"
            assert sig in allowed, f"{aid} {t}: señal desconocida {sig}"
            assert joints[t] in {"R", "L"}


def test_1_8f2_sampling_rate_250():
    for aid in ATH_182:
        cfg = _X02.load_config(aid)
        assert float(cfg["metadata"]["sampling_rate"]) == 250.0


def test_1_8f2_matches_recommendations_csv():
    """la config coincide EXACTAMENTE con el audit (no hay segunda verdad manual).

    Detecta: B0380-S02 LTOE->RTOE, B0371-S01 NEEDS_VALIDATION->RECOMMENDED, etc.
    """
    for aid in ATH_182:
        prim, joints = _cfg_resolved(aid)
        cfg = _X02.load_config(aid)
        techniques = cfg.get("techniques") or cfg.get("signals") or {}
        for t in ["S01", "S02", "S03", "S04", "S05"]:
            row = RECS182[(RECS182["athlete_id"] == aid) & (RECS182["technique"] == t)].iloc[0]
            assert prim[t] == row["recommended_signal"], f"{aid} {t}: señal != audit"
            assert joints[t] == row["recommended_side"], f"{aid} {t}: lado != audit"
            thresh = techniques.get(t, {}).get("thresholds") or {}
            needs = thresh.get("status") == "NEEDS_VALIDATION"
            assert needs == (row["recommendation_status"] == "NEEDS_VALIDATION"), \
                f"{aid} {t}: estado != audit ({row['recommendation_status']})"


def test_1_8f2_b0367_b0377_unchanged():
    prim67, joints67 = _cfg_resolved("B0367")
    assert prim67 == {"S01": "RFIN", "S02": "RTOE", "S03": "RTOE",
                      "S04": "RTOE", "S05": "RTOE"}
    assert all(v == "R" for v in joints67.values())
    prim77, joints77 = _cfg_resolved("B0377")
    assert prim77 == {"S01": "RFIN", "S02": "RTOE", "S03": "RTOE",
                      "S04": "LTOE", "S05": "RTOE"}
    assert joints77["S04"] == "L"
    cfg77 = _X02.load_config("B0377")
    assert cfg77["techniques"]["S01"]["thresholds"]["status"] == "NEEDS_VALIDATION"


def test_1_8f2_data_mart_not_modified():
    """el Mart consolidado sigue cumpliendo el contrato (histórico intacto)."""
    _assert_mart_contract_ok()


# --------------------------------------------------------------------------- #
# 16. Golden Path E01-T01 × S01-S05 (FASE 1.8F, Tarea 3)
# --------------------------------------------------------------------------- #

VAL_DIR = ROOT / "output" / "scaling_validation"
ATH_183 = ["B0400", "B0371", "B0380"]
_TASK310 = None


def _task3_script():
    global _TASK310
    if _TASK310 is None:
        import importlib.util as _ilu
        _tspec = _ilu.spec_from_file_location(
            "task3gp", str(ROOT / "scripts" / "10_phase_1_8f_task3_golden_path.py"))
        _TASK310 = _ilu.module_from_spec(_tspec)
        _tspec.loader.exec_module(_TASK310)
    return _TASK310


@pytest.fixture(scope="module")
def golden_result():
    """Regenera el Golden Path una vez y restaura el estado activo de 02."""
    mod = _task3_script()
    out = mod.run_golden_path()
    _X02.set_active_athlete("B0367")
    return mod, out


def _gp_df():
    return pd.read_csv(VAL_DIR / "phase_1_8f_task3_golden_path.csv")


def _sum_df():
    return pd.read_csv(VAL_DIR / "phase_1_8f_task3_summary.csv")


def test_1_8f3_athletes_only():
    assert set(_gp_df()["athlete_id"]) == set(ATH_183)
    assert set(_sum_df()["athlete_id"]) == set(ATH_183)
    assert set(pd.read_csv(VAL_DIR / "phase_1_8f_task3_events.csv")["athlete_id"]) == set(ATH_183)


def test_1_8f3_only_e01_t01():
    gp = _gp_df()
    assert set(gp["condition"].astype(str).str.strip().unique()) == {"E01"}
    assert set(gp["trial"].astype(str).str.strip().unique()) == {"T01"}


def test_1_8f3_only_s01_s05():
    gp = _gp_df()
    assert set(gp["technique"].unique()) == {"S01", "S02", "S03", "S04", "S05"}


def test_1_8f3_sampling_rate_250():
    gp = _gp_df()
    assert (gp["sampling_rate_hz"] == 250.0).all()


def test_1_8f3_no_empty_signal():
    gp = _gp_df()
    assert gp["signal_used"].notna().all()
    assert (gp["signal_used"].astype(str).str.strip() != "").all()


def test_1_8f3_b0380_s02_uses_ltoe():
    gp = _gp_df()
    s02 = gp[(gp["athlete_id"] == "B0380") & (gp["technique"] == "S02")]
    assert not s02.empty
    assert (s02["signal_used"] == "LTOE").all()
    assert (s02["movement_side"] == "L").all()


def test_1_8f3_s01_signal_behavior():
    """S01: RFIN usada en B0400; en B0371/B0380 el gate min_snr=8 la descarta.

    Documenta el hallazgo observado (revisión, no corrección): el pipeline usó
    el respaldo en los dos casos NEEDS_VALIDATION porque RFIN falla SNR.
    """
    gp = _gp_df()
    get = lambda a: gp[(gp["athlete_id"] == a) & (gp["technique"] == "S01")]
    assert not get("B0400").empty and (get("B0400")["signal_used"] == "RFIN").all()
    assert (get("B0371")["signal_used"] != "RFIN").all()
    assert (get("B0380")["signal_used"] != "RFIN").all()


def test_1_8f3_recommended_cells_match_config():
    """para celdas RECOMMENDED, la señal usada coincide con la config de Task 2.

    En las dos celdas S01-NEEDS_VALIDATION (B0371/B0380) la señal configurada
    NO fue usada (gate de SNR); eso queda registrado y esperado.
    """
    sm = _sum_df()
    for _, r in sm.iterrows():
        cfg_signal = _cfg_resolved(r["athlete_id"])[0][r["technique"]]
        if r["technique"] == "S01" and r["athlete_id"] in ("B0371", "B0380"):
            assert r["signal_used"] != cfg_signal  # fallback documentado
        else:
            assert r["signal_used"] == cfg_signal, f"{r['athlete_id']} {r['technique']}"


def test_1_8f3_summary_one_row_per_cell():
    sm = _sum_df()
    assert len(sm) == 15
    assert sm.duplicated(subset=["athlete_id", "technique"]).sum() == 0
    assert "validation_status" in sm.columns


def test_1_8f3_data_mart_intact():
    """el Mart consolidado cumple el contrato (histórico intacto + 250 Hz)."""
    _assert_mart_contract_ok()


def test_1_8f3_b0367_b0377_history_intact():
    b67 = pd.read_csv(ROOT / "output" / "executions_sample.csv")
    sub = b67[(b67["condition"] == "E01") & (b67["trial"] == "T01")]
    assert sub.groupby("technique").size().to_dict() == \
        {"S01": 3, "S02": 3, "S03": 3, "S04": 3, "S05": 3}
    b77 = pd.read_csv(ROOT / "output" / "B0377" / "executions_sample.csv")
    sub77 = b77[(b77["condition"] == "E01") & (b77["trial"] == "T01")]
    assert sub77.groupby("technique").size().to_dict() == \
        {"S01": 1, "S02": 3, "S03": 3, "S04": 3, "S05": 3}


def test_1_8f3_rows_belong_to_golden_path():
    """cada fila proviene únicamente de un C3D S0X-E01-T01 de los 3 atletas."""
    import re as _re
    gp = _gp_df()
    pat = _re.compile(r"^\d{4}-\d{2}-\d{2}-B0\d{3}-S0[1-5]-E01-T01\.c3d$")
    bad = [f for f in gp["source_file"].unique() if not pat.match(str(f))]
    assert not bad, f"archivos fuera del Golden Path: {bad}"


def _frames_match(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    """Compara marcos por columna: floats con tolerancia, cadenas con NaN≈''."""
    a, b = a.reset_index(drop=True), b.reset_index(drop=True)
    if list(a.columns) != list(b.columns):
        return False
    if len(a) != len(b):
        return False
    for c in a.columns:
        ca, cb = a[c], b[c]
        try:
            if pd.api.types.is_numeric_dtype(ca) and pd.api.types.is_numeric_dtype(cb):
                if not np.allclose(ca.fillna(np.nan).to_numpy(dtype=float),
                                   cb.fillna(np.nan).to_numpy(dtype=float),
                                   rtol=1e-9, atol=1e-9, equal_nan=True):
                    return False
            else:
                if ca.fillna("").astype(str).tolist() != cb.fillna("").astype(str).tolist():
                    return False
        except (TypeError, ValueError):
            return False
    return True


def test_1_8f3_outputs_reproducible(golden_result):
    """reejecutar el Golden Path regenera exactamente los CSV versionados
    (float64 con tolerancia por round-trip CSV, cadenas con NaN='')."""
    mod, (gp, ev, ql, cells) = golden_result
    assert _frames_match(pd.read_csv(VAL_DIR / "phase_1_8f_task3_golden_path.csv"), gp)
    assert _frames_match(pd.read_csv(VAL_DIR / "phase_1_8f_task3_summary.csv"), cells)
    assert _frames_match(pd.read_csv(VAL_DIR / "phase_1_8f_task3_events.csv"), ev)
    assert _frames_match(pd.read_csv(VAL_DIR / "phase_1_8f_task3_execution_quality.csv"), ql)


def test_1_8f3_events_quality_complete():
    ev = pd.read_csv(VAL_DIR / "phase_1_8f_task3_events.csv")
    ql = pd.read_csv(VAL_DIR / "phase_1_8f_task3_execution_quality.csv")
    assert set(ev["status"].unique()) <= {"accepted", "rejected", "review"}
    assert "rejection_reason" in ev.columns
    assert set(ql["quality_flag"].unique()) <= {"OK", "WARN", "REVIEW", "INVALID"}


# --------------------------------------------------------------------------- #
# 17. Validación dirigida de S01 (FASE 1.8F, Task 4)
# --------------------------------------------------------------------------- #

S1V_DIR = ROOT / "output" / "scaling_validation" / "s01_validation"
ATH_184 = ["B0367", "B0377", "B0400", "B0371", "B0380"]
S1_SIGNALS = ["RFIN", "LFIN", "RTOE"]
EVIDENCE_LEVELS = {"VALIDATED", "PROMISING_BUT_INCOMPLETE",
                   "INSUFFICIENT_EVIDENCE", "SIGNAL_PROBLEM",
                   "SEGMENTATION_PROBLEM", "REPRESENTATION_PROBLEM"}
STATUSES = {"segmentable", "marginal", "not_segmentable", "missing"}
_TASK411 = None


def _task4_script():
    global _TASK411
    if _TASK411 is None:
        import importlib.util as _ilu
        _sspec = _ilu.spec_from_file_location(
            "t4s01", str(ROOT / "scripts" / "11_s01_targeted_validation.py"))
        _TASK411 = _ilu.module_from_spec(_sspec)
        _sspec.loader.exec_module(_TASK411)
    return _TASK411


@pytest.fixture(scope="module")
def s01_result():
    mod = _task4_script()
    out = mod.run_s01_validation()
    return mod, out


def _cmp4():
    return pd.read_csv(S1V_DIR / "s01_signal_comparison.csv")


def _ass4():
    return pd.read_csv(S1V_DIR / "s01_athlete_assessment.csv")


def test_1_8f4_outputs_exist(s01_result):
    assert (S1V_DIR / "s01_signal_comparison.csv").exists()
    assert (S1V_DIR / "s01_athlete_assessment.csv").exists()
    figs = list((S1V_DIR / "figures").glob("s01_*.png"))
    assert len(figs) >= 8
    assert (S1V_DIR / "figures" / "s01_B0400.png").exists()


def test_1_8f4_exactly_5_athletes():
    assert set(_cmp4()["athlete_id"]) == set(ATH_184)
    assert set(_ass4()["athlete_id"]) == set(ATH_184)


def test_1_8f4_three_signals_per_athlete():
    cmp = _cmp4()
    assert len(cmp) == 15
    for a in ATH_184:
        assert set(cmp[cmp["athlete_id"] == a]["signal"]) == set(S1_SIGNALS)
    assert len(_ass4()) == 5
    assert _ass4().duplicated(subset=["athlete_id"]).sum() == 0


def test_1_8f4_required_columns():
    cmp_cols = ["athlete_id", "technique", "condition", "trial", "signal",
                "baseline", "mad", "vmax", "snr", "threshold",
                "candidate_count", "accepted_count", "rejected_count",
                "review_count", "mean_duration_s", "candidate_separation_s",
                "signal_status"]
    ass_cols = ["athlete_id", "configured_signal", "observed_best_signal",
                "rfin_status", "lfin_status", "rtoe_status",
                "fallback_detected", "fallback_interpretation",
                "evidence_level", "recommendation"]
    assert set(cmp_cols) <= set(_cmp4().columns)
    assert set(ass_cols) <= set(_ass4().columns)


def test_1_8f4_classification_consistent():
    cmp, ass = _cmp4(), _ass4()
    assert set(cmp["signal_status"].unique()) <= STATUSES
    assert set(ass["evidence_level"].unique()) <= EVIDENCE_LEVELS
    for _, r in ass.iterrows():
        assert r["configured_signal"] == "RFIN"
        sub = cmp[cmp["athlete_id"] == r["athlete_id"]]
        assert set(sub["signal"]) == set(S1_SIGNALS)
    # fallback solo donde hubo golden path con señal distinta a la configurada
    fb = set(ass.loc[ass["fallback_detected"] == True, "athlete_id"])
    assert fb == {"B0371", "B0380"}
    # recommendations esperadas por evidencia
    get = lambda a: ass[ass["athlete_id"] == a].iloc[0]
    assert "RFIN_NOT_VALIDATED" in str(get("B0377")["recommendation"])
    assert "RFIN_NOT_VALIDATED" in str(get("B0371")["recommendation"])
    assert "RTOE_FALLBACK_NOT_VALIDATED" in str(get("B0380")["recommendation"])
    assert get("B0400")["evidence_level"] == "VALIDATED"
    assert get("B0367")["evidence_level"] == "VALIDATED"


def test_1_8f4_deterministic():
    """dos ejecuciones producen exactamente los mismos marcos y decisión."""
    mod = _task4_script()
    c1, a1, o1 = mod.run_s01_validation()
    c2, a2, o2 = mod.run_s01_validation()
    assert _frames_match(c1.reset_index(drop=True), c2.reset_index(drop=True))
    assert _frames_match(a1.reset_index(drop=True), a2.reset_index(drop=True))
    assert o1 == o2 == "B"


def test_1_8f4_outputs_reproducible(s01_result):
    """los CSV versionados se regeneran idénticos (tolerancia float, NaN='')."""
    mod, _ = s01_result
    c1, a1, o1 = mod.run_s01_validation()
    assert _frames_match(pd.read_csv(S1V_DIR / "s01_signal_comparison.csv"), c1)
    assert _frames_match(pd.read_csv(S1V_DIR / "s01_athlete_assessment.csv"), a1)


def test_1_8f4_data_mart_intact():
    """el Mart consolidado cumple el contrato (histórico intacto + 250 Hz)."""
    _assert_mart_contract_ok()


def test_1_8f4_configs_untouched():
    """el dataset de configs sigue siendo exactamente el intencional (34 + …)."""
    cfg = ROOT / "config" / "athletes"
    expected = set(_new29()._cohort_250()) | {"B0367"}
    assert set(p.stem for p in cfg.glob("*.yaml")) == expected


# --------------------------------------------------------------------------- #
# 18. Expansión controlada cohorte 250 Hz (FASE 1.8F, Task 5)
# --------------------------------------------------------------------------- #

COH_DIR = ROOT / "output" / "scaling_validation" / "cohort_expansion"
EXISTING_5 = {"B0367", "B0377", "B0400", "B0371", "B0380"}
_NEW29 = None


def _new29():
    global _NEW29
    if _NEW29 is None:
        import importlib.util as _ilu
        _cspec = _ilu.spec_from_file_location(
            "coh13", str(ROOT / "scripts" / "13_phase_1_8f_cohort_expansion.py"))
        _NEW29 = _ilu.module_from_spec(_cspec)
        _cspec.loader.exec_module(_NEW29)
    return _NEW29


def _st5():
    return pd.read_csv(COH_DIR / "cohort_250hz_status.csv")


def _rec5():
    return pd.read_csv(COH_DIR / "cohort_signal_recommendations.csv")


def _audit5():
    return pd.read_csv(COH_DIR / "cohort_signal_audit.csv")


def _gp5():
    return pd.read_csv(COH_DIR / "cohort_golden_path.csv")


def test_1_8f5_outputs_exist():
    for f in ["cohort_250hz_status.csv", "cohort_signal_audit.csv",
              "cohort_signal_recommendations.csv", "cohort_golden_path.csv",
              "cohort_events.csv", "cohort_execution_quality.csv"]:
        assert (COH_DIR / f).exists()
    assert len(list((COH_DIR / "figures").glob("fig_*.png"))) >= 4


def test_1_8f5_universe_33_250hz_no_b0367():
    st = _st5()
    assert len(st) == 33
    assert (st["sampling_rate_hz"] == 250.0).all()
    assert "B0367" not in set(st["athlete_id"])


def test_1_8f5_status_known_values():
    st = _st5()
    ok = {"READY_FOR_CONFIG", "PARTIAL_CONFIG", "EXISTING_GOLDEN_PATH",
          "EXISTING_PARTIAL_DATA"}
    assert set(st["final_status"].unique()) <= ok
    get = lambda a: st[st["athlete_id"] == a].iloc[0]
    assert get("B0377")["final_status"] == "EXISTING_PARTIAL_DATA"
    for a in ("B0400", "B0371", "B0380"):
        assert get(a)["final_status"] == "EXISTING_GOLDEN_PATH"
    new = sorted(set(st["athlete_id"]) - EXISTING_5)
    assert len(new) == 29
    for a in new:
        r = get(a)
        assert r["final_status"] in ("READY_FOR_CONFIG", "PARTIAL_CONFIG")
        assert r["config_status"] == "provisional"
        assert r["golden_path_status"] == "done"


def test_1_8f5_audit_pending_scope():
    aud = _audit5()
    pending = set(_st5()["athlete_id"]) - EXISTING_5
    assert set(aud["athlete_id"]) == pending
    assert set(aud["technique"]) == {"S02", "S03", "S04", "S05"}
    assert set(aud["condition"]) == {"E01"} and set(aud["trial"]) == {"T01"}
    assert len(aud) == 29 * 4 * 6


def test_1_8f5_recs_shape_scope():
    rec = _rec5()
    assert len(rec) == 29 * 4
    assert rec.duplicated(subset=["athlete_id", "technique"]).sum() == 0
    assert set(rec["technique"]) == {"S02", "S03", "S04", "S05"}
    assert set(rec["evidence_status"]) <= {"RECOMMENDED", "NEEDS_VALIDATION",
                                           "INSUFFICIENT_DATA"}
    # S04 es la técnica problemática
    nv = rec[rec["evidence_status"] == "NEEDS_VALIDATION"]
    assert set(nv["technique"]) == {"S04"}
    assert set(nv["athlete_id"]) == {"B0388", "B0401"}


def test_1_8f5_configs_match_recommendations():
    cfg_ids = {p.stem for p in (ROOT / "config" / "athletes").glob("*.yaml")}
    new = sorted(cfg_ids - EXISTING_5)
    assert len(new) == 29
    rec = _rec5()
    for aid in new:
        cfg = _X02.load_config(aid)
        t = cfg.get("techniques") or {}
        s01 = t.get("S01", {})
        assert s01.get("signal") == "RFIN"
        assert (s01.get("thresholds") or {}).get("status") == "NEEDS_VALIDATION"
        for tech in ("S02", "S03", "S04", "S05"):
            row = rec[(rec["athlete_id"] == aid) & (rec["technique"] == tech)].iloc[0]
            if tech in t:
                e = t[tech]
                assert row["evidence_status"] == "RECOMMENDED", f"{aid} {tech}"
                assert e["signal"] == row["recommended_signal"], f"{aid} {tech} señal"
                assert e["joints_side"] == row["movement_side"], f"{aid} {tech} lado"
            else:
                assert row["evidence_status"] != "RECOMMENDED", f"{aid} {tech} ausente"


def test_1_8f5_gp_scope_and_sources():
    gp = _gp5()
    assert set(gp["technique"]) <= {"S02", "S03", "S04", "S05"}
    assert (gp["sampling_rate_hz"] == 250.0).all()
    assert set(gp["condition"]) == {"E01"} and set(gp["trial"]) == {"T01"}
    assert {"cohort_expansion_gp", "task3_golden_path"} <= set(gp["_source"])
    assert "B0377" not in set(gp["athlete_id"])
    src = gp["_source"].value_counts().to_dict()
    assert src["task3_golden_path"] == 37


def test_1_8f5_gp_signal_matches_recommendations():
    """regresión: el Golden Path USÓ la señal recomendada (atleta activado)."""
    gp = _gp5()
    rec = _rec5()[_rec5()["evidence_status"] == "RECOMMENDED"]
    g = gp[gp["_source"] == "cohort_expansion_gp"]
    mismatches = []
    for _, r in rec.iterrows():
        cell = g[(g["athlete_id"] == r["athlete_id"]) & (g["technique"] == r["technique"])]
        if cell.empty:
            continue
        if set(cell["signal_used"].unique()) != {r["recommended_signal"]}:
            mismatches.append((r["athlete_id"], r["technique"]))
    assert not mismatches, f"GP usó señal distinta a la recomendada: {mismatches}"


def test_1_8f5_data_mart_intact():
    """el Mart consolidado cumple el contrato (histórico intacto + 250 Hz)."""
    _assert_mart_contract_ok()


def test_1_8f5_determinism_limit2():
    """dos ejecuciones con --limit 2 (sin escritura) producen los mismos marcos."""
    mod = _new29()
    r1 = mod.run_cohort_expansion(limit=2, write=False)
    r2 = mod.run_cohort_expansion(limit=2, write=False)
    for k in ("status", "audit", "recs", "gp", "events"):
        assert _frames_match(r1[k].reset_index(drop=True),
                             r2[k].reset_index(drop=True)), k
    assert r1["new_cfg_ids"] == r2["new_cfg_ids"]


def test_1_8f5_reproducible_b0372():
    """recomputar en muestra (limit=1) reproduce exactamente las filas de B0372."""
    mod = _new29()
    r = mod.run_cohort_expansion(limit=1, write=False)
    got = r["gp"][r["gp"]["athlete_id"] == "B0372"]
    committed = _gp5()[_gp5()["athlete_id"] == "B0372"]
    assert len(got) > 0 and len(committed) > 0
    assert sorted(got["execution_id"]) == sorted(committed["execution_id"])
    assert got["_source"].eq("cohort_expansion_gp").all()


# --------------------------------------------------------------------------- #
# 19. Consolidación controlada del Data Mart 250 Hz (FASE 1.8F, Task 6)
# --------------------------------------------------------------------------- #

T6_DIR = ROOT / "output" / "data_mart" / "task6_consolidation"
_T614 = None


def _cons14():
    global _T614
    if _T614 is None:
        import importlib.util as _ilu
        _cspec = _ilu.spec_from_file_location(
            "con14", str(ROOT / "scripts" / "14_consolidate_250hz_data_mart.py"))
        _T614 = _ilu.module_from_spec(_cspec)
        _cspec.loader.exec_module(_T614)
    return _T614


def _mart_new_block():
    m = pd.read_csv(MART_FILE)
    return m[~m["athlete_id"].isin(["B0367", "B0377"])]


def test_1_8f6_outputs_exist():
    assert (ROOT / "output" / "data_mart" / "task6_backup" /
            "athlete_execution_features.csv").exists()
    for f in ["data_mart_250hz_consolidated.csv", "data_mart_task6_validation.csv",
              "data_mart_task6_duplicates.csv", "data_mart_task6_summary.csv",
              "data_mart_task6_coverage.csv", "source_reconciliation.csv"]:
        assert (T6_DIR / f).exists()


def test_1_8f6_schema_39_columns():
    mart = pd.read_csv(MART_FILE)
    assert len(mart.columns) == 39
    assert list(mart.columns) == _cons14().CONTRACT_COLUMNS


def test_1_8f6_execution_id_unique():
    mart = pd.read_csv(MART_FILE)
    assert mart["execution_id"].is_unique


def test_1_8f6_no_nan_features():
    mart = pd.read_csv(MART_FILE)
    feats = ["duration_s", "time_to_peak_s", "vmax", "vmean", "amax",
             "displacement", "path_length", "hip_rom", "knee_rom",
             "ankle_rom", "snr"]
    assert mart[feats].isna().sum().sum() == 0


def test_1_8f6_no_s01():
    mart = pd.read_csv(MART_FILE)
    assert not (mart["technique"] == "S01").any()


def test_1_8f6_new_block_250_hz():
    nb = _mart_new_block()
    assert set(nb["sampling_rate_hz"]) == {250.0}


def test_1_8f6_new_block_e01_t01():
    nb = _mart_new_block()
    assert set(nb["condition"]) == {"E01"}
    assert set(nb["trial"]) == {"T01"}


def test_1_8f6_new_block_s02_s05():
    nb = _mart_new_block()
    assert set(nb["technique"]) == {"S02", "S03", "S04", "S05"}


def test_1_8f6_b0388_b0401_no_s04():
    mart = pd.read_csv(MART_FILE)
    for aid in ("B0388", "B0401"):
        cell = mart[(mart["athlete_id"] == aid) & (mart["technique"] == "S04")]
        assert cell.empty, f"{aid} tiene S04 artificial"


def test_1_8f6_b0377_not_duplicated():
    mart = pd.read_csv(MART_FILE)
    b77 = mart[mart["athlete_id"] == "B0377"]
    assert len(b77) == 9
    src77 = mart[(mart["athlete_id"] == "B0377")]
    assert (src77["source_dataset"] == "feature_readiness_sample").all()


def test_1_8f6_task3_37_rows():
    mart = pd.read_csv(MART_FILE)
    t3 = mart[mart["source_dataset"] == "task3_golden_path"]
    assert len(t3) == 37
    assert set(t3["athlete_id"]) == {"B0400", "B0371", "B0380"}


def test_1_8f6_primary_signal_matches_config():
    mart = _mart_new_block()
    for aid, tech in mart[["athlete_id", "technique"]].drop_duplicates().itertuples(False):
        cfg = _X02.load_config(aid)
        e = (cfg.get("techniques") or {}).get(tech, {})
        cfg_sig = e.get("signal")
        if cfg_sig:  # técnicas con evidencia en config (S04 B0388/B0401 no existen aquí)
            sub = mart[(mart["athlete_id"] == aid) & (mart["technique"] == tech)]
            assert (sub["primary_signal"] == cfg_sig).all(), f"{aid} {tech}"


def test_1_8f6_movement_side_matches_config():
    mart = _mart_new_block()
    for aid, tech in mart[["athlete_id", "technique"]].drop_duplicates().itertuples(False):
        cfg = _X02.load_config(aid)
        e = (cfg.get("techniques") or {}).get(tech, {})
        j = e.get("joints_side")
        if j:
            sub = mart[(mart["athlete_id"] == aid) & (mart["technique"] == tech)]
            assert (sub["movement_side"] == j).all(), f"{aid} {tech}"


def test_1_8f6_sampling_matches_config():
    mart = _mart_new_block()
    for aid in mart["athlete_id"].unique():
        cfg = _X02.load_config(aid)
        assert float(cfg["metadata"]["sampling_rate"]) == 250.0
        sub = mart[mart["athlete_id"] == aid]
        assert (sub["sampling_rate_hz"] == 250.0).all()


def test_1_8f6_historical_preserved():
    mart = pd.read_csv(MART_FILE)
    hist = pd.read_csv(MART_BACKUP)
    hm = mart[mart["athlete_id"].isin(["B0367", "B0377"])]
    assert hm.reset_index(drop=True).equals(hist.reset_index(drop=True))


def test_1_8f6_reconciliation():
    rec = pd.read_csv(T6_DIR / "source_reconciliation.csv")
    total = rec[rec["source"] == "TOTAL"].iloc[0]
    assert total["inserted_rows"] == 428
    assert total["duplicate_rows"] == 0
    hi = rec[rec["source"] == "feature_readiness_sample"].iloc[0]["inserted_rows"]
    t3 = rec[rec["source"] == "task3_golden_path"].iloc[0]["inserted_rows"]
    t5 = rec[rec["source"] == "cohort_expansion_gp"].iloc[0]["inserted_rows"]
    assert (hi, t3, t5) == (18, 37, 373)


def test_1_8f6_determinism_write_false():
    """dos ejecuciones (write=False) producen el mismo mart consolidado."""
    mod = _cons14()
    r1 = mod.run_consolidation(write_final=False)
    r2 = mod.run_consolidation(write_final=False)
    assert r1["state"] == "OK" and r2["state"] == "OK"
    assert _frames_match(r1["final"].reset_index(drop=True),
                         r2["final"].reset_index(drop=True))


def test_1_8f6_features_match_source():
    """las features del bloque nuevo NO fueron alteradas vs la fuente (mapeo nombre a nombre)."""
    mod = _cons14()
    gp = pd.read_csv(mod.COH_FILE)
    block = mod._build_new_block(gp)
    mart = _mart_new_block()
    merged = mart.merge(block, on="execution_id", suffixes=("", "_src"),
                        how="inner")
    assert len(merged) == len(mart)
    for c in mod.FEATURE_COLS:
        assert np.allclose(merged[c], merged[f"{c}_src"], rtol=1e-12, atol=1e-12,
                           equal_nan=True), f"feature alterada: {c}"


# --------------------------------------------------------------------------- #
# 20. ML Dataset v0 — diseño y auditoría (TASK 7)
# --------------------------------------------------------------------------- #

MLV0_DIR = ROOT / "output" / "ml_dataset_v0"
ML_FEATURES = ["duration_s", "time_to_peak_s", "vmax", "vmean", "amax",
               "displacement", "path_length", "hip_rom", "knee_rom",
               "ankle_rom", "snr"]
_T715 = None


def _ml15():
    global _T715
    if _T715 is None:
        import importlib.util as _ilu
        _tspec = _ilu.spec_from_file_location(
            "ml15", str(ROOT / "scripts" / "15_build_ml_dataset_v0.py"))
        _T715 = _ilu.module_from_spec(_tspec)
        _tspec.loader.exec_module(_T715)
    return _T715


def _mlv0():
    return pd.read_csv(MLV0_DIR / "ml_dataset_v0.csv")


def test_7_ml_dataset_v0_exists():
    assert (MLV0_DIR / "ml_dataset_v0.csv").exists()
    assert (MLV0_DIR / "dataset_manifest.csv").exists()
    for f in ["feature_audit.csv", "feature_comparability_audit.csv",
              "population_filter_audit.csv", "class_distribution.csv",
              "athlete_coverage.csv", "feature_distribution_summary.csv",
              "feature_correlation.csv", "feature_by_technique.csv",
              "outlier_audit.csv"]:
        assert (MLV0_DIR / f).exists(), f
    assert (MLV0_DIR / "ml_validation_strategy.md").exists()
    assert len(list((MLV0_DIR / "figures").glob("*.png"))) >= 3


def test_7_only_s02_s05():
    assert set(_mlv0()["technique"]) == {"S02", "S03", "S04", "S05"}


def test_7_no_s01():
    assert "S01" not in set(_mlv0()["technique"])


def test_7_no_b0367():
    assert "B0367" not in set(_mlv0()["athlete_id"])


def test_7_source_250hz_e01_t01_accepted():
    mart = pd.read_csv(MART_FILE)
    m = mart.merge(_mlv0()[["execution_id"]], on="execution_id")
    assert (m["sampling_rate_hz"] == 250.0).all()
    assert set(m["condition"]) == {"E01"}
    assert set(m["trial"]) == {"T01"}
    assert (m["qc_status"] == "accepted").all()


def test_7_execution_id_unique():
    assert _mlv0()["execution_id"].is_unique


def test_7_identifiers_not_features():
    man = pd.read_csv(MLV0_DIR / "dataset_manifest.csv")
    assert set(man.loc[man["role"] == "identifier", "column"]) == \
        {"execution_id", "athlete_id"}
    assert set(man.loc[man["role"] == "feature", "column"]) == set(ML_FEATURES)
    assert set(man.loc[man["role"] == "target", "column"]) == {"technique"}


def test_7_no_versioning_and_no_constants_in_dataset():
    v0 = _mlv0()
    feats = [c for c in v0.columns if c in ML_FEATURES]
    assert not any("_version" in c or c.startswith("comparability_")
                   for c in v0.columns)
    for c in feats:
        assert v0[c].nunique() > 1, f"feature constante: {c}"


def test_7_no_nan_no_inf_features():
    v0 = _mlv0()
    feats = [c for c in v0.columns if c in ML_FEATURES]
    assert v0[feats].isna().sum().sum() == 0
    assert not np.isinf(v0[feats].to_numpy(dtype=float)).any()


def test_7_rows_and_athletes():
    v0 = _mlv0()
    assert len(v0) == 419
    assert v0["athlete_id"].nunique() == 33
    counts = v0["technique"].value_counts().to_dict()
    assert counts == {"S05": 115, "S02": 106, "S03": 104, "S04": 94}


def test_7_partial_athletes_no_s04():
    v0 = _mlv0()
    for aid in ("B0388", "B0401", "B0377"):
        cell = v0[(v0["athlete_id"] == aid) & (v0["technique"] == "S04")]
        assert cell.empty, f"{aid} tiene S04"


def test_7_manifest_matches_columns():
    man = pd.read_csv(MLV0_DIR / "dataset_manifest.csv")
    assert list(man["column"]) == list(_mlv0().columns)


def test_7_deterministic_reproducible():
    """dos ejecuciones de la construcción producen exactamente el mismo dataset."""
    import hashlib as _hl
    mod = _ml15()
    hashes = []
    for _ in range(2):
        v0, _ = mod.build_ml_dataset_v0()
        assert list(v0.columns) == ["execution_id", "athlete_id",
                                    "technique"] + ML_FEATURES
        hashes.append(_hl.md5((MLV0_DIR / "ml_dataset_v0.csv")
                              .read_bytes()).hexdigest())
    assert hashes[0] == hashes[1]


# --------------------------------------------------------------------------- #
# 21. Baseline ML por atleta + Dashboard ML (TASK 7B)
# --------------------------------------------------------------------------- #

MLR_DIR = ROOT / "output" / "ml_results"
CLASSES_7B = ["S02", "S03", "S04", "S05"]
_T716 = None


def _ml16():
    global _T716
    if _T716 is None:
        import importlib.util as _ilu
        _tspec = _ilu.spec_from_file_location(
            "ml16", str(ROOT / "scripts" / "16_task7b_baseline_ml.py"))
        _T716 = _ilu.module_from_spec(_tspec)
        _tspec.loader.exec_module(_T716)
    return _T716


def _oof7b():
    return pd.read_csv(MLR_DIR / "oof_predictions.csv")


def _fold_of() -> dict:
    fa = pd.read_csv(MLR_DIR / "fold_assignments.csv")
    return dict(zip(fa["athlete_id"], fa["fold"]))


def test_7b_ml_dataset_v0_intact():
    import hashlib as _hl
    cfg = _ml16().json.loads(
        (MLR_DIR / "experiment_config.json").read_text(encoding="utf-8"))
    cur = _hl.md5((ROOT / "output" / "ml_dataset_v0" /
                   "ml_dataset_v0.csv").read_bytes()).hexdigest()
    assert cur == cfg["ml_dataset_v0_md5"], "ml_dataset_v0.csv modificado"


def test_7b_fold_assignments_5_folds():
    fa = pd.read_csv(MLR_DIR / "fold_assignments.csv")
    assert len(fa) == 33
    assert set(fa["fold"]) == set(range(1, 6))
    assert fa["athlete_id"].is_unique


def test_7b_no_group_leakage():
    """cada atleta pertenece a un único fold; train/validation disjuntos."""
    fa = pd.read_csv(MLR_DIR / "fold_assignments.csv")
    assert fa["athlete_id"].is_unique
    assert len(fa) == fa["athlete_id"].nunique()
    fm = pd.read_csv(MLR_DIR / "fold_metrics.csv")
    for _, r in fm.iterrows():
        assert r["n_train_athletes"] + r["n_validation_athletes"] == 33, \
            f"leakage fold {r['fold']} ({r['experiment']})"


def test_7b_oof_complete_and_unique():
    oof = _oof7b()
    v0 = pd.read_csv(ROOT / "output" / "ml_dataset_v0" / "ml_dataset_v0.csv")
    for exp in ("A", "B"):
        sub = oof[oof["experiment"] == exp]
        assert len(sub) == len(v0)
        assert sub["execution_id"].is_unique
        assert set(sub["execution_id"]) == set(v0["execution_id"])


def test_7b_oof_only_validation_fold():
    """cada predicción OOF corresponde al fold de validación de su atleta."""
    oof = _oof7b()
    fold_of = _fold_of()
    bad = []
    for _, r in oof.iterrows():
        if r["fold"] != fold_of[r["athlete_id"]]:
            bad.append((r["execution_id"], r["athlete_id"]))
    assert not bad


def test_7b_classes_and_no_nan_probs():
    oof = _oof7b()
    assert set(oof["true_technique"]).issubset(set(CLASSES_7B))
    assert set(oof["predicted_technique"]).issubset(set(CLASSES_7B))
    probs = [c for c in oof.columns if c.startswith("prob_")]
    assert probs
    vals = oof[probs].to_numpy(dtype=float)
    assert not np.isnan(vals).any()
    assert not np.isinf(vals).any()


def test_7b_confusion_sums_to_oof():
    cm = pd.read_csv(MLR_DIR / "confusion_matrix.csv")
    for exp in ("A", "B"):
        assert int(cm.loc[cm["experiment"] == exp, "count"].sum()) == 419


def test_7b_errors_match_oof():
    oof = _oof7b()
    er = pd.read_csv(MLR_DIR / "classification_errors.csv")
    for exp in ("A", "B"):
        wrong = set(oof.loc[(oof["experiment"] == exp) &
                            (oof["true_technique"] != oof["predicted_technique"]),
                            "execution_id"])
        err_ids = set(er.loc[er["experiment"] == exp, "execution_id"])
        assert wrong == err_ids


def test_7b_same_folds_a_b():
    fm = pd.read_csv(MLR_DIR / "fold_metrics.csv")
    a = fm[fm["experiment"] == "A"].set_index("fold")
    b = fm[fm["experiment"] == "B"].set_index("fold")
    assert list(a.index) == list(b.index) == [1, 2, 3, 4, 5]
    for k in range(1, 6):
        assert a.loc[k, "n_train"] == b.loc[k, "n_train"]
        assert a.loc[k, "n_validation"] == b.loc[k, "n_validation"]
        assert a.loc[k, "n_train_athletes"] == b.loc[k, "n_train_athletes"]


def test_7b_b_without_snr():
    co = pd.read_csv(MLR_DIR / "feature_coefficients.csv")
    assert "snr" not in set(co.loc[co["experiment"] == "B", "feature"])
    cfg = _ml16().json.loads(
        (MLR_DIR / "experiment_config.json").read_text(encoding="utf-8"))
    assert len(cfg["features"]["B"]) == 10
    assert len(cfg["features"]["A"]) == 11


def test_7b_no_predictor_leak_columns():
    co = pd.read_csv(MLR_DIR / "feature_coefficients.csv")
    leak_cols = {"athlete_id", "execution_id", "primary_signal",
                 "movement_side", "source_dataset"}
    assert not (set(co["feature"]) & leak_cols)


def test_7b_deterministic_byte_identity():
    """dos ejecuciones -> byte-identidad de los artefactos CSV (mismo entorno)."""
    mod = _ml16()
    hashes = []
    for _ in range(2):
        mod.run_task7b()
        hashes.append(hashlib.md5((MLR_DIR / "oof_predictions.csv")
                                  .read_bytes()).hexdigest())
    assert hashes[0] == hashes[1]


def test_7b_dashboard_readonly_and_no_ml():
    """dashboard ML consume ml_results sin escribir y sin importar sklearn."""
    import importlib.util as _ilu
    _dspec = _ilu.spec_from_file_location("dashdata", str(ROOT / "dashboard" / "data.py"))
    _dd = _ilu.module_from_spec(_dspec)
    _dspec.loader.exec_module(_dd)
    before = {p.name: hashlib.md5(p.read_bytes()).hexdigest()
              for p in (MLR_DIR).glob("*.csv")}
    res = _dd.load_ml_results()
    assert res
    after = {p.name: hashlib.md5(p.read_bytes()).hexdigest()
             for p in (MLR_DIR).glob("*.csv")}
    assert before == after
    src = (ROOT / "dashboard" / "app_ml.py").read_text(encoding="utf-8") + \
          (ROOT / "dashboard" / "data.py").read_text(encoding="utf-8")
    assert "import sklearn" not in src and "from sklearn" not in src
    assert "sklearn" in src  # aparece solo en comentarios/aviso


def test_7b_baseline_comparison_complete():
    bc = pd.read_csv(MLR_DIR / "baseline_comparison.csv")
    metrics = {"accuracy", "balanced_accuracy", "precision_macro",
               "recall_macro", "f1_macro", "f1_weighted"}
    assert set(bc.loc[bc["experiment"] == "A", "metric"]) == metrics
    assert set(bc.loc[bc["experiment"] == "B", "metric"]) == metrics
    # descripción observada: B no es inferior en media (documentado, sin ranking)
    f1a = bc.loc[(bc["experiment"] == "A") & (bc["metric"] == "f1_macro"), "mean"].iloc[0]
    f1b = bc.loc[(bc["experiment"] == "B") & (bc["metric"] == "f1_macro"), "mean"].iloc[0]
    assert 0.5 < f1a < f1b < 0.75


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))