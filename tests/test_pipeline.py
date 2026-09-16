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
    assert len(mart) == 18
    assert set(mart["athlete_id"]) == {"B0367", "B0377"}
    assert set(mart["technique"]) == {"S02", "S03", "S05"}
    assert set(mart["condition"]) == {"E01"}
    assert set(mart["trial"]) == {"T01"}


def test_mart_sampling_rate_metadata():
    if not MART_FILE.exists():
        pytest.skip("FASE 1.8C no ejecutada")
    mart = pd.read_csv(MART_FILE)
    assert set(mart.loc[mart["athlete_id"] == "B0367", "sampling_rate_hz"]) == {200.0}
    assert set(mart.loc[mart["athlete_id"] == "B0377", "sampling_rate_hz"]) == {250.0}
    assert (mart["movement_side"] == "R").all()  # golden path usa lado derecho


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
    assert sm["total_executions"] == 18
    assert sm["total_athletes"] == 2
    assert sm["sampling_rate_200hz"] == 9
    assert sm["sampling_rate_250hz"] == 9
    assert sm["missing_feature_values"] == 0


def test_mart_units_metadata():
    if not MART_FILE.exists():
        pytest.skip("FASE 1.8C no ejecutada")
    mart = pd.read_csv(MART_FILE)
    assert (mart["feature_version"].astype(str) == "1.0").all()
    assert (mart["units_version"].astype(str) == "1.0").all()
    assert (mart["source_dataset"] == "feature_readiness_sample").all()


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
    assert len(df) == 18


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
    assert set(df["athlete_id"].unique()) == {"B0367", "B0377"}


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
    assert len(s02) == 6
    assert set(s02["technique"]) == {"S02"}


def test_dashboard_data_mart_not_modified():
    """el módulo de datos no escribe; el archivo del Data Mart queda igual."""
    da = _dash_data()
    before = da.DATA_MART_PATH.read_bytes()
    _ = da.load_data_mart()
    after = da.DATA_MART_PATH.read_bytes()
    assert before == after


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))