"""Smoke test for the standalone Streamlit app model and packaging."""
from __future__ import annotations
import sys
import types
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
APP_CODE = (ROOT / "app.py").read_text(encoding="utf-8")


def test_packaging():
    assert "from src.microfluidic_core import" not in APP_CODE
    assert (ROOT / "data" / "Comprehensive_normalized.xlsx").is_file()
    assert (ROOT / "requirements.txt").is_file()
    assert APP_CODE.startswith('"""Microfluidic Droplet AI - self-contained')


def _get_model_namespace():
    """Execute only the model portion (before UI), without a running webserver."""
    code = APP_CODE.split("# ============================= STREAMLIT UI ===============================", 1)[0]
    # app's import of streamlit is not needed for model logic; strip it for fast unit tests.
    code = code.replace("import streamlit as st\n", "")
    module = types.ModuleType("microfluidic_app_model")
    sys.modules[module.__name__] = module
    exec(compile(code, str(ROOT / "app.py"), "exec"), module.__dict__, module.__dict__)
    return module.__dict__


def test_real_data_prediction_and_inverse_design():
    m = _get_model_namespace()
    data = m["read_dataset"](ROOT / "data" / "Comprehensive_normalized.xlsx")
    assert len(data) == 868
    bundle = m["train_model"](data)
    assert len(bundle.target_columns) == 3
    assert len(bundle.base_feature_columns) >= 10
    assert np.isfinite(bundle.metrics["R2"]).all()
    trial = m["make_input_template"](bundle, rows=3)
    predictions = m["predict_outputs"](bundle, trial)
    assert predictions.shape == (3, 3)
    assert np.isfinite(predictions.to_numpy()).all()
    unc = m["prediction_uncertainty"](bundle, trial)
    assert unc.shape == (3, 3)
    assert (unc.to_numpy() >= 0).all()
    results = m["recommend_conditions"](bundle, {
        t: bundle.target_profile[t]["median"] for t in bundle.target_columns
    }, n_candidates=40, top_k=5)
    assert results.shape[0] == 5
    assert np.isfinite(results["Decision score (lower is better)"]).all()
    test_csv = m["read_dataset"](ROOT / "data" / "Comprehensive_normalized.csv")
    assert test_csv.shape == data.shape
