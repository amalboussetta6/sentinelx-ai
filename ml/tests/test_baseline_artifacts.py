"""Cohérence entre le notebook de baseline, ml/src et les fichiers sauvegardés (ignoré s'ils sont absents)."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from ml.src.data import add_rul, last_cycle_per_engine, load_cmapss
from ml.src.evaluation import compute_metrics, engine_folds
from ml.src.features import add_history_features, history_feature_names

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
BASELINE = ROOT / "artifacts" / "baseline_fd001"
pytestmark = pytest.mark.skipif(not (BASELINE / "baseline_rul_fd001.joblib").exists()
                                or not (RAW / "test_FD001.txt").exists(),
                                reason="artefacts de la baseline ou données absents")


@pytest.fixture(scope="module")
def metrics():
    return json.loads((BASELINE / "baseline_metrics.json").read_text(encoding="utf-8"))


def test_metrics_file_is_consistent(metrics):
    assert metrics["features"] == history_feature_names(metrics["sensors"])
    assert metrics["selected_model"] in metrics["test_metrics"]
    assert len(metrics["cross_validation"]["fold_engines"]) == 5


def test_cross_validation_folds_can_be_rebuilt(metrics):
    """S4 doit pouvoir retrouver exactement les 5 tours de validation croisée."""
    train = load_cmapss(RAW / "train_FD001.txt")
    split = json.loads((ROOT / "artifacts" / "eda_fd001" / "fd001_engine_split.json").read_text(encoding="utf-8"))
    train_part = train[train["unit_id"].isin(split["analysis_engines"])]
    rebuilt = []
    for fit_engines, check_engines in engine_folds(train_part):
        rebuilt.append(check_engines)
    assert rebuilt == metrics["cross_validation"]["fold_engines"]


def test_saved_model_reproduces_test_predictions(metrics):
    """Le modèle sauvegardé, rechargé, redonne les prédictions et le RMSE enregistrés."""
    bundle = joblib.load(BASELINE / "baseline_rul_fd001.joblib")
    assert bundle["model_name"] == metrics["selected_model"] and bundle["features"] == metrics["features"]
    test = add_history_features(load_cmapss(RAW / "test_FD001.txt"), bundle["sensors"],
                                window=bundle["window"], n_start=bundle["n_start"])
    last = last_cycle_per_engine(test)
    predictions = np.clip(bundle["model"].predict(last[bundle["features"]]), 0, None)
    saved = pd.read_csv(BASELINE / "test_predictions.csv")
    np.testing.assert_allclose(predictions, saved[bundle["model_name"]], atol=1e-3)
    rmse = compute_metrics(saved["RUL_true"], saved[bundle["model_name"]])["RMSE"]
    assert round(rmse, 4) == pytest.approx(metrics["test_metrics"][bundle["model_name"]]["RMSE"], abs=1e-3)


def test_training_features_match_rul(metrics):
    train = add_history_features(add_rul(load_cmapss(RAW / "train_FD001.txt")), metrics["sensors"])
    assert np.isfinite(train[metrics["features"]].to_numpy()).all()
    assert (train["RUL"] >= 0).all()


def test_saved_model_settings_match_training(metrics):
    """Les réglages enregistrés avec le modèle sont ceux de baseline_metrics.json, et le modèle
    attend exactement les 42 features, dans l'ordre utilisé pour l'entraînement."""
    bundle = joblib.load(BASELINE / "baseline_rul_fd001.joblib")
    model = bundle["model"]
    assert list(model.feature_names_in_) == bundle["features"] == metrics["features"]
    assert bundle["rul_cap"] == metrics["rul_cap"][bundle["model_name"]]
    assert bundle["window"] == metrics["window"] and bundle["n_start"] == metrics["n_start"]
    assert bundle["max_features"] == metrics["random_forest_params"]["max_features"] == model.max_features
