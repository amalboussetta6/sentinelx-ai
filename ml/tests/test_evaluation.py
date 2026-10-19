import numpy as np
import pandas as pd
import pytest

from ml.src.evaluation import (cap_target, clip_negative, cross_validate, evaluate_predictions, group_folds,
                               nasa_score, paired_gain, regression_metrics)
from ml.src.models import make_model


def test_regression_metrics_known_values():
    m = regression_metrics([0, 10, 20], [1, 10, 17], with_nasa=True)
    assert m["MAE"] == pytest.approx(4 / 3)
    assert m["RMSE"] == pytest.approx(np.sqrt(10 / 3))
    assert "NASA_score" in m


def test_nasa_score_penalises_late_predictions_more():
    assert nasa_score([50], [60]) > nasa_score([50], [40]) > 0
    assert nasa_score([50], [50]) == 0


def test_cap_and_clip():
    assert cap_target([10, 200], 125).tolist() == [10, 125]
    assert cap_target([10, 200], None).tolist() == [10, 200]
    assert clip_negative([-3, 4]).tolist() == [0, 4]


def test_evaluate_predictions_zone_uses_true_rul():
    res = evaluate_predictions(np.array([0, 50, 300]), np.array([0, 50, 125]), horizon=127)
    assert res["RMSE_zone"] == 0 and res["RMSE_global"] > 0


def test_group_folds_keep_engines_together():
    groups = np.repeat(np.arange(10), 5)
    for train_idx, test_idx in group_folds(groups, 5):
        assert set(groups[train_idx]).isdisjoint(groups[test_idx])


@pytest.mark.parametrize("name,params", [("LinearRegression", {}), ("Ridge", {"alpha": 1.0}),
                                         ("RandomForest", {"n_estimators": 10, "max_depth": 3})])
def test_cross_validate_runs_for_every_model(name, params):
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(10), 20)
    X = pd.DataFrame({"a": rng.normal(size=200), "b": rng.normal(size=200)})
    y = np.clip(50 + 10 * X["a"].to_numpy() + rng.normal(size=200), 0, None)
    folds = group_folds(groups, 5)
    per_fold, oof = cross_validate(lambda: make_model(name, n_jobs=1, **params), X, y, groups, folds, horizon=127)
    assert per_fold.shape[0] == 5 and (oof >= 0).all()


def test_paired_gain():
    ref = pd.DataFrame({"RMSE_zone": [10, 12, 11]})
    cand = pd.DataFrame({"RMSE_zone": [9, 11, 11.5]})
    g = paired_gain(ref, cand)
    assert g["folds_improved"] == 2 and g["gain_mean"] == pytest.approx((1 + 1 - 0.5) / 3)
