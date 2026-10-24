import numpy as np
import pandas as pd
import pytest

from ml.src.evaluation import compute_metrics, engine_folds, evaluate_on_zone
from ml.src.models import create_model


def test_metrics_on_small_example():
    metrics = compute_metrics([10, 50, 100], [12, 40, 110])
    assert metrics["MAE"] == pytest.approx(22 / 3)
    assert metrics["RMSE"] == pytest.approx(np.sqrt(204 / 3))
    assert metrics["R2"] == pytest.approx(1 - 204 / np.sum((np.array([10, 50, 100]) - 160 / 3) ** 2))


def test_perfect_prediction():
    metrics = compute_metrics([1, 2, 3], [1, 2, 3])
    assert metrics["MAE"] == 0 and metrics["RMSE"] == 0 and metrics["R2"] == 1


def test_zone_uses_true_rul_only():
    """Seules les lignes avec un RUL réel <= horizon comptent, et on compare au vrai RUL."""
    true_rul = pd.Series([0, 50, 127, 300])
    predictions = [0, 50, 127, 110]                     # le moteur sain (300) est hors zone
    assert evaluate_on_zone(true_rul, predictions)["RMSE"] == 0


def test_zone_ignores_predictions_outside_zone():
    """Une grosse erreur sur une ligne hors zone (RUL réel > horizon) ne change pas les métriques."""
    true_rul = [10, 20, 200]
    assert evaluate_on_zone(true_rul, [12, 18, 0], horizon=127) == evaluate_on_zone(true_rul, [12, 18, 999], horizon=127)


def test_engine_folds_keep_engines_together():
    df = pd.DataFrame({"unit_id": np.repeat(np.arange(1, 11), 5)})   # 10 moteurs, 5 lignes chacun
    folds = engine_folds(df, n_splits=5)
    assert len(folds) == 5
    controlled = []
    for fit_engines, check_engines in folds:
        assert set(fit_engines).isdisjoint(check_engines)              # aucun moteur des deux côtés
        assert sorted(fit_engines + check_engines) == list(range(1, 11))
        controlled += check_engines
    assert sorted(controlled) == list(range(1, 11))     # chaque moteur est contrôlé exactement une fois


def test_create_model():
    assert type(create_model("Régression linéaire")).__name__ == "LinearRegression"
    forest = create_model("Forêt aléatoire", max_features=1.0)
    assert type(forest).__name__ == "RandomForestRegressor" and forest.max_features == 1.0
    assert create_model("Forêt aléatoire").max_features == 0.33
    with pytest.raises(ValueError):
        create_model("inconnu")
