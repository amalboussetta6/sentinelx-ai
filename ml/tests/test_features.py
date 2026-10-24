import numpy as np
import pandas as pd
import pytest

from ml.src.features import (add_history_features, compute_slope, delta_from_start, history_feature_names,
                             rolling_mean, rolling_slope)

FEATURES = history_feature_names(["sensor_2"])


def test_compute_slope():
    assert compute_slope(np.array([10, 12, 14, 16])) == pytest.approx(2.0)
    assert compute_slope(np.array([5, 5, 5])) == pytest.approx(0.0)


def test_slope_recovers_known_trend(two_engines):
    df = add_history_features(two_engines, ["sensor_2"], window=4, n_start=3)
    engine_1 = df[df["unit_id"] == 1]["sensor_2_slope"].to_numpy()
    engine_2 = df[df["unit_id"] == 2]["sensor_2_slope"].to_numpy()
    assert engine_1[0] == 0                              # un seul cycle : pas de pente
    np.testing.assert_allclose(engine_1[1:], 0.5)
    np.testing.assert_allclose(engine_2[1:], -2.0)


def test_mean_and_delta_values(two_engines):
    df = add_history_features(two_engines, ["sensor_2"], window=3, n_start=3)
    engine_1 = df[df["unit_id"] == 1]
    assert engine_1["sensor_2_mean"].iloc[0] == pytest.approx(100.5)           # premier cycle seul
    assert engine_1["sensor_2_mean"].iloc[4] == pytest.approx(102.0)           # cycles 3, 4, 5
    start = np.mean([100.5, 101.0, 101.5])                                     # 3 premiers cycles
    assert engine_1["sensor_2_delta"].iloc[-1] == pytest.approx(106.0 - start)
    assert engine_1["sensor_2_delta"].iloc[1] == pytest.approx(101.0 - 100.75) # cycle 2 : moyenne des cycles déjà vus


def test_future_does_not_change_past(two_engines):
    """Modifier les cycles futurs ne doit pas changer les features passées (pas de fuite du futur)."""
    before = add_history_features(two_engines, ["sensor_2"], window=4, n_start=3)
    modified = two_engines.copy()
    modified.loc[modified["cycle"] > 5, "sensor_2"] += 1000
    after = add_history_features(modified, ["sensor_2"], window=4, n_start=3)
    past = before["cycle"] <= 5
    np.testing.assert_allclose(before.loc[past, FEATURES], after.loc[past, FEATURES])


def test_engines_are_not_mixed(two_engines):
    """Les features du moteur 2 ne dépendent pas du moteur 1."""
    before = add_history_features(two_engines, ["sensor_2"], window=4, n_start=3)
    modified = two_engines.copy()
    modified.loc[modified["unit_id"] == 1, "sensor_2"] *= 50
    after = add_history_features(modified, ["sensor_2"], window=4, n_start=3)
    engine_2 = before["unit_id"] == 2
    np.testing.assert_allclose(before.loc[engine_2, FEATURES], after.loc[engine_2, FEATURES])


def test_feature_names_contain_no_forbidden_variable():
    """Aucune variable qui révèle le futur ou l'identité du moteur parmi les features."""
    names = history_feature_names(["sensor_2", "sensor_3"])
    assert len(names) == 6 and len(set(names)) == 6
    for forbidden in ["unit_id", "RUL", "cycle", "last_cycle", "life_pct"]:
        assert forbidden not in names


def test_input_is_not_modified(two_engines):
    before = two_engines.copy()
    add_history_features(two_engines, ["sensor_2"])
    assert two_engines.equals(before)


def test_rolling_mean_uses_available_cycles_at_start():
    values = pd.Series([100.0, 102.0, 101.0, 105.0])
    np.testing.assert_allclose(rolling_mean(values, window=3), [100, 101, 101, 308 / 3])


def test_rolling_slope_first_cycle_and_window():
    """Premier cycle : pente 0 ; ensuite, seules les `window` dernières valeurs comptent."""
    values = np.array([0.0, 10.0, 11.0, 12.0])
    slopes = rolling_slope(values, window=3)
    assert slopes[0] == 0
    assert slopes[1] == pytest.approx(10.0)               # fenêtre [0, 10]
    assert slopes[3] == pytest.approx(1.0)                # fenêtre [10, 11, 12] : le 0 est sorti de la fenêtre


def test_delta_reference_before_and_after_start():
    """Avant n_start : moyenne des cycles déjà vus ; après : moyenne figée des n_start premiers cycles."""
    values = pd.Series([100.0, 102.0, 101.0, 105.0])
    cycles = pd.Series([1, 2, 3, 4])
    np.testing.assert_allclose(delta_from_start(values, cycles, n_start=2), [0, 1, 0, 4])


def test_fictional_engine_example_of_the_notebook():
    """Le moteur fictif de 5 cycles du notebook (section 3) : mêmes valeurs que le calcul à la main."""
    toy = pd.DataFrame({"unit_id": 1, "cycle": [1, 2, 3, 4, 5], "sensor_x": [100, 102, 101, 105, 108]})
    df = add_history_features(toy, ["sensor_x"], window=3, n_start=2)
    np.testing.assert_allclose(df["sensor_x_mean"], [100, 101, 101, 308 / 3, 314 / 3])
    np.testing.assert_allclose(df["sensor_x_slope"], [0, 2, 0.5, 1.5, 3.5])
    np.testing.assert_allclose(df["sensor_x_delta"], [0, 1, 0, 4, 7])


def test_feature_names_order():
    assert history_feature_names(["sensor_2", "sensor_3"]) == [
        "sensor_2_mean", "sensor_3_mean", "sensor_2_slope", "sensor_3_slope", "sensor_2_delta", "sensor_3_delta"]
