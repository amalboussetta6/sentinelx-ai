import numpy as np
import pytest

from ml.src.features import (add_early_life_deviation, add_rolling_mean, add_rolling_slope,
                             build_feature_set, check_features)

BUILDERS = [(add_rolling_mean, 3), (add_rolling_slope, 4), (add_early_life_deviation, 3)]


@pytest.mark.parametrize("builder,param", BUILDERS)
def test_features_are_causal(toy_engines, builder, param):
    """Modifier le futur d'un moteur ne doit pas changer ses features passées."""
    base, names = builder(toy_engines, ["sensor_2"], param)
    future = toy_engines.copy()
    future.loc[future["cycle"] > 5, "sensor_2"] += 1_000
    modified, _ = builder(future, ["sensor_2"], param)
    past = toy_engines["cycle"] <= 5
    np.testing.assert_allclose(base.loc[past, names], modified.loc[past, names])


@pytest.mark.parametrize("builder,param", BUILDERS)
def test_features_do_not_cross_engines(toy_engines, builder, param):
    """Les features du moteur 2 ne dépendent pas du moteur 1."""
    base, names = builder(toy_engines, ["sensor_2"], param)
    other = toy_engines.copy()
    other.loc[other["unit_id"] == 1, "sensor_2"] *= 50
    modified, _ = builder(other, ["sensor_2"], param)
    engine2 = toy_engines["unit_id"] == 2
    np.testing.assert_allclose(base.loc[engine2, names], modified.loc[engine2, names])


def test_rolling_mean_values(toy_engines):
    out, (name,) = add_rolling_mean(toy_engines, ["sensor_2"], 3)
    engine1 = out[out["unit_id"] == 1][name].to_numpy()
    assert engine1[0] == pytest.approx(100.5)                 # premier cycle : seule valeur disponible
    assert engine1[4] == pytest.approx(100 + 0.5 * 4)          # moyenne des cycles 3, 4, 5


def test_rolling_slope_recovers_linear_trend(toy_engines):
    out, (name,) = add_rolling_slope(toy_engines, ["sensor_2"], 4)
    e1, e2 = out[out["unit_id"] == 1][name].to_numpy(), out[out["unit_id"] == 2][name].to_numpy()
    assert e1[0] == 0.0                                        # un seul cycle : pente indéfinie -> 0
    np.testing.assert_allclose(e1[1:], 0.5)
    np.testing.assert_allclose(e2[1:], -2.0)


def test_early_life_deviation(toy_engines):
    out, (name,) = add_early_life_deviation(toy_engines, ["sensor_2"], 3)
    e1 = out[out["unit_id"] == 1][name].to_numpy()
    reference = np.mean([100.5, 101.0, 101.5])
    assert e1[-1] == pytest.approx(100 + 0.5 * 12 - reference)


def test_build_feature_set_keeps_order_and_rejects_leaks(toy_engines):
    out, feats = build_feature_set(toy_engines, ["setting_1", "sensor_2"], {"mean": 3, "slope": 3})
    assert feats == ["setting_1", "sensor_2", "sensor_2_mean3", "sensor_2_slope3"]
    assert out.index.equals(toy_engines.index)
    with pytest.raises(ValueError):
        check_features(["sensor_2", "unit_id"])
