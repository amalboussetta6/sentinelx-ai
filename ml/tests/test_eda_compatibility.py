"""Données réelles et décisions de l'EDA (tests ignorés si data/raw est vide)."""
import json
from pathlib import Path

import numpy as np
import pytest

from ml.src.data import COLUMN_NAMES, add_rul, last_cycle_per_engine, load_cmapss

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
EDA = ROOT / "artifacts" / "eda_fd001"
pytestmark = pytest.mark.skipif(not (RAW / "train_FD001.txt").exists(), reason="données FD001 absentes")


def test_load_real_files():
    train = load_cmapss(RAW / "train_FD001.txt")
    test = load_cmapss(RAW / "test_FD001.txt")
    assert list(train.columns) == COLUMN_NAMES and len(train) == 20631 and train["unit_id"].nunique() == 100
    assert len(test) == 13096 and test["unit_id"].nunique() == 100
    assert train.isna().sum().sum() == 0
    assert (train.groupby("unit_id")["cycle"].diff().dropna() == 1).all()       # cycles continus
    assert last_cycle_per_engine(test)["cycle"].min() == 31                     # historique minimum du test


def test_split_is_reproducible_and_by_engine():
    split = json.loads((EDA / "fd001_engine_split.json").read_text(encoding="utf-8"))
    rng = np.random.RandomState(split["random_state"])
    shuffled = rng.permutation(np.arange(1, 101))
    assert sorted(shuffled[:20].tolist()) == split["validation_engines"]
    assert set(split["validation_engines"]).isdisjoint(split["analysis_engines"])
    assert len(split["analysis_engines"]) + len(split["validation_engines"]) == 100


def test_selected_sensors_and_rul():
    groups = json.loads((EDA / "fd001_feature_groups.json").read_text(encoding="utf-8"))
    train = add_rul(load_cmapss(RAW / "train_FD001.txt"))
    assert len(groups["selected_sensors"]) == 14
    assert (train[groups["selected_sensors"]].nunique() > 2).all()
    assert (train[groups["constant_columns"]].nunique() == 1).all()
    assert (train.groupby("unit_id")["RUL"].min() == 0).all()
    assert not set(groups["selected_sensors"]) & set(groups["forbidden_as_feature"])
