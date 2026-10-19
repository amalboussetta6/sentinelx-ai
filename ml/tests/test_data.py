import numpy as np
import pandas as pd
import pytest

from ml.src.data import GroundTruthGuard, add_rul, check_rul, last_observed_cycles, split_by_engine


def test_add_rul_counts_down_to_zero(toy_engines):
    df = add_rul(toy_engines)
    check_rul(df)
    first = df.groupby("unit_id").first()
    assert first.loc[1, "RUL"] == 11 and first.loc[2, "RUL"] == 7
    assert "max_cycle" not in df.columns and "last_cycle" not in df.columns


def test_add_rul_does_not_modify_input(toy_engines):
    before = toy_engines.copy()
    add_rul(toy_engines)
    pd.testing.assert_frame_equal(before, toy_engines)


def test_check_rul_detects_broken_target(toy_engines):
    df = add_rul(toy_engines)
    df.loc[df.index[3], "RUL"] = 99
    with pytest.raises(AssertionError):
        check_rul(df)


def test_split_by_engine_is_disjoint_and_complete(toy_engines):
    analysis, validation = split_by_engine(toy_engines, [1], [2])
    assert set(analysis["unit_id"]) == {1} and set(validation["unit_id"]) == {2}
    with pytest.raises(ValueError):
        split_by_engine(toy_engines, [1, 2], [2])
    with pytest.raises(ValueError):
        split_by_engine(toy_engines, [1], [])


def test_last_observed_cycles(toy_engines):
    last = last_observed_cycles(toy_engines)
    assert last["unit_id"].tolist() == [1, 2] and last["cycle"].tolist() == [12, 8]


def test_ground_truth_guard_requires_freeze_and_single_read(tmp_path):
    path = tmp_path / "RUL_FD001.txt"
    path.write_text("112\n98\n")
    guard = GroundTruthGuard(path)
    with pytest.raises(RuntimeError):
        guard.read()
    guard.freeze()
    assert guard.read().tolist() == [112, 98]
    with pytest.raises(RuntimeError):
        guard.read()
