from ml.src.data import add_rul, last_cycle_per_engine


def test_rul_counts_down_to_zero(two_engines):
    df = add_rul(two_engines)
    engine_1 = df[df["unit_id"] == 1]
    assert engine_1["RUL"].tolist() == list(range(11, -1, -1))   # 11, 10, ..., 0
    assert (df.groupby("unit_id")["RUL"].min() == 0).all()


def test_rul_does_not_change_input(two_engines):
    before = two_engines.copy()
    add_rul(two_engines)
    assert two_engines.equals(before)


def test_last_cycle_per_engine(two_engines):
    last = last_cycle_per_engine(two_engines)
    assert last["unit_id"].tolist() == [1, 2] and last["cycle"].tolist() == [12, 8]
