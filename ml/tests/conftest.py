import pandas as pd
import pytest


@pytest.fixture
def two_engines():
    """Deux petits moteurs synthétiques : sensor_2 suit une droite de pente connue."""
    rows = []
    for unit_id, n_cycles, slope in [(1, 12, 0.5), (2, 8, -2.0)]:
        for cycle in range(1, n_cycles + 1):
            rows.append({"unit_id": unit_id, "cycle": cycle, "sensor_2": 100 + slope * cycle})
    return pd.DataFrame(rows)
