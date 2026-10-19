import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def toy_engines():
    """Deux moteurs synthétiques : capteur linéaire (pente connue) + bruit déterministe."""
    rows = []
    for unit, length, slope in ((1, 12, 0.5), (2, 8, -2.0)):
        for cycle in range(1, length + 1):
            rows.append({"unit_id": unit, "cycle": cycle, "setting_1": 0.0,
                         "sensor_2": 100 + slope * cycle, "sensor_3": float(cycle % 3)})
    return pd.DataFrame(rows)
