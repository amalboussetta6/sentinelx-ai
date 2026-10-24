"""Features d'historique calculées moteur par moteur, avec le passé uniquement.

Mêmes fonctions, mêmes noms et mêmes calculs que dans notebooks/02_baseline_fd001.ipynb (section 3).
Le notebook vérifie à la fin (section 11) que ce module produit exactement les mêmes features.
"""
import numpy as np
import pandas as pd


def compute_slope(values):
    """Pente de la droite qui passe au mieux par ces valeurs (en unités par cycle)."""
    cycles = np.arange(len(values))                  # 0, 1, 2, ... : la position de chaque cycle
    slope, intercept = np.polyfit(cycles, values, 1)  # droite y = slope * x + intercept
    return slope


def rolling_mean(values, window):
    """Moyenne des `window` dernières valeurs (ou des valeurs disponibles en début de vie)."""
    return values.rolling(window, min_periods=1).mean()


def rolling_slope(values, window):
    """Pour chaque cycle : pente des `window` dernières valeurs (0 au premier cycle).

    `values` est un tableau NumPy contenant les mesures d'un seul moteur, dans l'ordre des cycles.
    """
    slopes = []
    for position in range(len(values)):
        first = max(0, position - window + 1)        # début de la fenêtre (jamais avant le 1er cycle)
        recent_values = values[first:position + 1]   # valeurs déjà observées, au plus `window`
        if len(recent_values) < 2:
            slopes.append(0.0)                       # une seule valeur : pas de pente
        else:
            slopes.append(compute_slope(recent_values))
    return slopes


def delta_from_start(values, cycles, n_start):
    """Écart entre la mesure actuelle et le niveau de départ du moteur.

    Référence = moyenne des `n_start` premiers cycles ; avant le cycle `n_start`, moyenne des cycles déjà vus.
    """
    reference = values.expanding().mean()                # avant le cycle n_start : moyenne des cycles déjà vus
    start_level = values[cycles <= n_start].mean()       # moyenne des n_start premiers cycles
    reference[cycles > n_start] = start_level            # ensuite : référence figée
    return values - reference


def add_history_features(df, sensors, window=30, n_start=20):
    """Ajoute les colonnes <capteur>_mean, <capteur>_slope et <capteur>_delta, en traitant les moteurs un par un."""
    df = df.sort_values(["unit_id", "cycle"])
    engine_tables = []
    for engine_id, engine in df.groupby("unit_id"):
        engine = engine.copy()
        for sensor in sensors:
            values = engine[sensor]
            engine[sensor + "_mean"] = rolling_mean(values, window)
            engine[sensor + "_slope"] = rolling_slope(values.to_numpy(dtype=float), window)
            engine[sensor + "_delta"] = delta_from_start(values, engine["cycle"], n_start)
        engine_tables.append(engine)
    return pd.concat(engine_tables)


def history_feature_names(sensors):
    """Noms des 3 x len(sensors) features, dans l'ordre utilisé par la baseline (toutes les _mean, puis _slope, puis _delta)."""
    names = []
    for suffix in ["_mean", "_slope", "_delta"]:
        for sensor in sensors:
            names.append(sensor + suffix)
    return names
