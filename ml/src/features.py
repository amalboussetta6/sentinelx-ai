"""Sélection des features et features temporelles CAUSALES, calculées moteur par moteur.

Règles (README §9) :
- aucune variable future (unit_id, RUL, dernier cycle, progression de vie) parmi les features ;
- les features temporelles n'utilisent que le passé du moteur (cycles <= t) ;
- aucune fenêtre ne traverse deux trajectoires.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import ID_COLUMNS, SENSOR_COLUMNS, SETTING_COLUMNS

FORBIDDEN_DEFAULT = ("RUL", "last_cycle", "life_pct", "max_cycle", "progress_pct", "unit_id")


def candidate_features(feature_groups: dict) -> list[str]:
    """Les variables non constantes de l'EDA (17 pour FD001), dans l'ordre du schéma."""
    return [c for c in SETTING_COLUMNS + SENSOR_COLUMNS if c not in feature_groups["constant"]]


def check_features(features, forbidden=FORBIDDEN_DEFAULT) -> None:
    """Garde-fou anti-fuite : aucune variable interdite, aucune feature dupliquée."""
    features = list(features)
    leaked = set(forbidden).intersection(features)
    if leaked:
        raise ValueError(f"Variables interdites comme features : {sorted(leaked)}")
    if len(set(features)) != len(features):
        raise ValueError("Feature dupliquée")


def _sorted(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(ID_COLUMNS)


def add_rolling_mean(df: pd.DataFrame, columns, window: int) -> tuple[pd.DataFrame, list[str]]:
    """Moyenne des `window` derniers cycles du même moteur (passé uniquement ; début de vie : cycles disponibles)."""
    out = _sorted(df).copy()
    rolled = (out.groupby("unit_id")[list(columns)]
                 .rolling(window=window, min_periods=1).mean()
                 .reset_index(level=0, drop=True))
    names = [f"{c}_mean{window}" for c in columns]
    out[names] = rolled.loc[out.index, list(columns)].to_numpy()
    return out.loc[df.index], names


def add_rolling_slope(df: pd.DataFrame, columns, window: int) -> tuple[pd.DataFrame, list[str]]:
    """Pente (unités par cycle) de la droite des moindres carrés sur les `window` derniers cycles du moteur.

    Calcul vectorisé par sommes glissantes : pente = cov(t, x) / var(t). Avec un seul cycle disponible,
    la pente est indéfinie et vaut 0.
    """
    out = _sorted(df).copy()
    columns = list(columns)
    grouped = out.groupby("unit_id")
    t = out["cycle"].astype(float)

    def rolling_sum(series_frame):
        return (series_frame.groupby(out["unit_id"]).rolling(window=window, min_periods=1).sum()
                .reset_index(level=0, drop=True).loc[out.index])

    n = grouped["cycle"].rolling(window=window, min_periods=1).count().reset_index(level=0, drop=True).loc[out.index]
    sum_t = rolling_sum(t.to_frame("t"))["t"]
    sum_tt = rolling_sum((t * t).to_frame("tt"))["tt"]
    x = out[columns].astype(float)
    sum_x = rolling_sum(x)
    sum_tx = rolling_sum(x.mul(t, axis=0))
    denom = n * sum_tt - sum_t ** 2
    numer = sum_tx.mul(n, axis=0) - sum_x.mul(sum_t, axis=0)
    slope = numer.div(denom.where(denom != 0), axis=0).fillna(0.0)
    names = [f"{c}_slope{window}" for c in columns]
    out[names] = slope[columns].to_numpy()
    return out.loc[df.index], names


def add_early_life_deviation(df: pd.DataFrame, columns, n_early: int) -> tuple[pd.DataFrame, list[str]]:
    """Écart entre la mesure courante et la moyenne des `n_early` premiers cycles du même moteur.

    Corrige en partie l'usure initiale inconnue de chaque moteur (EDA §2). Causal : avant le cycle
    `n_early`, la référence est la moyenne des cycles déjà observés.
    """
    out = _sorted(df).copy()
    columns = list(columns)
    x = out[columns].astype(float)
    early_rank = out.groupby("unit_id").cumcount() < n_early            # True pendant les n_early premiers cycles
    cum_sum = x.where(early_rank, 0.0).groupby(out["unit_id"]).cumsum()  # somme des cycles de référence vus
    cum_n = early_rank.astype(int).groupby(out["unit_id"]).cumsum()
    reference = cum_sum.div(cum_n, axis=0)
    names = [f"{c}_dev{n_early}" for c in columns]
    out[names] = (x - reference).to_numpy()
    return out.loc[df.index], names


FEATURE_BUILDERS = {"mean": add_rolling_mean, "slope": add_rolling_slope, "dev": add_early_life_deviation}


def build_feature_set(df: pd.DataFrame, base_features, spec: dict) -> tuple[pd.DataFrame, list[str]]:
    """Construit un jeu de features à partir d'une spécification déclarative.

    Exemple : spec = {"mean": 30, "slope": 30} ajoute les moyennes et pentes sur 30 cycles des capteurs
    de `base_features`. Les réglages (setting_*) ne reçoivent pas de features temporelles.
    """
    sensors = [f for f in base_features if f.startswith("sensor_")]
    out, features = df, list(base_features)
    for kind, param in spec.items():
        out, names = FEATURE_BUILDERS[kind](out, sensors, param)
        features += names
    check_features(features)
    return out, features
