"""Fabrique des modèles baseline : Linear Regression, Ridge et Random Forest.

Chaque appel renvoie un Pipeline NEUF. Les modèles linéaires sont précédés d'un StandardScaler
(ajusté sur les données d'apprentissage uniquement) ; la Random Forest n'en a pas besoin,
car un arbre ne compare une feature qu'à des seuils sur cette même feature.
"""
from __future__ import annotations

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODEL_NAMES = ("LinearRegression", "Ridge", "RandomForest")


def make_model(name: str, random_state: int = 42, n_jobs: int = -1, **params) -> Pipeline:
    """Pipeline prêt à l'emploi pour le modèle `name` avec ses hyperparamètres `params`."""
    if name == "LinearRegression":
        return Pipeline([("scaler", StandardScaler()), ("regressor", LinearRegression(**params))])
    if name == "Ridge":
        return Pipeline([("scaler", StandardScaler()), ("regressor", Ridge(**params))])
    if name == "RandomForest":
        forest = RandomForestRegressor(random_state=random_state, n_jobs=n_jobs, **params)
        return Pipeline([("regressor", forest)])
    raise ValueError(f"Modèle inconnu : {name} (attendus : {MODEL_NAMES})")
