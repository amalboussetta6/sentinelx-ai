"""Métriques RUL, post-traitement des prédictions et validation croisée par moteur (GroupKFold)."""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold


def cap_target(y, cap):
    """RUL plafonné min(RUL, cap) pour l'APPRENTISSAGE ; cap=None -> RUL brut."""
    y = np.asarray(y, dtype=float)
    return y if cap is None else np.minimum(y, cap)


def clip_negative(y_pred):
    """Un RUL ne peut pas être négatif : les prédictions < 0 sont ramenées à 0."""
    return np.clip(np.asarray(y_pred, dtype=float), 0, None)


def nasa_score(y_true, y_pred) -> float:
    """Score asymétrique PHM08 (plus bas = meilleur) : d = prédit - réel ; d > 0 (tardif) pénalisé plus fort."""
    d = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))


def regression_metrics(y_true, y_pred, with_nasa: bool = False) -> dict:
    """MAE, RMSE, R² (et score NASA si demandé)."""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    metrics = {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }
    if with_nasa:
        metrics["NASA_score"] = nasa_score(y_true, y_pred)
    return metrics


def evaluate_predictions(y_true, y_pred, horizon: int) -> dict:
    """Métriques sur toutes les lignes (`*_global`) et sur la zone RUL réel <= horizon (`*_zone`)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    zone = y_true <= horizon
    glob = regression_metrics(y_true, y_pred)
    zon = regression_metrics(y_true[zone], y_pred[zone])
    return {**{f"{k}_global": v for k, v in glob.items()}, **{f"{k}_zone": v for k, v in zon.items()}}


def rul_band_table(rul_true, errors, bands) -> pd.DataFrame:
    """Erreurs (prédit - réel) résumées par tranche de RUL réel."""
    rul_true, errors = np.asarray(rul_true), np.asarray(errors, dtype=float)
    labels = [f"{int(lo)}–{int(hi)}" if np.isfinite(hi) else f"> {int(lo)}" for lo, hi in zip(bands[:-1], bands[1:])]
    band = pd.cut(rul_true, bins=bands, labels=labels, include_lowest=True)
    frame = pd.DataFrame({"tranche": band, "erreur": errors})
    return frame.groupby("tranche", observed=True)["erreur"].agg(
        lignes="size",
        MAE=lambda e: e.abs().mean(),
        RMSE=lambda e: float(np.sqrt((e ** 2).mean())),
        biais_moyen="mean",
    )


def group_folds(groups, n_splits: int):
    """Folds GroupKFold déterministes : toutes les lignes d'un moteur dans le même fold."""
    groups = np.asarray(groups)
    folds = list(GroupKFold(n_splits=n_splits).split(np.zeros(len(groups)), groups=groups))
    for train_idx, test_idx in folds:
        if not set(groups[train_idx]).isdisjoint(groups[test_idx]):
            raise AssertionError("fuite : un moteur est à la fois en apprentissage et en évaluation")
    return folds


def cross_validate(make_estimator: Callable[[], object], X: pd.DataFrame, y, groups, folds, horizon: int,
                   cap=None, clip: bool = True):
    """Validation croisée par moteur d'une configuration.

    - `make_estimator()` renvoie un estimateur NEUF (pipeline) : rien n'est partagé entre folds ;
    - le plafond `cap` ne s'applique qu'à la cible d'apprentissage ;
    - l'évaluation se fait toujours contre le RUL réel.

    Retourne (métriques par fold [DataFrame], prédictions out-of-fold [ndarray]).
    """
    y = np.asarray(y)
    y_fit = cap_target(y, cap)
    oof = np.empty(len(y), dtype=float)
    rows = []
    for fold_id, (train_idx, test_idx) in enumerate(folds, start=1):
        estimator = make_estimator().fit(X.iloc[train_idx], y_fit[train_idx])
        pred = estimator.predict(X.iloc[test_idx])
        pred = clip_negative(pred) if clip else np.asarray(pred, dtype=float)
        oof[test_idx] = pred
        rows.append({"fold": fold_id, **evaluate_predictions(y[test_idx], pred, horizon)})
    return pd.DataFrame(rows).set_index("fold"), oof


def paired_gain(reference_folds: pd.DataFrame, candidate_folds: pd.DataFrame, metric: str = "RMSE_zone") -> dict:
    """Comparaison APPARIÉE de deux configurations évaluées sur les mêmes folds.

    gain = métrique(référence) - métrique(candidat), fold par fold (> 0 : candidat meilleur).
    """
    gains = (reference_folds[metric] - candidate_folds[metric]).to_numpy()
    return {"gain_mean": float(gains.mean()), "gain_min": float(gains.min()),
            "folds_improved": int((gains > 0).sum()), "n_folds": int(len(gains))}
