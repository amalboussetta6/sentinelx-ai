"""Métriques et validation croisée par moteur, utilisées dans tout le projet.

Mêmes fonctions et mêmes calculs que dans notebooks/02_baseline_fd001.ipynb (section 4).
"""
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold


def compute_metrics(y_true, y_pred):
    """MAE, RMSE et R2 (les deux premiers en cycles)."""
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
    }


def evaluate_on_zone(true_rul, predictions, horizon=127):
    """Métriques sur la zone utile (RUL réel <= horizon), en comparant au vrai RUL (jamais plafonné)."""
    true_rul = np.asarray(true_rul)
    predictions = np.asarray(predictions)
    in_zone = true_rul <= horizon                       # True pour les lignes de la zone utile
    return compute_metrics(true_rul[in_zone], predictions[in_zone])


def engine_folds(df, n_splits=5):
    """Les n_splits tours de la validation croisée par moteur (GroupKFold).

    Renvoie une liste de (moteurs pour apprendre, moteurs pour contrôler), chaque liste étant triée.
    Appliquée au tableau des 80 moteurs d'entraînement (lignes triées par moteur puis par cycle),
    elle redonne exactement les 5 tours de la baseline, aussi enregistrés dans
    artifacts/baseline_fd001/baseline_metrics.json ("fold_engines" = moteurs contrôlés à chaque tour).
    """
    folds = []
    for fit_index, check_index in GroupKFold(n_splits=n_splits).split(df, groups=df["unit_id"]):
        fit_engines = sorted(df["unit_id"].iloc[fit_index].unique().tolist())
        check_engines = sorted(df["unit_id"].iloc[check_index].unique().tolist())
        folds.append((fit_engines, check_engines))
    return folds
