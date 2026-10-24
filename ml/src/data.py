"""Chargement des fichiers NASA C-MAPSS et calcul de la cible RUL."""
import pandas as pd

COLUMN_NAMES = (["unit_id", "cycle", "setting_1", "setting_2", "setting_3"]
                + [f"sensor_{i}" for i in range(1, 22)])


def load_cmapss(path):
    """Charge un fichier C-MAPSS (valeurs séparées par des espaces, sans en-tête)."""
    df = pd.read_csv(path, sep=r"\s+", header=None, names=COLUMN_NAMES)
    return df.sort_values(["unit_id", "cycle"]).reset_index(drop=True)


def add_rul(df):
    """Ajoute la colonne RUL = dernier cycle du moteur - cycle courant (moteurs suivis jusqu'à la panne)."""
    df = df.copy()
    df["RUL"] = df.groupby("unit_id")["cycle"].transform("max") - df["cycle"]
    return df


def last_cycle_per_engine(df):
    """Garde une ligne par moteur : son dernier cycle observé (protocole du test NASA)."""
    return df.sort_values(["unit_id", "cycle"]).groupby("unit_id").tail(1).reset_index(drop=True)
