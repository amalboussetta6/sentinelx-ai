"""Chargement de NASA C-MAPSS, décisions de l'EDA, cible RUL et vérité terrain du test.

Toutes les fonctions sont pures (elles ne modifient pas le DataFrame reçu) et
reprennent exactement le schéma et les règles du notebook 01_eda_fd001_final_fusion.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ID_COLUMNS = ["unit_id", "cycle"]
SETTING_COLUMNS = [f"setting_{i}" for i in range(1, 4)]
SENSOR_COLUMNS = [f"sensor_{i}" for i in range(1, 22)]
COLUMN_NAMES = ID_COLUMNS + SETTING_COLUMNS + SENSOR_COLUMNS
REQUIRED_RAW_FILES = ("train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt")


def find_project_root(start: Path | None = None) -> Path:
    """Remonte depuis `start` (dossier courant par défaut) jusqu'au dossier qui contient data/raw et artifacts.

    Fonctionne que le notebook soit lancé depuis la racine du dépôt ou depuis notebooks/.
    """
    start = Path.cwd() if start is None else Path(start)
    for candidate in (start, *start.parents):
        if (candidate / "data" / "raw").is_dir() and (candidate / "artifacts").is_dir():
            return candidate
    raise FileNotFoundError(
        "Racine du projet SentinelX AI introuvable : lancer le notebook depuis le dépôt "
        "(dossier contenant data/raw/ et artifacts/)."
    )


def check_raw_files(raw_dir: Path) -> None:
    """Vérifie la présence des trois fichiers FD001 (non versionnés, voir README §7)."""
    missing = [name for name in REQUIRED_RAW_FILES if not (raw_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Fichiers manquants dans {raw_dir} : {missing}. Copier les fichiers C-MAPSS FD001 "
            "dans data/raw/ sans les modifier (README §7)."
        )


def file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    """Empreinte SHA-256 d'un fichier : prouve que les données n'ont pas changé depuis l'EDA."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_cmapss(path: Path) -> pd.DataFrame:
    """Charge un fichier C-MAPSS (espaces, sans en-tête) avec le schéma du projet, trié par moteur puis cycle."""
    df = pd.read_csv(path, sep=r"\s+", header=None, names=COLUMN_NAMES)
    if not all(pd.api.types.is_integer_dtype(df[c]) for c in ID_COLUMNS):
        raise ValueError(f"{Path(path).name} : unit_id et cycle doivent être entiers — schéma inattendu.")
    return df.sort_values(ID_COLUMNS).reset_index(drop=True)


def load_eda_decisions(eda_artifact_dir: Path) -> tuple[dict, dict, dict]:
    """Relit les décisions figées de l'EDA : (résumé, split par moteur, groupes de variables)."""
    def read(name: str) -> dict:
        return json.loads((Path(eda_artifact_dir) / name).read_text(encoding="utf-8"))

    return read("eda_fd001_summary.json"), read("fd001_engine_split.json"), read("fd001_feature_groups.json")


def split_by_engine(df: pd.DataFrame, analysis_engines, validation_engines) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sépare un DataFrame en (analyse, validation) uniquement par identifiant de moteur."""
    analysis_engines, validation_engines = set(analysis_engines), set(validation_engines)
    if not analysis_engines.isdisjoint(validation_engines):
        raise ValueError("Un moteur ne peut pas être à la fois en analyse et en validation.")
    if analysis_engines | validation_engines != set(df["unit_id"]):
        raise ValueError("Le split doit couvrir exactement les moteurs du DataFrame.")
    analysis = df[df["unit_id"].isin(analysis_engines)].copy()
    validation = df[df["unit_id"].isin(validation_engines)].copy()
    return analysis, validation


def add_rul(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute RUL = dernier cycle du moteur - cycle courant (trajectoires complètes uniquement).

    Le dernier cycle T_i sert à construire la cible ; il n'est jamais ajouté comme colonne.
    """
    out = df.copy()
    last_cycle = out.groupby("unit_id")["cycle"].transform("max")
    out["RUL"] = (last_cycle - out["cycle"]).astype(int)
    return out


def check_rul(df: pd.DataFrame) -> None:
    """Contrôles de la cible : jamais négative, 0 au dernier cycle, -1 par cycle."""
    if (df["RUL"] < 0).any():
        raise AssertionError("RUL négatif")
    if not (df.groupby("unit_id")["RUL"].min() == 0).all():
        raise AssertionError("chaque moteur doit finir à RUL = 0")
    ordered = df.sort_values(ID_COLUMNS)
    steps = ordered.groupby("unit_id")["RUL"].diff().dropna()
    if not (steps == -1).all():
        raise AssertionError("le RUL doit diminuer d'exactement 1 par cycle")


def last_observed_cycles(df: pd.DataFrame) -> pd.DataFrame:
    """Une ligne par moteur : son dernier cycle observé (protocole d'évaluation du test NASA)."""
    idx = df.groupby("unit_id")["cycle"].idxmax()
    return df.loc[idx].sort_values("unit_id").reset_index(drop=True)


class GroundTruthGuard:
    """Accès unique et contrôlé à RUL_FD001.txt (README §9, règle 2).

    La vérité terrain ne peut être lue qu'après `freeze()`, et une seule fois.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.frozen = False
        self.n_reads = 0

    def freeze(self) -> None:
        """Déclare le pipeline figé : plus aucun choix de modélisation après cet appel."""
        self.frozen = True

    def read(self) -> np.ndarray:
        if not self.frozen:
            raise RuntimeError("RUL_FD001.txt ne peut être lu qu'après avoir figé le pipeline (freeze()).")
        if self.n_reads:
            raise RuntimeError("RUL_FD001.txt a déjà été lu : une seule lecture est autorisée.")
        self.n_reads += 1
        values = pd.read_csv(self.path, sep=r"\s+", header=None)[0].to_numpy()
        if values.ndim != 1 or (values < 0).any():
            raise ValueError("RUL_FD001.txt doit contenir une valeur de RUL positive par moteur.")
        return values
