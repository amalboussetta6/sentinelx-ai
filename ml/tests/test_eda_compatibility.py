"""Compatibilité avec les artefacts de l'EDA (ignorés si les données brutes sont absentes)."""
import pytest

from ml.src.data import add_rul, check_rul, file_sha256, find_project_root, load_cmapss, load_eda_decisions, split_by_engine
from ml.src.features import candidate_features

ROOT = find_project_root()
RAW = ROOT / "data" / "raw"
pytestmark = pytest.mark.skipif(not (RAW / "train_FD001.txt").exists(), reason="données FD001 absentes de data/raw/")


def test_eda_split_and_features_are_reproduced():
    summary, split, groups = load_eda_decisions(ROOT / "artifacts" / "eda_fd001")
    assert file_sha256(RAW / "train_FD001.txt") == summary["raw_files_sha256"]["train_FD001.txt"]
    assert file_sha256(RAW / "test_FD001.txt") == summary["raw_files_sha256"]["test_FD001.txt"]
    train = load_cmapss(RAW / "train_FD001.txt")
    analysis, validation = split_by_engine(train, split["analysis_engines"], split["validation_engines"])
    assert (len(analysis), len(validation)) == (16_561, 4_070)
    analysis = add_rul(analysis)
    check_rul(analysis)
    assert round(analysis["RUL"].mean(), 4) == summary["rul_analysis_train"]["mean"]
    feats = candidate_features(groups)
    assert len(feats) == 17 and not set(feats) & set(groups["forbidden_as_feature"])
