"""
Tests du module nanuq.loading.

Stratégie : on crée des fichiers temporaires (CSV, JSON, Parquet, SQLite)
en mémoire ou sur disque temporaire, on les charge avec nos fonctions,
on vérifie le contenu.

Le fixture ``tmp_path`` est fourni par pytest et nous donne un dossier
temporaire propre pour chaque test (auto-nettoyé après).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from nanuq import (
    load_csv,
    load_dataset_pipeline,
    load_features,
    load_json,
    load_parquet,
    load_sql,
)

# =============================================================================
# Fixtures : données d'exemple réutilisables entre les tests
# =============================================================================


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Petit DataFrame d'exemple commun à plusieurs tests."""
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "name": ["alice", "bob", "carol", "dave"],
            "score": [10.5, 20.0, 15.3, 8.1],
        }
    )


# =============================================================================
# load_csv
# =============================================================================


def test_load_csv_default(tmp_path: Path, sample_df: pd.DataFrame) -> None:
    """load_csv lit correctement un CSV virgule / utf-8."""
    csv_path = tmp_path / "data.csv"
    sample_df.to_csv(csv_path, index=False)

    loaded = load_csv(str(csv_path))

    pd.testing.assert_frame_equal(loaded, sample_df)


def test_load_csv_custom_separator(tmp_path: Path, sample_df: pd.DataFrame) -> None:
    """load_csv supporte un séparateur custom (ex: ';')."""
    csv_path = tmp_path / "data.csv"
    sample_df.to_csv(csv_path, sep=";", index=False)

    loaded = load_csv(str(csv_path), separator=";")

    pd.testing.assert_frame_equal(loaded, sample_df)


# =============================================================================
# load_json / load_parquet
# =============================================================================


def test_load_json(tmp_path: Path, sample_df: pd.DataFrame) -> None:
    """load_json lit un JSON tabulaire."""
    json_path = tmp_path / "data.json"
    sample_df.to_json(json_path)

    loaded = load_json(str(json_path))

    # On compare colonne par colonne (pas l'ordre de lignes nécessairement)
    assert set(loaded.columns) == set(sample_df.columns)
    assert len(loaded) == len(sample_df)


def test_load_parquet(tmp_path: Path, sample_df: pd.DataFrame) -> None:
    """load_parquet lit un Parquet."""
    parquet_path = tmp_path / "data.parquet"
    sample_df.to_parquet(parquet_path)

    loaded = load_parquet(str(parquet_path))

    pd.testing.assert_frame_equal(loaded, sample_df)


# =============================================================================
# load_sql
# =============================================================================


def test_load_sql(tmp_path: Path, sample_df: pd.DataFrame) -> None:
    """load_sql exécute une requête sur une base SQLite."""
    db_path = tmp_path / "test.db"

    # On crée une petite base avec une table "users"
    with sqlite3.connect(str(db_path)) as conn:
        sample_df.to_sql("users", conn, index=False)

    loaded = load_sql(str(db_path), "SELECT * FROM users ORDER BY id")

    pd.testing.assert_frame_equal(loaded, sample_df)


# =============================================================================
# load_dataset_pipeline
# =============================================================================


def test_load_dataset_pipeline_detects_csv(tmp_path: Path, sample_df: pd.DataFrame) -> None:
    """load_dataset_pipeline détecte automatiquement l'extension .csv."""
    csv_path = tmp_path / "data.csv"
    sample_df.to_csv(csv_path, index=False)

    loaded = load_dataset_pipeline(str(csv_path), drop_missing_rows=False)

    pd.testing.assert_frame_equal(loaded, sample_df)


def test_load_dataset_pipeline_unsupported_extension(tmp_path: Path) -> None:
    """Une extension inconnue déclenche une ValueError explicite."""
    bad_path = tmp_path / "data.xyz"
    bad_path.write_text("garbage")

    with pytest.raises(ValueError, match="Extension non supportée"):
        load_dataset_pipeline(str(bad_path))


def test_load_dataset_pipeline_drops_nan(tmp_path: Path) -> None:
    """drop_missing_rows=True supprime bien les lignes avec NaN."""
    df_with_nan = pd.DataFrame({"a": [1, 2, None, 4], "b": [10, None, 30, 40]})
    csv_path = tmp_path / "data.csv"
    df_with_nan.to_csv(csv_path, index=False)

    loaded = load_dataset_pipeline(str(csv_path), drop_missing_rows=True)

    assert loaded.isna().sum().sum() == 0
    assert len(loaded) == 2  # seules les lignes 1 et 4 sont complètes


# =============================================================================
# load_features
# =============================================================================


def test_load_features_returns_x_only(tmp_path: Path, sample_df: pd.DataFrame) -> None:
    """Sans target, load_features renvoie un seul DataFrame."""
    csv_path = tmp_path / "data.csv"
    sample_df.to_csv(csv_path, index=False)

    X = load_features(str(csv_path), features=["id", "score"])

    assert isinstance(X, pd.DataFrame)
    assert list(X.columns) == ["id", "score"]


def test_load_features_returns_x_and_y(tmp_path: Path, sample_df: pd.DataFrame) -> None:
    """Avec target, load_features renvoie le tuple (X, y)."""
    csv_path = tmp_path / "data.csv"
    sample_df.to_csv(csv_path, index=False)

    X, y = load_features(
        str(csv_path),
        features=["id", "name"],
        target="score",
    )

    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert list(X.columns) == ["id", "name"]
    assert y.name == "score"


def test_load_features_missing_column_raises(tmp_path: Path, sample_df: pd.DataFrame) -> None:
    """Une colonne inexistante doit déclencher une ValueError lisible."""
    csv_path = tmp_path / "data.csv"
    sample_df.to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="Colonnes introuvables"):
        load_features(str(csv_path), features=["id", "colonne_qui_nexiste_pas"])
