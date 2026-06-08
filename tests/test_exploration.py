"""
Tests du module nanuq.exploration.
"""

from __future__ import annotations

import pandas as pd
import pytest

from nanuq import dataset_overview, get_column_types, split_features_target


@pytest.fixture
def mixed_df() -> pd.DataFrame:
    """DataFrame avec colonnes numériques ET catégorielles."""
    return pd.DataFrame(
        {
            "age": [25, 32, 47, 19],
            "salaire": [30000.0, 45000.0, 80000.0, 22000.0],
            "ville": ["Paris", "Lyon", "Paris", "Marseille"],
            "actif": [True, True, False, True],
        }
    )


# =============================================================================
# dataset_overview
# =============================================================================


def test_dataset_overview_does_not_raise(
    mixed_df: pd.DataFrame, capsys: pytest.CaptureFixture[str]
) -> None:
    """dataset_overview doit tourner sans erreur et imprimer quelque chose."""
    result = dataset_overview(mixed_df)

    # La fonction n'a pas de valeur de retour
    assert result is None

    # On capture stdout et on vérifie quelques éléments attendus
    captured = capsys.readouterr()
    assert "SHAPE" in captured.out
    assert "VALEURS MANQUANTES" in captured.out


# =============================================================================
# get_column_types
# =============================================================================


def test_get_column_types_separates_numeric_and_categorical(
    mixed_df: pd.DataFrame,
) -> None:
    """Les numériques (int/float) et le reste doivent être bien séparés."""
    types = get_column_types(mixed_df)

    assert "numeric" in types
    assert "categorical" in types

    # int + float = numeric
    assert set(types["numeric"]) == {"age", "salaire"}

    # object + bool = categorical (par exclusion des numériques)
    assert set(types["categorical"]) == {"ville", "actif"}


def test_get_column_types_empty_dataframe() -> None:
    """Sur un DataFrame vide, on doit renvoyer des listes vides."""
    types = get_column_types(pd.DataFrame())

    assert types == {"numeric": [], "categorical": []}


# =============================================================================
# split_features_target
# =============================================================================


def test_split_features_target_returns_correct_shapes(
    mixed_df: pd.DataFrame,
) -> None:
    """X doit être un DataFrame avec les colonnes demandées, y une Series."""
    X, y = split_features_target(
        mixed_df,
        features=["age", "salaire"],
        target="ville",
    )

    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert list(X.columns) == ["age", "salaire"]
    assert y.name == "ville"
    assert len(X) == len(y) == len(mixed_df)
