"""Tests du module nanuq.cleaning."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nanuq import (
    clean_nan,
    fillna_category,
    fillna_mean,
    fillna_median,
    missing_values_report,
)


@pytest.fixture
def df_with_nans() -> pd.DataFrame:
    """DataFrame avec des NaN à différents endroits."""
    return pd.DataFrame(
        {
            "age": [25, np.nan, 47, 19, 30],
            "salaire": [30000.0, 45000.0, np.nan, 22000.0, 38000.0],
            "ville": ["Paris", None, "Paris", "Marseille", None],
        }
    )


# =============================================================================
# clean_nan
# =============================================================================


def test_clean_nan_removes_all_nan_rows(df_with_nans: pd.DataFrame) -> None:
    """clean_nan supprime toutes les lignes contenant au moins un NaN."""
    result = clean_nan(df_with_nans)

    assert result.isna().sum().sum() == 0
    # Seules 2 lignes sont complètes (indices 0 et 3)
    assert len(result) == 2


def test_clean_nan_does_not_modify_input(df_with_nans: pd.DataFrame) -> None:
    """clean_nan ne modifie pas le DataFrame d'entrée."""
    original_shape = df_with_nans.shape
    _ = clean_nan(df_with_nans)
    assert df_with_nans.shape == original_shape


# =============================================================================
# missing_values_report
# =============================================================================


def test_missing_values_report_structure(df_with_nans: pd.DataFrame) -> None:
    """Le rapport a les bonnes colonnes et est trié décroissant."""
    report = missing_values_report(df_with_nans)

    assert list(report.columns) == ["missing_count", "missing_percent"]
    # Tri décroissant
    percentages = report["missing_percent"].tolist()
    assert percentages == sorted(percentages, reverse=True)


def test_missing_values_report_values(df_with_nans: pd.DataFrame) -> None:
    """Les compteurs et pourcentages sont corrects."""
    report = missing_values_report(df_with_nans)

    # 'ville' a 2 NaN sur 5 lignes = 40%
    assert report.loc["ville", "missing_count"] == 2
    assert report.loc["ville", "missing_percent"] == 40.0
    # 'age' a 1 NaN
    assert report.loc["age", "missing_count"] == 1


# =============================================================================
# fillna_mean / fillna_median
# =============================================================================


def test_fillna_mean_replaces_with_mean(df_with_nans: pd.DataFrame) -> None:
    """Les NaN sont remplacés par la moyenne de la colonne."""
    result = fillna_mean(df_with_nans, columns=["age"])

    expected_mean = df_with_nans["age"].mean()
    # La ligne d'origine en NaN (index 1) doit maintenant valoir expected_mean
    assert result.loc[1, "age"] == pytest.approx(expected_mean)
    assert result["age"].isna().sum() == 0


def test_fillna_median_replaces_with_median(df_with_nans: pd.DataFrame) -> None:
    """Les NaN sont remplacés par la médiane de la colonne."""
    result = fillna_median(df_with_nans, columns=["salaire"])

    expected_median = df_with_nans["salaire"].median()
    assert result.loc[2, "salaire"] == pytest.approx(expected_median)
    assert result["salaire"].isna().sum() == 0


def test_fillna_does_not_modify_input(df_with_nans: pd.DataFrame) -> None:
    """fillna_mean ne modifie pas le DataFrame d'entrée."""
    nan_count_before = df_with_nans["age"].isna().sum()
    _ = fillna_mean(df_with_nans, columns=["age"])
    assert df_with_nans["age"].isna().sum() == nan_count_before


# =============================================================================
# fillna_category
# =============================================================================


def test_fillna_category_default_value(df_with_nans: pd.DataFrame) -> None:
    """fillna_category remplit avec 'Unknown' par défaut."""
    result = fillna_category(df_with_nans, column="ville")

    assert result["ville"].isna().sum() == 0
    assert (result["ville"] == "Unknown").sum() == 2


def test_fillna_category_custom_value(df_with_nans: pd.DataFrame) -> None:
    """fillna_category accepte une valeur custom."""
    result = fillna_category(df_with_nans, column="ville", fill_value="Inconnu")

    assert (result["ville"] == "Inconnu").sum() == 2
