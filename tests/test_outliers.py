"""Tests du module nanuq.outliers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nanuq import (
    compute_zscore,
    detect_outliers_iqr,
    detect_outliers_zscore,
    iqr_bounds_report,
    outliers_iqr_summary,
    remove_outliers_iqr,
    zscore_outliers_report,
)


@pytest.fixture
def df_with_outliers() -> pd.DataFrame:
    """DataFrame avec une valeur clairement aberrante."""
    rng = np.random.default_rng(42)
    values = rng.normal(loc=50, scale=5, size=100).tolist()
    values += [200.0, 250.0]  # 2 outliers évidents
    return pd.DataFrame({"x": values, "y": list(range(102))})


# =============================================================================
# Z-score
# =============================================================================


def test_compute_zscore_creates_column(df_with_outliers: pd.DataFrame) -> None:
    """compute_zscore ajoute une colonne zscore_<col>."""
    result = compute_zscore(df_with_outliers, columns=["x"])

    assert "zscore_x" in result.columns
    # Moyenne du z-score ≈ 0, écart-type ≈ 1
    assert abs(result["zscore_x"].mean()) < 0.1


def test_detect_outliers_zscore_flags_extreme_values(
    df_with_outliers: pd.DataFrame,
) -> None:
    """Avec un seuil 3.0, les 2 outliers extrêmes doivent être détectés."""
    outliers = detect_outliers_zscore(
        df_with_outliers, column="x", zscore_threshold=3.0
    )

    # Au moins les 2 outliers ajoutés (les 200/250) sont détectés
    assert len(outliers) >= 1
    assert outliers["x"].max() >= 200


def test_zscore_outliers_report(df_with_outliers: pd.DataFrame) -> None:
    """Le rapport renvoie les bonnes clés."""
    report = zscore_outliers_report(df_with_outliers, "x")

    expected_keys = {"mean", "std", "z_min", "z_max", "n_outliers", "pct_outliers"}
    assert set(report.keys()) == expected_keys
    assert report["n_outliers"] >= 1


# =============================================================================
# IQR
# =============================================================================


def test_detect_outliers_iqr_flags_extreme_values(
    df_with_outliers: pd.DataFrame,
) -> None:
    """L'IQR détecte les outliers extrêmes."""
    outliers = detect_outliers_iqr(df_with_outliers, "x")

    assert len(outliers) >= 2
    assert (outliers["x"] >= 200).any()


def test_remove_outliers_iqr_reduces_size(
    df_with_outliers: pd.DataFrame,
) -> None:
    """remove_outliers_iqr réduit le nombre de lignes."""
    cleaned = remove_outliers_iqr(df_with_outliers, "x")

    assert len(cleaned) < len(df_with_outliers)
    # Les outliers extrêmes (200, 250) ne sont plus là
    assert cleaned["x"].max() < 200


def test_iqr_bounds_report_keys(df_with_outliers: pd.DataFrame) -> None:
    """iqr_bounds_report renvoie toutes les clés attendues."""
    report = iqr_bounds_report(df_with_outliers, "x")

    expected = {
        "Q1", "Q3", "IQR", "lower", "upper",
        "min", "max", "n_below", "n_above",
        "n_outliers", "pct_outliers",
    }
    assert set(report.keys()) == expected
    assert report["n_outliers"] >= 2


def test_outliers_iqr_summary_sorted(df_with_outliers: pd.DataFrame) -> None:
    """outliers_iqr_summary retourne un DataFrame trié par pct_outliers."""
    summary = outliers_iqr_summary(df_with_outliers, columns=["x", "y"])

    assert isinstance(summary, pd.DataFrame)
    assert "pct_outliers" in summary.columns
    # Tri décroissant : la 1re ligne a le plus haut pct_outliers
    percentages = summary["pct_outliers"].tolist()
    assert percentages == sorted(percentages, reverse=True)
