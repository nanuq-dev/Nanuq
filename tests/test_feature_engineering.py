"""Tests du module nanuq.feature_engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import PolynomialFeatures

from nanuq import (
    binarize_feature,
    create_interaction_feature,
    create_polynomial_features,
    discretize_equal_freq,
    discretize_equal_width,
)


@pytest.fixture
def df_simple() -> pd.DataFrame:
    """Petit DataFrame numérique pour les tests."""
    return pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5],
            "b": [10, 20, 30, 40, 50],
            "c": [0, 1000, 2000, 5000, 10000],
        }
    )


# =============================================================================
# Polynomial features
# =============================================================================


def test_polynomial_features_shape(df_simple: pd.DataFrame) -> None:
    """degree=2 sur 2 colonnes => 5 features (a, b, a², ab, b²)."""
    X_poly, poly = create_polynomial_features(df_simple[["a", "b"]], degree=2)

    assert isinstance(poly, PolynomialFeatures)
    # 2 features, degree=2, include_bias=False : 5 colonnes
    assert X_poly.shape == (len(df_simple), 5)


def test_polynomial_features_with_bias(df_simple: pd.DataFrame) -> None:
    """include_bias=True ajoute une colonne de 1."""
    X_poly, _ = create_polynomial_features(df_simple[["a", "b"]], degree=2, include_bias=True)

    assert X_poly.shape == (len(df_simple), 6)
    # La 1re colonne ne contient que des 1
    np.testing.assert_array_equal(X_poly[:, 0], np.ones(len(df_simple)))


# =============================================================================
# Interaction
# =============================================================================


def test_interaction_multiply(df_simple: pd.DataFrame) -> None:
    """multiply crée la colonne a*b."""
    result = create_interaction_feature(df_simple, "a", "b", operation="multiply")

    assert "a_b_interaction" in result.columns
    np.testing.assert_array_equal(result["a_b_interaction"], df_simple["a"] * df_simple["b"])


def test_interaction_add(df_simple: pd.DataFrame) -> None:
    """add crée la colonne a+b."""
    result = create_interaction_feature(df_simple, "a", "b", operation="add")
    np.testing.assert_array_equal(result["a_b_interaction"], df_simple["a"] + df_simple["b"])


def test_interaction_divide(df_simple: pd.DataFrame) -> None:
    """divide crée la colonne a/b."""
    result = create_interaction_feature(df_simple, "b", "a", operation="divide")
    np.testing.assert_allclose(result["b_a_interaction"], df_simple["b"] / df_simple["a"])


# =============================================================================
# Binarisation
# =============================================================================


def test_binarize_feature(df_simple: pd.DataFrame) -> None:
    """Valeurs > seuil = 1, sinon 0."""
    result = binarize_feature(df_simple, "c", threshold=1500)

    assert "c_binary" in result.columns
    expected = [0, 0, 1, 1, 1]
    assert result["c_binary"].tolist() == expected


# =============================================================================
# Discrétisation
# =============================================================================


def test_discretize_equal_width(df_simple: pd.DataFrame) -> None:
    """discretize_equal_width crée une colonne <col>_cut."""
    result = discretize_equal_width(df_simple, "c", n_bins=3)

    assert "c_cut" in result.columns
    # Chaque ligne est dans un des 3 bins (pas de NaN)
    assert result["c_cut"].notna().all()


def test_discretize_equal_freq(df_simple: pd.DataFrame) -> None:
    """discretize_equal_freq crée une colonne <col>_qcut."""
    result = discretize_equal_freq(df_simple, "c", n_bins=2, labels=False)

    assert "c_qcut" in result.columns
    # Avec 5 lignes et 2 bins, on a a peu près 2-3 par bin
    counts = result["c_qcut"].value_counts()
    assert len(counts) == 2


def test_discretize_equal_freq_custom_name(df_simple: pd.DataFrame) -> None:
    """new_column_name remplace le suffixe par défaut."""
    result = discretize_equal_freq(df_simple, "c", n_bins=2, new_column_name="c_tier")
    assert "c_tier" in result.columns
    assert "c_qcut" not in result.columns
