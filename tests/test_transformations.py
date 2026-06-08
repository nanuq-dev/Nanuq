"""Tests du module nanuq.transformations."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import PowerTransformer, QuantileTransformer

from nanuq import (
    boxcox_transform,
    log_transform,
    power_transform_features,
    quantile_transform_features,
    sqrt_transform,
)


@pytest.fixture
def df_positive() -> pd.DataFrame:
    """DataFrame avec valeurs strictement positives (compatible Box-Cox).

    Taille >= 1000 pour éviter le warning sklearn de QuantileTransformer
    (qui veut au moins n_quantiles=1000 échantillons par défaut).
    """
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "revenu": rng.exponential(scale=3000.0, size=1000) + 100,
            "age": rng.uniform(20, 60, size=1000),
        }
    )


# =============================================================================
# Transformations simples
# =============================================================================


def test_log_transform_creates_suffixed_column(df_positive: pd.DataFrame) -> None:
    """log_transform crée des colonnes <col>_log."""
    result = log_transform(df_positive, columns=["revenu"])

    assert "revenu_log" in result.columns
    # log(x+1) >= 0 si x >= 0
    assert (result["revenu_log"] >= 0).all()


def test_sqrt_transform_creates_suffixed_column(df_positive: pd.DataFrame) -> None:
    """sqrt_transform crée des colonnes <col>_sqrt."""
    result = sqrt_transform(df_positive, columns=["revenu"])

    assert "revenu_sqrt" in result.columns
    # sqrt(x) = original ** 0.5
    np.testing.assert_allclose(
        result["revenu_sqrt"], np.sqrt(df_positive["revenu"]), rtol=1e-10
    )


# =============================================================================
# Box-Cox
# =============================================================================


def test_boxcox_returns_lambdas(df_positive: pd.DataFrame) -> None:
    """boxcox_transform renvoie un dict des lambdas."""
    result, lambdas = boxcox_transform(df_positive, columns=["revenu"])

    assert "revenu_boxcox" in result.columns
    assert "revenu" in lambdas
    assert isinstance(lambdas["revenu"], float)


def test_boxcox_reduces_skewness(df_positive: pd.DataFrame) -> None:
    """Box-Cox réduit l'asymétrie d'une distribution skewed."""
    original_skew = abs(df_positive["revenu"].skew())
    result, _ = boxcox_transform(df_positive, columns=["revenu"])
    transformed_skew = abs(result["revenu_boxcox"].skew())

    # Sur une distribution exponentielle, Box-Cox doit améliorer la symétrie
    assert transformed_skew < original_skew


# =============================================================================
# Quantile Transformer
# =============================================================================


def test_quantile_transform_returns_transformer(
    df_positive: pd.DataFrame,
) -> None:
    """quantile_transform renvoie le transformer pour réutilisation."""
    result, qt = quantile_transform_features(df_positive, columns=["revenu"])

    assert isinstance(qt, QuantileTransformer)
    # Avec output_distribution='normal', les valeurs sont centrées
    assert abs(result["revenu"].mean()) < 0.5


# =============================================================================
# Power Transform
# =============================================================================


def test_power_transform_yeo_johnson(df_positive: pd.DataFrame) -> None:
    """Yeo-Johnson (défaut) fonctionne et renvoie un transformer."""
    result, pt = power_transform_features(df_positive, columns=["revenu", "age"])

    assert isinstance(pt, PowerTransformer)
    # Après PowerTransformer, moyenne ≈ 0
    np.testing.assert_allclose(
        result[["revenu", "age"]].mean(), [0, 0], atol=1e-10
    )


def test_power_transform_accepts_negative_values() -> None:
    """Yeo-Johnson accepte les valeurs négatives (contrairement à Box-Cox)."""
    df = pd.DataFrame({"x": [-10.0, -5.0, 0.0, 5.0, 10.0, 100.0]})

    # Ne doit pas lever d'erreur
    result, _ = power_transform_features(df, columns=["x"], method="yeo-johnson")
    assert len(result) == len(df)
