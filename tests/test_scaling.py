"""Tests du module nanuq.scaling."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from nanuq import scale_minmax, scale_standard


@pytest.fixture
def df_numeric() -> pd.DataFrame:
    """DataFrame numérique pour tester le scaling."""
    return pd.DataFrame(
        {
            "age": [25, 32, 47, 19, 30],
            "salaire": [30000.0, 45000.0, 80000.0, 22000.0, 38000.0],
        }
    )


# =============================================================================
# scale_minmax
# =============================================================================


def test_scale_minmax_range(df_numeric: pd.DataFrame) -> None:
    """Les valeurs scalées sont dans [0, 1]."""
    X_scaled, scaler = scale_minmax(df_numeric, features=["age", "salaire"])

    assert isinstance(scaler, MinMaxScaler)
    assert X_scaled.min() >= 0.0
    assert X_scaled.max() <= 1.0


def test_scale_minmax_shape(df_numeric: pd.DataFrame) -> None:
    """Le tableau scalé a la même forme que l'input."""
    X_scaled, _ = scale_minmax(df_numeric, features=["age", "salaire"])

    assert X_scaled.shape == (len(df_numeric), 2)


# =============================================================================
# scale_standard
# =============================================================================


def test_scale_standard_zero_mean(df_numeric: pd.DataFrame) -> None:
    """Après standardisation, la moyenne par colonne est ~0."""
    X_scaled, scaler = scale_standard(df_numeric, features=["age", "salaire"])

    assert isinstance(scaler, StandardScaler)
    # Moyenne quasi-nulle par colonne (tolérance flottante)
    np.testing.assert_allclose(X_scaled.mean(axis=0), [0, 0], atol=1e-10)


def test_scale_standard_unit_std(df_numeric: pd.DataFrame) -> None:
    """Après standardisation, l'écart-type par colonne est ~1."""
    X_scaled, _ = scale_standard(df_numeric, features=["age", "salaire"])

    np.testing.assert_allclose(X_scaled.std(axis=0), [1, 1], atol=1e-10)


# =============================================================================
# Réutilisation des scalers (test du contrat clé : pas de data leakage)
# =============================================================================


def test_scaler_can_be_reused_on_new_data(df_numeric: pd.DataFrame) -> None:
    """Le scaler retourné peut s'appliquer à de nouvelles données."""
    _, scaler = scale_standard(df_numeric, features=["age", "salaire"])

    # Nouvelle donnée (1 ligne)
    new_data = pd.DataFrame({"age": [50], "salaire": [60000.0]})
    transformed = scaler.transform(new_data)

    assert transformed.shape == (1, 2)
