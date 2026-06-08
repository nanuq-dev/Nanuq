"""Tests du module nanuq.encoding."""

from __future__ import annotations

import pandas as pd
import pytest
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from nanuq import (
    encode_binary,
    encode_binary_representation,
    encode_multiclass,
    label_encode_column,
    one_hot_encode,
    one_hot_encode_sklearn,
)


@pytest.fixture
def df_categorical() -> pd.DataFrame:
    """DataFrame avec différents types de variables catégorielles."""
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6],
            "heure_supp": ["Oui", "Non", "Oui", "Non", "Oui", "Non"],
            "niveau": ["Faible", "Moyen", "Élevé", "Faible", "Moyen", "Élevé"],
            "departement": ["RH", "Tech", "RH", "Commerce", "Tech", "Commerce"],
        }
    )


# =============================================================================
# encode_binary
# =============================================================================


def test_encode_binary(df_categorical: pd.DataFrame) -> None:
    """Le mapping Oui/Non → 1/0 fonctionne."""
    result = encode_binary(df_categorical, "heure_supp", mapping={"Oui": 1, "Non": 0})

    assert result["heure_supp"].dtype.kind in ("i", "u")  # int
    assert result["heure_supp"].tolist() == [1, 0, 1, 0, 1, 0]


# =============================================================================
# encode_multiclass
# =============================================================================


def test_encode_multiclass(df_categorical: pd.DataFrame) -> None:
    """L'encoding ordinal préserve l'ordre fourni."""
    result = encode_multiclass(
        df_categorical,
        "niveau",
        mapping={"Faible": 0, "Moyen": 1, "Élevé": 2},
    )

    assert result["niveau"].tolist() == [0, 1, 2, 0, 1, 2]


# =============================================================================
# label_encode_column
# =============================================================================


def test_label_encode_returns_encoder(df_categorical: pd.DataFrame) -> None:
    """label_encode renvoie un tuple (df, encoder) avec un LabelEncoder."""
    result, encoder = label_encode_column(df_categorical, "departement")

    assert isinstance(encoder, LabelEncoder)
    # Les valeurs deviennent des entiers >= 0
    assert result["departement"].dtype.kind in ("i", "u")
    assert result["departement"].min() >= 0


def test_label_encode_classes_match(df_categorical: pd.DataFrame) -> None:
    """encoder.classes_ contient les modalités originales triées."""
    _, encoder = label_encode_column(df_categorical, "departement")

    assert set(encoder.classes_) == {"Commerce", "RH", "Tech"}


# =============================================================================
# one_hot_encode (pandas)
# =============================================================================


def test_one_hot_encode_creates_columns(df_categorical: pd.DataFrame) -> None:
    """one_hot_encode crée une colonne par modalité."""
    result = one_hot_encode(df_categorical, columns=["departement"])

    # On doit avoir 3 colonnes one-hot
    one_hot_cols = [c for c in result.columns if c.startswith("departement_")]
    assert len(one_hot_cols) == 3


def test_one_hot_encode_drop_first(df_categorical: pd.DataFrame) -> None:
    """drop_first=True réduit le nombre de colonnes de 1."""
    result = one_hot_encode(df_categorical, columns=["departement"], drop_first=True)

    one_hot_cols = [c for c in result.columns if c.startswith("departement_")]
    assert len(one_hot_cols) == 2  # 3 modalités - 1


# =============================================================================
# one_hot_encode_sklearn
# =============================================================================


def test_one_hot_encode_sklearn_returns_encoder(
    df_categorical: pd.DataFrame,
) -> None:
    """Avec return_encoder=True, on récupère un OneHotEncoder."""
    result, encoder = one_hot_encode_sklearn(
        df_categorical, columns=["departement"], return_encoder=True
    )

    assert isinstance(encoder, OneHotEncoder)
    one_hot_cols = [c for c in result.columns if c.startswith("departement_")]
    assert len(one_hot_cols) == 3


def test_one_hot_encode_sklearn_no_columns_present(
    df_categorical: pd.DataFrame,
) -> None:
    """Si aucune colonne demandée n'existe, renvoie le DataFrame tel quel."""
    result = one_hot_encode_sklearn(df_categorical, columns=["colonne_inexistante"])

    pd.testing.assert_frame_equal(result, df_categorical)


# =============================================================================
# encode_binary_representation
# =============================================================================


def test_encode_binary_representation_n_bits(df_categorical: pd.DataFrame) -> None:
    """3 modalités → ⌈log₂(3)⌉ = 2 colonnes binaires."""
    result, mapping = encode_binary_representation(
        df_categorical, "departement", drop_original=True
    )

    bit_cols = [c for c in result.columns if "departement_bit_" in c]
    assert len(bit_cols) == 2
    assert "departement" not in result.columns  # dropped


def test_encode_binary_representation_mapping(
    df_categorical: pd.DataFrame,
) -> None:
    """Le mapping retourné couvre toutes les modalités uniques."""
    _, mapping = encode_binary_representation(df_categorical, "departement")

    assert set(mapping.keys()) == {"RH", "Tech", "Commerce"}
