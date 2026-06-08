"""Tests du module nanuq.bivariate."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nanuq import (
    chi2_features,
    compare_means_by_class,
    mann_whitney_features,
    rank_features_by_target_association,
    target_correlations,
)

# =============================================================================
# Fixtures : un dataset synthétique avec signaux contrôlés
# =============================================================================


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    """DataFrame synthétique avec signaux statistiques contrôlés.

    - ``strong_signal_num`` : numérique, différence forte entre les 2 classes
    - ``weak_signal_num``   : numérique, signal faible
    - ``no_signal_num``     : numérique, bruit pur (indépendant de la cible)
    - ``strong_signal_cat`` : catégorielle, fortement liée à la cible
    - ``no_signal_cat``     : catégorielle, indépendante de la cible
    - ``target``            : 0/1 binaire
    """
    rng = np.random.default_rng(42)
    n = 500
    target = rng.integers(0, 2, size=n)

    # Numérique avec signal fort : class 0 ~ N(50, 5), class 1 ~ N(70, 5)
    strong_num = np.where(target == 0, rng.normal(50, 5, size=n), rng.normal(70, 5, size=n))

    # Numérique avec signal faible : moyennes proches
    weak_num = np.where(target == 0, rng.normal(50, 10, size=n), rng.normal(52, 10, size=n))

    # Numérique sans signal : bruit pur
    no_signal_num = rng.normal(50, 10, size=n)

    # Catégorielle avec signal fort
    strong_cat = np.where(
        target == 0,
        rng.choice(["A", "B"], size=n, p=[0.8, 0.2]),
        rng.choice(["A", "B"], size=n, p=[0.2, 0.8]),
    )

    # Catégorielle sans signal
    no_signal_cat = rng.choice(["X", "Y", "Z"], size=n)

    return pd.DataFrame(
        {
            "strong_signal_num": strong_num,
            "weak_signal_num": weak_num,
            "no_signal_num": no_signal_num,
            "strong_signal_cat": strong_cat,
            "no_signal_cat": no_signal_cat,
            "target": target,
        }
    )


# =============================================================================
# compare_means_by_class
# =============================================================================


def test_compare_means_returns_correct_columns(synthetic_df: pd.DataFrame) -> None:
    """compare_means_by_class renvoie les colonnes attendues."""
    result = compare_means_by_class(
        synthetic_df,
        features=["strong_signal_num", "weak_signal_num", "no_signal_num"],
        target="target",
    )

    assert "mean_class_0" in result.columns
    assert "mean_class_1" in result.columns
    assert "median_class_0" in result.columns
    assert "median_class_1" in result.columns
    assert "delta_mean_pct" in result.columns
    assert len(result) == 3


def test_compare_means_sorted_by_abs_delta(synthetic_df: pd.DataFrame) -> None:
    """Le résultat est trié par |delta_mean_pct| décroissant."""
    result = compare_means_by_class(
        synthetic_df,
        features=["weak_signal_num", "strong_signal_num", "no_signal_num"],
        target="target",
    )
    # Le signal fort doit être en première position
    assert result.index[0] == "strong_signal_num"


# =============================================================================
# mann_whitney_features
# =============================================================================


def test_mann_whitney_detects_strong_signal(synthetic_df: pd.DataFrame) -> None:
    """La feature avec signal fort doit avoir la p-value la plus basse."""
    result = mann_whitney_features(
        synthetic_df,
        features=["strong_signal_num", "weak_signal_num", "no_signal_num"],
        target="target",
    )

    assert result.iloc[0]["feature"] == "strong_signal_num"
    assert result.iloc[0]["p_value"] < 0.001
    assert result.iloc[0]["signif"] == "***"


def test_mann_whitney_ranks_correctly(synthetic_df: pd.DataFrame) -> None:
    """Les p-values sont triées en ordre croissant."""
    result = mann_whitney_features(
        synthetic_df,
        features=["strong_signal_num", "weak_signal_num", "no_signal_num"],
        target="target",
    )
    p_values = result["p_value"].tolist()
    assert p_values == sorted(p_values)


def test_mann_whitney_non_binary_target_raises(
    synthetic_df: pd.DataFrame,
) -> None:
    """Une cible non binaire doit lever ValueError."""
    df = synthetic_df.copy()
    df["target"] = np.random.default_rng(0).integers(0, 5, size=len(df))

    with pytest.raises(ValueError, match="binaire"):
        mann_whitney_features(df, features=["strong_signal_num"], target="target")


# =============================================================================
# chi2_features
# =============================================================================


def test_chi2_detects_strong_signal(synthetic_df: pd.DataFrame) -> None:
    """La catégorielle avec signal fort doit avoir la p-value la plus basse."""
    result = chi2_features(
        synthetic_df,
        features=["strong_signal_cat", "no_signal_cat"],
        target="target",
    )

    assert result.iloc[0]["feature"] == "strong_signal_cat"
    assert result.iloc[0]["p_value"] < 0.001
    assert result.iloc[0]["signif"] == "***"


def test_chi2_no_signal_marker_is_ns(synthetic_df: pd.DataFrame) -> None:
    """Une catégorielle sans signal a un marqueur 'ns'."""
    result = chi2_features(
        synthetic_df,
        features=["no_signal_cat"],
        target="target",
    )
    assert result.iloc[0]["signif"] == "ns"


# =============================================================================
# target_correlations
# =============================================================================


def test_target_correlations_default_methods(synthetic_df: pd.DataFrame) -> None:
    """Sans argument methods, renvoie Pearson et Spearman."""
    result = target_correlations(
        synthetic_df,
        features=["strong_signal_num", "no_signal_num"],
        target="target",
    )
    assert set(result.columns) == {"pearson", "spearman"}


def test_target_correlations_strong_signal_first(
    synthetic_df: pd.DataFrame,
) -> None:
    """Le signal fort est trié en premier."""
    result = target_correlations(
        synthetic_df,
        features=["strong_signal_num", "weak_signal_num", "no_signal_num"],
        target="target",
    )
    assert result.index[0] == "strong_signal_num"


def test_target_correlations_string_target_raises(
    synthetic_df: pd.DataFrame,
) -> None:
    """Une cible string déclenche une erreur claire."""
    df = synthetic_df.copy()
    df["target"] = df["target"].astype(str)  # "0", "1"
    # On la rend vraiment non-numérique
    df["target"] = df["target"].replace({"0": "Non", "1": "Oui"})

    with pytest.raises(ValueError, match="numérique"):
        target_correlations(
            df,
            features=["strong_signal_num"],
            target="target",
        )


# =============================================================================
# rank_features_by_target_association — la fonction maîtresse
# =============================================================================


def test_rank_features_combines_both_tests(synthetic_df: pd.DataFrame) -> None:
    """La fonction maîtresse combine numériques + catégorielles dans un classement."""
    result = rank_features_by_target_association(
        synthetic_df,
        numeric_features=["strong_signal_num", "no_signal_num"],
        categorical_features=["strong_signal_cat", "no_signal_cat"],
        target="target",
    )

    # Toutes les features sont là
    assert set(result["feature"]) == {
        "strong_signal_num",
        "no_signal_num",
        "strong_signal_cat",
        "no_signal_cat",
    }

    # Les 2 signaux forts sont en tête
    top_2_features = set(result.head(2)["feature"])
    assert "strong_signal_num" in top_2_features
    assert "strong_signal_cat" in top_2_features


def test_rank_features_test_column(synthetic_df: pd.DataFrame) -> None:
    """La colonne 'test' indique le bon test selon le type de feature."""
    result = rank_features_by_target_association(
        synthetic_df,
        numeric_features=["strong_signal_num"],
        categorical_features=["strong_signal_cat"],
        target="target",
    )

    test_by_feature = result.set_index("feature")["test"].to_dict()
    assert test_by_feature["strong_signal_num"] == "mann_whitney"
    assert test_by_feature["strong_signal_cat"] == "chi2"
