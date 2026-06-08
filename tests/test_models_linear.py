"""Tests du module nanuq.models.linear."""

from __future__ import annotations

import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge

from nanuq import (
    lasso_regression_pipeline,
    linear_regression_multivariate,
    linear_regression_univariate,
    logistic_regression_pipeline,
    ridge_regression_pipeline,
)


@pytest.fixture
def regression_df() -> pd.DataFrame:
    """DataFrame de régression : 5 features + target continue."""
    X, y = make_regression(n_samples=300, n_features=5, noise=10, random_state=42)
    df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(5)])
    df["target"] = y
    return df


@pytest.fixture
def classification_df() -> pd.DataFrame:
    """DataFrame de classification binaire : 5 features + target 0/1."""
    X, y = make_classification(n_samples=300, n_features=5, n_informative=3, random_state=42)
    df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(5)])
    df["target"] = y
    return df


# =============================================================================
# linear_regression_univariate
# =============================================================================


def test_linear_regression_univariate(regression_df: pd.DataFrame) -> None:
    """Renvoie un modèle entraîné + prédictions + métriques."""
    reg, y_pred, results = linear_regression_univariate(regression_df, "feat_0", "target")

    assert isinstance(reg, LinearRegression)
    assert len(y_pred) == len(regression_df)
    assert set(results.keys()) == {"r2", "rmse", "mae", "mape"}


# =============================================================================
# linear_regression_multivariate
# =============================================================================


def test_linear_regression_multivariate_with_scaling(
    regression_df: pd.DataFrame,
) -> None:
    """Avec scaling minmax, le scaler est renvoyé."""
    features = [c for c in regression_df.columns if c != "target"]
    reg, scaler, y_pred, results = linear_regression_multivariate(
        regression_df, features, "target", scaling="minmax"
    )

    assert isinstance(reg, LinearRegression)
    assert scaler is not None  # un MinMaxScaler
    assert results["r2"] > 0  # le modèle apprend


def test_linear_regression_multivariate_no_scaling(
    regression_df: pd.DataFrame,
) -> None:
    """Sans scaling, scaler est None."""
    features = [c for c in regression_df.columns if c != "target"]
    _, scaler, _, _ = linear_regression_multivariate(
        regression_df, features, "target", scaling=None
    )
    assert scaler is None


# =============================================================================
# logistic_regression_pipeline
# =============================================================================


def test_logistic_regression_pipeline(classification_df: pd.DataFrame) -> None:
    """Pipeline complet : structure du résultat."""
    features = [c for c in classification_df.columns if c != "target"]
    result = logistic_regression_pipeline(classification_df, features, "target", random_state=42)

    expected_keys = {
        "model",
        "scaler",
        "y_test",
        "y_pred",
        "y_proba",
        "confusion_matrix",
        "metrics",
    }
    assert set(result.keys()) == expected_keys
    assert isinstance(result["model"], LogisticRegression)
    assert set(result["metrics"].keys()) == {"accuracy", "precision", "recall", "f1"}
    # Sur ce dataset synthétique, l'accuracy doit être > 0.7
    assert result["metrics"]["accuracy"] > 0.7


def test_logistic_regression_pipeline_threshold(
    classification_df: pd.DataFrame,
) -> None:
    """Un seuil très haut diminue le nombre de prédictions positives."""
    features = [c for c in classification_df.columns if c != "target"]

    res_low = logistic_regression_pipeline(
        classification_df, features, "target", decision_threshold=0.2, random_state=42
    )
    res_high = logistic_regression_pipeline(
        classification_df, features, "target", decision_threshold=0.8, random_state=42
    )

    n_pos_low = sum(res_low["y_pred"])
    n_pos_high = sum(res_high["y_pred"])
    assert n_pos_high <= n_pos_low


# =============================================================================
# ridge_regression_pipeline
# =============================================================================


def test_ridge_pipeline_structure(regression_df: pd.DataFrame) -> None:
    """Ridge renvoie modèle + coefs + diagnostic."""
    features = [c for c in regression_df.columns if c != "target"]
    result = ridge_regression_pipeline(regression_df, features, "target", alpha=1.0)

    expected_keys = {
        "model",
        "scaler",
        "coefs",
        "intercept",
        "train",
        "test",
        "diagnosis",
    }
    assert set(result.keys()) == expected_keys
    assert isinstance(result["model"], Ridge)
    assert isinstance(result["coefs"], pd.Series)
    assert len(result["coefs"]) == len(features)


# =============================================================================
# lasso_regression_pipeline
# =============================================================================


def test_lasso_pipeline_structure(regression_df: pd.DataFrame) -> None:
    """Lasso renvoie en plus n_features_used + selected_features."""
    features = [c for c in regression_df.columns if c != "target"]
    result = lasso_regression_pipeline(regression_df, features, "target", alpha=0.5)

    expected_keys = {
        "model",
        "scaler",
        "coefs",
        "intercept",
        "train",
        "test",
        "diagnosis",
        "n_features_used",
        "selected_features",
    }
    assert set(result.keys()) == expected_keys
    assert isinstance(result["model"], Lasso)
    assert result["n_features_used"] == len(result["selected_features"])
    assert result["n_features_used"] <= len(features)


def test_lasso_high_alpha_eliminates_features(regression_df: pd.DataFrame) -> None:
    """Avec un alpha très élevé, Lasso met tous les coefficients à 0."""
    features = [c for c in regression_df.columns if c != "target"]
    result = lasso_regression_pipeline(regression_df, features, "target", alpha=1e6)
    # Avec une régularisation extrême, tout est à 0
    assert result["n_features_used"] == 0
