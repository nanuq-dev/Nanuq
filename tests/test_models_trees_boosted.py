"""
Tests des modules nanuq.models.trees et nanuq.models.boosted.

Random Forest et Gradient Boosting partagent une structure de résultat
similaire (issue de la migration), donc on teste les deux ensemble pour
éviter la duplication.
"""

from __future__ import annotations

import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)

from nanuq import gradient_boosting_pipeline, random_forest_pipeline

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def classification_df() -> pd.DataFrame:
    """Classification binaire, 5 features."""
    X, y = make_classification(n_samples=300, n_features=5, n_informative=3, random_state=42)
    df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(5)])
    df["target"] = y
    return df


@pytest.fixture
def regression_df() -> pd.DataFrame:
    """Régression continue, 5 features."""
    X, y = make_regression(n_samples=300, n_features=5, noise=10, random_state=42)
    df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(5)])
    df["target"] = y
    return df


# =============================================================================
# Random Forest
# =============================================================================


def test_random_forest_classification_structure(
    classification_df: pd.DataFrame,
) -> None:
    """RF classification renvoie un classifieur + matrice de confusion."""
    features = [c for c in classification_df.columns if c != "target"]
    result = random_forest_pipeline(
        classification_df,
        features,
        "target",
        task="classification",
        n_estimators=20,
        random_state=42,
    )

    expected_keys = {
        "model",
        "feature_importances",
        "train",
        "test",
        "diagnosis",
        "confusion_matrix",
        "classification_report",
    }
    assert set(result.keys()) == expected_keys
    assert isinstance(result["model"], RandomForestClassifier)
    assert len(result["feature_importances"]) == len(features)


def test_random_forest_regression_structure(regression_df: pd.DataFrame) -> None:
    """RF régression n'a PAS de confusion_matrix."""
    features = [c for c in regression_df.columns if c != "target"]
    result = random_forest_pipeline(
        regression_df,
        features,
        "target",
        task="regression",
        n_estimators=20,
        random_state=42,
    )

    expected_keys = {
        "model",
        "feature_importances",
        "train",
        "test",
        "diagnosis",
    }
    assert set(result.keys()) == expected_keys
    assert isinstance(result["model"], RandomForestRegressor)
    assert "confusion_matrix" not in result


def test_random_forest_invalid_task_raises(classification_df: pd.DataFrame) -> None:
    """task='foo' déclenche une ValueError explicite."""
    features = [c for c in classification_df.columns if c != "target"]
    with pytest.raises(ValueError, match="task='foo' invalide"):
        random_forest_pipeline(
            classification_df,
            features,
            "target",
            task="foo",
        )


def test_random_forest_feature_importances_sum_to_one(
    classification_df: pd.DataFrame,
) -> None:
    """Les feature importances de RF somment à 1."""
    features = [c for c in classification_df.columns if c != "target"]
    result = random_forest_pipeline(
        classification_df,
        features,
        "target",
        n_estimators=20,
        random_state=42,
    )
    assert abs(result["feature_importances"].sum() - 1.0) < 1e-9


# =============================================================================
# Gradient Boosting
# =============================================================================


def test_gradient_boosting_classification_structure(
    classification_df: pd.DataFrame,
) -> None:
    """GB classification renvoie aussi X_test et y_test pour staged loss."""
    features = [c for c in classification_df.columns if c != "target"]
    result = gradient_boosting_pipeline(
        classification_df,
        features,
        "target",
        task="classification",
        n_estimators=20,
        random_state=42,
    )

    expected_keys = {
        "model",
        "feature_importances",
        "train",
        "test",
        "diagnosis",
        "X_test",
        "y_test",
        "confusion_matrix",
        "classification_report",
    }
    assert set(result.keys()) == expected_keys
    assert isinstance(result["model"], GradientBoostingClassifier)


def test_gradient_boosting_regression_structure(regression_df: pd.DataFrame) -> None:
    """GB régression : pas de confusion_matrix, mais X_test/y_test présents."""
    features = [c for c in regression_df.columns if c != "target"]
    result = gradient_boosting_pipeline(
        regression_df,
        features,
        "target",
        task="regression",
        n_estimators=20,
        random_state=42,
    )

    assert "X_test" in result and "y_test" in result
    assert "confusion_matrix" not in result
    assert isinstance(result["model"], GradientBoostingRegressor)


def test_gradient_boosting_invalid_task(classification_df: pd.DataFrame) -> None:
    """task invalide => ValueError."""
    features = [c for c in classification_df.columns if c != "target"]
    with pytest.raises(ValueError, match="invalide"):
        gradient_boosting_pipeline(
            classification_df,
            features,
            "target",
            task="invalid",
        )
