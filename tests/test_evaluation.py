"""Tests du module nanuq.evaluation."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split

from nanuq import (
    benchmark_models,
    cross_validate_model,
    diagnose_bias_overfit,
    evaluate_classifier,
    evaluate_regressor,
    grid_search_cv,
)

# =============================================================================
# Fixtures : petits datasets synthétiques propres
# =============================================================================


@pytest.fixture
def regression_data():
    """Dataset synthétique de régression."""
    X, y = make_regression(n_samples=200, n_features=5, noise=10, random_state=42)
    return train_test_split(X, y, test_size=0.2, random_state=42)


@pytest.fixture
def classification_data():
    """Dataset synthétique de classification binaire."""
    X, y = make_classification(
        n_samples=200, n_features=5, n_informative=3, random_state=42
    )
    return train_test_split(X, y, test_size=0.2, random_state=42)


# =============================================================================
# diagnose_bias_overfit
# =============================================================================


def test_diagnose_good_model() -> None:
    """Train et test proches et hauts => bon modèle."""
    diag, gap = diagnose_bias_overfit(train_score=0.88, test_score=0.85)
    assert "BON" in diag
    assert abs(gap - 0.03) < 1e-9


def test_diagnose_overfit() -> None:
    """Écart train-test important => overfit."""
    diag, gap = diagnose_bias_overfit(train_score=0.95, test_score=0.70)
    assert "OVERFIT" in diag
    assert gap > 0.05


def test_diagnose_bias() -> None:
    """Deux scores faibles => biais."""
    diag, _ = diagnose_bias_overfit(train_score=0.55, test_score=0.50)
    assert "BIAIS" in diag


def test_diagnose_anomaly() -> None:
    """Test nettement meilleur que train => anomalie."""
    diag, gap = diagnose_bias_overfit(train_score=0.70, test_score=0.90)
    assert "Anomalie" in diag
    assert gap < 0


# =============================================================================
# evaluate_regressor
# =============================================================================


def test_evaluate_regressor_keys(regression_data) -> None:
    """evaluate_regressor renvoie les bonnes clés."""
    X_train, X_test, y_train, y_test = regression_data
    model = LinearRegression().fit(X_train, y_train)

    result = evaluate_regressor(
        model, X_train, y_train, X_test, y_test, verbose=False
    )

    assert set(result.keys()) == {"train", "test", "diagnosis", "gap"}
    assert set(result["train"].keys()) == {"r2", "rmse", "mae", "mape"}
    assert set(result["test"].keys()) == {"r2", "rmse", "mae", "mape"}


def test_evaluate_regressor_metrics_are_floats(regression_data) -> None:
    """Toutes les métriques sont des floats convertibles."""
    X_train, X_test, y_train, y_test = regression_data
    model = LinearRegression().fit(X_train, y_train)
    result = evaluate_regressor(
        model, X_train, y_train, X_test, y_test, verbose=False
    )

    for split in ("train", "test"):
        for metric_value in result[split].values():
            assert isinstance(metric_value, float)
            assert not np.isnan(metric_value)


# =============================================================================
# evaluate_classifier
# =============================================================================


def test_evaluate_classifier_keys(classification_data) -> None:
    """evaluate_classifier renvoie les bonnes clés."""
    X_train, X_test, y_train, y_test = classification_data
    model = LogisticRegression(max_iter=1000).fit(X_train, y_train)

    result = evaluate_classifier(
        model, X_train, y_train, X_test, y_test, verbose=False
    )

    expected_keys = {
        "train", "test", "confusion_matrix",
        "classification_report", "diagnosis", "gap",
    }
    assert set(result.keys()) == expected_keys


def test_evaluate_classifier_metrics_panel(classification_data) -> None:
    """train/test contiennent accuracy, precision, recall, f1, auc."""
    X_train, X_test, y_train, y_test = classification_data
    model = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    result = evaluate_classifier(
        model, X_train, y_train, X_test, y_test, verbose=False
    )

    expected_metrics = {"accuracy", "precision", "recall", "f1", "auc"}
    assert set(result["train"].keys()) == expected_metrics
    assert set(result["test"].keys()) == expected_metrics

    # AUC est non-None car LogisticRegression a predict_proba
    assert result["test"]["auc"] is not None


# =============================================================================
# cross_validate_model
# =============================================================================


def test_cross_validate_model_structure(classification_data) -> None:
    """cross_validate_model renvoie scores + statistiques."""
    X_train, _, y_train, _ = classification_data
    model = LogisticRegression(max_iter=1000)

    result = cross_validate_model(
        model, X_train, y_train, cv=3, scoring="accuracy", verbose=False
    )

    expected_keys = {"scores", "mean", "std", "min", "max", "cv", "scoring"}
    assert set(result.keys()) == expected_keys
    assert len(result["scores"]) == 3
    assert 0 <= result["mean"] <= 1
    assert result["cv"] == 3


# =============================================================================
# grid_search_cv
# =============================================================================


def test_grid_search_cv_finds_best_params(classification_data) -> None:
    """grid_search_cv retourne best_params et best_estimator."""
    from sklearn.tree import DecisionTreeClassifier

    X_train, _, y_train, _ = classification_data

    result = grid_search_cv(
        DecisionTreeClassifier,
        X_train, y_train,
        param_grid={"max_depth": [3, 5, 10]},
        cv=3,
        scoring="accuracy",
        fixed_params={"random_state": 42},
        verbose=False,
    )

    expected_keys = {
        "best_params", "best_score", "best_estimator", "results", "cv_results"
    }
    assert set(result.keys()) == expected_keys
    # Le best_params doit contenir max_depth
    assert "max_depth" in result["best_params"]
    # results est un DataFrame avec une ligne par combinaison
    assert len(result["results"]) == 3


# =============================================================================
# benchmark_models
# =============================================================================


def test_benchmark_classification_structure(classification_data) -> None:
    """benchmark_models renvoie un DataFrame trié par test_acc."""
    from sklearn.ensemble import RandomForestClassifier

    X_train, X_test, y_train, y_test = classification_data
    models = {
        "logreg": LogisticRegression(max_iter=1000),
        "rf": RandomForestClassifier(n_estimators=20, random_state=42),
    }

    bench = benchmark_models(
        models, X_train, y_train, X_test, y_test,
        task="classification", verbose=False
    )

    assert len(bench) == 2
    # Colonnes attendues
    expected_cols = {
        "train_acc", "test_acc", "train_f1", "test_f1",
        "train_auc", "test_auc", "gap"
    }
    assert expected_cols.issubset(set(bench.columns))
    # Trié par test_acc décroissant
    assert bench["test_acc"].is_monotonic_decreasing


def test_benchmark_invalid_task_raises(classification_data) -> None:
    """task invalide => ValueError."""
    X_train, X_test, y_train, y_test = classification_data
    models = {"logreg": LogisticRegression(max_iter=1000)}

    with pytest.raises(ValueError, match="invalide"):
        benchmark_models(
            models, X_train, y_train, X_test, y_test,
            task="invalid", verbose=False
        )
