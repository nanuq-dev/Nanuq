"""
Tests du module nanuq.plots.

Approche : **smoke tests** principalement. Pour des fonctions de
visualisation, on vérifie surtout :

1. La fonction ne lève pas d'exception sur des inputs valides.
2. Le type / la structure du retour quand il y en a un.
3. Les cas d'erreur explicites (AttributeError sur modèle inapproprié).

Note : le backend matplotlib `Agg` est forcé dans conftest.py, donc
``plt.show()`` ne bloque pas et n'affiche aucune fenêtre.

Après chaque test on appelle ``plt.close("all")`` pour libérer les
figures et éviter une accumulation mémoire si on en crée beaucoup.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from nanuq import (
    compare_distribution,
    plot_boosting_staged_loss,
    plot_boxplot,
    plot_confusion_matrix_pretty,
    plot_correlation_matrix,
    plot_feature_importances,
    plot_histogram,
    plot_train_test_curve,
)

# =============================================================================
# Fixtures + auto-cleanup des figures matplotlib
# =============================================================================


@pytest.fixture(autouse=True)
def _close_figures():
    """Ferme toutes les figures matplotlib après chaque test."""
    yield
    plt.close("all")


@pytest.fixture
def simple_df() -> pd.DataFrame:
    """DataFrame avec colonnes numériques + 1 catégorielle."""
    return pd.DataFrame(
        {
            "age": [25, 32, 47, 19, 30, 40, 55],
            "salaire": [30000, 45000, 80000, 22000, 38000, 52000, 91000],
            "ville": ["Paris", "Lyon", "Paris", "Marseille", "Lyon", "Paris", "Lyon"],
        }
    )


@pytest.fixture
def classification_data():
    """Petit dataset de classification binaire."""
    X, y = make_classification(
        n_samples=200,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        n_repeated=0,
        random_state=42,
    )
    return X, y


@pytest.fixture
def regression_data():
    """Petit dataset de régression."""
    return make_regression(n_samples=200, n_features=4, noise=10, random_state=42)


# =============================================================================
# EDA — plots simples
# =============================================================================


def test_plot_boxplot_y_only(simple_df: pd.DataFrame) -> None:
    """plot_boxplot avec uniquement y ne plante pas."""
    plot_boxplot(simple_df, y="salaire")


def test_plot_boxplot_grouped(simple_df: pd.DataFrame) -> None:
    """plot_boxplot par catégorie ne plante pas."""
    plot_boxplot(simple_df, x="ville", y="salaire")


def test_plot_histogram(simple_df: pd.DataFrame) -> None:
    """plot_histogram ne plante pas et utilise le bon titre."""
    plot_histogram(simple_df, column="age", bins=5)
    # Le titre de la dernière figure doit être 'age'
    assert plt.gca().get_title() == "age"


def test_plot_correlation_matrix(simple_df: pd.DataFrame) -> None:
    """plot_correlation_matrix sur 2 colonnes numériques ne plante pas."""
    plot_correlation_matrix(simple_df)


def test_compare_distribution() -> None:
    """compare_distribution génère une figure à 2 panneaux."""
    original = np.random.default_rng(0).exponential(size=100)
    transformed = np.log1p(original)

    compare_distribution(original, transformed)
    # 2 axes attendus dans la figure
    assert len(plt.gcf().axes) == 2


# =============================================================================
# plot_train_test_curve
# =============================================================================


def test_plot_train_test_curve_returns_dataframe(classification_data) -> None:
    """plot_train_test_curve renvoie un DataFrame avec les bonnes colonnes."""
    X, y = classification_data
    # On split simplement la moitié
    half = len(X) // 2
    X_train, X_test = X[:half], X[half:]
    y_train, y_test = y[:half], y[half:]

    df = plot_train_test_curve(
        DecisionTreeClassifier,
        X_train, y_train, X_test, y_test,
        param_name="max_depth",
        param_values=[2, 5, 10],
        scoring="accuracy",
        fixed_params={"random_state": 42},
        plot=False,  # pas d'affichage pour ce test
    )

    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) == {"max_depth", "train", "test", "gap"}
    assert len(df) == 3


def test_plot_train_test_curve_with_plot(classification_data) -> None:
    """Avec plot=True, la figure est bien créée sans erreur."""
    X, y = classification_data
    half = len(X) // 2

    plot_train_test_curve(
        DecisionTreeClassifier,
        X[:half], y[:half], X[half:], y[half:],
        param_name="max_depth",
        param_values=[2, 5, 10],
        scoring="accuracy",
        fixed_params={"random_state": 42},
        plot=True,
    )
    # Une figure existe
    assert plt.get_fignums()


# =============================================================================
# plot_confusion_matrix_pretty
# =============================================================================


def test_plot_confusion_matrix_pretty(classification_data) -> None:
    """plot_confusion_matrix_pretty ne plante pas."""
    X, y = classification_data
    model = LogisticRegression(max_iter=1000).fit(X, y)
    y_pred = model.predict(X)

    plot_confusion_matrix_pretty(y, y_pred)


def test_plot_confusion_matrix_pretty_normalized(classification_data) -> None:
    """Mode normalisation 'true' ne plante pas."""
    X, y = classification_data
    model = LogisticRegression(max_iter=1000).fit(X, y)
    y_pred = model.predict(X)

    plot_confusion_matrix_pretty(y, y_pred, normalize="true")


# =============================================================================
# plot_feature_importances
# =============================================================================


def test_plot_feature_importances_returns_series(classification_data) -> None:
    """plot_feature_importances renvoie une Series triée décroissante."""
    X, y = classification_data
    model = RandomForestClassifier(n_estimators=10, random_state=42).fit(X, y)
    features = [f"feat_{i}" for i in range(X.shape[1])]

    importances = plot_feature_importances(model, features)

    assert isinstance(importances, pd.Series)
    assert importances.is_monotonic_decreasing


def test_plot_feature_importances_top_n(classification_data) -> None:
    """top_n limite le graphique mais la Series renvoyée est complète."""
    X, y = classification_data
    model = RandomForestClassifier(n_estimators=10, random_state=42).fit(X, y)
    features = [f"feat_{i}" for i in range(X.shape[1])]

    importances = plot_feature_importances(model, features, top_n=2)
    # On garde toutes les features dans le retour (le top_n affecte seulement l'affichage)
    assert len(importances) == len(features)


def test_plot_feature_importances_raises_on_invalid_model() -> None:
    """Un modèle sans feature_importances_ déclenche AttributeError."""
    model = LogisticRegression()  # pas de feature_importances_
    with pytest.raises(AttributeError, match="feature_importances_"):
        plot_feature_importances(model, feature_names=["a", "b"])


# =============================================================================
# plot_boosting_staged_loss
# =============================================================================


def test_plot_boosting_staged_loss_returns_scores(classification_data) -> None:
    """plot_boosting_staged_loss renvoie un array de longueur n_estimators."""
    X, y = classification_data
    model = GradientBoostingClassifier(
        n_estimators=20, learning_rate=0.1, random_state=42
    ).fit(X, y)

    scores = plot_boosting_staged_loss(model, X, y)

    assert isinstance(scores, np.ndarray)
    assert len(scores) == 20


def test_plot_boosting_staged_loss_raises_on_invalid_model(
    classification_data,
) -> None:
    """Un modèle non-boosté déclenche AttributeError."""
    X, y = classification_data
    model = LogisticRegression(max_iter=1000).fit(X, y)

    with pytest.raises(AttributeError, match="staged_predict_proba"):
        plot_boosting_staged_loss(model, X, y)
