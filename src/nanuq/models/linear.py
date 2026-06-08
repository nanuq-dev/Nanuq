"""
models.linear — Modèles linéaires.

Fonctions disponibles :

- ``linear_regression_univariate``    : régression linéaire à 1 feature
- ``linear_regression_multivariate``  : régression linéaire à N features (+ scaling)
- ``logistic_regression_pipeline``    : pipeline classification logistique complet
- ``ridge_regression_pipeline``       : régression linéaire **régularisée L2** (Ridge)
- ``lasso_regression_pipeline``       : régression linéaire **régularisée L1** (Lasso),
                                        avec sélection automatique de variables

Régularisation L1 vs L2, en bref :

- **Ridge (L2)** : pénalise les coefficients par leur carré → les
  "rétrécit" tous, sans en mettre à 0. Bon contre le surapprentissage
  quand on a beaucoup de features corrélées.
- **Lasso (L1)** : pénalise par valeur absolue → met certains
  coefficients **exactement à 0**, ce qui fait office de sélection
  automatique de variables. Donne un modèle parcimonieux.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from nanuq.evaluation import evaluate_regressor
from nanuq.scaling import scale_minmax, scale_standard

# =============================================================================
# Régression linéaire
# =============================================================================


def linear_regression_univariate(
    df: pd.DataFrame,
    feature: str,
    target: str,
) -> tuple[LinearRegression, np.ndarray, dict[str, float]]:
    """Entraîne une régression linéaire simple (1 feature → 1 target).

    Évalue le modèle sur l'ensemble d'entraînement uniquement (pas de
    split). Pour une évaluation propre, utiliser une cross-validation
    via ``cross_validate_model`` (à venir).

    Args:
        df: DataFrame d'entrée.
        feature: Nom de la colonne explicative (X).
        target: Nom de la colonne cible (y).

    Returns:
        Tuple ``(reg, y_pred, results)`` :

        - ``reg`` : modèle ``LinearRegression`` entraîné.
        - ``y_pred`` : prédictions sur les données d'entraînement.
        - ``results`` : dict des métriques ``{r2, rmse, mae, mape}``.

    Example:
        >>> reg, y_pred, res = linear_regression_univariate(df, "age", "salaire")
        >>> print(f"R² = {res['r2']:.3f}")
    """
    # df[[feature]] (double crochet) pour garder 2D
    X = df[[feature]]
    y = df[target]

    reg = LinearRegression()
    reg.fit(X, y)

    y_pred = reg.predict(X)

    results = {
        "r2": r2_score(y, y_pred),
        "rmse": np.sqrt(mean_squared_error(y, y_pred)),
        "mae": mean_absolute_error(y, y_pred),
        "mape": mean_absolute_percentage_error(y, y_pred),
    }

    return reg, y_pred, results


def linear_regression_multivariate(
    df: pd.DataFrame,
    features: list[str],
    target: str,
    scaling: str | None = "minmax",
) -> tuple[LinearRegression, Any | None, np.ndarray, dict[str, float]]:
    """Entraîne une régression linéaire multivariée avec scaling optionnel.

    Args:
        df: DataFrame d'entrée.
        features: Liste des colonnes explicatives.
        target: Nom de la colonne cible.
        scaling: Méthode de mise à l'échelle des features.

            - ``'minmax'`` (défaut) : ``MinMaxScaler`` [0, 1]
            - ``'standard'``        : ``StandardScaler`` (z-score)
            - ``None`` ou autre     : pas de scaling

    Returns:
        Tuple ``(reg, scaler, y_pred, results)`` :

        - ``reg`` : modèle ``LinearRegression`` entraîné.
        - ``scaler`` : scaler entraîné (ou ``None`` si pas de scaling).
        - ``y_pred`` : prédictions sur les données d'entraînement.
        - ``results`` : dict des métriques ``{r2, rmse, mae, mape}``.

    Example:
        >>> reg, scaler, y_pred, res = linear_regression_multivariate(
        ...     df, features=["age", "anciennete"], target="salaire"
        ... )
    """
    # Application du scaling demandé
    if scaling == "minmax":
        X, scaler = scale_minmax(df, features)
    elif scaling == "standard":
        X, scaler = scale_standard(df, features)
    else:
        X = df[features].values
        scaler = None

    y = df[target]

    reg = LinearRegression()
    reg.fit(X, y)

    y_pred = reg.predict(X)

    results = {
        "r2": r2_score(y, y_pred),
        "rmse": np.sqrt(mean_squared_error(y, y_pred)),
        "mae": mean_absolute_error(y, y_pred),
        "mape": mean_absolute_percentage_error(y, y_pred),
    }

    return reg, scaler, y_pred, results


# =============================================================================
# Régression logistique (classification)
# =============================================================================


def logistic_regression_pipeline(
    df: pd.DataFrame,
    features: list[str],
    target: str,
    scaling: str | None = "minmax",
    test_size: float = 0.2,
    random_state: int = 42,
    decision_threshold: float = 0.5,
) -> dict[str, Any]:
    """Pipeline complet de régression logistique.

    Étapes :

    1. Scaling des features
    2. Split train / test
    3. Entraînement du modèle
    4. Prédictions avec seuil de décision personnalisable
    5. Calcul des métriques (accuracy, precision, recall, F1)

    Args:
        df: DataFrame d'entrée.
        features: Liste des colonnes explicatives.
        target: Nom de la colonne cible (binaire ou multi-classes).
        scaling: Méthode de scaling (``'minmax'``, ``'standard'`` ou
            ``None``).
        test_size: Proportion du jeu de test (``0.2`` = 20 %).
        random_state: Graine aléatoire pour la reproductibilité.
        decision_threshold: Seuil de décision pour la classe positive
            (ignoré en multi-classes). ``0.5`` par défaut. À baisser
            pour favoriser le **rappel**, à monter pour favoriser la
            **précision**.

    Returns:
        Dictionnaire contenant :

        - ``'model'`` : modèle ``LogisticRegression`` entraîné.
        - ``'scaler'`` : scaler entraîné (ou ``None``).
        - ``'y_test'`` : valeurs réelles du test.
        - ``'y_pred'`` : prédictions.
        - ``'y_proba'`` : probabilités prédites.
        - ``'confusion_matrix'`` : matrice de confusion.
        - ``'metrics'`` : dict ``{accuracy, precision, recall, f1}``.

    Example:
        >>> result = logistic_regression_pipeline(
        ...     df, features=["age", "revenu"], target="a_quitte_l_entreprise",
        ... )
        >>> print(result["metrics"])
    """
    # ----- 1. Scaling -----
    if scaling == "minmax":
        X, scaler = scale_minmax(df, features)
    elif scaling == "standard":
        X, scaler = scale_standard(df, features)
    else:
        X = df[features].values
        scaler = None

    y = df[target]

    # ----- 2. Split train / test -----
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    # ----- 3. Entraînement -----
    # max_iter=1000 pour éviter les warnings de non-convergence
    clf = LogisticRegression(random_state=random_state, max_iter=1000)
    clf.fit(X_train, y_train)

    # ----- 4. Prédictions avec seuil personnalisé -----
    y_proba_full = clf.predict_proba(X_test)

    # Cas binaire : on applique le seuil sur la proba de la classe 1
    if y_proba_full.shape[1] == 2:
        y_proba = y_proba_full[:, 1]
        y_pred = [0 if proba < decision_threshold else 1 for proba in y_proba]
    # Cas multi-classes : on prend argmax (predict standard)
    else:
        y_pred = clf.predict(X_test)
        y_proba = y_proba_full

    # ----- 5. Évaluation -----
    cm = confusion_matrix(y_test, y_pred)

    # average='weighted' pour gérer le déséquilibre des classes
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted"),
        "recall": recall_score(y_test, y_pred, average="weighted"),
        "f1": f1_score(y_test, y_pred, average="weighted"),
    }

    return {
        "model": clf,
        "scaler": scaler,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "confusion_matrix": cm,
        "metrics": metrics,
    }


# =============================================================================
# Régressions régularisées : Ridge et Lasso
# =============================================================================


def ridge_regression_pipeline(
    df: pd.DataFrame,
    features: list[str],
    target: str,
    alpha: float = 1.0,
    scaling: str | None = "standard",
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, Any]:
    """Pipeline complet de régression Ridge (régularisation **L2**).

    Étapes : scaling → split train/test → fit Ridge → évaluation sur
    les deux ensembles avec diagnostic biais/overfit automatique.

    À utiliser quand une régression linéaire simple overfit (ex : sur
    une régression polynomiale de haut degré, ou un dataset avec
    beaucoup de features corrélées).

    Args:
        df: DataFrame d'entrée.
        features: Liste des colonnes explicatives.
        target: Nom de la colonne cible (continue).
        alpha: Intensité de la régularisation L2.

            - ``0`` : équivalent à une régression linéaire classique
            - ``1.0`` (défaut) : régularisation modérée
            - ``>> 1`` : forte contrainte sur les coefficients

            À optimiser par grid search en pratique.

        scaling: Mise à l'échelle des features. ``'standard'``
            fortement recommandé pour Ridge (sinon les variables
            d'échelles différentes sont pénalisées inégalement).
            ``'minmax'`` aussi valide. ``None`` à éviter.
        test_size: Proportion du jeu de test (``0.2`` par défaut).
        random_state: Graine pour la reproductibilité.

    Returns:
        Dict contenant :

        - ``'model'`` : modèle ``Ridge`` entraîné.
        - ``'scaler'`` : scaler entraîné (ou ``None``).
        - ``'coefs'`` : coefficients du modèle (``pd.Series``).
        - ``'intercept'`` : ordonnée à l'origine.
        - ``'train'`` : métriques sur train ``{r2, rmse, mae, mape}``.
        - ``'test'`` : métriques sur test.
        - ``'diagnosis'`` : str (biais / overfit / bon).

    Example:
        >>> result = ridge_regression_pipeline(
        ...     df, features=features, target="price", alpha=1.0,
        ... )
        >>> print(result["diagnosis"])
    """
    # ----- Scaling -----
    if scaling == "minmax":
        X, scaler = scale_minmax(df, features)
    elif scaling == "standard":
        X, scaler = scale_standard(df, features)
    else:
        X = df[features].values
        scaler = None

    y = df[target]

    # ----- Split train/test -----
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    # ----- Entraînement -----
    model = Ridge(alpha=alpha, random_state=random_state)
    model.fit(X_train, y_train)

    # ----- Évaluation -----
    eval_result = evaluate_regressor(
        model,
        X_train,
        y_train,
        X_test,
        y_test,
        verbose=False,
    )

    return {
        "model": model,
        "scaler": scaler,
        "coefs": pd.Series(model.coef_, index=features),
        "intercept": float(model.intercept_),
        "train": eval_result["train"],
        "test": eval_result["test"],
        "diagnosis": eval_result["diagnosis"],
    }


def lasso_regression_pipeline(
    df: pd.DataFrame,
    features: list[str],
    target: str,
    alpha: float = 1.0,
    scaling: str | None = "standard",
    test_size: float = 0.2,
    random_state: int = 42,
    max_iter: int = 10000,
) -> dict[str, Any]:
    """Pipeline complet de régression Lasso (régularisation **L1**).

    Particularité du Lasso : il met certains coefficients **exactement
    à 0**, ce qui en fait un outil de **sélection automatique de
    variables**. À utiliser quand on veut un modèle parcimonieux (peu
    de features actives) ou pour identifier les variables réellement
    utiles.

    Args:
        df: DataFrame d'entrée.
        features: Liste des colonnes explicatives.
        target: Nom de la colonne cible (continue).
        alpha: Intensité de la régularisation L1. Plus ``alpha`` est
            élevé, plus de coefficients sont mis à 0.
        scaling: Mise à l'échelle (``'standard'`` fortement recommandé).
        test_size: Proportion du jeu de test.
        random_state: Graine pour la reproductibilité.
        max_iter: Nombre maximum d'itérations pour la convergence.
            Lasso converge moins vite que Ridge, défaut ``10000``.

    Returns:
        Dict contenant les mêmes clés que ``ridge_regression_pipeline``,
        plus :

        - ``'n_features_used'`` : nombre de coefficients non nuls.
        - ``'selected_features'`` : liste des features avec ``coef != 0``.

    Example:
        >>> result = lasso_regression_pipeline(df, features, "price", alpha=0.5)
        >>> print(f"Retenues : {result['n_features_used']}/{len(features)}")
        >>> print(result["selected_features"])
    """
    # Scaling
    if scaling == "minmax":
        X, scaler = scale_minmax(df, features)
    elif scaling == "standard":
        X, scaler = scale_standard(df, features)
    else:
        X = df[features].values
        scaler = None

    y = df[target]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    # Entraînement
    model = Lasso(alpha=alpha, random_state=random_state, max_iter=max_iter)
    model.fit(X_train, y_train)

    # Évaluation
    eval_result = evaluate_regressor(
        model,
        X_train,
        y_train,
        X_test,
        y_test,
        verbose=False,
    )

    # Identification des variables réellement utilisées (coef != 0)
    coefs = pd.Series(model.coef_, index=features)
    selected = coefs[coefs != 0].index.tolist()

    return {
        "model": model,
        "scaler": scaler,
        "coefs": coefs,
        "intercept": float(model.intercept_),
        "train": eval_result["train"],
        "test": eval_result["test"],
        "diagnosis": eval_result["diagnosis"],
        "n_features_used": len(selected),
        "selected_features": selected,
    }
