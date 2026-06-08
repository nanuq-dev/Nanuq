"""
models.boosted — Modèles à arbres boostés (Gradient Boosting).

Le boosting entraîne les arbres **les uns après les autres**, chacun
corrigeant les erreurs du précédent. Sur données tabulaires, c'est
souvent le modèle le plus performant — mais plus délicat à régler que
Random Forest (overfit facilement si on monte le ``learning_rate``).

Fonctions disponibles :

- ``gradient_boosting_pipeline`` : pipeline complet avec le
  ``GradientBoostingClassifier`` / ``Regressor`` de sklearn.

À venir : wrappers XGBoost, LightGBM, et ``compare_boosted_models``
pour les benchmarker entre eux.

Stratégie d'optimisation recommandée :

1. Fixer ``max_depth=3`` et ``learning_rate=0.1``
2. Optimiser ``n_estimators`` par grid search + CV
3. Optimiser ``learning_rate`` (plus bas = plus lent mais meilleur)
4. Ajuster ``max_depth`` en dernier (impact mineur)
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split

from nanuq.evaluation import evaluate_classifier, evaluate_regressor


def gradient_boosting_pipeline(
    df: pd.DataFrame,
    features: list[str],
    target: str,
    task: str = "classification",
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    max_depth: int = 3,
    subsample: float = 1.0,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, Any]:
    """Pipeline complet Gradient Boosting (sklearn) pour classification
    ou régression.

    Args:
        df: DataFrame d'entrée.
        features: Liste des colonnes explicatives.
        target: Nom de la colonne cible.
        task: ``'classification'`` ou ``'regression'``.
        n_estimators: Nombre d'arbres séquentiels.
            Contrairement à Random Forest, augmenter trop ce paramètre
            **overfit** ici (chaque arbre ajoute de la complexité).
        learning_rate: Taux d'apprentissage.

            - ``0.1`` (défaut) : valeur classique
            - ``0.01`` – ``0.05`` : convergence plus lente mais meilleurs
              résultats (souvent à combiner avec plus d'arbres)
            - ``>= 0.5`` : risque de divergence

        max_depth: Profondeur max de chaque arbre faible.

            - ``3`` (défaut) : recommandé pour boosting
            - Au-delà de ``5`` : risque accru d'overfitting

        subsample: Fraction d'échantillons utilisée par arbre.

            - ``1.0`` (défaut) : tout le dataset
            - ``< 1.0`` : *stochastic gradient boosting* (régularisation)

        test_size: Proportion du jeu de test.
        random_state: Graine.

    Returns:
        Dict avec les mêmes clés que ``random_forest_pipeline``, plus :

        - ``'X_test'`` et ``'y_test'`` : conservés pour pouvoir tracer
          ensuite la "staged loss" (perte en fonction du nombre
          d'arbres) via ``plot_boosting_staged_loss``.

    Raises:
        ValueError: Si ``task`` n'est ni ``'classification'`` ni
            ``'regression'``.

    Example:
        >>> result = gradient_boosting_pipeline(
        ...     df, features, target="a_quitte_l_entreprise",
        ...     n_estimators=200, learning_rate=0.05, max_depth=3,
        ... )
    """
    X = df[features].values
    y = df[target].values

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    common_params = dict(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        random_state=random_state,
    )

    if task == "classification":
        model = GradientBoostingClassifier(**common_params)
    elif task == "regression":
        model = GradientBoostingRegressor(**common_params)
    else:
        raise ValueError(f"task='{task}' invalide. Utiliser 'classification' ou 'regression'.")

    model.fit(X_train, y_train)

    if task == "classification":
        eval_result = evaluate_classifier(
            model,
            X_train,
            y_train,
            X_test,
            y_test,
            verbose=False,
        )
    else:
        eval_result = evaluate_regressor(
            model,
            X_train,
            y_train,
            X_test,
            y_test,
            verbose=False,
        )

    importances = pd.Series(
        model.feature_importances_,
        index=features,
    ).sort_values(ascending=False)

    result: dict[str, Any] = {
        "model": model,
        "feature_importances": importances,
        "train": eval_result["train"],
        "test": eval_result["test"],
        "diagnosis": eval_result["diagnosis"],
        # On garde les données de test pour pouvoir tracer la staged loss
        "X_test": X_test,
        "y_test": y_test,
    }
    if task == "classification":
        result["confusion_matrix"] = eval_result["confusion_matrix"]
        result["classification_report"] = eval_result["classification_report"]

    return result
