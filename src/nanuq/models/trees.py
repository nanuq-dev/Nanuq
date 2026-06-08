"""
models.trees — Modèles à base d'arbres : Random Forest.

Random Forest est un excellent choix par défaut sur données tabulaires :

- Pas besoin de scaling (insensible à l'échelle des features)
- Gère bien les non-linéarités et les interactions
- Peu d'overfitting tant qu'il y a assez d'arbres
- Accepte les variables numériques et encodées en même temps
- Fournit nativement des feature importances

Ce module fournit ``random_forest_pipeline``, un pipeline complet
train/test + évaluation, utilisable indifféremment en classification ou
régression via le paramètre ``task``.

À venir : ``decision_tree_pipeline`` et ``compare_trees_methods``.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split

from nanuq.evaluation import evaluate_classifier, evaluate_regressor


def random_forest_pipeline(
    df: pd.DataFrame,
    features: list[str],
    target: str,
    task: str = "classification",
    n_estimators: int = 100,
    max_depth: int | None = None,
    max_features: str | int | float | None = "sqrt",
    min_samples_split: int = 2,
    bootstrap: bool = True,
    test_size: float = 0.2,
    random_state: int = 42,
    n_jobs: int = -1,
) -> dict[str, Any]:
    """Pipeline complet Random Forest pour classification ou régression.

    Args:
        df: DataFrame d'entrée.
        features: Liste des colonnes explicatives.
        target: Nom de la colonne cible.
        task: Type de problème.

            - ``'classification'`` : utilise ``RandomForestClassifier``
            - ``'regression'``     : utilise ``RandomForestRegressor``

        n_estimators: Nombre d'arbres dans la forêt.

            - ``100`` (défaut) : bon compromis temps / performance
            - Plus = meilleur mais avec rendements décroissants
            - Saturation typique vers 200-500 arbres

        max_depth: Profondeur max de chaque arbre.

            - ``None`` (défaut) : pas de limite (les arbres overfittent
              individuellement, mais la moyenne corrige)
            - Valeur entière : régularisation par élagage

        max_features: Nombre de features tirées au sort par split.

            - ``'sqrt'`` (défaut classif) : √(n_features), recommandé
            - ``'log2'``                  : log₂(n_features)
            - ``int``                     : nombre exact
            - ``float``                   : fraction de n_features
            - ``None``                    : toutes (réduit la diversité)

        min_samples_split: Nombre min d'échantillons pour splitter
            un nœud (régularisation, défaut ``2``).
        bootstrap: Si ``True``, chaque arbre est entraîné sur un
            échantillon avec remise (recommandé).
        test_size: Proportion du jeu de test.
        random_state: Graine pour la reproductibilité.
        n_jobs: Nombre de cores CPU.

            - ``-1`` (défaut) : tous les cores disponibles
            - ``1``           : single thread

    Returns:
        Dict contenant :

        - ``'model'`` : modèle entraîné.
        - ``'feature_importances'`` : ``pd.Series`` triée par importance.
        - ``'train'`` : métriques sur train.
        - ``'test'`` : métriques sur test.
        - ``'diagnosis'`` : str (biais / overfit / bon).
        - ``'confusion_matrix'`` : seulement si ``task='classification'``.
        - ``'classification_report'`` : seulement si ``task='classification'``.

    Raises:
        ValueError: Si ``task`` n'est ni ``'classification'`` ni
            ``'regression'``.

    Example:
        >>> result = random_forest_pipeline(
        ...     df,
        ...     features=["age", "anciennete", "revenu_mensuel"],
        ...     target="a_quitte_l_entreprise",
        ...     n_estimators=200, max_depth=10,
        ... )
        >>> print(result["diagnosis"])
        >>> print(result["feature_importances"].head())
    """
    # ----- Préparation des données (pas de scaling, tree-based) -----
    X = df[features].values
    y = df[target].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
    )

    # ----- Hyperparamètres communs aux deux variantes -----
    common_params = dict(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features=max_features,
        min_samples_split=min_samples_split,
        bootstrap=bootstrap,
        random_state=random_state,
        n_jobs=n_jobs,
    )

    # ----- Instanciation selon la tâche -----
    if task == "classification":
        model = RandomForestClassifier(**common_params)
    elif task == "regression":
        model = RandomForestRegressor(**common_params)
    else:
        raise ValueError(
            f"task='{task}' invalide. Utiliser 'classification' ou 'regression'."
        )

    # ----- Entraînement -----
    model.fit(X_train, y_train)

    # ----- Évaluation -----
    if task == "classification":
        eval_result = evaluate_classifier(
            model, X_train, y_train, X_test, y_test, verbose=False,
        )
    else:
        eval_result = evaluate_regressor(
            model, X_train, y_train, X_test, y_test, verbose=False,
        )

    # ----- Feature importances (toujours dispo pour RF) -----
    importances = pd.Series(
        model.feature_importances_, index=features,
    ).sort_values(ascending=False)

    result: dict[str, Any] = {
        "model": model,
        "feature_importances": importances,
        "train": eval_result["train"],
        "test": eval_result["test"],
        "diagnosis": eval_result["diagnosis"],
    }
    if task == "classification":
        result["confusion_matrix"] = eval_result["confusion_matrix"]
        result["classification_report"] = eval_result["classification_report"]

    return result
