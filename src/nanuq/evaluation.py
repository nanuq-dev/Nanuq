"""
evaluation — Évaluation de modèles : métriques, diagnostic, CV, benchmark.

Fonctions disponibles :

- ``diagnose_bias_overfit``   : diagnostic automatique biais/overfit
- ``evaluate_classifier``     : panel complet pour la classification
- ``evaluate_regressor``      : panel complet pour la régression
- ``cross_validate_model``    : validation croisée K-fold
- ``grid_search_cv``          : recherche d'hyperparamètres + CV
- ``benchmark_models``        : comparaison côte-à-côte de plusieurs modèles

À venir : ``precision_recall_curve_threshold`` (seuil optimal pour la
classification déséquilibrée), ``stratified_cv_multiple_metrics``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, cross_val_score

# =============================================================================
# Diagnostic biais / overfit
# =============================================================================


def diagnose_bias_overfit(
    train_score: float,
    test_score: float,
    bias_threshold: float = 0.7,
    overfit_gap: float = 0.05,
) -> tuple[str, float]:
    """Diagnostic automatique biais / overfit / bon modèle.

    Compare les scores du modèle sur train et test pour identifier
    rapidement la situation :

    - Les deux scores ``<`` ``bias_threshold`` → **BIAIS** (sous-apprentissage)
    - Écart train-test ``>`` ``overfit_gap``  → **OVERFIT** (sur-apprentissage)
    - Test ``>`` train avec écart significatif → **anomalie statistique**
    - Sinon                                    → **modèle équilibré**

    Args:
        train_score: Score sur l'ensemble d'entraînement (entre 0 et 1).
        test_score: Score sur l'ensemble de test (entre 0 et 1).
        bias_threshold: Seuil sous lequel les deux scores sont considérés
            comme faibles (biais). ``0.7`` par défaut.
        overfit_gap: Écart train-test au-delà duquel on parle d'overfit.
            ``0.05`` par défaut (5 points).

    Returns:
        Tuple ``(diagnostic, gap)`` :

        - ``diagnostic`` : chaîne avec emoji et description.
        - ``gap`` : ``train_score - test_score`` (float).

    Example:
        >>> diagnostic, gap = diagnose_bias_overfit(0.92, 0.78)
        >>> print(diagnostic)
        ⚠️  OVERFIT — écart train-test = 0.14 (train=0.92, test=0.78)
    """
    gap = train_score - test_score

    if train_score < bias_threshold and test_score < bias_threshold:
        return (
            f"⚠️  BIAIS — les deux scores sont faibles "
            f"(train={train_score:.2f}, test={test_score:.2f})",
            gap,
        )

    if gap > overfit_gap:
        return (
            f"⚠️  OVERFIT — écart train-test = {gap:.2f} "
            f"(train={train_score:.2f}, test={test_score:.2f})",
            gap,
        )

    if gap < -overfit_gap:
        return (
            f"❓  Anomalie — test > train de {-gap:.2f} "
            f"(vérifier le split ou la fuite de données)",
            gap,
        )

    return (
        f"✅ BON — scores équilibrés "
        f"(train={train_score:.2f}, test={test_score:.2f})",
        gap,
    )


# =============================================================================
# Métriques de classification (helper interne)
# =============================================================================


def _classification_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    model: Any,
    X: np.ndarray | pd.DataFrame,
    average: str = "weighted",
) -> dict[str, float | None]:
    """Calcule accuracy / precision / recall / F1 / AUC pour un classificateur.

    AUC n'est calculée que si le modèle expose ``predict_proba``. Gère
    automatiquement le cas binaire vs multi-classes.

    Helper interne : utilise plutôt ``evaluate_classifier`` directement
    pour avoir le panel complet (matrice de confusion, rapport, etc.).
    """
    metrics: dict[str, float | None] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, average=average, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, average=average, zero_division=0)
        ),
        "f1": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
    }

    metrics["auc"] = None
    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X)
            if y_proba.shape[1] == 2:
                metrics["auc"] = float(roc_auc_score(y_true, y_proba[:, 1]))
            else:
                metrics["auc"] = float(
                    roc_auc_score(y_true, y_proba, multi_class="ovr")
                )
        except Exception:
            metrics["auc"] = None

    return metrics


# =============================================================================
# Évaluation complète : classification
# =============================================================================


def evaluate_classifier(
    model: Any,
    X_train: np.ndarray | pd.DataFrame,
    y_train: np.ndarray | pd.Series,
    X_test: np.ndarray | pd.DataFrame,
    y_test: np.ndarray | pd.Series,
    average: str = "weighted",
    verbose: bool = True,
) -> dict[str, Any]:
    """Évaluation complète d'un classificateur sur train ET test.

    Calcule un panel complet de métriques sur les deux ensembles, la
    matrice de confusion, le classification report, et un diagnostic
    automatique (biais / overfit / bon).

    Args:
        model: Modèle scikit-learn **déjà entraîné**.
        X_train: Features d'entraînement.
        y_train: Cible d'entraînement.
        X_test: Features de test.
        y_test: Cible de test.
        average: Méthode d'agrégation pour les métriques multi-classes
            (``'weighted'``, ``'macro'``, ``'micro'``).
        verbose: Si ``True``, affiche un résumé lisible dans la console.

    Returns:
        Dict contenant ``'train'``, ``'test'``, ``'confusion_matrix'``,
        ``'classification_report'``, ``'diagnosis'``, ``'gap'``.

    Example:
        >>> clf = DecisionTreeClassifier(max_depth=9).fit(X_train, y_train)
        >>> result = evaluate_classifier(clf, X_train, y_train, X_test, y_test)
        >>> print(result["diagnosis"])
    """
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_metrics = _classification_metrics(
        y_train, y_train_pred, model, X_train, average=average
    )
    test_metrics = _classification_metrics(
        y_test, y_test_pred, model, X_test, average=average
    )

    cm = confusion_matrix(y_test, y_test_pred)
    cr = classification_report(y_test, y_test_pred, zero_division=0)

    diagnosis, gap = diagnose_bias_overfit(
        train_metrics["accuracy"],
        test_metrics["accuracy"],
    )

    if verbose:
        print("=" * 60)
        print("ÉVALUATION DU CLASSIFICATEUR")
        print("=" * 60)
        for split_name, m in [("TRAIN", train_metrics), ("TEST", test_metrics)]:
            print(f"\n{split_name} :")
            for k, v in m.items():
                if v is not None:
                    print(f"  {k:10s} : {v:.4f}")
        print(f"\nDiagnostic : {diagnosis}")
        print("\nClassification report (test) :")
        print(cr)

    return {
        "train": train_metrics,
        "test": test_metrics,
        "confusion_matrix": cm,
        "classification_report": cr,
        "diagnosis": diagnosis,
        "gap": gap,
    }


# =============================================================================
# Évaluation complète : régression
# =============================================================================


def evaluate_regressor(
    model: Any,
    X_train: np.ndarray | pd.DataFrame,
    y_train: np.ndarray | pd.Series,
    X_test: np.ndarray | pd.DataFrame,
    y_test: np.ndarray | pd.Series,
    verbose: bool = True,
) -> dict[str, Any]:
    """Évaluation complète d'un modèle de régression sur train ET test.

    Calcule R², RMSE, MAE, MAPE sur les deux ensembles + diagnostic
    biais/overfit basé sur le R².

    Args:
        model: Modèle de régression **déjà entraîné**.
        X_train: Features d'entraînement.
        y_train: Cible d'entraînement.
        X_test: Features de test.
        y_test: Cible de test.
        verbose: Si ``True``, affiche un résumé lisible.

    Returns:
        Dict contenant ``'train'``, ``'test'``, ``'diagnosis'``, ``'gap'``.
    """
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    def _metrics(y_true, y_pred):
        return {
            "r2": float(r2_score(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "mape": float(mean_absolute_percentage_error(y_true, y_pred)),
        }

    train_metrics = _metrics(y_train, y_train_pred)
    test_metrics = _metrics(y_test, y_test_pred)

    diagnosis, gap = diagnose_bias_overfit(
        train_metrics["r2"],
        test_metrics["r2"],
    )

    if verbose:
        print("=" * 60)
        print("ÉVALUATION DU MODÈLE DE RÉGRESSION")
        print("=" * 60)
        for split_name, m in [("TRAIN", train_metrics), ("TEST", test_metrics)]:
            print(f"\n{split_name} :")
            for k, v in m.items():
                print(f"  {k:5s} : {v:.4f}")
        print(f"\nDiagnostic : {diagnosis}")

    return {
        "train": train_metrics,
        "test": test_metrics,
        "diagnosis": diagnosis,
        "gap": gap,
    }


# =============================================================================
# Cross-validation
# =============================================================================


def cross_validate_model(
    model: Any,
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    cv: int = 5,
    scoring: str = "accuracy",
    n_jobs: int = -1,
    verbose: bool = True,
) -> dict[str, Any]:
    """Évaluation par validation croisée K-fold.

    Découpe le dataset en ``cv`` plis et entraîne / évalue le modèle K
    fois, chaque pli servant tour à tour de test. Beaucoup plus robuste
    qu'un simple split, surtout sur des petits datasets ou des datasets
    déséquilibrés.

    Args:
        model: Modèle scikit-learn **non entraîné** (``cross_val_score``
            entraîne lui-même à chaque pli).
        X: Données complètes (pas de split préalable nécessaire).
        y: Cible.
        cv: Nombre de plis. ``5`` par défaut. Valeurs typiques : 3, 5, 10.
        scoring: Métrique d'évaluation. Exemples :

            - Classification : ``'accuracy'``, ``'f1_weighted'``,
              ``'roc_auc'``, ``'roc_auc_ovr'`` (multi-classes),
              ``'precision'``, ``'recall'``.
            - Régression : ``'r2'``, ``'neg_mean_squared_error'``,
              ``'neg_mean_absolute_error'``.

            Liste complète : ``sklearn.metrics.get_scorer_names()``.

        n_jobs: Parallélisation (``-1`` = tous les cores).
        verbose: Affiche un résumé.

    Returns:
        Dict contenant :

        - ``'scores'`` : array des K scores
        - ``'mean'`` : score moyen
        - ``'std'`` : écart-type (stabilité du modèle)
        - ``'min'`` / ``'max'`` : extrêmes observés
        - ``'cv'`` : nombre de plis
        - ``'scoring'`` : métrique utilisée

    Example:
        >>> clf = RandomForestClassifier(n_estimators=100, random_state=42)
        >>> result = cross_validate_model(clf, X, y, cv=5, scoring="roc_auc_ovr")
        >>> print(f"AUC = {result['mean']:.3f} ± {result['std']:.3f}")
    """
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs)

    result = {
        "scores": scores,
        "mean": float(scores.mean()),
        "std": float(scores.std()),
        "min": float(scores.min()),
        "max": float(scores.max()),
        "cv": cv,
        "scoring": scoring,
    }

    if verbose:
        print(f"Cross-validation {cv}-fold sur {type(model).__name__}")
        print(f"  Scoring : {scoring}")
        print(f"  Scores  : {[f'{s:.4f}' for s in scores]}")
        print(f"  Mean    : {result['mean']:.4f}")
        print(f"  Std     : {result['std']:.4f}")
        print(f"  Range   : [{result['min']:.4f}, {result['max']:.4f}]")

    return result


# =============================================================================
# Grid Search + Cross Validation
# =============================================================================


def grid_search_cv(
    model_class: Any,
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    param_grid: dict[str, list[Any]],
    cv: int = 5,
    scoring: str = "accuracy",
    fixed_params: dict[str, Any] | None = None,
    n_jobs: int = -1,
    verbose: bool = True,
) -> dict[str, Any]:
    """Recherche d'hyperparamètres par grid search + cross-validation.

    Wrapper de ``GridSearchCV`` qui teste exhaustivement toutes les
    combinaisons de ``param_grid``, évalue chacune par CV à K plis,
    et retourne la meilleure combinaison.

    Args:
        model_class: **Classe** du modèle (pas une instance).
            Ex : ``DecisionTreeClassifier``, ``RandomForestClassifier``.
        X: Données complètes.
        y: Cible.
        param_grid: Dict ``{nom_param: liste_valeurs}``.
            Ex : ``{'max_depth': [3, 5, 10], 'min_samples_split': [2, 5]}``.
        cv: Nombre de plis (``5`` par défaut).
        scoring: Métrique d'évaluation (voir ``cross_validate_model``).
        fixed_params: Paramètres fixés pour toutes les combinaisons.
            Ex : ``{'random_state': 42, 'n_jobs': -1}``.
        n_jobs: Parallélisation du grid search.
        verbose: Affiche les résultats principaux.

    Returns:
        Dict contenant :

        - ``'best_params'`` : meilleurs paramètres trouvés.
        - ``'best_score'`` : meilleur score moyen en CV.
        - ``'best_estimator'`` : modèle entraîné avec ces paramètres.
        - ``'results'`` : DataFrame des combinaisons explorées
          (params + ``mean_test_score``, ``std_test_score``,
          ``rank_test_score``), trié par rang.
        - ``'cv_results'`` : objet brut de sklearn (debug avancé).

    Warning:
        Le nombre de modèles entraînés est ``|param_combinations| × cv``.
        Exemple : 3 valeurs × 4 valeurs × 5 plis = 60 entraînements.
        Avec un Random Forest de 200 arbres, ça peut être long.

    Example:
        >>> result = grid_search_cv(
        ...     DecisionTreeClassifier, X, y,
        ...     param_grid={"max_depth": list(range(2, 30, 2))},
        ...     cv=5, scoring="roc_auc_ovr",
        ...     fixed_params={"random_state": 808},
        ... )
        >>> print(result["best_params"])
    """
    fixed_params = fixed_params or {}
    model = model_class(**fixed_params)

    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=1 if verbose else 0,
    )

    grid.fit(X, y)

    # Tableau résumé : les params explorés + score moyen et std
    results_df = pd.DataFrame(grid.cv_results_)
    interest_cols = (
        [f"param_{p}" for p in param_grid.keys()]
        + ["mean_test_score", "std_test_score", "rank_test_score"]
    )
    results_df = results_df[interest_cols].sort_values("rank_test_score")

    if verbose:
        print("\n" + "=" * 60)
        print("RÉSULTATS GRID SEARCH")
        print("=" * 60)
        print(f"Meilleurs paramètres : {grid.best_params_}")
        print(f"Meilleur score ({scoring}) : {grid.best_score_:.4f}")

    return {
        "best_params": grid.best_params_,
        "best_score": float(grid.best_score_),
        "best_estimator": grid.best_estimator_,
        "results": results_df,
        "cv_results": grid.cv_results_,
    }


# =============================================================================
# Benchmark de modèles
# =============================================================================


def benchmark_models(
    models: dict[str, Any],
    X_train: np.ndarray | pd.DataFrame,
    y_train: np.ndarray | pd.Series,
    X_test: np.ndarray | pd.DataFrame,
    y_test: np.ndarray | pd.Series,
    task: str = "classification",
    use_cv: bool = False,
    cv: int = 5,
    verbose: bool = True,
) -> pd.DataFrame:
    """Compare plusieurs modèles entraînés sur les mêmes données.

    Entraîne chaque modèle, mesure ses performances train et test (et
    optionnellement par cross-validation), et retourne un DataFrame
    trié par performance test décroissante.

    Stratégie recommandée :

    1. Régression linéaire/logistique comme **baseline**
    2. Random Forest comme **gain rapide**
    3. Gradient Boosting (sklearn / XGBoost / LightGBM) si besoin de
       **pousser plus loin**

    Args:
        models: Dict ``{nom: instance_modèle_non_entraîné}``.
        X_train: Features d'entraînement.
        y_train: Cible d'entraînement.
        X_test: Features de test.
        y_test: Cible de test.
        task: ``'classification'`` ou ``'regression'``.
        use_cv: Si ``True``, ajoute une colonne de score CV (sur train).
            Plus robuste mais plus lent.
        cv: Nombre de plis pour la CV (si ``use_cv=True``).
        verbose: Affiche le tableau récapitulatif.

    Returns:
        DataFrame avec une ligne par modèle, colonnes :

        - **classification** : ``train_acc``, ``test_acc``, ``train_f1``,
          ``test_f1``, ``train_auc``, ``test_auc``, ``gap``.
        - **régression** : ``train_r2``, ``test_r2``, ``train_rmse``,
          ``test_rmse``, ``gap``.

        Si ``use_cv=True`` : ajoute ``cv_mean`` et ``cv_std``. Trié par
        score test décroissant.

    Raises:
        ValueError: Si ``task`` n'est ni ``'classification'`` ni
            ``'regression'``.

    Example:
        >>> models = {
        ...     "logreg": LogisticRegression(max_iter=1000),
        ...     "rf": RandomForestClassifier(n_estimators=100, random_state=42),
        ...     "gb": GradientBoostingClassifier(n_estimators=100, random_state=42),
        ... }
        >>> bench = benchmark_models(models, X_train, y_train, X_test, y_test)
    """
    if task not in ("classification", "regression"):
        raise ValueError(
            f"task='{task}' invalide. Utiliser 'classification' ou 'regression'."
        )

    rows = []
    for name, model in models.items():
        # Entraînement
        model.fit(X_train, y_train)

        # Évaluation sur train et test
        if task == "classification":
            ev = evaluate_classifier(
                model, X_train, y_train, X_test, y_test, verbose=False,
            )
            row = {
                "model": name,
                "train_acc": ev["train"]["accuracy"],
                "test_acc": ev["test"]["accuracy"],
                "train_f1": ev["train"]["f1"],
                "test_f1": ev["test"]["f1"],
                "train_auc": ev["train"]["auc"],
                "test_auc": ev["test"]["auc"],
                "gap": ev["gap"],
            }
            sort_col = "test_acc"
        else:
            ev = evaluate_regressor(
                model, X_train, y_train, X_test, y_test, verbose=False,
            )
            row = {
                "model": name,
                "train_r2": ev["train"]["r2"],
                "test_r2": ev["test"]["r2"],
                "train_rmse": ev["train"]["rmse"],
                "test_rmse": ev["test"]["rmse"],
                "gap": ev["gap"],
            }
            sort_col = "test_r2"

        # Cross-validation optionnelle
        if use_cv:
            cv_scoring = "accuracy" if task == "classification" else "r2"
            cv_scores = cross_val_score(
                type(model)(**model.get_params()),
                X_train, y_train,
                cv=cv, scoring=cv_scoring, n_jobs=-1,
            )
            row["cv_mean"] = float(cv_scores.mean())
            row["cv_std"] = float(cv_scores.std())

        rows.append(row)

    bench = pd.DataFrame(rows).set_index("model").sort_values(
        sort_col, ascending=False,
    )

    if verbose:
        print("\n" + "=" * 60)
        print(f"BENCHMARK - {task.upper()} ({len(models)} modèles)")
        print("=" * 60)
        print(bench.round(4))

    return bench
