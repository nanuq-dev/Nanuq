"""
plots — Visualisations classiques pour l'EDA et la modélisation.

Fonctions disponibles :

**EDA (exploration)**

- ``plot_boxplot`` : boxplot (distribution + outliers)
- ``plot_histogram`` : histogramme d'une colonne numérique
- ``plot_correlation_matrix`` : heatmap de corrélations
- ``compare_distribution`` : histogramme avant/après transformation

**Modélisation (classification, régression)**

- ``plot_train_test_curve`` : courbes train/test selon un hyperparamètre
- ``plot_confusion_matrix_pretty`` : heatmap annotée d'une matrice de
  confusion
- ``plot_feature_importances`` : barres horizontales des importances
- ``plot_boosting_staged_loss`` : courbe de convergence du boosting

Toutes ces fonctions appellent ``plt.show()`` à la fin pour rendre la
figure visible en notebook. En tests automatisés, le backend matplotlib
``Agg`` (non-interactif) évite tout blocage.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)

# =============================================================================
# EDA — visualisations classiques d'exploration
# =============================================================================


def plot_boxplot(
    df: pd.DataFrame,
    x: str | None = None,
    y: str | None = None,
    figsize: tuple[int, int] = (8, 5),
) -> None:
    """Trace un boxplot pour visualiser distribution et outliers.

    Args:
        df: DataFrame source.
        x: Nom de la colonne pour l'axe X (souvent catégorielle si
            on regroupe par modalité).
        y: Nom de la colonne pour l'axe Y (numérique).
        figsize: Taille de la figure en pouces (largeur, hauteur).

    Example:
        >>> # Boxplot de la colonne 'revenu' globale :
        >>> plot_boxplot(df, y="revenu_mensuel")
        >>> # Comparaison de la distribution par département :
        >>> plot_boxplot(df, x="departement", y="revenu_mensuel")
    """
    plt.figure(figsize=figsize)
    sns.boxplot(data=df, x=x, y=y)
    plt.show()


def plot_histogram(
    df: pd.DataFrame,
    column: str,
    bins: int = 30,
    figsize: tuple[int, int] = (8, 5),
) -> None:
    """Trace l'histogramme d'une colonne numérique.

    Args:
        df: DataFrame source.
        column: Nom de la colonne à visualiser.
        bins: Nombre de classes (barres) de l'histogramme.

            - ``30`` (défaut) : bon compromis général
            - Augmenter pour des séries longues
            - Diminuer pour des séries courtes

        figsize: Taille de la figure (largeur, hauteur).

    Example:
        >>> plot_histogram(df, "revenu_mensuel", bins=50)
    """
    plt.figure(figsize=figsize)
    plt.hist(df[column], bins=bins)
    plt.title(column)
    plt.show()


def plot_correlation_matrix(
    df: pd.DataFrame,
    figsize: tuple[int, int] = (10, 8),
) -> None:
    """Trace la matrice de corrélation (heatmap) des variables numériques.

    Utilise la corrélation de Pearson par défaut (linéaire). Les
    colonnes non numériques sont ignorées.

    Args:
        df: DataFrame source.
        figsize: Taille de la figure (largeur, hauteur).

    Note:
        Pour des relations non-linéaires, préférer la corrélation de
        Spearman (basée sur les rangs) : faire ``df.corr(method='spearman')``
        en amont.

    Example:
        >>> plot_correlation_matrix(df)
    """
    plt.figure(figsize=figsize)
    sns.heatmap(
        df.corr(numeric_only=True),
        annot=True,           # affiche les coefficients
        cmap="coolwarm",      # rouge = positif, bleu = négatif
    )
    plt.show()


def compare_distribution(
    original: np.ndarray | pd.Series,
    transformed: np.ndarray | pd.Series,
) -> None:
    """Compare visuellement deux distributions (avant / après transformation).

    Utile pour vérifier l'effet d'une transformation (log, Box-Cox,
    Yeo-Johnson, etc.).

    Args:
        original: Distribution avant transformation.
        transformed: Distribution après transformation.

    Example:
        >>> from nanuq import log_transform
        >>> df_t = log_transform(df, columns=["revenu_mensuel"])
        >>> compare_distribution(df["revenu_mensuel"], df_t["revenu_mensuel_log"])
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(original, bins=30)
    axes[0].set_title("Avant transformation")

    axes[1].hist(transformed, bins=30)
    axes[1].set_title("Après transformation")

    plt.show()


# =============================================================================
# Modélisation — courbes train/test et matrice de confusion
# =============================================================================


def plot_train_test_curve(
    model_class: Any,
    X_train: np.ndarray | pd.DataFrame,
    y_train: np.ndarray | pd.Series,
    X_test: np.ndarray | pd.DataFrame,
    y_test: np.ndarray | pd.Series,
    param_name: str,
    param_values: list[Any],
    scoring: str = "auc",
    fixed_params: dict[str, Any] | None = None,
    figsize: tuple[int, int] = (10, 6),
    plot: bool = True,
) -> pd.DataFrame:
    """Trace la courbe train/test en fonction d'un hyperparamètre.

    Pour chaque valeur de ``param_name``, entraîne un nouveau modèle et
    mesure le score sur train et test. Permet de visualiser le **sweet
    spot** entre biais (à gauche) et overfit (à droite).

    Args:
        model_class: **Classe** du modèle (pas une instance) à
            instancier pour chaque valeur testée.
            Ex : ``DecisionTreeClassifier``.
        X_train: Features d'entraînement.
        y_train: Cible d'entraînement.
        X_test: Features de test.
        y_test: Cible de test.
        param_name: Nom de l'hyperparamètre à faire varier
            (``'max_depth'``, ``'min_samples_split'``, ``'C'``,
            ``'n_estimators'``, …).
        param_values: Liste / range des valeurs à tester.
        scoring: Métrique d'évaluation.

            - ``'auc'`` : roc_auc_score (nécessite ``predict_proba``)
            - ``'accuracy'`` : accuracy_score (toujours dispo)

        fixed_params: Autres paramètres du modèle, fixes pour toutes
            les itérations (ex : ``{'random_state': 42}``).
        figsize: Taille de la figure.
        plot: Si ``False``, calcule et retourne le DataFrame sans
            afficher (utile pour automatisation ou tests).

    Returns:
        DataFrame avec colonnes :

        - ``<param_name>`` : valeur testée
        - ``'train'`` : score sur train
        - ``'test'`` : score sur test
        - ``'gap'`` : écart train − test

    Example:
        >>> scores = plot_train_test_curve(
        ...     DecisionTreeClassifier,
        ...     X_train, y_train, X_test, y_test,
        ...     param_name="max_depth",
        ...     param_values=list(range(2, 30, 2)),
        ...     fixed_params={"random_state": 808},
        ...     scoring="auc",
        ... )
        >>> best = scores.loc[scores["test"].idxmax()]
    """
    fixed_params = fixed_params or {}
    rows = []

    for value in param_values:
        # Entraînement d'un modèle avec la valeur courante
        params = {**fixed_params, param_name: value}
        clf = model_class(**params)
        clf.fit(X_train, y_train)

        # Calcul du score selon la métrique demandée
        if scoring == "auc" and hasattr(clf, "predict_proba"):
            y_train_proba = clf.predict_proba(X_train)
            y_test_proba = clf.predict_proba(X_test)
            if y_train_proba.shape[1] == 2:
                train_score = roc_auc_score(y_train, y_train_proba[:, 1])
                test_score = roc_auc_score(y_test, y_test_proba[:, 1])
            else:
                train_score = roc_auc_score(
                    y_train, y_train_proba, multi_class="ovr"
                )
                test_score = roc_auc_score(
                    y_test, y_test_proba, multi_class="ovr"
                )
        else:
            # accuracy par défaut (ou si predict_proba indisponible)
            train_score = accuracy_score(y_train, clf.predict(X_train))
            test_score = accuracy_score(y_test, clf.predict(X_test))

        rows.append({
            param_name: value,
            "train": float(train_score),
            "test": float(test_score),
            "gap": float(train_score - test_score),
        })

    df_scores = pd.DataFrame(rows)

    if plot:
        # Identification du sweet spot (test maximal)
        best_idx = df_scores["test"].idxmax()
        best_value = df_scores.loc[best_idx, param_name]
        best_score = df_scores.loc[best_idx, "test"]

        plt.figure(figsize=figsize)
        plt.plot(
            df_scores[param_name], df_scores["train"],
            marker="o", label="train", linewidth=2,
        )
        plt.plot(
            df_scores[param_name], df_scores["test"],
            marker="s", label="test", linewidth=2,
        )
        plt.axvline(
            best_value, linestyle="--", color="green", alpha=0.6,
            label=f"Sweet spot : {param_name}={best_value} "
                  f"(test={best_score:.3f})",
        )
        plt.xlabel(param_name)
        plt.ylabel(scoring.upper())
        plt.title(f"Train vs Test ({scoring.upper()}) selon {param_name}")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    return df_scores


def plot_confusion_matrix_pretty(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    labels: list[Any] | None = None,
    figsize: tuple[int, int] = (8, 6),
    normalize: str | None = None,
    cmap: str = "Blues",
) -> None:
    """Affiche la matrice de confusion en heatmap annotée.

    Args:
        y_true: Vraies classes.
        y_pred: Prédictions.
        labels: Noms des classes à afficher (par défaut : indices triés).
        figsize: Taille de la figure.
        normalize: Mode de normalisation des cellules.

            - ``None`` (défaut) : comptes bruts
            - ``'true'`` : par ligne → recall par classe sur la diagonale
            - ``'pred'`` : par colonne → précision par classe sur la diagonale
            - ``'all'`` : sur le total

        cmap: Palette matplotlib (``'Blues'``, ``'Greens'``,
            ``'coolwarm'``, …).

    Example:
        >>> plot_confusion_matrix_pretty(y_test, y_pred, normalize="true")
    """
    cm = confusion_matrix(y_true, y_pred, normalize=normalize)

    plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        # Format : 2 décimales si normalisé, entier sinon
        fmt=".2f" if normalize else "d",
        cmap=cmap,
        xticklabels=labels if labels else "auto",
        yticklabels=labels if labels else "auto",
        cbar=True,
    )
    plt.xlabel("Prédiction")
    plt.ylabel("Réalité")

    title = "Matrice de confusion"
    if normalize:
        title += f' (normalisée par "{normalize}")'
    plt.title(title)
    plt.tight_layout()
    plt.show()


# =============================================================================
# Modélisation — importances de features et convergence
# =============================================================================


def plot_feature_importances(
    model: Any,
    feature_names: list[str],
    top_n: int | None = None,
    figsize: tuple[int, int] = (10, 6),
    color: str = "steelblue",
) -> pd.Series:
    """Trace l'importance des variables d'un modèle tree-based.

    Fonctionne avec tout modèle exposant l'attribut
    ``feature_importances_`` (Random Forest, Gradient Boosting,
    Decision Tree, XGBoost, LightGBM…).

    Args:
        model: Modèle entraîné avec attribut ``feature_importances_``.
        feature_names: Noms des features (même ordre que les colonnes
            de ``X`` utilisées à l'entraînement).
        top_n: Si fourni, n'affiche que les ``top_n`` variables les
            plus importantes. ``None`` (défaut) = toutes.
        figsize: Taille de la figure.
        color: Couleur des barres.

    Returns:
        ``pd.Series`` des importances triée par valeur décroissante
        (utile aussi pour exploration sans affichage).

    Raises:
        AttributeError: Si le modèle n'a pas ``feature_importances_``.

    Example:
        >>> rf_result = random_forest_pipeline(df, features, "target")
        >>> plot_feature_importances(rf_result["model"], features, top_n=10)
    """
    if not hasattr(model, "feature_importances_"):
        raise AttributeError(
            f"Le modèle {type(model).__name__} n'expose pas "
            f"`feature_importances_`. Cette fonction est destinée aux "
            f"modèles tree-based (RF, GB, DecisionTree, XGBoost, LightGBM)."
        )

    importances = pd.Series(
        model.feature_importances_, index=feature_names,
    ).sort_values(ascending=False)

    to_plot = importances.head(top_n) if top_n else importances

    plt.figure(figsize=figsize)
    # Barres horizontales triées de la plus importante en haut
    to_plot.iloc[::-1].plot(kind="barh", color=color)
    plt.xlabel("Importance relative")
    plt.title(f"Importance des variables ({type(model).__name__})")
    plt.tight_layout()
    plt.show()

    return importances


def plot_boosting_staged_loss(
    model: Any,
    X_test: np.ndarray | pd.DataFrame,
    y_test: np.ndarray | pd.Series,
    figsize: tuple[int, int] = (10, 6),
) -> np.ndarray:
    """Trace l'évolution du log_loss à chaque itération du boosting.

    Permet de visualiser comment le modèle apprend (convergence) ou
    diverge (overfit) au fil des arbres ajoutés. Très utile pour
    diagnostiquer un ``learning_rate`` trop élevé.

    Args:
        model: ``GradientBoostingClassifier`` déjà entraîné.
            Doit exposer ``staged_predict_proba()``.
        X_test: Features de test.
        y_test: Cible de test.
        figsize: Taille de la figure.

    Returns:
        Array NumPy des log_loss à chaque itération (longueur =
        ``n_estimators`` du modèle).

    Raises:
        AttributeError: Si le modèle n'a pas ``staged_predict_proba``.

    Example:
        >>> result = gradient_boosting_pipeline(
        ...     df, features, "label",
        ...     n_estimators=500, learning_rate=0.1,
        ... )
        >>> plot_boosting_staged_loss(
        ...     result["model"], result["X_test"], result["y_test"],
        ... )
    """
    if not hasattr(model, "staged_predict_proba"):
        raise AttributeError(
            f"Le modèle {type(model).__name__} n'expose pas "
            f"`staged_predict_proba`. Cette fonction nécessite un "
            f"GradientBoostingClassifier."
        )

    # Calcul du log_loss à chaque itération
    scores = np.zeros(model.n_estimators, dtype=np.float64)
    for i, y_proba in enumerate(model.staged_predict_proba(X_test)):
        # log_loss accepte 2 colonnes (binaire) OU N colonnes (multi-classes)
        scores[i] = log_loss(y_test, y_proba)

    # Identification du minimum (sweet spot)
    best_iter = int(scores.argmin())
    best_loss = float(scores.min())

    plt.figure(figsize=figsize)
    plt.plot(range(1, len(scores) + 1), scores, linewidth=2)
    plt.axvline(
        best_iter + 1, linestyle="--", color="green", alpha=0.6,
        label=f"Min : iter={best_iter + 1} (loss={best_loss:.4f})",
    )
    plt.xlabel("Itération (nombre d'arbres)")
    plt.ylabel("Log loss (test)")
    plt.title(f"Convergence du boosting (lr={model.learning_rate})")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    return scores
