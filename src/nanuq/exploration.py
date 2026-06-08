"""
exploration — Exploration et diagnostic d'un DataFrame.

Fonctions de "premier regard" à utiliser dès qu'on découvre un nouveau
dataset :

    - ``dataset_overview``       : vue d'ensemble en console
    - ``get_column_types``       : sépare numériques / catégorielles
    - ``split_features_target``  : sépare X (features) et y (cible)

Pour un rapport d'analyse exploratoire plus poussé (multi-pages,
recommandations automatiques par colonne), voir le module
``nanuq.reporting``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# =============================================================================
# Vue d'ensemble
# =============================================================================


def dataset_overview(df: pd.DataFrame) -> None:
    """Affiche une vue d'ensemble rapide du DataFrame en console.

    Affiche successivement :

    1. La forme du DataFrame (lignes × colonnes)
    2. Les types de colonnes et la mémoire utilisée (``df.info()``)
    3. Les statistiques descriptives des colonnes numériques
       (``df.describe()``)
    4. Les 5 premières lignes (``df.head()``)
    5. Le nombre de valeurs manquantes par colonne

    Cette fonction n'a **pas de valeur de retour** : elle affiche
    uniquement. Elle est conçue pour un usage interactif (notebook,
    REPL) au tout début de l'exploration d'un dataset.

    Args:
        df: DataFrame à inspecter.

    Example:
        >>> dataset_overview(df)
        ============================================================
        SHAPE : (1470, 35)
        ============================================================
        ...
    """
    print("=" * 60)
    print("SHAPE :", df.shape)
    print("=" * 60)
    print(df.info())
    print("=" * 60)
    print(df.describe())
    print("=" * 60)
    print("HEAD :")
    print(df.head())
    print("=" * 60)
    print("VALEURS MANQUANTES PAR COLONNE :")
    print(df.isna().sum())


# =============================================================================
# Typage des colonnes
# =============================================================================


def get_column_types(df: pd.DataFrame) -> dict[str, list[str]]:
    """Sépare les colonnes en deux groupes : numériques / catégorielles.

    Utile pour appliquer automatiquement des traitements différents
    selon le type de colonne (scaling pour les numériques, encodage
    pour les catégorielles, etc.).

    Args:
        df: DataFrame à analyser.

    Returns:
        Dictionnaire avec deux clés :

        - ``'numeric'``     : liste des colonnes numériques (int, float)
        - ``'categorical'`` : liste des autres colonnes (object, str,
          bool, datetime, category, ...)

    Note:
        Les booléens et datetimes sont rangés dans ``'categorical'``.
        Si tu veux les isoler, sélectionne-les explicitement avec
        ``df.select_dtypes(include=["bool"])``.

    Example:
        >>> types = get_column_types(df)
        >>> types["numeric"]
        ['age', 'revenu_mensuel', 'annees_dans_l_entreprise']
        >>> types["categorical"]
        ['genre', 'statut_marital', 'departement', 'poste']
    """
    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = df.select_dtypes(exclude=np.number).columns.tolist()

    return {
        "numeric": numeric_columns,
        "categorical": categorical_columns,
    }


# =============================================================================
# Séparation features / target
# =============================================================================


def split_features_target(
    df: pd.DataFrame,
    features: list[str],
    target: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Sépare le DataFrame en matrice de features X et vecteur cible y.

    Args:
        df: DataFrame source contenant features et target.
        features: Liste des noms de colonnes à utiliser comme features.
        target: Nom de la colonne cible.

    Returns:
        Tuple ``(X, y)`` où :

        - ``X`` est un DataFrame contenant les colonnes ``features``
        - ``y`` est une Series contenant la colonne ``target``

    Example:
        >>> X, y = split_features_target(
        ...     df,
        ...     features=["age", "revenu_mensuel", "anciennete"],
        ...     target="a_quitte_l_entreprise",
        ... )
    """
    X = df[features]
    y = df[target]
    return X, y
