"""
transformations — Transformations de distribution.

Pourquoi transformer la distribution d'une feature ? Beaucoup de modèles
(notamment linéaires) supposent que les features ont une distribution
**à peu près symétrique**, idéalement normale. Quand on a une variable
fortement asymétrique (revenus, prix, populations…), une transformation
peut considérablement améliorer les performances.

Méthodes disponibles :

- ``log_transform``        : ``log(x + 1)``, simple, x ≥ 0
- ``sqrt_transform``       : racine carrée, effet plus doux
- ``boxcox_transform``     : Box-Cox, λ optimal, x > 0 strict
- ``power_transform_features``    : Box-Cox **ou** Yeo-Johnson (accepte
  les valeurs négatives, recommandé par défaut)
- ``quantile_transform_features`` : transformation par quantiles, très
  robuste aux outliers, force une distribution cible

À venir : ``compare_distribution_transforms`` (mise en regard des
transformations sur une même variable avec mesures de skewness).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import boxcox
from sklearn.preprocessing import PowerTransformer, QuantileTransformer

# =============================================================================
# Transformations simples (création d'une colonne dérivée)
# =============================================================================


def log_transform(
    df: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """Applique une transformation ``log(x + 1)`` sur chaque colonne.

    Utile pour les distributions à forte queue droite (revenus, prix,
    populations). Le ``+ 1`` évite ``log(0) = -inf``.

    Args:
        df: DataFrame d'entrée.
        columns: Liste des colonnes numériques à transformer
            (doivent être ``>= 0``).

    Returns:
        DataFrame enrichi de colonnes ``<col>_log``.

    Example:
        >>> df_t = log_transform(df, columns=["revenu_mensuel"])
        >>> # Nouvelle colonne : "revenu_mensuel_log"
    """
    df = df.copy()
    for col in columns:
        # log1p = log(x + 1), évite -inf si x = 0
        df[f"{col}_log"] = np.log(df[col] + 1)
    return df


def sqrt_transform(
    df: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """Applique une racine carrée sur chaque colonne.

    Effet adoucissant moins fort que le log : utile pour les comptages
    (lois de Poisson) ou des distributions modérément asymétriques.

    Args:
        df: DataFrame d'entrée.
        columns: Liste des colonnes numériques (``>= 0``).

    Returns:
        DataFrame enrichi de colonnes ``<col>_sqrt``.

    Example:
        >>> df_t = sqrt_transform(df, columns=["nb_formations_suivies"])
    """
    df = df.copy()
    for col in columns:
        df[f"{col}_sqrt"] = np.sqrt(df[col])
    return df


# =============================================================================
# Box-Cox : λ optimal automatique (valeurs strictement positives)
# =============================================================================


def boxcox_transform(
    df: pd.DataFrame,
    columns: list[str],
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Applique la transformation Box-Cox sur chaque colonne.

    Trouve automatiquement le ``λ`` optimal pour rendre la distribution
    la plus proche possible d'une normale.

    Requiert des valeurs **strictement positives**. Pour des données
    contenant des zéros ou des négatifs, utiliser
    ``power_transform_features`` avec ``method='yeo-johnson'``.

    Args:
        df: DataFrame d'entrée.
        columns: Liste des colonnes numériques (strictement > 0).

    Returns:
        Tuple ``(df_transformed, lambdas)`` :

        - ``df_transformed`` : DataFrame avec colonnes ``<col>_boxcox``.
        - ``lambdas`` : ``dict`` ``{col: λ_optimal}`` pour réappliquer
          la même transformation à d'autres données.

    Example:
        >>> df_t, lambdas = boxcox_transform(df, columns=["revenu_mensuel"])
        >>> lambdas["revenu_mensuel"]
        0.34  # exemple
    """
    df = df.copy()
    lambdas = {}

    for col in columns:
        # boxcox retourne (array_transformé, lambda_optimal)
        transformed, lambda_ = boxcox(df[col])
        df[f"{col}_boxcox"] = transformed
        lambdas[col] = lambda_

    return df, lambdas


# =============================================================================
# Quantile Transformer (force une distribution cible)
# =============================================================================


def quantile_transform_features(
    df: pd.DataFrame,
    columns: list[str],
    output_distribution: str = "normal",
) -> tuple[pd.DataFrame, QuantileTransformer]:
    """Transformation par quantiles vers une distribution cible.

    Force chaque colonne à suivre une distribution uniforme ou normale.
    **Très robuste aux outliers** : ils sont simplement projetés sur
    les quantiles extrêmes.

    Args:
        df: DataFrame d'entrée.
        columns: Liste des colonnes à transformer (les colonnes du
            DataFrame retourné sont **remplacées** sur place, pas de
            suffixe ajouté).
        output_distribution: Distribution cible.

            - ``'normal'`` (défaut) : N(0, 1)
            - ``'uniform'``         : U(0, 1)

    Returns:
        Tuple ``(df_transformed, qt)`` :

        - ``df_transformed`` : DataFrame avec colonnes transformées
          (les originales sont écrasées).
        - ``qt`` : ``QuantileTransformer`` entraîné, réutilisable sur
          d'autres données via ``qt.transform(...)``.

    Example:
        >>> df_t, qt = quantile_transform_features(df, columns=["revenu_mensuel"])
    """
    df = df.copy()
    qt = QuantileTransformer(output_distribution=output_distribution)
    df[columns] = qt.fit_transform(df[columns])
    return df, qt


# =============================================================================
# Power Transform (Yeo-Johnson ou Box-Cox)
# =============================================================================


def power_transform_features(
    df: pd.DataFrame,
    columns: list[str],
    method: str = "yeo-johnson",
) -> tuple[pd.DataFrame, PowerTransformer]:
    """Power transform : Yeo-Johnson ou Box-Cox via sklearn.

    Args:
        df: DataFrame d'entrée.
        columns: Liste des colonnes à transformer (remplacées sur place).
        method: Méthode utilisée.

            - ``'yeo-johnson'`` (défaut) : accepte les valeurs négatives.
              **Recommandé par défaut**.
            - ``'box-cox'`` : valeurs strictement positives uniquement.

    Returns:
        Tuple ``(df_transformed, pt)`` :

        - ``df_transformed`` : DataFrame avec colonnes transformées.
        - ``pt`` : ``PowerTransformer`` entraîné, réutilisable.

    Example:
        >>> df_t, pt = power_transform_features(df, columns=["revenu_mensuel"])
    """
    df = df.copy()
    pt = PowerTransformer(method=method)
    df[columns] = pt.fit_transform(df[columns])
    return df, pt
