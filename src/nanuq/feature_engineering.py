"""
feature_engineering — Création de nouvelles features à partir des existantes.

Le feature engineering, c'est l'art de transformer ou combiner des
colonnes existantes pour aider le modèle à mieux apprendre. Souvent
plus impactant qu'un changement d'algorithme.

Fonctions disponibles :

- ``create_polynomial_features`` : génère les puissances et croisements
  (``a, b → a, b, a², ab, b²``). Aide les modèles linéaires à capturer
  de la non-linéarité.

- ``create_interaction_feature`` : croise deux colonnes (``×``, ``+``,
  ``÷``) pour créer une feature dérivée.

- ``binarize_feature`` : transforme une variable numérique en binaire
  via un seuil (``revenu > 50k → 1 sinon 0``).

- ``discretize_equal_width`` (``pd.cut``)  : découpe en intervalles
  d'amplitude égale.

- ``discretize_equal_freq`` (``pd.qcut``)  : découpe en quantiles
  (effectifs égaux). **Recommandé** sur les distributions asymétriques.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

# =============================================================================
# Features polynomiales et interactions
# =============================================================================


def create_polynomial_features(
    X: np.ndarray | pd.DataFrame,
    degree: int = 2,
    include_bias: bool = False,
) -> tuple[np.ndarray, PolynomialFeatures]:
    """Génère des features polynomiales et croisées.

    Avec ``degree=2`` et 2 features ``[a, b]``, génère
    ``[a, b, a², ab, b²]``. Permet à un modèle linéaire de capturer
    des relations non linéaires et des interactions entre variables.

    Args:
        X: Matrice de features (``numpy.ndarray`` ou ``pd.DataFrame``).
        degree: Degré maximal du polynôme.

            - ``2`` (défaut) : interactions + carrés
            - ``3+`` : à utiliser avec parcimonie

        include_bias: Si ``True``, ajoute une colonne de 1 (intercept).
            ``False`` est généralement préférable car les modèles sklearn
            gèrent déjà l'intercept de leur côté.

    Returns:
        Tuple ``(X_poly, poly)`` :

        - ``X_poly`` : nouvelle matrice élargie (numpy array).
        - ``poly`` : objet ``PolynomialFeatures`` entraîné, réutilisable
          sur d'autres données via ``poly.transform(X_test)``.

    Warning:
        Explosion combinatoire avec ``degree`` élevé : ``N`` features
        au degré ``d`` produisent ``C(N+d, d)`` colonnes. Pour 20
        features au degré 3 : 1 770 colonnes !

    Example:
        >>> X_poly, poly = create_polynomial_features(X, degree=2)
        >>> # Pour appliquer la même transformation au test set :
        >>> X_test_poly = poly.transform(X_test)
    """
    poly = PolynomialFeatures(degree=degree, include_bias=include_bias)
    X_poly = poly.fit_transform(X)
    return X_poly, poly


def create_interaction_feature(
    df: pd.DataFrame,
    col1: str,
    col2: str,
    operation: str = "multiply",
) -> pd.DataFrame:
    """Crée une feature d'interaction entre deux colonnes.

    Args:
        df: DataFrame d'entrée.
        col1: Première colonne.
        col2: Seconde colonne.
        operation: Type d'interaction.

            - ``'multiply'`` (défaut) : ``col1 * col2``
            - ``'add'``               : ``col1 + col2``
            - ``'divide'``            : ``col1 / col2`` (attention au /0)

    Returns:
        DataFrame enrichi de la colonne ``<col1>_<col2>_interaction``.

    Example:
        >>> df_eng = create_interaction_feature(
        ...     df, "annees_dans_l_entreprise", "niveau_hierarchique_poste"
        ... )
    """
    df = df.copy()
    new_col_name = f"{col1}_{col2}_interaction"

    if operation == "multiply":
        df[new_col_name] = df[col1] * df[col2]
    elif operation == "add":
        df[new_col_name] = df[col1] + df[col2]
    elif operation == "divide":
        # Attention aux divisions par zéro : à filtrer en amont si besoin
        df[new_col_name] = df[col1] / df[col2]

    return df


# =============================================================================
# Binarisation
# =============================================================================


def binarize_feature(
    df: pd.DataFrame,
    column: str,
    threshold: float,
) -> pd.DataFrame:
    """Transforme une feature numérique en variable binaire (0/1).

    Args:
        df: DataFrame d'entrée.
        column: Nom de la colonne numérique.
        threshold: Seuil de binarisation.
            Valeurs ``> threshold`` → ``1``, sinon ``0``.

    Returns:
        DataFrame enrichi de la colonne ``<column>_binary``.

    Example:
        >>> df_bin = binarize_feature(df, "revenu_mensuel", threshold=5000)
    """
    df = df.copy()
    df[f"{column}_binary"] = (df[column] > threshold).astype(int)
    return df


# =============================================================================
# Discrétisation (continu → bins)
# =============================================================================


def discretize_equal_width(
    df: pd.DataFrame,
    column: str,
    n_bins: int = 10,
    labels: bool | list[Any] | None = None,
    new_column_name: str | None = None,
) -> pd.DataFrame:
    """Discrétisation en intervalles d'amplitude égale (wrapper ``pd.cut``).

    Découpe le range ``[min, max]`` en ``n_bins`` intervalles de même
    largeur.

    Args:
        df: DataFrame d'entrée.
        column: Nom de la colonne numérique à discrétiser.
        n_bins: Nombre d'intervalles (``10`` par défaut).
        labels: Étiquettes des bins.

            - ``None`` (défaut) : objets ``pd.Interval`` (lisibles)
            - ``False`` : index entiers 0, 1, 2, …
            - ``list`` : labels personnalisés (longueur = ``n_bins``)

        new_column_name: Nom de la nouvelle colonne créée.
            Par défaut : ``<column>_cut``.

    Returns:
        DataFrame enrichi de la colonne discrétisée.

    Warning:
        Sur les distributions très asymétriques (queue lourde), la
        majorité des échantillons peut se retrouver dans un seul bin.
        Dans ce cas, préférer ``discretize_equal_freq``.

    Example:
        >>> df_bin = discretize_equal_width(df, "age", n_bins=5)
    """
    df = df.copy()
    new_col = new_column_name or f"{column}_cut"
    df[new_col] = pd.cut(df[column], bins=n_bins, labels=labels)
    return df


def discretize_equal_freq(
    df: pd.DataFrame,
    column: str,
    n_bins: int = 10,
    labels: bool | list[Any] | None = None,
    new_column_name: str | None = None,
) -> pd.DataFrame:
    """Discrétisation en intervalles d'effectifs égaux (wrapper ``pd.qcut``).

    Découpe les données en quantiles : chaque bin contient à peu près
    le même nombre d'échantillons. **Robuste aux distributions
    asymétriques** — à privilégier dans la grande majorité des cas.

    Args:
        df: DataFrame d'entrée.
        column: Nom de la colonne numérique à discrétiser.
        n_bins: Nombre de quantiles (``10`` = déciles par défaut).
            Cas usuels : ``4`` (quartiles), ``5`` (quintiles), ``10``
            (déciles).
        labels: Étiquettes des bins (voir ``discretize_equal_width``).
        new_column_name: Nom de la nouvelle colonne.
            Par défaut : ``<column>_qcut``.

    Returns:
        DataFrame enrichi de la colonne discrétisée.

    Note:
        Si beaucoup de valeurs égales existent (ex: beaucoup de 0),
        ``pd.qcut`` peut ne pas créer le nombre exact de bins demandé.
        L'option ``duplicates='drop'`` (utilisée ici) évite l'erreur
        dans ce cas.

    Example:
        >>> df_bin = discretize_equal_freq(df, "revenu_mensuel", n_bins=5)
    """
    df = df.copy()
    new_col = new_column_name or f"{column}_qcut"
    df[new_col] = pd.qcut(
        df[column],
        q=n_bins,
        labels=labels,
        duplicates="drop",  # tolère les bornes dupliquées
    )
    return df
