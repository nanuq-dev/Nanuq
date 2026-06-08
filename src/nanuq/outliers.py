"""
outliers — Détection et traitement des valeurs aberrantes.

Deux approches complémentaires, sensibles à des choses différentes :

- **Z-score** (``compute_zscore``, ``detect_outliers_zscore``,
  ``zscore_outliers_report``) : suppose une distribution proche d'une
  normale. Mesure combien d'écarts-types séparent un point de la moyenne.
  Sur des distributions très asymétriques, l'écart-type est gonflé par
  les valeurs extrêmes elles-mêmes et le z-score sous-estime les outliers.

- **IQR** (``detect_outliers_iqr``, ``remove_outliers_iqr``,
  ``iqr_bounds_report``, ``outliers_iqr_summary``) : basé sur les
  quartiles. Une valeur est outlier si elle est hors de
  ``[Q1 − k·IQR, Q3 + k·IQR]`` (avec k = 1.5 par défaut).
  C'est le même critère que le boxplot. **Recommandé par défaut** car
  robuste aux distributions asymétriques.

À venir : ``detect_outliers_isolation_forest`` (modèle non supervisé) et
``compare_outlier_detection_methods``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

# =============================================================================
# Z-score : calcul, détection, rapport
# =============================================================================


def compute_zscore(
    df: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """Ajoute une colonne z-score pour chaque feature spécifiée.

    Le z-score mesure le nombre d'écarts-types qui sépare une valeur
    de la moyenne : ``z = (x - mean) / std``.

    Args:
        df: DataFrame d'entrée.
        columns: Liste des colonnes numériques.

    Returns:
        DataFrame enrichi de colonnes ``zscore_<col>``.

    Example:
        >>> df_z = compute_zscore(df, columns=["age", "salaire"])
        >>> df_z[["zscore_age", "zscore_salaire"]].describe()
    """
    df = df.copy()
    for col in columns:
        df[f"zscore_{col}"] = stats.zscore(df[col])
    return df


def detect_outliers_zscore(
    df: pd.DataFrame,
    column: str,
    zscore_threshold: float = 3.0,
) -> pd.DataFrame:
    """Détecte les outliers d'une colonne par la méthode z-score.

    Args:
        df: DataFrame d'entrée.
        column: Nom de la colonne numérique.
        zscore_threshold: Seuil ``|z|`` au-delà duquel une valeur est
            considérée comme outlier.

            - ``3.0`` (défaut) : ~0.3 % des valeurs sous loi normale
            - ``2.5`` : plus strict
            - ``4.0`` : plus tolérant

    Returns:
        Sous-DataFrame contenant **uniquement les lignes outliers**.

    Example:
        >>> outliers = detect_outliers_zscore(df, "revenu_mensuel")
        >>> print(f"{len(outliers)} outliers détectés")
    """
    z = np.abs(stats.zscore(df[column]))
    return df[z > zscore_threshold]


def zscore_outliers_report(
    df: pd.DataFrame,
    column: str,
    zscore_threshold: float = 3.0,
) -> dict[str, float]:
    """Rapport synthétique des outliers par z-score pour une colonne.

    Args:
        df: DataFrame d'entrée.
        column: Nom de la colonne numérique.
        zscore_threshold: Seuil ``|z|`` au-delà duquel un point est
            considéré comme outlier (``3.0`` par défaut).

    Returns:
        Dictionnaire contenant :

        - ``'mean'``, ``'std'``        : moyenne et écart-type
        - ``'z_min'``, ``'z_max'``     : z-scores extrêmes observés
        - ``'n_outliers'``             : nombre de points avec ``|z|`` > seuil
        - ``'pct_outliers'``           : pourcentage correspondant

    Note:
        Sur des distributions très asymétriques (longue queue), le
        z-score sous-estime souvent les outliers car l'écart-type est
        gonflé par les valeurs extrêmes elles-mêmes. Dans ce cas,
        préférer ``iqr_bounds_report``.

    Example:
        >>> report = zscore_outliers_report(df, "revenu_mensuel")
        >>> print(f"{report['n_outliers']} outliers ({report['pct_outliers']:.1f} %)")
    """
    series = df[column]

    # z-score : (x - mean) / std
    z = stats.zscore(series)

    n_outliers = int((np.abs(z) > zscore_threshold).sum())

    return {
        "mean": float(series.mean()),
        "std": float(series.std()),
        "z_min": float(z.min()),
        "z_max": float(z.max()),
        "n_outliers": n_outliers,
        "pct_outliers": float(n_outliers / len(df) * 100),
    }


# =============================================================================
# IQR : détection, suppression, rapports
# =============================================================================


def detect_outliers_iqr(
    df: pd.DataFrame,
    column: str,
    iqr_multiplier: float = 1.5,
) -> pd.DataFrame:
    """Détecte les outliers d'une colonne par la méthode IQR (boxplot).

    Une valeur est considérée comme outlier si elle est en dehors de ::

        [Q1 - k * IQR, Q3 + k * IQR]

    où ``k = iqr_multiplier`` (1.5 par défaut, le standard du boxplot).

    Args:
        df: DataFrame d'entrée.
        column: Nom de la colonne numérique.
        iqr_multiplier: Multiplicateur de l'IQR.

            - ``1.5`` (défaut) : outliers classiques du boxplot
            - ``3.0``          : outliers extrêmes uniquement

    Returns:
        Sous-DataFrame contenant **uniquement les lignes outliers**.

    Example:
        >>> outliers = detect_outliers_iqr(df, "revenu_mensuel")
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - iqr_multiplier * IQR
    upper = Q3 + iqr_multiplier * IQR

    return df[(df[column] < lower) | (df[column] > upper)]


def remove_outliers_iqr(
    df: pd.DataFrame,
    column: str,
    iqr_multiplier: float = 1.5,
) -> pd.DataFrame:
    """Supprime les outliers d'une colonne par la méthode IQR.

    Args:
        df: DataFrame d'entrée.
        column: Nom de la colonne numérique sur laquelle filtrer.
        iqr_multiplier: Multiplicateur de l'IQR (voir
            ``detect_outliers_iqr``).

    Returns:
        DataFrame **sans** les outliers de la colonne ciblée.

    Example:
        >>> df_clean = remove_outliers_iqr(df, "revenu_mensuel", iqr_multiplier=3.0)
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - iqr_multiplier * IQR
    upper = Q3 + iqr_multiplier * IQR

    return df[(df[column] >= lower) & (df[column] <= upper)].copy()


def iqr_bounds_report(
    df: pd.DataFrame,
    column: str,
    iqr_multiplier: float = 1.5,
) -> dict[str, float]:
    """Rapport synthétique IQR pour une colonne numérique.

    Contrairement à ``detect_outliers_iqr`` qui retourne les LIGNES
    outliers, cette fonction renvoie uniquement les STATISTIQUES :
    quartiles, bornes, comptes d'outliers, etc. Utile pour décider
    d'une stratégie de nettoyage avant de filtrer.

    Args:
        df: DataFrame d'entrée.
        column: Nom de la colonne numérique à analyser.
        iqr_multiplier: Multiplicateur de l'IQR.

            - ``1.5`` (défaut) : outliers classiques du boxplot
            - ``3.0``          : outliers extrêmes uniquement

    Returns:
        Dictionnaire contenant :

        - ``'Q1'``, ``'Q3'``, ``'IQR'`` : quartiles et écart interquartile
        - ``'lower'``, ``'upper'``      : bornes basse et haute
        - ``'min'``, ``'max'``          : min et max observés
        - ``'n_below'``, ``'n_above'``  : outliers sous / au-dessus
        - ``'n_outliers'``              : total des outliers
        - ``'pct_outliers'``            : pourcentage du dataset

    Example:
        >>> rep = iqr_bounds_report(df, "revenu_mensuel")
        >>> print(f"Borne haute : {rep['upper']:.2f}")
        >>> print(f"{rep['n_outliers']} outliers ({rep['pct_outliers']:.1f} %)")
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - iqr_multiplier * IQR
    upper = Q3 + iqr_multiplier * IQR

    n_below = int((df[column] < lower).sum())
    n_above = int((df[column] > upper).sum())
    n_outliers = n_below + n_above

    return {
        "Q1": float(Q1),
        "Q3": float(Q3),
        "IQR": float(IQR),
        "lower": float(lower),
        "upper": float(upper),
        "min": float(df[column].min()),
        "max": float(df[column].max()),
        "n_below": n_below,
        "n_above": n_above,
        "n_outliers": n_outliers,
        "pct_outliers": float(n_outliers / len(df) * 100),
    }


def outliers_iqr_summary(
    df: pd.DataFrame,
    columns: list[str],
    iqr_multiplier: float = 1.5,
) -> pd.DataFrame:
    """Rapport IQR comparatif sur plusieurs colonnes.

    Pratique en phase exploratoire pour repérer d'un coup d'œil quelles
    variables présentent le plus d'outliers.

    Args:
        df: DataFrame d'entrée.
        columns: Liste des colonnes numériques à analyser.
        iqr_multiplier: Multiplicateur de l'IQR (``1.5`` par défaut).

    Returns:
        DataFrame avec une ligne par colonne, indexé par nom de colonne,
        trié par ``pct_outliers`` décroissant. Colonnes :

        ``Q1``, ``Q3``, ``IQR``, ``lower``, ``upper``, ``min``, ``max``,
        ``n_below``, ``n_above``, ``n_outliers``, ``pct_outliers``.

    Example:
        >>> summary = outliers_iqr_summary(df, ["age", "revenu_mensuel"])
        >>> summary["pct_outliers"]
    """
    rows = []
    for col in columns:
        report = iqr_bounds_report(df, col, iqr_multiplier=iqr_multiplier)
        report["column"] = col
        rows.append(report)

    result = pd.DataFrame(rows).set_index("column")
    return result.sort_values("pct_outliers", ascending=False)
