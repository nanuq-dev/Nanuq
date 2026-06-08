"""
cleaning — Nettoyage des données et gestion des valeurs manquantes.

Fonctions disponibles :

- ``clean_nan``               : supprime toutes les lignes contenant un NaN
- ``missing_values_report``   : rapport tabulaire des NaN par colonne
- ``fillna_mean``             : imputation par la moyenne (numérique)
- ``fillna_median``           : imputation par la médiane (numérique)
- ``fillna_category``         : imputation par une valeur fixe (catégoriel)

À venir : ``fillna_mode``, ``fillna_knn``, ``fillna_iterative``, et
``compare_imputation_methods`` (mise en regard de plusieurs stratégies).
"""

from __future__ import annotations

import pandas as pd

# =============================================================================
# Rapports et nettoyage global
# =============================================================================


def clean_nan(df: pd.DataFrame) -> pd.DataFrame:
    """Supprime toutes les lignes contenant au moins un NaN.

    Équivalent à ``df.dropna()`` mais retourne une copie pour préserver
    le DataFrame d'entrée.

    Args:
        df: DataFrame d'entrée (non modifié).

    Returns:
        Nouveau DataFrame sans aucune ligne NaN.

    Warning:
        Peut supprimer une grosse partie du dataset si les NaN sont
        dispersés. Préférer une imputation ciblée (``fillna_*``) pour
        les colonnes peu touchées.

    Example:
        >>> df_clean = clean_nan(df)
    """
    return df.dropna().copy()


def missing_values_report(df: pd.DataFrame) -> pd.DataFrame:
    """Génère un rapport détaillé des valeurs manquantes par colonne.

    Args:
        df: DataFrame à analyser.

    Returns:
        DataFrame trié par % de NaN décroissant, avec deux colonnes :

        - ``missing_count``   : nombre absolu de NaN
        - ``missing_percent`` : pourcentage de NaN (0 à 100)

    Example:
        >>> report = missing_values_report(df)
        >>> report.head()
                          missing_count  missing_percent
        revenu_mensuel              147              10.0
        annees_experience            44               3.0
        age                           0               0.0
    """
    # Compte absolu de NaN par colonne
    missing_count = df.isna().sum()

    # Pourcentage de NaN (mean() sur bool = proportion)
    missing_percent = df.isna().mean() * 100

    report = pd.DataFrame(
        {
            "missing_count": missing_count,
            "missing_percent": missing_percent,
        }
    )

    # Tri du plus problématique au moins problématique
    return report.sort_values(by="missing_percent", ascending=False)


# =============================================================================
# Imputation des valeurs manquantes
# =============================================================================


def fillna_mean(
    df: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """Imputation des NaN par la moyenne de chaque colonne.

    Sensible aux outliers : un seul revenu de 1 M€ va tirer la moyenne
    vers le haut. Préférer ``fillna_median`` si la distribution est
    asymétrique (revenus, prix immobiliers, etc.).

    Args:
        df: DataFrame d'entrée.
        columns: Liste des colonnes **numériques** à imputer.

    Returns:
        DataFrame avec NaN remplacés par la moyenne de chaque colonne.

    Example:
        >>> df_clean = fillna_mean(df, columns=["age", "revenu_mensuel"])
    """
    df = df.copy()
    for col in columns:
        df[col] = df[col].fillna(df[col].mean())
    return df


def fillna_median(
    df: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """Imputation des NaN par la médiane de chaque colonne.

    Robuste aux outliers : la médiane est insensible aux valeurs
    extrêmes. **Recommandé par défaut** pour les distributions
    asymétriques (revenus, prix, ancienneté…).

    Args:
        df: DataFrame d'entrée.
        columns: Liste des colonnes **numériques** à imputer.

    Returns:
        DataFrame avec NaN remplacés par la médiane de chaque colonne.

    Example:
        >>> df_clean = fillna_median(df, columns=["revenu_mensuel"])
    """
    df = df.copy()
    for col in columns:
        df[col] = df[col].fillna(df[col].median())
    return df


def fillna_category(
    df: pd.DataFrame,
    column: str,
    fill_value: str = "Unknown",
) -> pd.DataFrame:
    """Imputation des NaN pour une colonne catégorielle.

    Args:
        df: DataFrame d'entrée.
        column: Nom de la colonne catégorielle à imputer.
        fill_value: Valeur de remplacement.

            - ``'Unknown'`` (défaut)
            - Alternatives : ``'Missing'``, ``'NA'``
            - Ou la modalité la plus fréquente : ``df[col].mode()[0]``

    Returns:
        DataFrame avec NaN remplacés par ``fill_value`` dans la colonne.

    Example:
        >>> df_clean = fillna_category(df, column="domaine_etude")
        >>> df_clean = fillna_category(df, column="statut_marital",
        ...                            fill_value="Non renseigné")
    """
    df = df.copy()
    df[column] = df[column].fillna(fill_value)
    return df
