"""
loading — Chargement de données depuis différents formats.

Fonctions de lecture qui retournent toutes un ``pandas.DataFrame`` prêt
à l'emploi. Couvre les formats les plus courants en Data Science :

    - CSV       (texte délimité)
    - Excel     (.xlsx, .xls)
    - JSON      (tabulaire)
    - Parquet   (colonnaire, compressé — recommandé pour les gros volumes)
    - SQL       (SQLite ; pour PostgreSQL / MySQL, utiliser SQLAlchemy)

Ce module fournit aussi deux fonctions plus haut niveau :

    - ``load_dataset_pipeline`` : dispatch automatique selon l'extension.
    - ``load_features``         : charge uniquement les colonnes utiles,
                                  avec validation des noms en amont.
"""

from __future__ import annotations

import sqlite3
from typing import Any

import pandas as pd

from nanuq._utils import _check_columns_exist

# =============================================================================
# Lecture par format
# =============================================================================


def load_csv(
    filepath: str,
    separator: str = ",",
    encoding: str = "utf-8",
) -> pd.DataFrame:
    """Charge un fichier CSV dans un DataFrame.

    Args:
        filepath: Chemin (relatif ou absolu) vers le fichier CSV.
        separator: Caractère séparateur des colonnes.
            ``','`` par défaut, ``';'`` fréquent en France/Europe,
            ``'\\t'`` pour les fichiers TSV.
        encoding: Encodage du fichier.
            ``'utf-8'`` par défaut, ``'latin-1'`` ou ``'cp1252'`` pour
            les CSV exportés depuis Excel sous Windows.

    Returns:
        DataFrame contenant les données du fichier.

    Example:
        >>> df = load_csv("data/sales.csv", separator=";")
    """
    return pd.read_csv(filepath, sep=separator, encoding=encoding)


def load_excel(
    filepath: str,
    sheet_name: str | int = 0,
) -> pd.DataFrame:
    """Charge une feuille d'un fichier Excel (.xlsx / .xls).

    Args:
        filepath: Chemin vers le fichier Excel.
        sheet_name: Nom (``str``) ou index (``int``) de la feuille à
            charger. Par défaut, la première feuille (index 0) est lue.

    Returns:
        DataFrame correspondant à la feuille demandée.

    Example:
        >>> df = load_excel("data/ventes.xlsx", sheet_name="2024")
    """
    return pd.read_excel(filepath, sheet_name=sheet_name)


def load_json(filepath: str) -> pd.DataFrame:
    """Charge un fichier JSON dans un DataFrame.

    Args:
        filepath: Chemin vers le fichier JSON.
            Le JSON doit être tabulaire (liste d'objets ou
            dictionnaire de listes). Pour un JSON imbriqué/hiérarchique,
            utiliser ``pd.json_normalize`` en amont.

    Returns:
        DataFrame contenant les données JSON.

    Example:
        >>> df = load_json("data/users.json")
    """
    return pd.read_json(filepath)


def load_parquet(filepath: str) -> pd.DataFrame:
    """Charge un fichier Parquet dans un DataFrame.

    Le format Parquet est colonnaire, compressé et conserve les types :
    préférable au CSV pour les gros volumes (jusqu'à 10x plus rapide
    en lecture, fichiers 5 à 10x plus compacts).

    Args:
        filepath: Chemin vers le fichier ``.parquet``.

    Returns:
        DataFrame contenant les données Parquet.

    Example:
        >>> df = load_parquet("data/transactions.parquet")
    """
    return pd.read_parquet(filepath)


def load_sql(
    database_path: str,
    query: str,
) -> pd.DataFrame:
    """Exécute une requête SQL sur une base SQLite et renvoie le résultat.

    Args:
        database_path: Chemin vers le fichier ``.db`` / ``.sqlite``.
        query: Requête SQL valide (``SELECT ... FROM ...``).

    Returns:
        DataFrame contenant le résultat de la requête.

    Note:
        La connexion est ouverte puis refermée automatiquement.
        Pour PostgreSQL / MySQL, remplacer ``sqlite3.connect`` par un
        moteur SQLAlchemy : ``sqlalchemy.create_engine("postgresql://...")``.

    Example:
        >>> df = load_sql(
        ...     "data/shop.db",
        ...     "SELECT customer_id, total FROM orders WHERE year = 2024"
        ... )
    """
    # Ouverture de la connexion à la base SQLite
    conn = sqlite3.connect(database_path)

    # Exécution de la requête et chargement direct en DataFrame
    df = pd.read_sql_query(query, conn)

    # On referme proprement la connexion (libère le fichier)
    conn.close()

    return df


# =============================================================================
# Chargement haut niveau (dispatch automatique + sélection)
# =============================================================================


def load_dataset_pipeline(
    filepath: str,
    drop_missing_rows: bool = True,
) -> pd.DataFrame:
    """Pipeline générique de chargement avec détection d'extension.

    Détecte automatiquement le format à partir de l'extension du fichier
    (csv, xlsx, xls, json) et applique éventuellement un nettoyage des
    lignes contenant des NaN.

    Args:
        filepath: Chemin vers le fichier source.
        drop_missing_rows: Si ``True``, supprime toutes les lignes
            contenant au moins un NaN (équivalent ``df.dropna()``).

            ATTENTION : peut supprimer beaucoup de lignes. Préférer
            une imputation ciblée (voir ``nanuq.cleaning``) pour
            les datasets sérieux.

    Returns:
        DataFrame chargé (et éventuellement nettoyé).

    Raises:
        ValueError: Si l'extension du fichier n'est pas supportée.
            Formats acceptés : csv, xlsx, xls, json.

    Example:
        >>> df = load_dataset_pipeline("data/raw.csv", drop_missing_rows=False)
    """
    # Extraction de l'extension (en minuscules par sécurité)
    extension = filepath.split(".")[-1].lower()

    # Dispatch selon le format de fichier
    if extension == "csv":
        df = pd.read_csv(filepath)
    elif extension in ["xlsx", "xls"]:
        df = pd.read_excel(filepath)
    elif extension == "json":
        df = pd.read_json(filepath)
    else:
        raise ValueError(
            f"Extension non supportée : {extension}. Formats acceptés : csv, xlsx, xls, json."
        )

    # Nettoyage optionnel des NaN
    if drop_missing_rows:
        df = df.dropna()

    # Feedback console utile en exploration interactive
    print("Dataset loaded successfully")
    print("Shape :", df.shape)

    return df


def load_features(
    filepath: str,
    features: list[str],
    target: str | None = None,
    drop_missing_rows: bool = False,
    **read_kwargs: Any,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.Series]:
    """Charge un dataset et sélectionne directement les colonnes utiles.

    Détecte automatiquement le format du fichier, ne charge en mémoire
    que les colonnes demandées (gain de mémoire sur les gros datasets
    grâce à ``usecols`` pour les CSV), et retourne soit X seul, soit
    le tuple ``(X, y)``.

    Équivaut au pattern courant ::

        df = pd.read_csv(filepath)
        X = df[features]
        y = df[target]

    …mais en un seul appel, avec gestion d'erreurs explicite, chargement
    plus efficace, et validation en amont des noms de colonnes.

    Args:
        filepath: Chemin vers le fichier (csv, xlsx, xls, json).
        features: Liste des colonnes à utiliser comme features (X).
        target: Nom de la colonne cible (y), optionnel.

            - Si ``None`` (défaut)  : retourne uniquement X.
            - Si renseigné          : retourne le tuple ``(X, y)``.

        drop_missing_rows: Si ``True``, supprime les lignes contenant
            au moins un NaN parmi les colonnes sélectionnées.
            L'alignement entre X et y est préservé.
        **read_kwargs: Arguments supplémentaires transmis au lecteur
            pandas (``read_csv`` / ``read_excel`` / ``read_json``).
            Utile notamment pour CSV : ``sep=';'``, ``encoding='latin-1'``,
            ``decimal=','``, etc.

    Returns:
        - ``DataFrame`` X si ``target=None``.
        - ``Tuple[DataFrame, Series]`` ``(X, y)`` si ``target`` est fourni.

    Raises:
        ValueError: Si l'extension n'est pas supportée OU si une colonne
            demandée n'existe pas dans le fichier (message d'erreur
            explicite avec la liste des colonnes disponibles).

    Example:
        Cas 1 — CSV anglo-saxon (séparateur ``,`` par défaut) ::

            >>> X = load_features(
            ...     "arbres.csv",
            ...     features=["domanialite", "genre", "espece",
            ...               "circonference_cm", "hauteur_m"],
            ... )

        Cas 2 — CSV français avec ``;`` et encoding spécifique ::

            >>> X, y = load_features(
            ...     "arbres.csv",
            ...     features=["DOMANIALITE", "GENRE", "CIRCONFERENCE (cm)"],
            ...     target="HAUTEUR (m)",
            ...     drop_missing_rows=True,
            ...     sep=";",
            ...     encoding="utf-8",
            ... )
    """
    extension = filepath.split(".")[-1].lower()

    # Liste complète des colonnes à charger (features + target éventuel)
    cols_to_load = list(features) + ([target] if target else [])

    # ----- Étape 1 : lire l'en-tête pour valider les noms en amont -----
    # Plus robuste qu'attendre l'erreur de usecols qui est cryptique
    # quand le séparateur est mauvais.
    if extension == "csv":
        header_df = pd.read_csv(filepath, nrows=0, **read_kwargs)
    elif extension in ["xlsx", "xls"]:
        header_df = pd.read_excel(filepath, nrows=0, **read_kwargs)
    elif extension == "json":
        header_df = pd.read_json(filepath, **read_kwargs)
    else:
        raise ValueError(
            f"Extension non supportée : {extension}. Formats acceptés : csv, xlsx, xls, json."
        )

    available_columns = list(header_df.columns)

    # Détection d'un mauvais séparateur (cas typique : une seule colonne avec
    # tous les noms collés)
    if len(available_columns) == 1 and any(sep in available_columns[0] for sep in [";", "\t", "|"]):
        raise ValueError(
            f"Le fichier ne semble contenir qu'une seule colonne :\n"
            f"  {available_columns[0][:200]}...\n"
            f"=> Le séparateur n'est probablement pas ','. "
            f"Essayer load_features(..., sep=';') ou sep='\\t'."
        )

    _check_columns_exist(header_df, cols_to_load)

    # ----- Étape 2 : chargement réel des colonnes sélectionnées -----
    if extension == "csv":
        # usecols économise la mémoire en n'allouant pas les autres colonnes
        df = pd.read_csv(filepath, usecols=cols_to_load, **read_kwargs)
    elif extension in ["xlsx", "xls"]:
        df = pd.read_excel(filepath, **read_kwargs)[cols_to_load]
    else:  # json
        df = pd.read_json(filepath, **read_kwargs)[cols_to_load]

    # ----- Étape 3 : drop NaN éventuel (préserve l'alignement X/y) -----
    if drop_missing_rows:
        df = df.dropna()

    # ----- Étape 4 : split features / target -----
    X = df[features].copy()

    if target is not None:
        y = df[target].copy()
        return X, y

    return X
