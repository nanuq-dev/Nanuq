"""
encoding — Encodage des variables catégorielles.

Pourquoi encoder ? La plupart des modèles de Machine Learning attendent
des nombres en entrée, pas des chaînes de caractères. Encoder, c'est
transformer ``["Paris", "Lyon", "Paris"]`` en quelque chose que le modèle
peut digérer (``[0, 1, 0]``, ou trois colonnes binaires, etc.).

Quelle méthode choisir ?

- **Binaire** (``encode_binary``) : une variable à 2 modalités, mapping explicite.
  Ex : ``{"Oui": 1, "Non": 0}``.

- **Ordinal** (``encode_multiclass``) : modalités avec ordre logique.
  Ex : ``{"Faible": 0, "Moyen": 1, "Élevé": 2}``.

- **Label** (``label_encode_column``) : encodage automatique alphabétique.
  Recommandé pour la **cible** d'un problème multi-classe.

- **One-Hot** (``one_hot_encode`` / ``one_hot_encode_sklearn``) :
  une colonne binaire par modalité. À utiliser pour les variables
  **nominales** (sans ordre : ville, couleur…). Recommandé par défaut.

- **Binaire compact** (``encode_binary_representation``) : encode N
  modalités en ⌈log₂(N)⌉ colonnes. Utile en haute cardinalité (>15).
  À réserver aux modèles à base d'arbres.

À venir : ``target_encoding``, ``frequency_encoding``, et
``compare_encoding_methods``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# =============================================================================
# Encodage par mapping explicite (binaire ou ordinal)
# =============================================================================


def encode_binary(
    df: pd.DataFrame,
    column: str,
    mapping: dict[Any, int],
) -> pd.DataFrame:
    """Encode une variable **binaire** selon un mapping manuel.

    À utiliser pour les variables à exactement 2 modalités, quand on
    connaît leur sémantique et qu'on veut maîtriser l'encodage.

    Args:
        df: DataFrame d'entrée.
        column: Nom de la colonne à encoder.
        mapping: Dictionnaire ``{valeur_originale: valeur_encodée}``.
            Exemple : ``{"Oui": 1, "Non": 0}``.

    Returns:
        DataFrame avec la colonne encodée en ``int``.

    Example:
        >>> df_enc = encode_binary(df, "heure_supplementaires",
        ...                        mapping={"Oui": 1, "Non": 0})
    """
    df = df.copy()

    # En pandas 3.0+, les colonnes string sont strictement typées : on
    # convertit explicitement en object avant d'y assigner des entiers.
    df[column] = df[column].astype(object)

    # Remplacement par mapping explicite (plus contrôlé que .map())
    for original_value, encoded_value in mapping.items():
        df.loc[df[column] == original_value, column] = encoded_value

    # Conversion en int pour cohérence des types
    df[column] = df[column].astype(int)
    return df


def encode_multiclass(
    df: pd.DataFrame,
    column: str,
    mapping: dict[Any, int],
) -> pd.DataFrame:
    """Encode une variable catégorielle multi-classes selon un mapping ordinal.

    À utiliser quand un **ordre existe** entre les modalités.
    Exemple : ``{"Faible": 0, "Moyen": 1, "Élevé": 2}``.

    Pour des variables **nominales** (sans ordre — ville, couleur…),
    préférer ``one_hot_encode`` afin de ne pas introduire d'ordre fictif
    que le modèle exploiterait à tort.

    Args:
        df: DataFrame d'entrée.
        column: Nom de la colonne à encoder.
        mapping: Dictionnaire ``{modalité: code_numérique}`` ordonné.

    Returns:
        DataFrame avec la colonne encodée en ``int``.

    Example:
        >>> df_enc = encode_multiclass(df, "frequence_deplacement",
        ...                            mapping={"Jamais": 0,
        ...                                     "Occasionnel": 1,
        ...                                     "Fréquent": 2})
    """
    df = df.copy()

    # En pandas 3.0+, les colonnes string sont strictement typées : on
    # convertit explicitement en object avant d'y assigner des entiers.
    df[column] = df[column].astype(object)

    for original_value, encoded_value in mapping.items():
        df.loc[df[column] == original_value, column] = encoded_value

    df[column] = df[column].astype(int)
    return df


# =============================================================================
# Label encoding (sklearn) — pour la cible
# =============================================================================


def label_encode_column(
    df: pd.DataFrame,
    column: str,
) -> tuple[pd.DataFrame, LabelEncoder]:
    """Applique un ``LabelEncoder`` sklearn à une colonne.

    Encodage automatique : chaque modalité reçoit un entier (ordre
    alphabétique). Particulièrement adapté à la **colonne cible** d'un
    problème de classification multi-classes.

    Args:
        df: DataFrame d'entrée.
        column: Nom de la colonne à encoder.

    Returns:
        Tuple ``(df_encoded, encoder)`` :

        - ``df_encoded`` : DataFrame avec la colonne encodée en int.
        - ``encoder`` : ``LabelEncoder`` entraîné, réutilisable pour :

          - décoder : ``encoder.inverse_transform(y_pred)``
          - transformer de nouvelles données : ``encoder.transform(...)``

    Example:
        >>> df_enc, encoder = label_encode_column(df, "departement")
        >>> encoder.classes_
        array(['Commercial', 'R&D', 'RH'], dtype=object)
    """
    df = df.copy()
    encoder = LabelEncoder()
    df[column] = encoder.fit_transform(df[column])
    return df, encoder


# =============================================================================
# Encodage binaire compact (haute cardinalité)
# =============================================================================


def encode_binary_representation(
    df: pd.DataFrame,
    column: str,
    drop_original: bool = True,
) -> tuple[pd.DataFrame, dict[Any, int]]:
    """Encode une variable catégorielle en représentation binaire compacte.

    Pour ``N`` modalités, génère ``⌈log₂(N)⌉`` colonnes binaires (au lieu
    de ``N`` pour le one-hot). Utile en haute cardinalité (>15).

    Args:
        df: DataFrame d'entrée.
        column: Nom de la colonne catégorielle.
        drop_original: Si ``True``, supprime la colonne d'origine.

    Returns:
        Tuple ``(df_encoded, mapping)`` :

        - ``df_encoded`` : DataFrame avec les colonnes
          ``<column>_bit_0``, ``<column>_bit_1``, …
        - ``mapping`` : ``dict`` ``{modalité: index}`` pour décoder ou
          réappliquer l'encodage ailleurs.

    Warning:
        Introduit un ordre fictif sur les bits, ce qui peut biaiser les
        modèles linéaires. À réserver de préférence aux **modèles à
        base d'arbres** (Random Forest, XGBoost, LightGBM).

        À ne pas confondre avec ``encode_binary`` qui sert à encoder
        une variable à 2 modalités (Oui/Non → 0/1) via mapping explicite.

    Example:
        >>> df_enc, mapping = encode_binary_representation(df, "departement")
        >>> # 3 modalités => 2 colonnes binaires (bit_0, bit_1)
    """
    df = df.copy()

    # 1. Construire un mapping ordinal (chaque modalité reçoit un index)
    unique_values = df[column].dropna().unique()
    mapping = {val: idx for idx, val in enumerate(unique_values)}

    # 2. Convertir chaque valeur en son index entier
    #    (NaN -> -1 temporairement pour permettre les opérations bitwise)
    nan_mask = df[column].isna().to_numpy()
    indices_int = df[column].map(mapping).fillna(-1).astype(int).to_numpy()

    # 3. Calculer le nombre de bits nécessaires
    n_categories = len(unique_values)
    n_bits = max(1, int(np.ceil(np.log2(n_categories))))

    # 4. Décomposer chaque index en bits (poids faible -> poids fort)
    for bit_pos in range(n_bits):
        col_name = f"{column}_bit_{bit_pos}"
        # Décalage à droite + ET logique pour extraire le bit demandé
        bit_array = (indices_int >> bit_pos) & 1
        # On remet NaN là où la valeur d'origine était manquante
        bit_series = pd.Series(bit_array, index=df.index, dtype="Int64")
        bit_series[nan_mask] = pd.NA
        df[col_name] = bit_series

    if drop_original:
        df = df.drop(columns=[column])

    return df, mapping


# =============================================================================
# One-Hot Encoding
# =============================================================================


def one_hot_encode(
    df: pd.DataFrame,
    columns: list[str],
    drop_first: bool = False,
) -> pd.DataFrame:
    """One-Hot Encoding de colonnes catégorielles (via ``pd.get_dummies``).

    Crée une colonne binaire par modalité. Idéal pour variables
    **nominales** sans ordre (couleur, ville, département…).

    Simple et rapide. Pour un usage "production" (gestion des modalités
    inconnues, compatibilité avec les Pipelines sklearn), préférer
    ``one_hot_encode_sklearn``.

    Args:
        df: DataFrame d'entrée.
        columns: Liste des colonnes catégorielles à encoder.
        drop_first: Si ``True``, supprime la première modalité pour
            éviter la colinéarité parfaite (utile en régression linéaire
            / logistique).

            - **Modèles linéaires** : ``True`` recommandé.
            - **Modèles à base d'arbres** : ``False`` (la colinéarité ne
              les gêne pas, et on garde toute l'information).

    Returns:
        DataFrame élargi avec les colonnes one-hot.

    Example:
        >>> df_enc = one_hot_encode(df, columns=["departement", "poste"],
        ...                         drop_first=True)
    """
    return pd.get_dummies(df, columns=columns, drop_first=drop_first)


def one_hot_encode_sklearn(
    df: pd.DataFrame,
    columns: list[str],
    drop_first: bool = False,
    handle_unknown: str = "ignore",
    return_encoder: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, OneHotEncoder | None]:
    """One-Hot Encoding via le ``OneHotEncoder`` de sklearn.

    Avantages par rapport à ``one_hot_encode`` (``pd.get_dummies``) :

    - Gère proprement les **modalités inconnues** au moment du
      ``transform`` (utile quand on déploie le modèle sur de nouvelles
      données).
    - L'encoder peut être **retourné et réutilisé** sur le test set.
    - Compatible avec les ``Pipeline`` sklearn pour un workflow
      industriel.

    Args:
        df: DataFrame d'entrée.
        columns: Liste des colonnes catégorielles à encoder.
        drop_first: Si ``True``, supprime la première modalité (évite la
            colinéarité parfaite). Idem que ``one_hot_encode``.
        handle_unknown: Comportement face à une modalité inconnue au
            moment du ``transform`` :

            - ``'ignore'`` (défaut) : crée un vecteur 0 partout (safe en
              production).
            - ``'error'`` : lève une exception.

        return_encoder: Si ``True``, retourne aussi le ``OneHotEncoder``
            entraîné (utile pour réappliquer la même transformation sur
            de nouvelles données plus tard).

    Returns:
        - Si ``return_encoder=False`` : DataFrame élargi.
        - Si ``return_encoder=True``  : ``(DataFrame élargi, OneHotEncoder)``.

        Si aucune colonne demandée n'est présente dans ``df``, la
        fonction renvoie le DataFrame tel quel (et ``None`` pour
        l'encoder le cas échéant).

    Example:
        >>> df_enc, encoder = one_hot_encode_sklearn(
        ...     df, columns=["departement", "poste"], return_encoder=True
        ... )
        >>> # Réutilisation sur le test set :
        >>> X_test_enc = encoder.transform(df_test[["departement", "poste"]])
    """
    df = df.copy()

    # Filtrer pour ne garder que les colonnes effectivement présentes
    cols_present = [c for c in columns if c in df.columns]
    if not cols_present:
        return (df, None) if return_encoder else df

    # Initialisation du OneHotEncoder
    encoder = OneHotEncoder(
        drop="first" if drop_first else None,
        sparse_output=False,            # array dense, plus simple à manipuler
        handle_unknown=handle_unknown,
        dtype=int,                      # 0/1 plutôt que True/False
    )

    # Fit + transform
    encoded_array = encoder.fit_transform(df[cols_present])

    # Reconstruction d'un DataFrame avec les bons noms de colonnes
    encoded_df = pd.DataFrame(
        encoded_array,
        columns=encoder.get_feature_names_out(cols_present),
        index=df.index,
    )

    # Remplacer les colonnes originales par leurs versions encodées
    df_result = pd.concat(
        [df.drop(columns=cols_present), encoded_df],
        axis=1,
    )

    if return_encoder:
        return df_result, encoder
    return df_result
