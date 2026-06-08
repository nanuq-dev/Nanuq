"""
scaling — Mise à l'échelle des variables numériques.

Pourquoi scaler ? Les modèles à base de **distance** (KNN, KMeans, SVM)
et de **gradient** (régression logistique, réseaux de neurones) sont
sensibles à l'échelle des variables. Sans scaling, la variable "salaire"
(de 0 à 100 000) va écraser la variable "âge" (de 18 à 65).

**Inutile pour les modèles à base d'arbres** (RandomForest, XGBoost,
LightGBM) qui prennent des décisions sur des seuils par colonne et
sont invariants aux transformations monotones.

Méthodes disponibles :

- ``scale_minmax``   : ramène dans [0, 1] — sensible aux outliers
- ``scale_standard`` : moyenne 0, écart-type 1 — robuste, recommandé par défaut

À venir : ``scale_robust`` (basé sur l'IQR), ``scale_quantile``, et
``compare_scaling_methods`` (mise en regard).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# =============================================================================
# Scalers
# =============================================================================


def scale_minmax(
    df: pd.DataFrame,
    features: list[str],
) -> tuple[np.ndarray, MinMaxScaler]:
    """Met à l'échelle des features dans [0, 1] avec MinMaxScaler.

    Formule : ``X_scaled = (X - X_min) / (X_max - X_min)``

    Sensible aux outliers (un seul outlier compresse tout le reste).
    À privilégier quand les bornes sont connues / stables (ex : âges
    bornés entre 18 et 100).

    Args:
        df: DataFrame contenant les features.
        features: Liste des colonnes numériques à scaler.

    Returns:
        Tuple ``(X_scaled, scaler)`` :

        - ``X_scaled`` : numpy array de shape ``(n_samples, n_features)``
          avec toutes les valeurs dans ``[0, 1]``.
        - ``scaler`` : ``MinMaxScaler`` entraîné, à réutiliser sur le
          jeu de test via ``scaler.transform(X_test)``.

    Note:
        Le retour est un **numpy array**, pas un DataFrame. Si tu veux
        garder les noms de colonnes, refaire un DataFrame ::

            X_scaled, scaler = scale_minmax(df, features)
            X_scaled = pd.DataFrame(X_scaled, columns=features, index=df.index)

    Example:
        >>> X_scaled, scaler = scale_minmax(df, features=["age", "anciennete"])
        >>> # Sur le test set : X_test_scaled = scaler.transform(df_test[features])
    """
    scaler = MinMaxScaler()
    X = scaler.fit_transform(df[features])
    return X, scaler


def scale_standard(
    df: pd.DataFrame,
    features: list[str],
) -> tuple[np.ndarray, StandardScaler]:
    """Standardise les features : moyenne 0, écart-type 1.

    Formule : ``X_scaled = (X - mean) / std``

    Plus robuste aux outliers que ``MinMaxScaler``. **Recommandé par
    défaut** pour la plupart des algorithmes nécessitant un scaling.

    Args:
        df: DataFrame contenant les features.
        features: Liste des colonnes numériques à scaler.

    Returns:
        Tuple ``(X_scaled, scaler)`` :

        - ``X_scaled`` : numpy array standardisé.
        - ``scaler`` : ``StandardScaler`` entraîné, réutilisable.

    Example:
        >>> X_scaled, scaler = scale_standard(df, features=["age", "revenu_mensuel"])
    """
    scaler = StandardScaler()
    X = scaler.fit_transform(df[features])
    return X, scaler
