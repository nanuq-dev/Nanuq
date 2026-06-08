"""
models.clustering — Clustering non supervisé.

Le clustering, c'est regrouper des observations qui se ressemblent,
sans avoir de cible à prédire. Utile pour segmenter une clientèle,
détecter des familles de produits, ou explorer la structure d'un
dataset avant modélisation supervisée.

Fonctions disponibles :

- ``train_kmeans`` : entraîne un modèle ``KMeans`` avec un nombre de
  clusters donné.
- ``compute_silhouette_scores`` : balaye plusieurs valeurs de ``k``
  et calcule le score de silhouette pour choisir le ``k`` optimal.

Bonnes pratiques :

- Les variables doivent être **scalées** (KMeans est basé sur des
  distances euclidiennes).
- ``random_state`` fixé pour la reproductibilité (KMeans utilise un
  tirage aléatoire pour les centres initiaux).
- ``n_init=10`` (utilisé ici) lance 10 initialisations et garde la
  meilleure, pour éviter de tomber sur un minimum local.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def train_kmeans(
    X: np.ndarray | pd.DataFrame,
    n_clusters: int = 3,
    random_state: int = 42,
) -> KMeans:
    """Entraîne un modèle ``KMeans``.

    Args:
        X: Matrice de features. **Doit être scalée** au préalable
            (sinon les variables d'échelles différentes dominent les
            distances).
        n_clusters: Nombre de clusters cible. Choisir avec la méthode
            du coude ou ``compute_silhouette_scores``.
        random_state: Graine aléatoire pour la reproductibilité.

    Returns:
        Modèle ``KMeans`` entraîné. Attributs utiles :

        - ``km.cluster_centers_`` : coordonnées des centres
        - ``km.labels_`` : cluster assigné à chaque échantillon
          d'entraînement
        - ``km.predict(X_new)`` : assigner un cluster à un nouvel
          échantillon

    Example:
        >>> from nanuq import scale_standard, train_kmeans
        >>> X_scaled, _ = scale_standard(df, features)
        >>> km = train_kmeans(X_scaled, n_clusters=4)
        >>> df["cluster"] = km.labels_
    """
    km = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10,  # 10 initialisations pour éviter les minima locaux
    )
    km.fit(X)
    return km


def compute_silhouette_scores(
    X: np.ndarray | pd.DataFrame,
    k_min: int = 2,
    k_max: int = 10,
) -> list[float]:
    """Calcule le score de silhouette pour différentes valeurs de ``k``.

    Le **score de silhouette** mesure la qualité du clustering :

    - proche de  ``1`` : clusters bien séparés
    - proche de  ``0`` : clusters qui se chevauchent
    - proche de ``-1`` : points probablement mal assignés

    Permet de choisir un ``k`` optimal : on prend la valeur de ``k``
    qui maximise le score (ou un coude visible sur la courbe).

    Args:
        X: Matrice de features (scalée de préférence).
        k_min: Nombre minimum de clusters à tester (``>= 2``).
        k_max: Nombre maximum de clusters à tester (inclus).

    Returns:
        Liste des scores de silhouette, dans l'ordre des ``k`` testés
        (de ``k_min`` à ``k_max`` inclus).

    Example:
        >>> scores = compute_silhouette_scores(X_scaled, k_min=2, k_max=8)
        >>> best_k = 2 + scores.index(max(scores))
        >>> print(f"k optimal : {best_k}")
    """
    scores = []

    for n in range(k_min, k_max + 1):
        km = KMeans(
            n_clusters=n,
            random_state=42,
            n_init=10,
        )
        km.fit(X)
        labels = km.predict(X)

        # Score de silhouette moyen sur tous les points
        score = silhouette_score(X, labels)
        scores.append(score)

    return scores
