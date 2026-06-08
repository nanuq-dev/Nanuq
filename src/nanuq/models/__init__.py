"""
nanuq.models — Sous-package des modèles de Machine Learning.

Organisé par famille de modèles :

- ``linear``     : régression linéaire, logistique, Ridge, Lasso
- ``trees``      : Random Forest
- ``boosted``    : Gradient Boosting (sklearn), XGBoost et LightGBM
                   à venir
- ``clustering`` : KMeans + sélection automatique du k optimal
"""

from __future__ import annotations

# Réexports pour usage direct depuis nanuq.models
from nanuq.models.boosted import gradient_boosting_pipeline
from nanuq.models.clustering import compute_silhouette_scores, train_kmeans
from nanuq.models.linear import (
    lasso_regression_pipeline,
    linear_regression_multivariate,
    linear_regression_univariate,
    logistic_regression_pipeline,
    ridge_regression_pipeline,
)
from nanuq.models.trees import random_forest_pipeline

__all__ = [
    # linear
    "linear_regression_univariate",
    "linear_regression_multivariate",
    "logistic_regression_pipeline",
    "ridge_regression_pipeline",
    "lasso_regression_pipeline",
    # trees
    "random_forest_pipeline",
    # boosted
    "gradient_boosting_pipeline",
    # clustering
    "train_kmeans",
    "compute_silhouette_scores",
]
