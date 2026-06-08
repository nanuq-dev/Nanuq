"""Tests du module nanuq.models.clustering."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

from nanuq import compute_silhouette_scores, train_kmeans


@pytest.fixture
def blob_data() -> np.ndarray:
    """3 blobs bien séparés -> k=3 doit être optimal."""
    X, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.6, random_state=42)
    return X


# =============================================================================
# train_kmeans
# =============================================================================


def test_train_kmeans_returns_fitted_model(blob_data: np.ndarray) -> None:
    """train_kmeans renvoie un KMeans entraîné."""
    km = train_kmeans(blob_data, n_clusters=3)

    assert isinstance(km, KMeans)
    # Le modèle a été fit : il a des centres
    assert hasattr(km, "cluster_centers_")
    assert km.cluster_centers_.shape == (3, blob_data.shape[1])


def test_train_kmeans_assigns_labels(blob_data: np.ndarray) -> None:
    """Les labels du modèle couvrent tous les n_clusters."""
    km = train_kmeans(blob_data, n_clusters=3)
    labels = km.labels_

    assert len(labels) == len(blob_data)
    assert set(labels) == {0, 1, 2}


# =============================================================================
# compute_silhouette_scores
# =============================================================================


def test_compute_silhouette_scores_length(blob_data: np.ndarray) -> None:
    """compute_silhouette_scores renvoie k_max - k_min + 1 scores."""
    scores = compute_silhouette_scores(blob_data, k_min=2, k_max=5)
    assert len(scores) == 4  # k=2, 3, 4, 5


def test_silhouette_finds_correct_k_on_blobs(blob_data: np.ndarray) -> None:
    """Sur 3 blobs bien séparés, le score est maximal en k=3."""
    scores = compute_silhouette_scores(blob_data, k_min=2, k_max=6)
    # Trouve k qui maximise (offset = k_min)
    best_k = 2 + scores.index(max(scores))
    assert best_k == 3


def test_silhouette_scores_are_in_valid_range(blob_data: np.ndarray) -> None:
    """Tous les scores sont dans [-1, 1]."""
    scores = compute_silhouette_scores(blob_data, k_min=2, k_max=4)
    for s in scores:
        assert -1.0 <= s <= 1.0
