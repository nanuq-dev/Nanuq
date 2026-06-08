"""Tests pour nanuq.persistence."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from nanuq.persistence import (
    build_metadata,
    load_artifacts,
    load_model_bundle,
    predict_with_threshold,
    save_artifacts,
    save_model_bundle,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def trained_classifier():
    """Un classifieur entraîné sur des données synthétiques (rapide)."""
    np.random.seed(42)
    X = pd.DataFrame(
        np.random.randn(200, 5),
        columns=[f"feat_{i}" for i in range(5)],
    )
    y = (X["feat_0"] + X["feat_1"] > 0).astype(int)
    model = GradientBoostingClassifier(n_estimators=10, max_depth=2, random_state=42)
    model.fit(X, y)
    return model, X, y


@pytest.fixture
def valid_metadata(trained_classifier):
    """Métadonnées valides associées au classifieur de test."""
    model, X, _ = trained_classifier
    return build_metadata(
        model,
        feature_order=X.columns.tolist(),
        threshold=0.4,
        metrics={"auc": 0.85, "f1": 0.72},
        hyperparameters={"max_depth": 2, "n_estimators": 10},
    )


# =============================================================================
# Tests : build_metadata
# =============================================================================


class TestBuildMetadata:
    def test_returns_dict_with_required_keys(self, trained_classifier):
        model, X, _ = trained_classifier
        meta = build_metadata(model, X.columns.tolist(), threshold=0.5)
        assert isinstance(meta, dict)
        assert "model_type" in meta
        assert "feature_order" in meta
        assert "threshold" in meta
        assert "created_at" in meta
        assert "sklearn_version" in meta

    def test_model_type_extracted_from_class(self, trained_classifier):
        model, X, _ = trained_classifier
        meta = build_metadata(model, X.columns.tolist(), threshold=0.5)
        assert meta["model_type"] == "GradientBoostingClassifier"

    def test_feature_order_is_list(self, trained_classifier):
        model, X, _ = trained_classifier
        # Même si on passe un Index pandas, ça doit ressortir en list
        meta = build_metadata(model, X.columns, threshold=0.5)
        assert isinstance(meta["feature_order"], list)
        assert meta["feature_order"] == ["feat_0", "feat_1", "feat_2", "feat_3", "feat_4"]

    def test_threshold_out_of_range_raises(self, trained_classifier):
        model, X, _ = trained_classifier
        with pytest.raises(ValueError, match="entre 0 et 1"):
            build_metadata(model, X.columns.tolist(), threshold=1.5)
        with pytest.raises(ValueError, match="entre 0 et 1"):
            build_metadata(model, X.columns.tolist(), threshold=-0.1)

    def test_optional_fields_default_to_empty(self, trained_classifier):
        model, X, _ = trained_classifier
        meta = build_metadata(model, X.columns.tolist(), threshold=0.5)
        assert meta["metrics"] == {}
        assert meta["hyperparameters"] == {}
        assert meta["training_info"] == {}
        assert meta["notes"] is None


# =============================================================================
# Tests : save_artifacts / load_artifacts
# =============================================================================


class TestSaveLoadArtifacts:
    def test_creates_both_files(self, tmp_path, trained_classifier, valid_metadata):
        model, _, _ = trained_classifier
        paths = save_artifacts(model, valid_metadata, tmp_path)
        assert paths["model"].exists()
        assert paths["metadata"].exists()
        assert paths["model"].name == "model.joblib"
        assert paths["metadata"].name == "metadata.json"

    def test_metadata_is_readable_json(self, tmp_path, trained_classifier, valid_metadata):
        model, _, _ = trained_classifier
        save_artifacts(model, valid_metadata, tmp_path)
        with open(tmp_path / "metadata.json", encoding="utf-8") as f:
            loaded = json.load(f)
        assert loaded["threshold"] == 0.4
        assert loaded["model_type"] == "GradientBoostingClassifier"
        # Le hash a été ajouté automatiquement
        assert "model_sha256" in loaded

    def test_roundtrip_preserves_model_and_metadata(
        self, tmp_path, trained_classifier, valid_metadata
    ):
        model, X, _ = trained_classifier
        save_artifacts(model, valid_metadata, tmp_path)

        loaded_model, loaded_meta = load_artifacts(tmp_path)
        # Le modèle rechargé doit produire les mêmes prédictions
        np.testing.assert_array_equal(model.predict_proba(X), loaded_model.predict_proba(X))
        # Les métadonnées doivent contenir les mêmes clés essentielles
        assert loaded_meta["threshold"] == valid_metadata["threshold"]
        assert loaded_meta["feature_order"] == valid_metadata["feature_order"]
        assert loaded_meta["metrics"] == valid_metadata["metrics"]

    def test_overwrite_false_raises_if_exists(self, tmp_path, trained_classifier, valid_metadata):
        model, _, _ = trained_classifier
        save_artifacts(model, valid_metadata, tmp_path)
        with pytest.raises(FileExistsError):
            save_artifacts(model, valid_metadata, tmp_path)

    def test_overwrite_true_allows_replacement(self, tmp_path, trained_classifier, valid_metadata):
        model, _, _ = trained_classifier
        save_artifacts(model, valid_metadata, tmp_path)
        # Une 2e sauvegarde avec overwrite=True doit fonctionner
        save_artifacts(model, valid_metadata, tmp_path, overwrite=True)

    def test_creates_output_dir_if_missing(self, tmp_path, trained_classifier, valid_metadata):
        model, _, _ = trained_classifier
        target = tmp_path / "nouveau" / "sous" / "dossier"
        save_artifacts(model, valid_metadata, target)
        assert target.exists()
        assert (target / "model.joblib").exists()

    def test_hash_verification_catches_corruption(
        self, tmp_path, trained_classifier, valid_metadata
    ):
        model, _, _ = trained_classifier
        save_artifacts(model, valid_metadata, tmp_path)
        # Corrompre le modèle (le remplacer par un autre)
        other = LogisticRegression()
        from joblib import dump

        dump(other, tmp_path / "model.joblib")
        with pytest.raises(ValueError, match="Hash du modèle"):
            load_artifacts(tmp_path)

    def test_hash_verification_can_be_disabled(self, tmp_path, trained_classifier, valid_metadata):
        model, _, _ = trained_classifier
        save_artifacts(model, valid_metadata, tmp_path)
        from joblib import dump

        dump(LogisticRegression(), tmp_path / "model.joblib")
        # Avec verify_hash=False, ne lève pas
        loaded_model, _ = load_artifacts(tmp_path, verify_hash=False)
        assert isinstance(loaded_model, LogisticRegression)

    def test_load_missing_files_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_artifacts(tmp_path)


# =============================================================================
# Tests : save_model_bundle / load_model_bundle
# =============================================================================


class TestBundle:
    def test_roundtrip(self, tmp_path, trained_classifier, valid_metadata):
        model, X, _ = trained_classifier
        bundle_path = tmp_path / "bundle.joblib"
        save_model_bundle(model, valid_metadata, bundle_path)

        loaded_model, loaded_meta = load_model_bundle(bundle_path)
        np.testing.assert_array_equal(model.predict_proba(X), loaded_model.predict_proba(X))
        assert loaded_meta["threshold"] == 0.4

    def test_overwrite_protection(self, tmp_path, trained_classifier, valid_metadata):
        model, _, _ = trained_classifier
        path = tmp_path / "bundle.joblib"
        save_model_bundle(model, valid_metadata, path)
        with pytest.raises(FileExistsError):
            save_model_bundle(model, valid_metadata, path)

    def test_load_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_model_bundle(tmp_path / "absent.joblib")


# =============================================================================
# Tests : predict_with_threshold
# =============================================================================


class TestPredictWithThreshold:
    def test_threshold_05_matches_default_predict(self, trained_classifier):
        model, X, _ = trained_classifier
        preds_custom = predict_with_threshold(model, X, threshold=0.5)
        preds_default = model.predict(X)
        np.testing.assert_array_equal(preds_custom, preds_default)

    def test_lower_threshold_predicts_more_positives(self, trained_classifier):
        model, X, _ = trained_classifier
        preds_low = predict_with_threshold(model, X, threshold=0.2)
        preds_high = predict_with_threshold(model, X, threshold=0.8)
        # Avec un seuil bas, on prédit plus de positifs
        assert preds_low.sum() >= preds_high.sum()

    def test_return_proba_returns_tuple(self, trained_classifier):
        model, X, _ = trained_classifier
        result = predict_with_threshold(model, X, threshold=0.5, return_proba=True)
        assert isinstance(result, tuple)
        assert len(result) == 2
        preds, probas = result
        assert preds.shape == (len(X),)
        assert probas.shape == (len(X),)
        assert 0 <= probas.min() and probas.max() <= 1

    def test_invalid_threshold_raises(self, trained_classifier):
        model, X, _ = trained_classifier
        with pytest.raises(ValueError, match="entre 0 et 1"):
            predict_with_threshold(model, X, threshold=1.5)

    def test_feature_order_reorders_columns(self, trained_classifier):
        model, X, _ = trained_classifier
        # Mélanger les colonnes de X
        X_shuffled = X[X.columns[::-1]]  # ordre inversé
        # Sans feature_order, les prédictions seraient FAUSSES
        # Avec feature_order, elles doivent être identiques
        feature_order = ["feat_0", "feat_1", "feat_2", "feat_3", "feat_4"]
        preds_correct = predict_with_threshold(
            model, X_shuffled, threshold=0.5, feature_order=feature_order
        )
        preds_ref = predict_with_threshold(model, X, threshold=0.5)
        np.testing.assert_array_equal(preds_correct, preds_ref)

    def test_missing_column_raises(self, trained_classifier):
        model, X, _ = trained_classifier
        X_incomplete = X.drop(columns=["feat_0"])
        with pytest.raises(ValueError, match="Colonnes manquantes"):
            predict_with_threshold(
                model,
                X_incomplete,
                threshold=0.5,
                feature_order=X.columns.tolist(),
            )

    def test_non_probabilistic_model_raises(self):
        from sklearn.svm import LinearSVC

        # LinearSVC n'a pas de predict_proba
        model = LinearSVC()
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(50, 3))
        y = np.random.randint(0, 2, 50)
        model.fit(X, y)
        with pytest.raises(AttributeError, match="predict_proba"):
            predict_with_threshold(model, X, threshold=0.5)
