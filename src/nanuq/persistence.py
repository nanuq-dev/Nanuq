"""
persistence — Sauvegarde et chargement de modèles ML pour mise en production.

Fonctions disponibles :

- ``save_artifacts``           : sauvegarde modèle (.joblib) + métadonnées (.json) séparés
- ``load_artifacts``           : charge la paire (modèle, métadonnées)
- ``save_model_bundle``        : sauvegarde tout-en-un (modèle + métadonnées dans un seul fichier)
- ``load_model_bundle``        : charge un bundle tout-en-un
- ``predict_with_threshold``   : applique un modèle avec un seuil de décision personnalisé
- ``build_metadata``           : helper pour construire un dict de métadonnées propres

Bonne pratique recommandée : utiliser ``save_artifacts``/``load_artifacts`` plutôt
que la version bundle. La séparation modèle/métadonnées permet :

- d'inspecter les métadonnées sans charger sklearn (lecture JSON simple)
- de versionner les métadonnées dans Git (le .joblib reste binaire)
- de diffuser les métriques sans diffuser le modèle
- de mettre à jour les seuils de décision sans toucher au modèle
"""

from __future__ import annotations

import hashlib
import json
import platform
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn

# Schéma minimal attendu dans les métadonnées
_REQUIRED_METADATA_KEYS = {"feature_order", "threshold", "model_type"}


# =============================================================================
# Helpers internes
# =============================================================================


def _file_hash(path: Path) -> str:
    """Calcule le SHA256 d'un fichier (pour vérification d'intégrité)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _validate_metadata(metadata: dict[str, Any]) -> None:
    """Vérifie qu'un dict de métadonnées contient les clés minimales requises."""
    missing = _REQUIRED_METADATA_KEYS - set(metadata.keys())
    if missing:
        raise ValueError(
            f"Métadonnées invalides : clés manquantes {sorted(missing)}. "
            f"Clés requises : {sorted(_REQUIRED_METADATA_KEYS)}."
        )

    threshold = metadata["threshold"]
    if not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
        raise ValueError(f"Le seuil doit être un nombre entre 0 et 1, reçu : {threshold!r}.")

    feature_order = metadata["feature_order"]
    if not isinstance(feature_order, list) or not all(isinstance(f, str) for f in feature_order):
        raise ValueError("feature_order doit être une liste de chaînes (noms de colonnes).")


# =============================================================================
# Construction des métadonnées
# =============================================================================


def build_metadata(
    model: Any,
    feature_order: list[str],
    threshold: float,
    *,
    metrics: dict[str, float] | None = None,
    hyperparameters: dict[str, Any] | None = None,
    training_info: dict[str, Any] | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    """Construit un dictionnaire de métadonnées prêt à être sauvegardé en JSON.

    Args:
        model: Le modèle sklearn entraîné (utilisé pour extraire le type).
        feature_order: Liste ordonnée des noms de colonnes d'entrée du modèle.
        threshold: Seuil de décision pour la classification binaire (entre 0 et 1).
        metrics: Optionnel - dict de métriques d'évaluation (auc, f1, etc.).
        hyperparameters: Optionnel - dict des hyperparamètres du modèle.
        training_info: Optionnel - dict d'infos sur l'entraînement (n_samples, date, etc.).
        notes: Optionnel - commentaire libre.

    Returns:
        dict prêt à être sérialisé en JSON.

    Examples:
        >>> from sklearn.ensemble import GradientBoostingClassifier
        >>> model = GradientBoostingClassifier().fit(X_train, y_train)
        >>> meta = build_metadata(
        ...     model,
        ...     feature_order=X_train.columns.tolist(),
        ...     threshold=0.286,
        ...     metrics={"auc": 0.83, "f1": 0.54},
        ... )
    """
    if not 0 <= threshold <= 1:
        raise ValueError(f"Le seuil doit être entre 0 et 1, reçu : {threshold}.")

    metadata = {
        # ---- Schéma minimal ----
        "model_type": type(model).__name__,
        "feature_order": list(feature_order),  # garantit que c'est une liste
        "threshold": float(threshold),
        # ---- Versionnage ----
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        # ---- Optionnel ----
        "metrics": metrics or {},
        "hyperparameters": hyperparameters or {},
        "training_info": training_info or {},
        "notes": notes,
    }
    _validate_metadata(metadata)
    return metadata


# =============================================================================
# Sauvegarde / Chargement : approche RECOMMANDÉE (artefacts séparés)
# =============================================================================


def save_artifacts(
    model: Any,
    metadata: dict[str, Any],
    output_dir: str | Path,
    *,
    model_filename: str = "model.joblib",
    metadata_filename: str = "metadata.json",
    overwrite: bool = False,
) -> dict[str, Path]:
    """Sauvegarde un modèle + ses métadonnées dans deux fichiers séparés.

    C'est la **bonne pratique recommandée** pour la mise en production :

    - ``model.joblib`` reste léger et binaire (le modèle sklearn brut)
    - ``metadata.json`` est lisible humainement (feature_order, threshold, métriques)

    On peut versionner les métadonnées dans Git, ajuster le seuil de décision
    sans réentraîner, ou inspecter les métriques sans charger sklearn.

    Args:
        model: Le modèle sklearn entraîné à sauvegarder.
        metadata: Dictionnaire de métadonnées (utiliser ``build_metadata`` pour le créer).
        output_dir: Dossier de destination. Sera créé s'il n'existe pas.
        model_filename: Nom du fichier modèle (par défaut : ``model.joblib``).
        metadata_filename: Nom du fichier métadonnées (par défaut : ``metadata.json``).
        overwrite: Si False (défaut), lève une erreur si les fichiers existent déjà.

    Returns:
        dict avec deux clés ``model`` et ``metadata`` pointant vers les Path créés.

    Raises:
        ValueError: Si les métadonnées sont invalides.
        FileExistsError: Si overwrite=False et qu'un fichier existe déjà.

    Examples:
        >>> meta = build_metadata(model, feature_order, threshold=0.29)
        >>> paths = save_artifacts(model, meta, "artifacts/")
        >>> print(paths["model"])
        PosixPath('artifacts/model.joblib')
    """
    _validate_metadata(metadata)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / model_filename
    metadata_path = output_dir / metadata_filename

    if not overwrite:
        for p in (model_path, metadata_path):
            if p.exists():
                raise FileExistsError(
                    f"Le fichier {p} existe déjà. Passez overwrite=True pour écraser."
                )

    # 1. Sauvegarder le modèle
    joblib.dump(model, model_path, compress=3)

    # 2. Enrichir les métadonnées avec un hash du modèle (intégrité)
    enriched_metadata = {**metadata, "model_sha256": _file_hash(model_path)}

    # 3. Sauvegarder les métadonnées en JSON lisible (indenté + UTF-8)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(enriched_metadata, f, indent=2, ensure_ascii=False, default=str)

    return {"model": model_path, "metadata": metadata_path}


def load_artifacts(
    input_dir: str | Path,
    *,
    model_filename: str = "model.joblib",
    metadata_filename: str = "metadata.json",
    verify_hash: bool = True,
) -> tuple[Any, dict[str, Any]]:
    """Charge un modèle + ses métadonnées depuis un dossier.

    Args:
        input_dir: Dossier contenant les artefacts.
        model_filename: Nom du fichier modèle attendu.
        metadata_filename: Nom du fichier métadonnées attendu.
        verify_hash: Si True (défaut), vérifie que le SHA256 du modèle correspond
            à celui sauvegardé dans les métadonnées (détecte les corruptions ou
            substitutions de fichier).

    Returns:
        Tuple ``(model, metadata)`` : le modèle sklearn et le dict de métadonnées.

    Raises:
        FileNotFoundError: Si l'un des fichiers est manquant.
        ValueError: Si verify_hash=True et que le hash ne correspond pas.

    Examples:
        >>> model, meta = load_artifacts("artifacts/")
        >>> print(f"Seuil : {meta['threshold']}, AUC : {meta['metrics']['auc']}")
    """
    input_dir = Path(input_dir)
    model_path = input_dir / model_filename
    metadata_path = input_dir / metadata_filename

    if not model_path.exists():
        raise FileNotFoundError(f"Modèle introuvable : {model_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Métadonnées introuvables : {metadata_path}")

    # Charger les métadonnées d'abord (pour pouvoir vérifier le hash)
    with open(metadata_path, encoding="utf-8") as f:
        metadata = json.load(f)

    if verify_hash and "model_sha256" in metadata:
        expected = metadata["model_sha256"]
        actual = _file_hash(model_path)
        if expected != actual:
            raise ValueError(
                f"Hash du modèle ne correspond pas aux métadonnées. "
                f"Attendu : {expected[:16]}..., obtenu : {actual[:16]}.... "
                f"Le fichier modèle a peut-être été modifié ou corrompu."
            )

    # Charger le modèle
    model = joblib.load(model_path)

    return model, metadata


# =============================================================================
# Sauvegarde / Chargement : approche BUNDLE (tout-en-un)
# =============================================================================


def save_model_bundle(
    model: Any,
    metadata: dict[str, Any],
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Sauvegarde le modèle et ses métadonnées dans UN SEUL fichier joblib.

    Pratique pour le partage rapide, mais moins flexible que ``save_artifacts``
    (qui sépare modèle binaire et métadonnées lisibles).

    Args:
        model: Le modèle à sauvegarder.
        metadata: Dictionnaire de métadonnées.
        path: Chemin de destination (typiquement ``.joblib`` ou ``.pkl``).
        overwrite: Si False, lève une erreur si le fichier existe déjà.

    Returns:
        Le ``Path`` du fichier créé.
    """
    _validate_metadata(metadata)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists() and not overwrite:
        raise FileExistsError(f"Le fichier {path} existe déjà.")

    bundle = {"model": model, "metadata": metadata}
    joblib.dump(bundle, path, compress=3)
    return path


def load_model_bundle(path: str | Path) -> tuple[Any, dict[str, Any]]:
    """Charge un bundle tout-en-un sauvegardé par ``save_model_bundle``.

    Args:
        path: Chemin du fichier bundle.

    Returns:
        Tuple ``(model, metadata)``.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Bundle introuvable : {path}")

    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or "model" not in bundle or "metadata" not in bundle:
        raise ValueError(
            f"Le fichier {path} ne contient pas un bundle valide "
            "(dict avec clés 'model' et 'metadata')."
        )
    return bundle["model"], bundle["metadata"]


# =============================================================================
# Prédiction avec seuil personnalisé
# =============================================================================


def predict_with_threshold(
    model: Any,
    X: pd.DataFrame | np.ndarray,
    threshold: float,
    *,
    feature_order: list[str] | None = None,
    return_proba: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Applique un modèle de classification avec un seuil de décision custom.

    Reproduit le comportement de ``model.predict()`` mais avec un seuil ajustable,
    ce qui est essentiel pour les datasets déséquilibrés (où le seuil par défaut
    de 0.5 est souvent sous-optimal).

    Args:
        model: Modèle sklearn avec une méthode ``predict_proba``.
        X: Données d'entrée (DataFrame ou array). Si DataFrame ET ``feature_order``
            est fourni, les colonnes sont réordonnées dans l'ordre attendu.
        threshold: Seuil de décision entre 0 et 1.
        feature_order: Optionnel - liste ordonnée des colonnes attendues par le modèle.
            Si fourni et X est un DataFrame, X est réindexé. Garantit la cohérence
            entre entraînement et inférence.
        return_proba: Si True, renvoie aussi les probabilités prédites.

    Returns:
        - Si ``return_proba=False`` : array de prédictions binaires (0/1)
        - Si ``return_proba=True``  : tuple (predictions, probabilities)

    Raises:
        AttributeError: Si le modèle n'a pas de méthode ``predict_proba``.
        ValueError: Si le seuil est hors [0, 1] ou si les colonnes ne correspondent pas.

    Examples:
        >>> model, meta = load_artifacts("artifacts/")
        >>> preds = predict_with_threshold(
        ...     model, X_new,
        ...     threshold=meta["threshold"],
        ...     feature_order=meta["feature_order"],
        ... )
    """
    if not 0 <= threshold <= 1:
        raise ValueError(f"Le seuil doit être entre 0 et 1, reçu : {threshold}.")

    if not hasattr(model, "predict_proba"):
        raise AttributeError(
            f"Le modèle {type(model).__name__} n'a pas de méthode predict_proba. "
            "predict_with_threshold ne fonctionne que pour les classifieurs probabilistes."
        )

    # Réordonner les colonnes si on a un DataFrame + un feature_order
    if feature_order is not None and isinstance(X, pd.DataFrame):
        missing = set(feature_order) - set(X.columns)
        if missing:
            raise ValueError(
                f"Colonnes manquantes dans X : {sorted(missing)}. Attendues : {feature_order}."
            )
        X = X[feature_order]

    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    if return_proba:
        return y_pred, y_proba
    return y_pred
