# Changelog

Toutes les évolutions notables de ce projet sont documentées dans ce fichier.

Le format s'inspire de [Keep a Changelog](https://keepachangelog.com/fr/1.1.0/),
et le projet suit le [versionnage sémantique](https://semver.org/lang/fr/).

## [0.2.0] - 2026-06-08

### Ajouté
- Module `persistence` pour la mise en production des modèles :
  - `build_metadata()` : construit des métadonnées normalisées (seuil, ordre des
    features, métriques, versions) avec validation de schéma.
  - `save_artifacts()` / `load_artifacts()` : sauvegarde recommandée en deux
    fichiers séparés (`model.joblib` binaire + `metadata.json` lisible), avec
    vérification d'intégrité par hash SHA256.
  - `save_model_bundle()` / `load_model_bundle()` : alternative tout-en-un.
  - `predict_with_threshold()` : prédiction avec seuil de décision personnalisé
    et réordonnancement automatique des colonnes.
- 24 tests unitaires pour le module `persistence` (suite totale : 161 tests).

### Modifié
- Mise à jour du README avec une section d'exemple pour `persistence`.

## [0.1.0] - 2026-05-31

### Ajouté
- Première version du toolkit avec les modules :
  - `loading` : chargement de données (CSV, Excel, JSON, Parquet, SQL).
  - `exploration` : aperçu de dataset, typage des colonnes, split features/cible.
  - `cleaning` : gestion des valeurs manquantes.
  - `encoding` : encodage binaire, ordinal, one-hot.
  - `scaling` : standardisation et normalisation.
  - `outliers` : détection par z-score et IQR.
  - `transformations` : log, sqrt, Box-Cox, quantile, power.
  - `feature_engineering` : features polynomiales, interactions, discrétisation.
  - `evaluation` : diagnostic biais/overfit, évaluation classif/régression,
    cross-validation, grid search, benchmark de modèles.
  - `models` : régressions linéaires, logistique, arbres, boosting, clustering.
  - `plots` : visualisations (boxplots, histogrammes, matrices de confusion,
    importances de features, courbes train/test).
  - `reporting` : génération de rapports EDA.
  - `bivariate` : analyses bivariées (chi², Mann-Whitney, corrélations).
  - `interpretation` : SHAP, permutation importance, comparaison de méthodes.
- Suite de 137 tests unitaires.
- Configuration qualité : ruff, mypy, pytest avec gestion stricte des warnings.
