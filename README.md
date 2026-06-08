<p align="center">
  <img src="https://raw.githubusercontent.com/nanuq-dev/nanuq/main/assets/logo.png" alt="nanuq" width="420">
</p>

<p align="center">
  <strong>Boîte à outils Machine Learning réutilisable</strong><br>
  Accélère l'exploration, la préparation, la modélisation et la mise en production
  des projets de Data Science.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11%2B-blue.svg" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License MIT">
  <img src="https://img.shields.io/badge/tests-161%20passed-brightgreen.svg" alt="Tests">
  <img src="https://img.shields.io/badge/code%20style-ruff-black.svg" alt="Ruff">
</p>

---

## Installation

```bash
pip install nanuq
```

Ou avec [uv](https://docs.astral.sh/uv/) :

```bash
uv pip install nanuq
```

## Démarrage rapide

```python
import nanuq

# 1. Charger et explorer
df = nanuq.load_csv("data.csv")
nanuq.dataset_overview(df)

# 2. Analyser les liens avec la cible
ranking = nanuq.rank_features_by_target_association(df, target="churn")

# 3. Entraîner et évaluer un modèle
from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier().fit(X_train, y_train)
nanuq.evaluate_classifier(model, X_train, y_train, X_test, y_test)

# 4. Sauvegarder pour la production
meta = nanuq.build_metadata(model, X_train.columns.tolist(), threshold=0.29)
nanuq.save_artifacts(model, meta, "artifacts/")

# 5. Recharger et prédire avec un seuil custom
model, meta = nanuq.load_artifacts("artifacts/")
preds = nanuq.predict_with_threshold(
    model, X_new, threshold=meta["threshold"], feature_order=meta["feature_order"]
)
```

## Philosophie

- **Réutilisable** : pensé pour être installé comme une lib et servir sur plusieurs projets.
- **Comparatif** : quand plusieurs méthodes existent pour une même action (imputation,
  scaling, gestion du déséquilibre, etc.), le toolkit propose les fonctions atomiques
  *et* un comparateur qui les met en regard.
- **Pédagogique** : code documenté, docstrings systématiques, exemples inclus.
- **Pas de magie** : pas d'effets de bord, pas de modification en place des DataFrames.

## Modules

| Module | Contenu |
|---|---|
| `loading` | Lecture CSV / Excel / JSON / Parquet / SQL |
| `exploration` | Vue d'ensemble du dataset, types de colonnes |
| `cleaning` | Imputation des NaN (mean, median, mode) + rapports |
| `outliers` | Détection Z-score / IQR + comparateur |
| `encoding` | One-Hot, Label, Ordinal, binaire + comparateur |
| `scaling` | MinMax, Standard + comparateur |
| `transformations` | Log, sqrt, Box-Cox, quantile, power |
| `feature_engineering` | Polynomial features, interactions, discrétisation |
| `balancing` | class_weight, SMOTE, undersampling |
| `bivariate` | Chi², Mann-Whitney, corrélations Pearson/Spearman |
| `models` | Modèles linéaires, arbres, boostés |
| `evaluation` | Métriques, CV stratifiée, courbe précision-rappel, seuil optimal |
| `interpretation` | SHAP, Permutation Importance, feature importance native |
| `persistence` | Sauvegarde/chargement de modèles pour la production |
| `plots` | Visualisations (boxplots, matrices de confusion, importances) |
| `reporting` | Génération de rapports EDA |

## Mise en production avec `persistence`

Le module `persistence` industrialise la sauvegarde des modèles selon les bonnes
pratiques : séparation du modèle binaire (`model.joblib`) et des métadonnées
lisibles (`metadata.json`), avec vérification d'intégrité par hash SHA256.

```python
from nanuq import build_metadata, save_artifacts, load_artifacts, predict_with_threshold

# Sauvegarde : deux fichiers séparés
meta = build_metadata(model, X.columns.tolist(), threshold=0.286,
                      metrics={"auc": 0.83, "f1": 0.54})
save_artifacts(model, meta, "artifacts/")

# Chargement + inférence avec seuil personnalisé
model, meta = load_artifacts("artifacts/")  # vérifie le hash automatiquement
preds = predict_with_threshold(model, X_new, threshold=meta["threshold"],
                               feature_order=meta["feature_order"])
```

Avantage de la séparation : le seuil de décision peut être ajusté dans
`metadata.json` sans réentraîner ni toucher au modèle.

## Développement

```bash
# Installer en mode editable avec les outils de dev
uv pip install -e ".[dev]"

# Lancer les tests
pytest

# Vérifier le style
ruff check src/ tests/
```

## Licence

[MIT](LICENSE) — (c) 2026 Olivier Gruwe

---

<p align="center">
  <em>nanuq</em> signifie « ours polaire » en inuktitut.
</p>
