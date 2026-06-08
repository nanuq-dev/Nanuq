<div align="center">
  <img src="https://raw.githubusercontent.com/nanuq-dev/nanuq/main/assets/logo.png" alt="nanuq" width="420">
</div>

**A reusable Machine Learning toolkit for the full data science lifecycle.**
From raw data to a deployed model — loading, exploration, cleaning, encoding,
modeling, interpretation, and production persistence, all in one coherent library.

![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)
![License MIT](https://img.shields.io/badge/license-MIT-green.svg)
![Tests](https://img.shields.io/badge/tests-161%20passed-brightgreen.svg)
![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-black.svg)

---

## Why nanuq?

Most data science projects re-implement the same plumbing over and over:
loading files, checking for missing values, encoding categories, comparing a few
models, tuning a decision threshold, then saving the result for production. That
boilerplate is rarely the interesting part, and rewriting it each time invites
subtle bugs and inconsistencies.

`nanuq` packages those recurring steps into a single, well-tested library you can
install and reuse across projects. It is intentionally **not** a black-box AutoML
tool: every function does one clear thing, and when several approaches exist for a
task (imputation, scaling, handling class imbalance, encoding), nanuq offers both
the atomic functions **and** a comparator that runs them side by side so you can
make an informed choice.

## Key features

- **Full lifecycle coverage** — 16 modules spanning loading, EDA, cleaning,
  encoding, scaling, transformations, feature engineering, balancing, modeling,
  evaluation, interpretation, plotting, reporting, and persistence.
- **Comparator pattern** — for tasks with multiple valid strategies, compare them
  on your data in a single call instead of guessing.
- **Production-ready persistence** — save a model and its metadata as separate,
  versionable artifacts, with SHA256 integrity checks and custom-threshold
  inference.
- **No surprises** — no in-place mutation of your DataFrames, no hidden global
  state, explicit return values.
- **Documented and tested** — systematic docstrings, runnable examples, 161 tests.

## Installation

```bash
pip install nanuq
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv pip install nanuq
```

Requires Python 3.11 or newer.

## Quick start

```python
import nanuq

# 1. Load and explore
df = nanuq.load_csv("data.csv")
nanuq.dataset_overview(df)

# 2. Rank features by their association with the target
ranking = nanuq.rank_features_by_target_association(df, target="churn")

# 3. Train and evaluate a model
from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier().fit(X_train, y_train)
nanuq.evaluate_classifier(model, X_train, y_train, X_test, y_test)

# 4. Save for production (model + metadata, kept separate)
meta = nanuq.build_metadata(model, X_train.columns.tolist(), threshold=0.29)
nanuq.save_artifacts(model, meta, "artifacts/")

# 5. Reload and predict with a custom decision threshold
model, meta = nanuq.load_artifacts("artifacts/")
preds = nanuq.predict_with_threshold(
    model, X_new,
    threshold=meta["threshold"],
    feature_order=meta["feature_order"],
)
```

## Philosophy

- **Reusable.** Designed to be installed as a dependency and shared across
  projects, not copy-pasted between notebooks.
- **Comparative.** When several methods exist for the same task, nanuq exposes
  the individual functions *and* a comparator that puts them side by side, so the
  choice is data-driven rather than habitual.
- **Educational.** Readable code, systematic docstrings, and worked examples — the
  library is meant to be learned from, not just called.
- **No magic.** No side effects, no in-place DataFrame mutation, predictable
  outputs you can reason about.

## Modules

| Module | Purpose |
| --- | --- |
| `loading` | Read CSV / Excel / JSON / Parquet / SQL into DataFrames |
| `exploration` | Dataset overview, column typing, quick diagnostics |
| `cleaning` | Missing-value imputation (mean, median, mode) with reports |
| `outliers` | Outlier detection (Z-score, IQR) and a comparator |
| `encoding` | One-Hot, Label, Ordinal, binary encoders and a comparator |
| `scaling` | MinMax, Standard scaling and a comparator |
| `transformations` | Log, sqrt, Box-Cox, quantile, power transforms |
| `feature_engineering` | Polynomial features, interactions, discretization |
| `balancing` | class_weight, SMOTE, undersampling for imbalanced data |
| `bivariate` | Chi², Mann-Whitney, Pearson/Spearman correlations |
| `models` | Ready-to-use linear, tree, boosted, and clustering pipelines |
| `evaluation` | Metrics, stratified CV, precision-recall curve, optimal threshold |
| `interpretation` | SHAP, permutation importance, native feature importance |
| `persistence` | Save/load models and metadata for production |
| `plots` | Boxplots, confusion matrices, importance charts |
| `reporting` | Automated EDA report generation |

## The comparator pattern

Rather than committing to a single model or transformation up front, nanuq lets
you put options side by side and decide from the results.

```python
# Benchmark several models on the same train/test split
results = nanuq.benchmark_models(
    {
        "logistic": LogisticRegression(),
        "random_forest": RandomForestClassifier(),
        "gradient_boosting": GradientBoostingClassifier(),
    },
    X_train, y_train, X_test, y_test,
    task="classification",
)
#   -> a DataFrame ranking each model by its metrics

# Compare a feature's distribution before and after a transformation
nanuq.compare_distribution(df["income"], nanuq.log_transform(df["income"]))

# Compare a feature's mean across target classes (quick signal check)
nanuq.compare_means_by_class(df, feature="overtime_hours", target="churn")
```

For atomic preprocessing you also get clear, single-purpose functions:
`scale_minmax`, `scale_standard`, `one_hot_encode`, `label_encode_column`,
`detect_outliers_iqr`, and more — no hidden defaults.

## Production with `persistence`

The `persistence` module follows a deliberate convention: keep the binary model
and its human-readable metadata in **separate files**.

```python
from nanuq import build_metadata, save_artifacts, load_artifacts, predict_with_threshold

# Save: two separate files
meta = build_metadata(
    model, X.columns.tolist(), threshold=0.286,
    metrics={"auc": 0.83, "f1": 0.54},
)
save_artifacts(model, meta, "artifacts/")
#   artifacts/model.joblib   -> the sklearn estimator (binary)
#   artifacts/metadata.json  -> threshold, feature order, metrics (readable)

# Load + predict with the stored threshold (hash is verified automatically)
model, meta = load_artifacts("artifacts/")
preds = predict_with_threshold(
    model, X_new,
    threshold=meta["threshold"],
    feature_order=meta["feature_order"],
)
```

Why separate the two? Because it lets you:

- inspect metrics and configuration without loading scikit-learn (plain JSON),
- version the metadata in Git while the model stays binary,
- adjust the decision threshold without retraining or touching the model,
- detect a corrupted or swapped model file via its SHA256 hash.

## Threshold optimization

On imbalanced datasets, the default 0.5 decision threshold is rarely optimal.
nanuq's `evaluate_classifier` reports the full set of metrics (including the
precision-recall trade-off) so you can pick the threshold that best fits your
objective — often dramatically improving recall on the minority class.

```python
# Full evaluation on train AND test, with the metrics you need to choose a threshold
metrics = nanuq.evaluate_classifier(model, X_train, y_train, X_test, y_test)

# Once chosen, the threshold travels with the model in metadata.json,
# and inference applies it consistently:
preds = nanuq.predict_with_threshold(
    model, X_new,
    threshold=meta["threshold"],   # e.g. 0.286 instead of 0.5
    feature_order=meta["feature_order"],
)
```

Storing the threshold alongside the model keeps inference consistent with the way
the model was tuned.

## Development

```bash
# Install in editable mode with the dev tools
uv pip install -e ".[dev]"

# Run the test suite
pytest

# Lint
ruff check src/ tests/
```

## License

[MIT](LICENSE) — © 2026 Olivier Gruwe

---

*nanuq* means "polar bear" in Inuktitut. 🐻‍❄️
