"""
nanuq — Boîte à outils Machine Learning réutilisable.

Ce package expose les fonctions principales au niveau racine, pour que
les utilisateurs puissent écrire ::

    from nanuq import load_csv, dataset_overview, evaluate_classifier

…sans avoir à connaître la structure interne des sous-modules.

Pour les utilisations avancées, on peut bien sûr importer directement
depuis les sous-modules ::

    from nanuq.interpretation import compare_feature_importance_methods
"""

from __future__ import annotations

# Version du package (lisible depuis Python, ex : nanuq.__version__)
__version__ = "0.2.1"


# =============================================================================
# Réexports publics
# =============================================================================
# Les fonctions principales sont ré-exportées au niveau racine du package
# pour offrir une API d'utilisation simple. Les modules vides ne sont pas
# importés tant qu'ils n'ont pas de fonctions publiques.
# =============================================================================

# --- loading ---
# --- cleaning ---
# --- bivariate ---
from nanuq.bivariate import (
    chi2_features,
    compare_means_by_class,
    mann_whitney_features,
    rank_features_by_target_association,
    target_correlations,
)
from nanuq.cleaning import (
    clean_nan,
    fillna_category,
    fillna_mean,
    fillna_median,
    missing_values_report,
)

# --- encoding ---
from nanuq.encoding import (
    encode_binary,
    encode_binary_representation,
    encode_multiclass,
    label_encode_column,
    one_hot_encode,
    one_hot_encode_sklearn,
)

# --- evaluation ---
from nanuq.evaluation import (
    benchmark_models,
    cross_validate_model,
    diagnose_bias_overfit,
    evaluate_classifier,
    evaluate_regressor,
    grid_search_cv,
)

# --- exploration ---
from nanuq.exploration import (
    dataset_overview,
    get_column_types,
    split_features_target,
)

# --- feature_engineering ---
from nanuq.feature_engineering import (
    binarize_feature,
    create_interaction_feature,
    create_polynomial_features,
    discretize_equal_freq,
    discretize_equal_width,
)
from nanuq.loading import (
    load_csv,
    load_dataset_pipeline,
    load_excel,
    load_features,
    load_json,
    load_parquet,
    load_sql,
)

# --- models.boosted ---
from nanuq.models.boosted import gradient_boosting_pipeline

# --- models.clustering ---
from nanuq.models.clustering import (
    compute_silhouette_scores,
    train_kmeans,
)

# --- models.linear ---
from nanuq.models.linear import (
    lasso_regression_pipeline,
    linear_regression_multivariate,
    linear_regression_univariate,
    logistic_regression_pipeline,
    ridge_regression_pipeline,
)

# --- models.trees ---
from nanuq.models.trees import random_forest_pipeline

# --- outliers ---
from nanuq.outliers import (
    compute_zscore,
    detect_outliers_iqr,
    detect_outliers_zscore,
    iqr_bounds_report,
    outliers_iqr_summary,
    remove_outliers_iqr,
    zscore_outliers_report,
)

# --- persistence ---
from nanuq.persistence import (
    build_metadata,
    load_artifacts,
    load_model_bundle,
    predict_with_threshold,
    save_artifacts,
    save_model_bundle,
)

# --- plots ---
from nanuq.plots import (
    compare_distribution,
    plot_boosting_staged_loss,
    plot_boxplot,
    plot_confusion_matrix_pretty,
    plot_correlation_matrix,
    plot_feature_importances,
    plot_histogram,
    plot_train_test_curve,
)

# --- reporting ---
from nanuq.reporting import generate_eda_report

# --- scaling ---
from nanuq.scaling import (
    scale_minmax,
    scale_standard,
)

# --- transformations ---
from nanuq.transformations import (
    boxcox_transform,
    log_transform,
    power_transform_features,
    quantile_transform_features,
    sqrt_transform,
)

# =============================================================================
# Liste explicite des objets publics
# =============================================================================

__all__: list[str] = [
    # version
    "__version__",
    # loading
    "load_csv",
    "load_excel",
    "load_json",
    "load_parquet",
    "load_sql",
    "load_dataset_pipeline",
    "load_features",
    # exploration
    "dataset_overview",
    "get_column_types",
    "split_features_target",
    # cleaning
    "clean_nan",
    "missing_values_report",
    "fillna_mean",
    "fillna_median",
    "fillna_category",
    # encoding
    "encode_binary",
    "encode_multiclass",
    "label_encode_column",
    "encode_binary_representation",
    "one_hot_encode",
    "one_hot_encode_sklearn",
    # scaling
    "scale_minmax",
    "scale_standard",
    # outliers
    "compute_zscore",
    "detect_outliers_zscore",
    "detect_outliers_iqr",
    "remove_outliers_iqr",
    "iqr_bounds_report",
    "outliers_iqr_summary",
    "zscore_outliers_report",
    # transformations
    "log_transform",
    "sqrt_transform",
    "boxcox_transform",
    "quantile_transform_features",
    "power_transform_features",
    # feature_engineering
    "create_polynomial_features",
    "create_interaction_feature",
    "binarize_feature",
    "discretize_equal_width",
    "discretize_equal_freq",
    # evaluation
    "diagnose_bias_overfit",
    "evaluate_classifier",
    "evaluate_regressor",
    "cross_validate_model",
    "grid_search_cv",
    "benchmark_models",
    # models.linear
    "linear_regression_univariate",
    "linear_regression_multivariate",
    "logistic_regression_pipeline",
    "ridge_regression_pipeline",
    "lasso_regression_pipeline",
    # models.trees
    "random_forest_pipeline",
    # models.boosted
    "gradient_boosting_pipeline",
    # models.clustering
    "train_kmeans",
    "compute_silhouette_scores",
    # plots
    "plot_boxplot",
    "plot_histogram",
    "plot_correlation_matrix",
    "compare_distribution",
    "plot_train_test_curve",
    "plot_confusion_matrix_pretty",
    "plot_feature_importances",
    "plot_boosting_staged_loss",
    # reporting
    "generate_eda_report",
    # bivariate
    "compare_means_by_class",
    "mann_whitney_features",
    "chi2_features",
    "target_correlations",
    "rank_features_by_target_association",
    # persistence
    "build_metadata",
    "save_artifacts",
    "load_artifacts",
    "save_model_bundle",
    "load_model_bundle",
    "predict_with_threshold",
]
