"""
reporting — Génération automatique de rapport d'analyse exploratoire.

Ce module produit un **rapport Markdown** complet pour un DataFrame :
vue d'ensemble, analyse colonne par colonne, détection de valeurs
suspectes, et plan d'action recommandé (drop / imputer / encoder /
transformer / scaler).

Fonction principale : ``generate_eda_report``.

Cette fonction remplace l'ancienne ``about()`` (renommée pour plus de
clarté : ``about`` était trop ambigu — il suggérait "à propos du
package" plutôt qu'un rapport sur les données).
"""

from __future__ import annotations

import datetime as _dt
from typing import Any

import pandas as pd
from scipy import stats

from nanuq.exploration import get_column_types

# =============================================================================
# Heuristiques internes pour la détection de valeurs suspectes
# =============================================================================

# Mots-clés (FR + EN) suggérant qu'une colonne devrait contenir des
# valeurs >= 0 (surface, prix, durée, etc.)
_POSITIVE_KEYWORDS = [
    "surface",
    "area",
    "aire",
    "count",
    "nombre",
    "nb_",
    "nb ",
    "price",
    "prix",
    "cost",
    "cout",
    "coût",
    "duree",
    "durée",
    "duration",
    "weight",
    "poids",
    "mass",
    "masse",
    "amount",
    "quantite",
    "quantité",
    "quantity",
    "energy",
    "energie",
    "énergie",
    "conso",
    "consommation",
    "emission",
    "émission",
    "volume",
    "distance",
    "occupant",
]


def _is_positive_expected(column_name: str) -> bool:
    """Heuristique : la colonne devrait-elle contenir des valeurs >= 0 ?

    Basée sur des mots-clés courants dans les noms de colonnes (FR + EN).
    """
    name = column_name.lower()
    return any(kw in name for kw in _POSITIVE_KEYWORDS)


def _is_year_column(column_name: str) -> bool:
    """Heuristique : la colonne contient-elle une année ?"""
    name = column_name.lower()
    return "année" in name or "annee" in name or "year" in name


# =============================================================================
# Détection de valeurs incohérentes
# =============================================================================


def _check_coherence(
    df: pd.DataFrame,
    column: str,
) -> list[str]:
    """Détecte les valeurs potentiellement incohérentes par heuristiques.

    Vérifie :

    1. Valeurs négatives sur variables censées être positives
    2. Valeurs nulles suspectes (peu nombreuses sur une variable positive)
    3. Années aberrantes (< 1800 ou > année courante + 5)
    4. Max très éloigné de la borne IQR extrême (probable erreur de saisie)

    Args:
        df: DataFrame d'entrée.
        column: Nom de la colonne numérique à vérifier.

    Returns:
        Liste de messages décrivant chaque anomalie trouvée.
        Liste vide si aucune anomalie.
    """
    issues: list[str] = []
    series = df[column].dropna()

    # Cas dégénérés
    if len(series) == 0:
        return ["Colonne entièrement vide"]
    # Bool est "numeric" pour pandas mais on n'a rien à vérifier dessus
    if not pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        return issues

    min_val = float(series.min())
    max_val = float(series.max())

    # 1. Négatifs sur variables censées être positives
    if _is_positive_expected(column) and min_val < 0:
        n_neg = int((series < 0).sum())
        issues.append(
            f"{n_neg} valeurs négatives (min = {min_val}) sur une variable censée être positive"
        )

    # 2. Zéros suspects (peu nombreux) sur variables positives
    if _is_positive_expected(column):
        n_zero = int((series == 0).sum())
        if 0 < n_zero / len(series) < 0.05:
            issues.append(
                f"{n_zero} valeurs à zéro (potentiellement aberrantes sur ce type de variable)"
            )

    # 3. Années aberrantes
    if _is_year_column(column):
        current_year = _dt.datetime.now().year
        if min_val < 1800:
            issues.append(f"Année minimale = {int(min_val)} (antérieure à 1800, à vérifier)")
        if max_val > current_year + 5:
            issues.append(f"Année maximale = {int(max_val)} (dans le futur — possible erreur)")

    # 4. Max très éloigné de la borne IQR extrême
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    if IQR > 0:
        extreme_upper = Q3 + 3 * IQR
        if max_val > extreme_upper * 5:
            issues.append(
                f"Max = {max_val:.1f}, très au-delà de la borne IQR "
                f"extrême ({extreme_upper:.1f}) — probable erreur de saisie"
            )

    return issues


# =============================================================================
# Analyse colonne par colonne + recommandation
# =============================================================================


def _analyze_column(
    df: pd.DataFrame,
    column: str,
    nan_drop_threshold: float,
    nan_high_threshold: float,
    high_cardinality_threshold: int,
    skew_threshold: float,
) -> dict[str, Any]:
    """Analyse une colonne et formule une recommandation.

    Logique de décision :

    - NaN ``>=`` ``drop_threshold`` → drop
    - NaN ``>=`` ``high_threshold`` → imputer par modèle
    - NaN > 0 → imputer simple (médiane / mode)
    - Numérique très skewée → recommander transformation
    - Numérique grande étendue → recommander scaling
    - Catégorielle (2 modalités) → encoder binaire 0/1
    - Catégorielle (3..N) → one-hot
    - Catégorielle (> N) → encodage binaire ou frequency

    Returns:
        Dict avec : ``type``, ``n_unique``, ``n_missing``,
        ``pct_missing``, ``recommendation`` (str),
        ``action_tags`` (list[str]).
    """
    series = df[column]
    n_missing = int(series.isna().sum())
    pct_missing = n_missing / len(df) * 100 if len(df) else 0.0
    n_unique = int(series.nunique(dropna=True))

    # Bool est considéré "numeric" par pandas, mais sémantiquement c'est
    # une variable binaire : on le bascule dans le bucket catégoriel pour
    # qu'il reçoive la recommandation "encoder binaire (0/1)" et qu'on
    # évite les calculs arithmétiques (skew, range) qui plantent sur bool.
    is_bool = pd.api.types.is_bool_dtype(series)
    is_numeric = pd.api.types.is_numeric_dtype(series) and not is_bool
    col_type = "num" if is_numeric else ("bool" if is_bool else "cat")

    recommendations: list[str] = []
    action_tags: list[str] = []

    # ----- 1. Décision sur les NaN -----
    if pct_missing >= nan_drop_threshold:
        recommendations.append(f"🗑️ Drop ({pct_missing:.0f}% NaN)")
        action_tags.append("drop")
        return {
            "type": col_type,
            "n_unique": n_unique,
            "n_missing": n_missing,
            "pct_missing": pct_missing,
            "recommendation": " / ".join(recommendations),
            "action_tags": action_tags,
        }
    elif pct_missing >= nan_high_threshold:
        recommendations.append(f"🧮 Imputer par modèle ({pct_missing:.0f}% NaN)")
        action_tags.append("impute_model")
    elif pct_missing > 0:
        if is_numeric:
            recommendations.append(f"📊 Imputer médiane ({pct_missing:.1f}% NaN)")
        else:
            recommendations.append(f"📊 Imputer 'Unknown' ({pct_missing:.1f}% NaN)")
        action_tags.append("impute_simple")

    # ----- 2. Décision sur le type / transformation -----
    if is_numeric:
        clean = series.dropna()

        # Skewness
        if len(clean) > 1 and clean.std() > 0:
            try:
                skew = float(stats.skew(clean))
                if abs(skew) > skew_threshold:
                    recommendations.append(f"📈 Transformer (skew = {skew:.1f})")
                    action_tags.append("transform_skew")
            except Exception:
                pass

        # Variance / étendue
        if len(clean) > 0:
            range_val = float(clean.max() - clean.min())
            if range_val > 100:
                recommendations.append("⚖️ Scaler")
                action_tags.append("scale")

    else:
        # Catégorielle : décision d'encodage selon la cardinalité
        if n_unique <= 1:
            recommendations.append("🗑️ Constante (drop)")
            action_tags.append("drop")
        elif n_unique == 2:
            recommendations.append("🔀 Encoder binaire (0/1)")
            action_tags.append("encode_binary")
        elif n_unique <= high_cardinality_threshold:
            recommendations.append(f"🏷️ One-hot ({n_unique} modalités)")
            action_tags.append("encode_onehot")
        else:
            recommendations.append(
                f"📦 Cardinalité élevée ({n_unique}) — encodage binaire ou frequency"
            )
            action_tags.append("encode_binary_repr")

    # ----- 3. Valeurs suspectes (numérique uniquement) -----
    if is_numeric and _check_coherence(df, column):
        recommendations.append("⚠️ Vérifier valeurs suspectes")
        action_tags.append("suspect")

    if not recommendations:
        recommendations.append("✅ OK")

    return {
        "type": col_type,
        "n_unique": n_unique,
        "n_missing": n_missing,
        "pct_missing": pct_missing,
        "recommendation": " / ".join(recommendations),
        "action_tags": action_tags,
    }


# =============================================================================
# Fonction publique : génération du rapport Markdown complet
# =============================================================================


def generate_eda_report(
    df: pd.DataFrame,
    output_path: str | None = None,
    nan_drop_threshold: float = 80.0,
    nan_high_threshold: float = 50.0,
    high_cardinality_threshold: int = 15,
    skew_threshold: float = 1.0,
) -> str:
    """Génère un rapport d'analyse exploratoire complet au format Markdown.

    Le rapport contient :

    1. **Vue d'ensemble** : shape, mémoire, types, NaN globaux, doublons.
    2. **Analyse colonne par colonne** : tableau (type, uniques, NaN,
       recommandation).
    3. **Valeurs suspectes détectées** : négatifs sur surfaces, années
       aberrantes, outliers extrêmes.
    4. **Plan d'action recommandé** : regroupé par stratégie (drop /
       imputer / encoder / transformer / scaler).

    Args:
        df: DataFrame à analyser.
        output_path: Chemin de sauvegarde du rapport (``.md``).
            Si ``None`` (défaut), le rapport n'est pas écrit sur disque
            mais est tout de même retourné comme chaîne.
        nan_drop_threshold: ``%`` de NaN au-delà duquel on recommande
            de supprimer la colonne (``80.0`` par défaut).
        nan_high_threshold: ``%`` de NaN au-delà duquel on recommande
            une imputation par modèle plutôt que par moyenne / médiane
            (``50.0`` par défaut).
        high_cardinality_threshold: Nombre de modalités au-delà duquel
            on recommande un encodage binaire / frequency plutôt que du
            one-hot (``15`` par défaut).
        skew_threshold: Seuil de skewness au-delà duquel on recommande
            une transformation (log, Box-Cox).
            ``1.0`` par défaut (asymétrie forte).

    Returns:
        Le rapport au format Markdown (``str``).

    Note:
        Les recommandations sont **des heuristiques**. Toujours les
        confronter à la connaissance métier du dataset.

    Example:
        >>> md = generate_eda_report(df, output_path="rapport.md")
        >>> print(md[:200])  # aperçu en console
    """
    lines: list[str] = []
    col_types = get_column_types(df)
    n_rows, n_cols = df.shape

    # ============ EN-TÊTE ============
    lines.append("# Rapport d'analyse du dataset")
    lines.append("")
    lines.append(
        f"_Généré le {_dt.datetime.now().strftime('%Y-%m-%d %H:%M')} "
        f"par `nanuq.generate_eda_report()`_"
    )
    lines.append("")

    # ============ 1. VUE D'ENSEMBLE ============
    n_num = len(col_types["numeric"])
    n_cat = len(col_types["categorical"])
    memory_mb = df.memory_usage(deep=True).sum() / 1024**2
    total_cells = n_rows * n_cols
    total_nan = int(df.isna().sum().sum())
    pct_nan = total_nan / total_cells * 100 if total_cells else 0.0
    n_duplicates = int(df.duplicated().sum())

    lines.append("## 1. Vue d'ensemble")
    lines.append("")
    lines.append(f"- **Forme** : {n_rows:,} lignes × {n_cols} colonnes")
    lines.append(f"- **Mémoire** : {memory_mb:.2f} MB")
    lines.append(f"- **Colonnes numériques** : {n_num}")
    lines.append(f"- **Colonnes catégorielles** : {n_cat}")
    lines.append(f"- **Valeurs manquantes globales** : {total_nan:,} ({pct_nan:.1f} % du total)")
    lines.append(f"- **Lignes dupliquées** : {n_duplicates}")
    lines.append("")

    # ============ 2. ANALYSE PAR COLONNE ============
    lines.append("## 2. Analyse colonne par colonne")
    lines.append("")
    lines.append("| Colonne | Type | Uniques | NaN | NaN % | Recommandation |")
    lines.append("|---|---|---:|---:|---:|---|")

    # Buckets pour le plan d'action final
    actions: dict[str, list[str]] = {
        "drop": [],
        "impute_model": [],
        "impute_simple": [],
        "encode_binary": [],
        "encode_onehot": [],
        "encode_binary_repr": [],
        "transform_skew": [],
        "scale": [],
        "suspect": [],
    }

    for col in df.columns:
        analysis = _analyze_column(
            df,
            col,
            nan_drop_threshold=nan_drop_threshold,
            nan_high_threshold=nan_high_threshold,
            high_cardinality_threshold=high_cardinality_threshold,
            skew_threshold=skew_threshold,
        )

        lines.append(
            f"| `{col}` | {analysis['type']} | {analysis['n_unique']:,} "
            f"| {analysis['n_missing']:,} | {analysis['pct_missing']:.1f} % "
            f"| {analysis['recommendation']} |"
        )

        for tag in analysis["action_tags"]:
            actions.setdefault(tag, []).append(col)

    lines.append("")

    # ============ 3. VALEURS SUSPECTES ============
    lines.append("## 3. Valeurs suspectes détectées")
    lines.append("")

    suspects_found = False
    for col in col_types["numeric"]:
        issues = _check_coherence(df, col)
        if issues:
            suspects_found = True
            lines.append(f"### `{col}`")
            lines.append("")
            for issue in issues:
                lines.append(f"- {issue}")
            lines.append("")

    if not suspects_found:
        lines.append(
            "_Aucune anomalie évidente détectée par les heuristiques "
            "(négatifs, années, outliers extrêmes)._"
        )
        lines.append("")

    # ============ 4. PLAN D'ACTION ============
    lines.append("## 4. Plan d'action recommandé")
    lines.append("")

    plan_sections = [
        (
            "drop",
            "🗑️ Colonnes à supprimer (trop de NaN ou constantes)",
            "Trop peu d'information exploitable. À drop avant tout traitement.",
        ),
        (
            "impute_model",
            "🧮 Colonnes à imputer par modèle (régression)",
            "Forte proportion de NaN : utiliser `linear_regression_multivariate` "
            "sur des features corrélées plutôt qu'une simple moyenne.",
        ),
        (
            "impute_simple",
            "📊 Colonnes à imputer par moyenne / médiane / mode",
            "Peu de NaN : `fillna_median` (numérique) ou `fillna_category` (catégoriel) suffisent.",
        ),
        (
            "encode_binary",
            "🔀 Variables binaires (2 modalités) — encoder en 0/1",
            "Utiliser `encode_binary` avec un mapping explicite (ex: {'Yes': 1, 'No': 0}).",
        ),
        (
            "encode_onehot",
            "🏷️ Variables catégorielles modérées — one-hot encoding",
            "Utiliser `one_hot_encode` avec `drop_first=True` pour les modèles linéaires.",
        ),
        (
            "encode_binary_repr",
            "📦 Variables à forte cardinalité",
            "Le one-hot exploserait la dimensionnalité. Utiliser "
            "`encode_binary_representation` ou un frequency/target encoding.",
        ),
        (
            "transform_skew",
            "📈 Variables très asymétriques — transformation à appliquer",
            "Utiliser `log_transform`, `boxcox_transform` (positifs) ou "
            "`power_transform_features` (yeo-johnson, accepte les négatifs).",
        ),
        (
            "scale",
            "⚖️ Variables d'étendue importante — à scaler",
            "Indispensable pour régression logistique, KNN, KMeans, SVM. "
            "Utiliser `scale_standard` ou `scale_minmax`.",
        ),
        (
            "suspect",
            "⚠️ Variables avec valeurs suspectes — vérification manuelle",
            "Voir section 3 ci-dessus. Décider au cas par cas : correction, "
            "suppression de lignes, ou conservation si justifié.",
        ),
    ]

    any_action = False
    for tag, title, advice in plan_sections:
        cols = actions.get(tag, [])
        if cols:
            any_action = True
            lines.append(f"### {title}")
            lines.append("")
            lines.append(f"_{advice}_")
            lines.append("")
            for c in cols:
                lines.append(f"- `{c}`")
            lines.append("")

    if not any_action:
        lines.append("_Aucune action particulière recommandée._")
        lines.append("")

    # ============ FOOTER ============
    lines.append("---")
    lines.append("")
    lines.append(
        "_Rapport généré automatiquement. Les recommandations sont "
        "des heuristiques — toujours confronter à la connaissance "
        "métier du dataset._"
    )

    md = "\n".join(lines)

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(md)

    return md
