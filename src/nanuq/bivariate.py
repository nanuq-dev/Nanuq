"""
bivariate — Analyses statistiques bivariées : features vs cible binaire.

Ce module aide à identifier rapidement les features les plus liées à
une cible binaire (classification). Il propose pour chaque type de
variable un test adapté, puis une fonction "orchestrateur" qui
ressort un classement unifié.

| Type de feature | Test approprié | Fonction |
|---|---|---|
| Numérique       | Mann-Whitney U (non paramétrique)  | ``mann_whitney_features`` |
| Catégorielle    | Chi² d'indépendance                | ``chi2_features``         |

Bonus :
- ``compare_means_by_class``  : stats moyenne/médiane par classe (lecture rapide)
- ``target_correlations``     : corrélations Pearson/Spearman avec la cible
- ``rank_features_by_target_association`` : classement unifié de **toutes** les features

Tous les tests retournent un DataFrame trié par p-value croissante, avec
une colonne ``signif`` indiquant le niveau de significativité (``***``,
``**``, ``*``, ``ns``) pour lecture rapide.

Important : ces fonctions supposent une **cible binaire**. La cible peut
être encodée en strings (``"Oui"``/``"Non"``) ou en numérique (0/1) — la
détection est automatique.
"""

from __future__ import annotations

import pandas as pd
from scipy import stats

# =============================================================================
# Helpers internes
# =============================================================================


def _significance_marker(p: float) -> str:
    """Convertit une p-value en marqueur lisible.

    - p < 0.001  →  ``***``  (très significatif)
    - p < 0.01   →  ``**``
    - p < 0.05   →  ``*``
    - sinon      →  ``ns``   (non significatif)
    """
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def _validate_binary_target(y: pd.Series) -> pd.Series:
    """Vérifie que la cible est binaire et renvoie ses 2 modalités.

    Args:
        y: Series de la cible.

    Returns:
        La même Series (pas de transformation).

    Raises:
        ValueError: Si la cible n'a pas exactement 2 modalités.
    """
    unique = y.dropna().unique()
    if len(unique) != 2:
        raise ValueError(
            f"La cible doit être binaire (2 modalités). "
            f"Trouvé {len(unique)} : {sorted(unique.tolist())}"
        )
    return y


# =============================================================================
# 1. Statistiques descriptives par classe
# =============================================================================


def compare_means_by_class(
    df: pd.DataFrame,
    features: list[str],
    target: str,
) -> pd.DataFrame:
    """Compare les statistiques descriptives de chaque feature par classe de cible.

    Pour chaque feature numérique, calcule la moyenne, la médiane et l'écart
    relatif entre les deux classes de la cible. Permet une lecture rapide
    de "qui a quelle moyenne dans chaque groupe".

    Args:
        df: DataFrame contenant features et cible.
        features: Liste des **colonnes numériques** à analyser.
        target: Nom de la colonne cible binaire.

    Returns:
        DataFrame indexé sur les noms de features, avec colonnes :

        - ``mean_class_0`` / ``mean_class_1``     : moyennes par classe
        - ``median_class_0`` / ``median_class_1`` : médianes par classe
        - ``delta_mean_pct`` : écart relatif des moyennes (% de class_0)

        Trié par ``|delta_mean_pct|`` décroissant.

    Raises:
        ValueError: Si la cible n'est pas binaire.

    Example:
        >>> stats_table = compare_means_by_class(df, numeric_features, "target")
        >>> stats_table.head()
                              mean_class_0  mean_class_1  delta_mean_pct
        nombre_participation_pee     0.85          0.53          -37.6
        annees_dans_le_poste         4.48          2.90          -35.3
        ...
    """
    y = _validate_binary_target(df[target])

    # On identifie les 2 classes en triant pour avoir un ordre stable
    classes = sorted(y.dropna().unique().tolist())
    class_0, class_1 = classes[0], classes[1]

    mask_0 = df[target] == class_0
    mask_1 = df[target] == class_1

    rows = []
    for col in features:
        mean_0 = float(df.loc[mask_0, col].mean())
        mean_1 = float(df.loc[mask_1, col].mean())
        median_0 = float(df.loc[mask_0, col].median())
        median_1 = float(df.loc[mask_1, col].median())

        # delta relatif en % de la classe 0 (référence)
        # On évite la division par zéro
        if mean_0 != 0:
            delta_pct = (mean_1 - mean_0) / abs(mean_0) * 100
        else:
            delta_pct = float("inf") if mean_1 != 0 else 0.0

        rows.append(
            {
                "feature": col,
                f"mean_class_{class_0}": mean_0,
                f"mean_class_{class_1}": mean_1,
                f"median_class_{class_0}": median_0,
                f"median_class_{class_1}": median_1,
                "delta_mean_pct": delta_pct,
            }
        )

    result = pd.DataFrame(rows).set_index("feature")
    return result.reindex(result["delta_mean_pct"].abs().sort_values(ascending=False).index)


# =============================================================================
# 2. Test Mann-Whitney U pour chaque feature numérique
# =============================================================================


def mann_whitney_features(
    df: pd.DataFrame,
    features: list[str],
    target: str,
    alternative: str = "two-sided",
) -> pd.DataFrame:
    """Applique le test Mann-Whitney U sur chaque feature numérique vs cible.

    Test **non paramétrique** (ne suppose pas la normalité) qui compare les
    distributions de deux groupes. À utiliser de préférence au test t de
    Student quand les données sont asymétriques (revenus, ancienneté…) — ce
    qui est presque toujours le cas en pratique.

    **Hypothèse nulle H₀** : les deux groupes ont la même distribution.
    **On rejette H₀** si p-value < 0.05 → la feature discrimine.

    Args:
        df: DataFrame contenant features et cible.
        features: Liste des **colonnes numériques** à tester.
        target: Nom de la colonne cible binaire.
        alternative: Hypothèse alternative.

            - ``"two-sided"`` (défaut) : les distributions diffèrent
            - ``"less"`` / ``"greater"`` : test directionnel

    Returns:
        DataFrame trié par p-value croissante, avec colonnes :

        - ``feature`` : nom de la feature
        - ``median_class_0`` / ``median_class_1`` : médianes par groupe
        - ``u_statistic`` : statistique U du test
        - ``p_value`` : probabilité que H₀ soit vraie
        - ``signif`` : ``***`` / ``**`` / ``*`` / ``ns``

    Raises:
        ValueError: Si la cible n'est pas binaire.

    Example:
        >>> mw = mann_whitney_features(df, numeric_features, "target")
        >>> mw.head(5)
                                    feature  median_class_0  median_class_1  p_value signif
        4               annee_experience_totale            10.0             7.0     0.0    ***
        1                        revenu_mensuel          5204.0          3202.0     0.0    ***
        ...
    """
    y = _validate_binary_target(df[target])
    classes = sorted(y.dropna().unique().tolist())
    class_0, class_1 = classes[0], classes[1]

    rows = []
    for col in features:
        group_0 = df.loc[df[target] == class_0, col].dropna()
        group_1 = df.loc[df[target] == class_1, col].dropna()

        # Si une variable est constante, le test n'est pas pertinent : on
        # met une p-value = 1.0 par convention pour qu'elle reste classée
        # tout en bas (non significative).
        try:
            u_stat, p_value = stats.mannwhitneyu(
                group_0,
                group_1,
                alternative=alternative,
            )
            u_stat, p_value = float(u_stat), float(p_value)
        except ValueError:
            u_stat, p_value = float("nan"), 1.0

        rows.append(
            {
                "feature": col,
                f"median_class_{class_0}": float(group_0.median()),
                f"median_class_{class_1}": float(group_1.median()),
                "u_statistic": u_stat,
                "p_value": p_value,
            }
        )

    result = pd.DataFrame(rows).sort_values("p_value").reset_index(drop=True)
    result["signif"] = result["p_value"].apply(_significance_marker)
    return result


# =============================================================================
# 3. Test du chi² pour chaque feature catégorielle
# =============================================================================


def chi2_features(
    df: pd.DataFrame,
    features: list[str],
    target: str,
) -> pd.DataFrame:
    """Applique le test du chi² d'indépendance sur chaque catégorielle vs cible.

    **Hypothèse nulle H₀** : la feature et la cible sont indépendantes.
    **On rejette H₀** si p-value < 0.05 → la feature est associée à la cible.

    Args:
        df: DataFrame contenant features et cible.
        features: Liste des **colonnes catégorielles** à tester (string,
            category, ou numériques discrètes encodant des modalités).
        target: Nom de la colonne cible.

    Returns:
        DataFrame trié par p-value croissante, avec colonnes :

        - ``feature`` : nom de la feature
        - ``chi2_statistic`` : statistique du test
        - ``p_value`` : probabilité que H₀ soit vraie
        - ``n_modalities`` : nombre de modalités de la feature
        - ``signif`` : marqueur de significativité

    Note:
        Le chi² requiert au moins 5 observations attendues par cellule
        de la table de contingence. Sur des modalités très rares, le
        résultat peut être moins fiable. Inspection visuelle du tableau
        de contingence recommandée pour les variables sensibles.

    Example:
        >>> chi2 = chi2_features(df, ["genre", "departement", "poste"], "target")
        >>> chi2.head()
                  feature  chi2_statistic  p_value  n_modalities signif
        4   heure_supplementaires        87.56      0.0             2    ***
        ...
    """
    _validate_binary_target(df[target])

    rows = []
    for col in features:
        contingency = pd.crosstab(df[col], df[target])

        try:
            chi2_stat, p_value, dof, _expected = stats.chi2_contingency(contingency)
        except ValueError:
            # Table de contingence invalide (modalité unique, etc.)
            chi2_stat, p_value = float("nan"), 1.0

        rows.append(
            {
                "feature": col,
                "chi2_statistic": float(chi2_stat),
                "p_value": float(p_value),
                "n_modalities": contingency.shape[0],
            }
        )

    result = pd.DataFrame(rows).sort_values("p_value").reset_index(drop=True)
    result["signif"] = result["p_value"].apply(_significance_marker)
    return result


# =============================================================================
# 4. Corrélations avec la cible
# =============================================================================


def target_correlations(
    df: pd.DataFrame,
    features: list[str],
    target: str,
    methods: list[str] | None = None,
) -> pd.DataFrame:
    """Corrélations de chaque feature numérique avec la cible.

    Calcule la corrélation entre chaque feature et la cible selon une ou
    plusieurs méthodes :

    - **Pearson** : relation **linéaire**. Sensible aux outliers.
    - **Spearman** : relation **monotone** (basée sur les rangs). Robuste.
      Si Spearman >> Pearson : signal **non linéaire**.

    Args:
        df: DataFrame contenant features et cible.
        features: Liste des **colonnes numériques**.
        target: Nom de la colonne cible (qui doit être numérique ou
            encodée — ex : 0/1 pour une binaire encodée).
        methods: Liste des méthodes. Par défaut ``["pearson", "spearman"]``.

    Returns:
        DataFrame indexé sur les features, avec une colonne par méthode
        (``pearson``, ``spearman``). Trié par ``|pearson|`` décroissant
        (ou par la première méthode demandée).

    Example:
        >>> corr = target_correlations(df, numeric_features, "target")
        >>> corr.head()
                                pearson  spearman
        heure_sup_bin              0.25      0.25
        annee_experience_totale   -0.17     -0.20
        ...
    """
    if methods is None:
        methods = ["pearson", "spearman"]

    # Le calcul de corrélation requiert la cible en numérique.
    # On vérifie ; si la cible est en string, on lève une erreur claire.
    if not pd.api.types.is_numeric_dtype(df[target]):
        raise ValueError(
            f"target_correlations requiert une cible numérique. "
            f"Type actuel de '{target}' : {df[target].dtype}.\n"
            f"Astuce : encode la cible en 0/1 avant : "
            f"df['{target}'] = (df['{target}'] == 'Oui').astype(int)"
        )

    columns = features + [target]
    result = pd.DataFrame(index=features)

    for method in methods:
        corr_with_target = df[columns].corr(method=method)[target].drop(target)
        result[method] = corr_with_target

    # Tri par |valeur absolue| de la première méthode
    sort_col = methods[0]
    result = result.reindex(result[sort_col].abs().sort_values(ascending=False).index)
    return result


# =============================================================================
# 5. Fonction maîtresse : classement unifié de toutes les features
# =============================================================================


def rank_features_by_target_association(
    df: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    target: str,
) -> pd.DataFrame:
    """Classement unifié de toutes les features par association avec la cible.

    Orchestre les tests adaptés selon le type :

    - **Numériques** → Mann-Whitney U
    - **Catégorielles** → Chi²

    Retourne un seul DataFrame trié par p-value croissante, **toutes
    features confondues**. C'est l'outil pour répondre rapidement à la
    question : *"quelles sont les variables les plus prédictives de ma cible ?"*

    Args:
        df: DataFrame contenant features et cible.
        numeric_features: Colonnes numériques (testées par Mann-Whitney).
        categorical_features: Colonnes catégorielles (testées par chi²).
        target: Nom de la colonne cible binaire.

    Returns:
        DataFrame trié par p-value croissante, avec colonnes :

        - ``feature`` : nom de la feature
        - ``test`` : ``"mann_whitney"`` ou ``"chi2"``
        - ``statistic`` : valeur de la statistique du test
        - ``p_value`` : probabilité que H₀ soit vraie
        - ``signif`` : marqueur de significativité

    Example:
        >>> ranking = rank_features_by_target_association(
        ...     df, numeric_features=num_cols, categorical_features=cat_cols,
        ...     target="a_quitte_l_entreprise",
        ... )
        >>> ranking.head(10)
                              feature          test  statistic  p_value signif
        0   heure_supplementaires            chi2      87.56      0.0    ***
        1                   poste            chi2      86.19      0.0    ***
        2  annee_experience_totale  mann_whitney   60540.50      0.0    ***
        ...
    """
    mw = mann_whitney_features(df, numeric_features, target)
    chi2 = chi2_features(df, categorical_features, target)

    # Harmonisation des colonnes pour pouvoir concaténer
    mw_subset = pd.DataFrame(
        {
            "feature": mw["feature"],
            "test": "mann_whitney",
            "statistic": mw["u_statistic"],
            "p_value": mw["p_value"],
        }
    )
    chi2_subset = pd.DataFrame(
        {
            "feature": chi2["feature"],
            "test": "chi2",
            "statistic": chi2["chi2_statistic"],
            "p_value": chi2["p_value"],
        }
    )

    result = (
        pd.concat(
            [mw_subset, chi2_subset],
            ignore_index=True,
        )
        .sort_values("p_value")
        .reset_index(drop=True)
    )
    result["signif"] = result["p_value"].apply(_significance_marker)
    return result
