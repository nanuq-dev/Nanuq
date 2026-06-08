"""
_utils — Utilitaires internes partagés entre les modules de nanuq.

Convention : le préfixe underscore (`_utils`) signale que ce module
est **interne** au package. Il ne fait pas partie de l'API publique
et son contenu peut changer sans préavis.

Les fonctions ici sont utilisées par plusieurs modules (loading,
cleaning, encoding...) et factorisées pour éviter la duplication.
"""

from __future__ import annotations

import pandas as pd

# =============================================================================
# Validation de colonnes
# =============================================================================


def _check_columns_exist(
    df: pd.DataFrame,
    requested_columns: list[str],
) -> None:
    """Vérifie que toutes les colonnes demandées existent dans le DataFrame.

    Lève une ``ValueError`` explicite avec la liste des colonnes
    disponibles si au moins une colonne manque. Beaucoup plus utile à
    déboguer qu'un ``KeyError`` standard de pandas, qui n'indique que
    le nom manquant sans contexte.

    Args:
        df: DataFrame à vérifier.
        requested_columns: Liste des noms de colonnes attendus.

    Raises:
        ValueError: Si au moins une des colonnes demandées est absente.
            Le message d'erreur liste les colonnes manquantes ET les
            colonnes disponibles, pour aider l'utilisateur à corriger
            (typo, casse, espaces…).

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({"a": [1], "b": [2]})
        >>> _check_columns_exist(df, ["a", "c"])
        Traceback (most recent call last):
            ...
        ValueError: Colonnes introuvables dans le fichier : ['c'].
        Colonnes disponibles : ['a', 'b']
    """
    missing = [c for c in requested_columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"Colonnes introuvables dans le fichier : {missing}.\n"
            f"Colonnes disponibles : {list(df.columns)}"
        )
