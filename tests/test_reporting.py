"""Tests du module nanuq.reporting."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nanuq import generate_eda_report
from nanuq.reporting import (
    _analyze_column,
    _check_coherence,
    _is_positive_expected,
    _is_year_column,
)

# =============================================================================
# Fixture : un DataFrame "réaliste" avec différents cas
# =============================================================================


@pytest.fixture
def eda_df() -> pd.DataFrame:
    """DataFrame avec mix de cas : NaN, doublons, asymétrie, suspects."""
    rng = np.random.default_rng(42)
    n = 200
    df = pd.DataFrame(
        {
            "age": rng.integers(18, 65, size=n).astype(float),
            "revenu": rng.exponential(scale=3000, size=n),  # skewed
            "anciennete": rng.integers(0, 20, size=n).astype(float),
            "surface_m2": rng.uniform(20, 150, size=n),
            "genre": rng.choice(["M", "F"], size=n),
            "departement": rng.choice(["RH", "Tech", "Commerce", "Marketing"], size=n),
            "id_unique": [f"emp_{i}" for i in range(n)],  # haute cardinalité
            "tres_vide": [np.nan] * (n - 5) + list(range(5)),  # > 80% NaN
        }
    )
    # On rajoute quelques NaN dans age (~5%)
    df.loc[rng.choice(n, size=10, replace=False), "age"] = np.nan
    return df


# =============================================================================
# Helpers : _is_positive_expected / _is_year_column
# =============================================================================


def test_is_positive_expected_matches_known_keywords() -> None:
    """Reconnaît les mots-clés français et anglais."""
    assert _is_positive_expected("surface_m2") is True
    assert _is_positive_expected("prix_unitaire") is True
    assert _is_positive_expected("PRICE") is True  # case-insensitive
    assert _is_positive_expected("Nombre_clients") is True
    assert _is_positive_expected("genre") is False
    assert _is_positive_expected("statut") is False


def test_is_year_column() -> None:
    """Reconnaît les variantes 'année', 'annee', 'year'."""
    assert _is_year_column("annee_naissance") is True
    assert _is_year_column("année_creation") is True
    assert _is_year_column("birth_year") is True
    assert _is_year_column("age") is False


# =============================================================================
# _check_coherence
# =============================================================================


def test_check_coherence_flags_negative_on_positive_var() -> None:
    """Détecte les valeurs négatives sur une variable censée être positive."""
    df = pd.DataFrame({"surface_m2": [50, 75, -10, 120]})
    issues = _check_coherence(df, "surface_m2")
    assert any("négatives" in msg for msg in issues)


def test_check_coherence_flags_bad_years() -> None:
    """Détecte les années aberrantes."""
    df = pd.DataFrame({"annee_naissance": [1990, 1750, 1985, 1970]})
    issues = _check_coherence(df, "annee_naissance")
    assert any("antérieure à 1800" in msg for msg in issues)


def test_check_coherence_empty_returns_full_warning() -> None:
    """Une colonne entièrement vide est signalée."""
    df = pd.DataFrame({"col": [np.nan, np.nan, np.nan]})
    issues = _check_coherence(df, "col")
    assert issues == ["Colonne entièrement vide"]


def test_check_coherence_clean_returns_empty() -> None:
    """Sur des données propres, pas d'anomalie."""
    df = pd.DataFrame({"age": [25, 32, 47, 19, 30]})
    issues = _check_coherence(df, "age")
    assert issues == []


# =============================================================================
# _analyze_column : logique de recommandation
# =============================================================================


def test_analyze_column_too_many_nan_recommends_drop() -> None:
    """> 80% de NaN => recommandation 'drop'."""
    df = pd.DataFrame({"x": [np.nan] * 90 + list(range(10))})
    result = _analyze_column(
        df, "x",
        nan_drop_threshold=80.0,
        nan_high_threshold=50.0,
        high_cardinality_threshold=15,
        skew_threshold=1.0,
    )
    assert "drop" in result["action_tags"]


def test_analyze_column_binary_recommends_encode_binary() -> None:
    """Catégorielle à 2 modalités => 'encode_binary'."""
    df = pd.DataFrame({"genre": ["M", "F", "M", "F", "M"]})
    result = _analyze_column(
        df, "genre",
        nan_drop_threshold=80.0,
        nan_high_threshold=50.0,
        high_cardinality_threshold=15,
        skew_threshold=1.0,
    )
    assert "encode_binary" in result["action_tags"]


def test_analyze_column_multi_modalities_recommends_onehot() -> None:
    """Catégorielle à 3-15 modalités => 'encode_onehot'."""
    df = pd.DataFrame({"ville": ["A", "B", "C", "A", "B", "C", "D"]})
    result = _analyze_column(
        df, "ville",
        nan_drop_threshold=80.0,
        nan_high_threshold=50.0,
        high_cardinality_threshold=15,
        skew_threshold=1.0,
    )
    assert "encode_onehot" in result["action_tags"]


def test_analyze_column_high_cardinality_recommends_binary_repr() -> None:
    """Cardinalité > 15 => 'encode_binary_repr'."""
    df = pd.DataFrame({"id": [f"x_{i}" for i in range(20)]})
    result = _analyze_column(
        df, "id",
        nan_drop_threshold=80.0,
        nan_high_threshold=50.0,
        high_cardinality_threshold=15,
        skew_threshold=1.0,
    )
    assert "encode_binary_repr" in result["action_tags"]


def test_analyze_column_skewed_numeric_recommends_transform() -> None:
    """Numérique très asymétrique => 'transform_skew'."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"revenu": rng.exponential(scale=1000, size=500)})
    result = _analyze_column(
        df, "revenu",
        nan_drop_threshold=80.0,
        nan_high_threshold=50.0,
        high_cardinality_threshold=15,
        skew_threshold=1.0,
    )
    assert "transform_skew" in result["action_tags"]


# =============================================================================
# generate_eda_report : fonction publique
# =============================================================================


def test_generate_eda_report_returns_markdown_string(eda_df: pd.DataFrame) -> None:
    """generate_eda_report renvoie une string Markdown contenant les sections."""
    md = generate_eda_report(eda_df)

    assert isinstance(md, str)
    # Les 4 grandes sections doivent être présentes
    assert "# Rapport d'analyse du dataset" in md
    assert "## 1. Vue d'ensemble" in md
    assert "## 2. Analyse colonne par colonne" in md
    assert "## 3. Valeurs suspectes détectées" in md
    assert "## 4. Plan d'action recommandé" in md


def test_generate_eda_report_writes_file(
    eda_df: pd.DataFrame, tmp_path: Path
) -> None:
    """Avec output_path, le rapport est écrit sur disque."""
    output = tmp_path / "rapport.md"
    md = generate_eda_report(eda_df, output_path=str(output))

    assert output.exists()
    assert output.read_text(encoding="utf-8") == md


def test_generate_eda_report_includes_all_columns(
    eda_df: pd.DataFrame,
) -> None:
    """Toutes les colonnes du DataFrame apparaissent dans le rapport."""
    md = generate_eda_report(eda_df)
    for col in eda_df.columns:
        assert f"`{col}`" in md


def test_generate_eda_report_flags_high_nan_column(
    eda_df: pd.DataFrame,
) -> None:
    """La colonne 'tres_vide' (>80% NaN) doit recevoir une reco 'Drop'."""
    md = generate_eda_report(eda_df)
    # On cherche la ligne contenant 'tres_vide' et on vérifie 'Drop' dedans
    for line in md.split("\n"):
        if "`tres_vide`" in line and "|" in line:
            assert "Drop" in line
            return
    pytest.fail("Ligne pour 'tres_vide' non trouvée dans le rapport")
