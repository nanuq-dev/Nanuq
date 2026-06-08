"""
scripts/clean.py — Nettoyage des fichiers temporaires (cross-platform).

Remplace les commandes Unix `rm -rf` et `find -delete` du Makefile par
leur équivalent Python. Fonctionne sur Windows, Linux et macOS sans
dépendance externe.

Usage :
    python scripts/clean.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

# Dossiers à supprimer (récursivement)
DIRS_TO_REMOVE = [
    "build",
    "dist",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    "htmlcov",
]

# Fichiers à supprimer
FILES_TO_REMOVE = [
    ".coverage",
]

# Patterns à supprimer récursivement depuis la racine du projet
GLOB_DIRS_TO_REMOVE = [
    "*.egg-info",  # à la racine
    "**/__pycache__",  # à tous les niveaux
]

GLOB_FILES_TO_REMOVE = [
    "**/*.pyc",
]


def _remove_dir(path: Path) -> None:
    """Supprime un dossier s'il existe (sans erreur si absent)."""
    if path.exists() and path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
        print(f"  supprimé   {path}")


def _remove_file(path: Path) -> None:
    """Supprime un fichier s'il existe (sans erreur si absent)."""
    if path.exists() and path.is_file():
        path.unlink(missing_ok=True)
        print(f"  supprimé   {path}")


def main() -> None:
    """Point d'entrée principal."""
    root = Path(".")

    print("Nettoyage des fichiers temporaires…")

    # Dossiers connus à la racine
    for d in DIRS_TO_REMOVE:
        _remove_dir(root / d)

    # Fichiers connus à la racine
    for f in FILES_TO_REMOVE:
        _remove_file(root / f)

    # Patterns globbing (egg-info, __pycache__ partout, etc.)
    for pattern in GLOB_DIRS_TO_REMOVE:
        for path in root.glob(pattern):
            if path.is_dir():
                _remove_dir(path)

    for pattern in GLOB_FILES_TO_REMOVE:
        for path in root.glob(pattern):
            if path.is_file():
                _remove_file(path)

    print("OK")


if __name__ == "__main__":
    main()
