"""
scripts/help.py — Affiche les cibles disponibles du Makefile (cross-platform).

Remplace le pipeline `grep | awk` du Makefile par un équivalent Python.
Parse le Makefile pour extraire les cibles documentées avec `## commentaire`.

Usage :
    python scripts/help.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Regex : capture "cible: ... ## description"
TARGET_PATTERN = re.compile(r"^([a-zA-Z_-]+):.*?## (.*)$", re.MULTILINE)


def main() -> None:
    """Lit le Makefile et affiche les cibles documentées."""
    makefile = Path("Makefile")

    if not makefile.exists():
        print("Erreur : Makefile introuvable dans le répertoire courant.")
        sys.exit(1)

    content = makefile.read_text(encoding="utf-8")

    print("Cibles disponibles :")
    for match in TARGET_PATTERN.finditer(content):
        target, description = match.groups()
        # Format : 2 espaces + cible alignée sur 20 chars + description
        print(f"  {target:<20} {description}")


if __name__ == "__main__":
    main()
