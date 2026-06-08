# =============================================================================
# Makefile - Automatisation des tâches de développement de nanuq
# =============================================================================
# Cross-platform : fonctionne sur Windows (Git Bash), Linux et macOS.
# Les opérations "exotiques" (suppression récursive, parsing de Makefile)
# sont déléguées à des petits scripts Python dans scripts/ pour éviter
# toute dépendance aux outils Unix (rm, find, grep, awk).
#
# Les commandes passent par `uv run` / `uv` : pas besoin d'activer un venv
# manuellement, uv résout l'environnement automatiquement.
#
# Usage : `make <cible>`, par exemple `make install-dev`.
# Pour voir toutes les cibles disponibles : `make help`.
# =============================================================================

# .PHONY = ces "cibles" ne correspondent pas à des fichiers, ce sont des commandes.
.PHONY: help install install-dev test test-cov lint format type-check check \
        build check-dist publish-test publish release clean all

# Cible par défaut quand on tape juste `make` : afficher l'aide.
.DEFAULT_GOAL := help

# Python pour lancer nos scripts cross-platform.
# Surcharge possible : `make PYTHON=python3.11 ...`.
PYTHON ?= python


help:  ## Affiche cette aide
	$(PYTHON) scripts/help.py


# -----------------------------------------------------------------------------
# Installation
# -----------------------------------------------------------------------------

install:  ## Installe le package en mode editable (utilisation normale)
	uv pip install -e .

install-dev:  ## Installe le package avec les dépendances de dev et notebooks
	uv pip install -e ".[dev,notebooks]"


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------

test:  ## Lance les tests unitaires
	uv run pytest

test-cov:  ## Lance les tests avec mesure de couverture (rapport HTML dans htmlcov/)
	uv run pytest --cov=nanuq --cov-report=term-missing --cov-report=html


# -----------------------------------------------------------------------------
# Qualité de code
# -----------------------------------------------------------------------------

lint:  ## Vérifie le style et les erreurs de code
	uv run ruff check src/ tests/ scripts/

format:  ## Reformate le code automatiquement
	uv run ruff format src/ tests/ scripts/
	uv run ruff check --fix src/ tests/ scripts/

type-check:  ## Vérifie les types statiques avec mypy
	uv run mypy src/nanuq

check:  ## Vérifie tout : lint + types (informatif) + tests (à lancer avant un commit)
	uv run ruff check src/ tests/ scripts/
	-uv run mypy src/nanuq
	uv run pytest


# -----------------------------------------------------------------------------
# Build et publication PyPI
# -----------------------------------------------------------------------------

build:  ## Construit le package distribuable (wheel + tar.gz dans dist/)
	uv build

check-dist:  ## Valide les artefacts construits (contrôle qualité PyPI)
	uvx twine check dist/*

publish-test:  ## Publie sur TestPyPI (bac à sable, sans risque)
	uv publish --publish-url https://test.pypi.org/legacy/

publish:  ## Publie sur le vrai PyPI (irréversible pour une version donnée)
	uv publish

release:  ## Workflow complet de release : clean -> build -> check-dist
	$(PYTHON) scripts/clean.py
	uv build
	uvx twine check dist/*
	@echo ""
	@echo "Artefacts prets dans dist/. Pour publier :"
	@echo "  make publish-test   (recommande d'abord)"
	@echo "  make publish        (PyPI officiel)"


# -----------------------------------------------------------------------------
# Vérification globale et nettoyage
# -----------------------------------------------------------------------------

all: check build  ## Vérifie tout puis construit le package

clean:  ## Supprime les fichiers temporaires et artefacts (cross-platform)
	$(PYTHON) scripts/clean.py
