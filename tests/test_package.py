"""
Tests de base du package nanuq.

Ce premier fichier de tests sert juste à vérifier que :
- le package s'importe sans erreur,
- la variable __version__ est correctement définie.

Les vrais tests fonctionnels seront ajoutés au fil de la migration.
"""

import nanuq


def test_package_imports():
    """Le package peut être importé sans lever d'exception."""
    assert nanuq is not None


def test_version_is_defined():
    """La version est définie et a un format raisonnable."""
    assert hasattr(nanuq, "__version__")
    assert isinstance(nanuq.__version__, str)
    assert len(nanuq.__version__.split(".")) >= 2
