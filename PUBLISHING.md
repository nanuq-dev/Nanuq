# Guide de publication sur PyPI

Ce document décrit les étapes pour publier `nanuq` sur PyPI.
Tout est déjà préparé : il reste surtout à choisir un nom et créer les comptes.

## Avant de publier : checklist

- [x] Nom unique choisi et vérifié libre sur PyPI : nanuq
- [x] Pseudo GitHub renseigné (nanuq-dev)
- [x] Email de contact renseigné (olivier.gruwe@gmail.com)
- [x] Nom complet renseigné dans LICENSE (Olivier Gruwe)
- [ ] Créer un compte PyPI : https://pypi.org/account/register/
- [ ] Créer un compte TestPyPI (bac à sable) : https://test.pypi.org/account/register/

## Étape 0 — Choisir un nom disponible

Le nom `nanuq` doit être vérifié sur `https://pypi.org/project/nanuq/`
(404 = disponible). S'il est pris, en choisir un autre et le reporter dans le
champ `name` de `pyproject.toml`.

### Note sur le logo dans le README

Le README utilise `assets/logo.png` en chemin relatif. Cela fonctionne sur
**GitHub**, mais **PyPI n'affiche pas les images en chemin relatif**. Pour que
le logo apparaisse sur la page PyPI, remplacer dans le README le chemin par une
URL absolue une fois le repo en ligne, par exemple :

```
https://raw.githubusercontent.com/nanuq-dev/nanuq/main/assets/logo.png
```

## Étape 1 — Construire le package

```bash
# Nettoyer les anciens builds
rm -rf dist/ build/

# Construire wheel + sdist
uv build
```

Cela crée deux fichiers dans `dist/` :
- `*.whl` (wheel — format binaire d'installation rapide)
- `*.tar.gz` (sdist — distribution source)

## Étape 2 — Valider le package

```bash
uv pip install twine
twine check dist/*
```

Les deux fichiers doivent afficher `PASSED`.

## Étape 3 — Tester sur TestPyPI (recommandé)

TestPyPI est une copie de PyPI pour s'entraîner sans polluer le vrai index.

```bash
# Publier sur TestPyPI
uv publish --publish-url https://test.pypi.org/legacy/

# Tester l'installation depuis TestPyPI dans un environnement neuf
uv pip install --index-url https://test.pypi.org/simple/ \
               --extra-index-url https://pypi.org/simple/ \
               LE-NOM-CHOISI
```

(L'`--extra-index-url` permet de récupérer les dépendances depuis le vrai PyPI,
car TestPyPI ne contient pas pandas, sklearn, etc.)

## Étape 4 — Publier sur le vrai PyPI

Une fois validé sur TestPyPI :

```bash
uv publish
```

Au premier `uv publish`, un token API sera demandé. Le créer ici :
https://pypi.org/manage/account/token/ (portée : "Entire account" la première
fois, puis on peut le restreindre au projet).

## Étape 5 (optionnel) — Publication automatique via GitHub Actions

Le workflow `.github/workflows/publish.yml` est déjà configuré pour publier
automatiquement à chaque tag de version. Pour l'activer :

1. Configurer le **Trusted Publishing** sur PyPI :
   https://pypi.org/manage/account/publishing/
   (lier le repo GitHub, le workflow `publish.yml`, l'environnement `pypi`)
2. Créer et pousser un tag de version :

```bash
git tag v0.2.0
git push origin v0.2.0
```

GitHub Actions construira et publiera automatiquement. Plus besoin de token :
le Trusted Publishing utilise une authentification sécurisée par OpenID Connect.

## Workflow de version pour les mises à jour futures

1. Modifier le code
2. Bumper la version dans `pyproject.toml` ET `src/nanuq/__init__.py`
3. Documenter les changements dans `CHANGELOG.md`
4. Commit + tag : `git tag vX.Y.Z && git push --tags`
5. La CI publie automatiquement (si Trusted Publishing configuré)

### Rappel du versionnage sémantique (X.Y.Z)

- **X (major)** : changement cassant l'API existante
- **Y (minor)** : nouvelle fonctionnalité rétrocompatible
- **Z (patch)** : correction de bug rétrocompatible
