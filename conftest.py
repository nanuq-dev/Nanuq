"""
conftest.py — Configuration globale pour pytest.

Ce fichier est automatiquement chargé par pytest **avant tout test**,
ce qui permet de configurer l'environnement très tôt — avant même
que les modules de tests ne soient importés.

Utilisation actuelle : définir ``LOKY_MAX_CPU_COUNT`` pour éviter le
warning de joblib sur Windows ("Could not find the number of physical
cores"). joblib lit cette variable au moment de l'import, donc il
faut la poser AVANT que quoi que ce soit n'importe sklearn ou joblib.
"""

from __future__ import annotations

import os

# joblib (utilisé par sklearn pour la parallélisation) lit cette variable
# à l'import. On la définit avant tout import sklearn dans les tests.
# `setdefault` respecte une valeur déjà présente dans l'environnement
# de l'utilisateur — on ne l'écrase pas.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

# matplotlib en backend "Agg" : pas d'affichage de fenêtre, pas de
# blocage lors des appels à plt.show() dans les tests. À configurer
# AVANT tout import de pyplot.
import matplotlib  # noqa: E402

matplotlib.use("Agg")

# Neutraliser plt.show() pendant les tests.
# Raison : sur Windows, en backend Agg, plt.show() émet le warning
# "FigureCanvasAgg is non-interactive" qui, remonté par la stack
# jusqu'à nanuq.plots, déclencherait notre règle
# `error::Warning:nanuq` et ferait échouer les tests.
# Approche standard utilisée par sklearn, seaborn, etc. en tests.
import matplotlib.pyplot as plt  # noqa: E402

plt.show = lambda *args, **kwargs: None
