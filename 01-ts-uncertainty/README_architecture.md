# README --- Architecture et orchestration du projet

> **Projet :** `01-ts-uncertainty`\
> **Package Python :** `tsunc`\
> **Objectif :** prévoir la température horaire à 24 heures et produire
> des intervalles de prédiction dont la couverture est évaluée et, si
> nécessaire, calibrée.

## 1. Pourquoi cette organisation ?

Le projet sépare deux activités complémentaires :

### 1. Le code métier --- le package `tsunc`

Le package contient les fonctions Python réutilisables : préparation des
données, modèles, calibration des intervalles et évaluation.

Il constitue la **boîte à outils** du projet. Une fonction est écrite
une fois dans le package, puis peut être utilisée dans plusieurs
notebooks, scripts ou environnements.

### 2. L'expérimentation --- les notebooks

Les notebooks servent à conduire les expériences : explorer les données,
entraîner et comparer les modèles, visualiser les résultats et commenter
les conclusions.

Ils constituent le **laboratoire** du projet. Ils appellent les
fonctions de `tsunc` au lieu de recopier leur implémentation.

**Avantage de cette séparation :** le code reste réutilisable et
testable, tandis que les notebooks restent lisibles et centrés sur les
questions scientifiques. Une correction dans une fonction commune
profite à toutes les expériences qui l'utilisent.

## 2. Déroulement général du projet

Le flux de travail prévu est le suivant :

1.  **Données** --- conserver les données originales, puis préparer les
    données utilisables pour les expériences.
2.  **Exploration** --- comprendre la série temporelle, sa fréquence,
    ses valeurs manquantes, sa saisonnalité et ses éventuelles
    anomalies.
3.  **Référence statistique (baseline)** --- construire un modèle simple
    et solide servant de point de comparaison.
4.  **Deep learning** --- entraîner un modèle neuronal, par exemple un
    LSTM, sans perdre les précautions propres aux séries temporelles.
5.  **Incertitude** --- produire des quantiles et construire/calibrer
    des intervalles de prédiction, notamment avec la méthode conformal
    prediction / CQR.
6.  **Évaluation** --- comparer les prévisions ponctuelles et les
    intervalles : qualité des prévisions, couverture observée, largeur
    des intervalles et stabilité.
7.  **Tests et documentation** --- vérifier les fonctions, expliquer les
    choix et conserver les résultats reproductibles.

Les données temporelles doivent être séparées dans l'ordre chronologique
: on ne mélange pas aléatoirement passé et futur. Les étapes de
préparation et de calibration doivent également éviter toute fuite
d'information depuis les données de test.

## 3. Organisation des dossiers et rôle des fichiers

Voici l'organisation cible. Certains modules pourront être ajoutés au
fur et à mesure, lorsque leur responsabilité sera clairement définie.

``` text
01-ts-uncertainty/
├── README.md
├── README_architecture.md
├── pyproject.toml
├── environment.yml
├── src/
│   └── tsunc/
│       ├── __init__.py
│       ├── config.py
│       ├── data.py
│       ├── models.py
│       ├── calibration.py
│       └── evaluation.py
├── notebooks/
│   ├── 01_exploration.ipynb
│   ├── 02_baseline.ipynb
│   ├── 03_deep_learning.ipynb
│   └── 04_uncertainty.ipynb
├── tests/
│   ├── test_data.py
│   └── test_calibration.py
├── data/
│   ├── raw/
│   └── processed/
└── docs/
    └── statistical_methods.md
```

  --------------------------------------------------------------------------
  Élément                    Rôle et utilisation     Pourquoi c'est utile
  -------------------------- ----------------------- -----------------------
  `README.md`                Présente le sujet, les  Donne une vue
                             objectifs,              d'ensemble rapide aux
                             l'installation et la    lecteurs.
                             manière de lancer le    
                             projet.                 

  `README_architecture.md`   Explique l'organisation Aide à comprendre où
                             technique et le         chercher et où ajouter
                             parcours d'exécution du du code.
                             projet.                 

  `pyproject.toml`           Décrit le package, ses  Rend `tsunc`
                             dépendances Python et   installable de manière
                             la façon de le          standard.
                             construire/installer.   

  `environment.yml`          Décrit un environnement Facilite la recréation
                             Conda reproductible     de l'environnement de
                             (version de Python et   travail.
                             bibliothèques).         

  `src/tsunc/`               Contient le code métier Évite de disperser ou
                             importable.             recopier les fonctions
                                                     dans les notebooks.

  `notebooks/*.ipynb`        Contiennent les         Conservent une trace
                             expériences,            interactive du
                             explications,           raisonnement
                             graphiques et           scientifique.
                             comparaisons.           

  `tests/*.py`               Vérifient               Détectent les
                             automatiquement le      régressions lorsqu'on
                             comportement des        modifie le code.
                             fonctions.              

  `data/raw/`                Accueille les données   Permet de repartir de
                             originales, conservées  la source et de tracer
                             sans modification.      les transformations.

  `data/processed/`          Accueille les données   Sépare les données
                             préparées pour          sources des données
                             l'analyse ou            transformées.
                             l'entraînement.         

  `docs/`                    Contient les            Évite de surcharger le
                             explications            README principal.
                             méthodologiques plus    
                             détaillées.             
  --------------------------------------------------------------------------

### Extensions courantes

-   **`.py`** : fichier de code Python. Il contient des fonctions,
    classes ou tests exécutables et importables.
-   **`.ipynb`** : notebook Jupyter. Il combine cellules de code,
    résultats, graphiques et texte explicatif.
-   **`.toml`** : fichier de configuration structuré ; ici, il décrit le
    package Python.
-   **`.yml` / `.yaml`** : fichier de configuration lisible ; ici, il
    décrit l'environnement Conda.
-   **`.md`** : fichier Markdown ; il sert à rédiger une documentation
    lisible sur GitHub.
-   **`.csv` / `.parquet`** : formats possibles de stockage de données
    tabulaires.
-   **`.pt` / `.pth`** : formats courants de sauvegarde de modèles ou
    poids PyTorch.

## 4. Comprendre les fichiers Python déjà présents

### `src/tsunc/__init__.py`

Contenu actuel :

``` python
"""tsunc: probabilistic hourly temperature forecasting with conformal prediction intervals."""

__version__ = "0.1.0"
```

-   La première ligne est une *docstring* : elle décrit brièvement le
    package.
-   `__version__` définit sa version.
-   La présence de `__init__.py` marque le dossier `tsunc` comme package
    Python classique.
-   Le fichier pourra plus tard exposer quelques fonctions publiques,
    mais il est préférable de le garder simple au départ.

La version est récupérée par `pyproject.toml` : elle n'est donc écrite
qu'à un seul endroit.

### `src/tsunc/config.py`

Contenu et signification :

``` python
"""Project-wide constants and reproducibility helpers."""

import random
import numpy as np

SEED: int = 42
HORIZON: int = 24
ALPHA: float = 0.10
QUANTILES: tuple[float, ...] = (ALPHA / 2, 0.5, 1 - ALPHA / 2)

def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
```

-   **`SEED = 42`** : graine utilisée pour rendre répétables les tirages
    aléatoires de Python et NumPy. Cela améliore la reproductibilité,
    sans garantir à elle seule des résultats identiques sur toutes les
    machines ou tous les accélérateurs.
-   **`HORIZON = 24`** : horizon de prévision de 24 heures. Il
    représente ici la durée future visée par la prévision.
-   **`ALPHA = 0.10`** : niveau de non-couverture visé de 10 %. La
    couverture nominale correspondante est `1 - ALPHA = 0.90`, soit 90
    %. La couverture réellement observée doit être mesurée sur des
    données de test ; elle n'est pas garantie par cette constante.
-   **`QUANTILES`** : niveaux de quantiles demandés au modèle. Avec
    `ALPHA = 0.10`, ils valent `(0.05, 0.5, 0.95)`.
    -   `0.05` : quantile inférieur ;
    -   `0.50` : médiane, utilisable comme prévision ponctuelle ;
    -   `0.95` : quantile supérieur.
-   **`set_seed()`** : initialise les générateurs aléatoires de Python
    et NumPy. Le commentaire annonce que la graine PyTorch sera ajoutée
    lorsque le LSTM sera introduit.

Le calcul des quantiles à partir de `ALPHA` évite de saisir séparément
`0.05` et `0.95`. Si la cible change, les quantiles extérieurs
s'adaptent automatiquement.

La fonction `set_seed()` est un outil de reproductibilité. Elle ne
charge pas les données et n'entraîne aucun modèle.

## 5. Comment installer et utiliser le package ?

Depuis le dossier `01-ts-uncertainty`, la commande :

``` bash
python -m pip install -e .
```

installe le package en mode **éditable**. Le `-e` signifie que Python
utilise le code du dossier de travail : après une modification d'un
fichier `.py`, il n'est généralement pas nécessaire de réinstaller le
package.

Dans un notebook ou un script :

``` python
import tsunc
from tsunc.config import SEED, HORIZON, ALPHA, QUANTILES, set_seed

set_seed()

print(tsunc.__version__)
print(HORIZON, ALPHA, QUANTILES)
```

Le fichier `pyproject.toml` indique notamment que le code se trouve dans
`src/` et que le package à installer s'appelle `tsunc`. C'est ce qui
permet d'importer `tsunc` sans ajouter manuellement `src/` à `sys.path`.

## 6. Rôle des deux fichiers de configuration

  -----------------------------------------------------------------------
  Fichier                             Question à laquelle il répond
  ----------------------------------- -----------------------------------
  `pyproject.toml`                    « Comment construire, installer et
                                      identifier mon package Python ? »

  `environment.yml`                   « Avec quel environnement logiciel
                                      exécuter le projet ? »

  `config.py`                         « Quels paramètres communs
                                      utilisent les expériences et les
                                      fonctions ? »
  -----------------------------------------------------------------------

Ils n'ont donc pas le même rôle et ne se remplacent pas.

## 7. Règles pratiques pour garder le projet propre

1.  Une fonction réutilisable va dans `src/tsunc/`, pas uniquement dans
    un notebook.
2.  Un notebook orchestre une expérience ; il ne doit pas devenir
    l'unique endroit où vit le code important.
3.  Les données originales ne sont pas écrasées par les données
    préparées.
4.  Les tests vérifient les fonctions importantes, en particulier les
    transformations temporelles et la calibration.
5.  Les paramètres communs sont centralisés dans `config.py`, plutôt que
    recopiés dans plusieurs fichiers.
6.  Les résultats doivent pouvoir être reliés à la version du code, aux
    données et aux paramètres qui les ont produits.
7.  On ajoute un module seulement lorsqu'il a une responsabilité
    identifiable ; il n'est pas nécessaire de créer tous les fichiers
    prévus dès le départ.

## 8. En résumé

-   **`src/tsunc/` :** le code métier, réutilisable et testable.
-   **`notebooks/` :** les expériences, les analyses et les explications
    scientifiques.
-   **`tests/` :** les vérifications automatiques.
-   **`data/` :** les données sources et préparées.
-   **`docs/` :** la documentation technique et méthodologique.
-   **`pyproject.toml` :** l'installation et la construction du package.
-   **`environment.yml` :** la description de l'environnement logiciel.
-   **`config.py` :** les paramètres communs et la reproductibilité.

Cette organisation est un peu plus structurée qu'un simple notebook,
mais elle devient avantageuse dès que le projet comporte plusieurs
modèles, plusieurs expériences et des fonctions réutilisées. Elle
facilite la maintenance, les tests, la collaboration et la reproduction
des résultats.
