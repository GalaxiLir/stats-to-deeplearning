# 01 — Prévision de températures horaires avec incertitude

> **Statut : en construction.** Socle (paquet `tsunc`, environnement), chargement des données
> et métriques d'évaluation sont écrits et testés ; la vérification sur les vrais fichiers
> Météo-France est en cours. Baselines, LSTM et intervalles conformaux suivent.

## Question

Un LSTM à sortie quantile, calibré par prédiction conforme, prévoit-il la température
horaire à Toulouse-Blagnac (horizon 24 h) mieux que des baselines statistiques classiques
(naïve saisonnière, ETS, ARIMA), et quelle incertitude peut-on annoncer honnêtement ?

## Démarche

1. Baselines classiques : naïve saisonnière (m = 24), ETS, ARIMA.
2. Modèle PyTorch : LSTM, sortie directe sur 24 h, perte pinball (régression quantile).
3. Incertitude : intervalles conformaux (CQR), calibrés chronologiquement.
4. Évaluation : origine glissante quotidienne ; MASE, pinball, couverture et largeur des
   intervalles ; comparaison finale, y compris si le modèle profond ne gagne pas.

## Installation en local

Depuis ce dossier :

```bash
conda env create -f environment.yml
conda activate tsunc
python -c "import tsunc; print(tsunc.__version__)"
```

## Structure

```
01-ts-uncertainty/
├── src/tsunc/
│   ├── config.py    # constantes (graine, horizon, niveaux de quantiles)
│   ├── data.py      # choix des fichiers, téléchargement, extraction station/variables/période
│   └── metrics.py   # pinball, MASE, couverture, largeur des intervalles
├── docs/            # guides de lecture du code (lire_data_py.md)
├── data/README.md   # fiche des données (source, licence, blocs) ; data/raw/ non versionné
├── notebooks/       # notebooks légers qui appellent src/ (00_data : contrôle des données)
├── tests/           # pytest : métriques, lecture des fichiers, absence de fuite temporelle
├── pyproject.toml   # définition du paquet `tsunc` et de ses dépendances
└── environment.yml  # environnement conda pour le travail en local
```

## Tests

```bash
pip install pytest
pytest
```

## English summary

Probabilistic forecasting of hourly temperature at Toulouse-Blagnac (24 h horizon):
classical baselines (seasonal naive, ETS, ARIMA) versus a quantile LSTM in PyTorch, with
chronologically calibrated conformal prediction intervals (CQR). Evaluation uses a daily
rolling origin and reports MASE, pinball loss, coverage and interval width. Work in progress.