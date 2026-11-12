Données : température horaire à Toulouse-Blagnac

> **Fiche provisoire.** Les points marqués « à confirmer » seront vérifiés lors du premier téléchargement, puis cette fiche sera complétée (trous, taille, période exacte de chaque fichier).

## Source

-   Jeu de données : [Données climatologiques de base - horaires](https://www.data.gouv.fr/datasets/donnees-climatologiques-de-base-horaires), publié par Météo-France sur data.gouv.fr.

-   Licence : Licence Ouverte 2.0. La source doit être citée : « Source : Météo-France, données climatologiques de base - horaires, Licence Ouverte 2.0 ».

-   Organisation : un fichier `.csv.gz` par département et par période, avec tous les paramètres. Les heures sont en **UTC**; on garde l'UTC pour éviter les décalages d'heure d'été.

-   Département utilisé : Haute-Garonne (31). Les fichiers sont téléchargés par `tsunc.data`, dans `data/raw/` (dossier ignoré par Git : les données brutes ne sont pas versionnées).

## Station et variable

-   Station : numéro `NUM_POSTE` = `31070004` (Blagnac). À confirmer avec `tsunc.data.list_stations`, qui affiche pour chaque station son nom, sa position et le nombre de températures disponibles.

-   Variables : colonnes des fichiers sources (`T` température sous abri, `TD` point de rosée, `U` humidité, ...). Unité de `T` (°C) à confirmer à l'ouverture du fichier : un contrôle de plausibilité (entre -60 et 60 °C) évite d'utiliser par erreur des valeurs dans une autre unité.

## Découpage chronologique (UTC)

Quatre blocs disjoints, dans cet ordre, sans mélange aléatoire :

| Bloc | Période | Rôle |
|----|----|----|
| `train` | 2010-01-01 à 2020-12-31 | apprentissage |
| `val` | année 2021 | arrêt précoce, choix des hyperparamètres |
| `calibration` | année 2022 | scores de la prédiction conforme |
| `test` | année 2023 | évaluation finale (365 origines quotidiennes) |

## Reproduire

Depuis le dossier `01-ts-uncertainty/` :

```         
from tsunc import data

# Une station, une ou plusieurs variables, une période (bornes incluses ; None = le maximum possible)
frame = data.extract("31070004", variables=["T", "U"], start="1990", end="2025")
print(data.gap_report(frame))               # heures manquantes et plus long trou, par variable
blocks = data.make_blocks(frame)            # train / val / calibration / test
```

`extract` demande à data.gouv.fr la liste des fichiers du département (`data.list_files`), garde ceux qui recouvrent la période, les télécharge dans `data/raw/` et assemble la table. Si deux fichiers se chevauchent, la ligne du fichier le plus récent est gardée. Si la liste automatique échoue, construire le tableau à la main avec `data.files_from_urls({nom: lien, ...})` et le passer à `extract(files=...)`.

## Limites connues

-   Les fichiers sont mis à jour régulièrement par Météo-France : noter la date de téléchargement pour chaque exécution de référence (à compléter).

-   Les heures manquantes restent `NaN` : aucune imputation n'est faite à ce stade.

-   Les codes qualité (`QT`, ...) ne sont pas lus ni exploités.

-   Une station n'existe pas sur toute la période : les années avant son ouverture n'apparaissent pas.
