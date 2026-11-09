
## Règles de projet
1. Une question claire par projet.
2. Une baseline classique (scikit-learn, statsmodels) avant tout modèle profond.
3. Un modèle PyTorch.
4. Une évaluation honnête : validation adaptée aux données (fenêtre glissante
   pour les séries temporelles, séparation par sujet ou par volume en
   imagerie), aucune fuite d'information, incertitude quantifiée.
5. Un tableau comparatif final, y compris quand le modèle profond ne gagne pas.

## Reproductibilité
- Seeds fixées (Python, NumPy, PyTorch) dans une fonction `set_seed`.
- Environnement décrit dans `environment.yml`.
- Chaque notebook s'exécute de bout en bout sur Colab ou Kaggle sans réglage.
- Données volumineuses et poids de modèles hors de Git (Hugging Face Hub, Zenodo).

## Code
- Noms de variables, fonctions et commentaires en anglais.
- Logique dans `src/`, notebooks légers qui l'appellent.
- Formatage et lint avec `ruff`.
- Tests avec `pytest` : formes des tenseurs, absence de fuite temporelle.

## Notebooks pédagogiques
- Une idée par notebook, ordre : motivation, théorie courte, code, expérience,
  limites.
- Lien explicite avec l'équivalent statistique (vraisemblance, estimation).
- Texte en français, code en anglais.

## Git
- Commits courts et atomiques : `feat:`, `fix:`, `docs:`, `test:`.
- Branche `main` toujours exécutable.
- Un README à jour avec le badge « Open in Colab ».