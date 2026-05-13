# MLOps - Suivi des expérimentations Pokebot

Ce document explique comment fonctionne l'environnement MLOps intégré au notebook `pokebot.ipynb` pour enregistrer les résultats de modélisation.

## Vue d'ensemble

Le projet utilise **MLflow** pour tracer automatiquement :
- Les **hyperparamètres** de chaque phase d'entraînement
- Les **métriques** d'évaluation (winrate vs Random, vs MaxPower)
- Les **artefacts** : modèles sauvegardés et graphiques

## Installation

```bash
pip install mlflow
# ou
pip install -r requirements.txt
```

## Structure du suivi

### 1. Configuration (cellule MLOps)

- **Tracking URI** : `file:./mlruns` — les runs sont stockés localement dans le dossier `mlruns/`
- **Expérience** : `pokebot`
- **Fonction `mlflow_ensure_run()`** : démarre une run si aucune n'est active (permet d'enchaîner les phases dans le même run)

### 2. Ce qui est enregistré par phase

| Phase | Paramètres | Artefacts |
|-------|------------|-----------|
| **Phase 0** (Imitation) | `phase0_min_rating`, `phase0_num_pages`, `phase0_target_switch_ratio`, `phase0_expert_actions`, `phase0_timesteps` | `model_imitation.zip` |
| **Phase 1** (vs Random) | `phase1_timesteps`, `phase1_opponent` | `model_phase1.zip` |
| **Phase 2** (vs MaxPower) | `phase2_timesteps`, `phase2_opponent` | `model_phase2.zip` |
| **Phase 3** (Self-Play) | `phase3_timesteps`, `phase3_opponent` | `model_final.zip` |
| **Évaluation** | — | `evolution_performances.png` |

### 3. Métriques d'évaluation

Pour chaque checkpoint (Phase 0 à 3) :
- `eval_phase0_imitation_winrate_vs_random`
- `eval_phase0_imitation_winrate_vs_maxpower`
- `eval_phase1_vs_random_winrate_vs_random`
- `eval_phase1_vs_random_winrate_vs_maxpower`
- ... (idem pour phase2 et phase3)

## Utilisation

### Exécution normale

1. Exécuter la cellule **MLflow setup** (une fois)
2. Exécuter les phases 0, 1, 2, 3 dans l'ordre
3. Exécuter la cellule d'évaluation

Toutes les phases partagent la **même run** MLflow (nommée `pokebot_YYYYMMDD_HHMM`).

### Consulter les résultats

```bash
# Démarrer l'interface MLflow
mlflow ui
```

Puis ouvrir http://localhost:5000 pour :
- Comparer les runs
- Voir les paramètres et métriques
- Télécharger les artefacts (modèles, graphiques)

### Serveur MLflow distant

Pour utiliser un serveur MLflow centralisé :

```python
# Dans la cellule MLOps, remplacer :
mlflow.set_tracking_uri("http://localhost:5000")  # ou votre URL
```

## Organisation des artefacts

```
mlruns/
    <experiment_id>/
        <run_id>/
            artifacts/
                phase0/
                    model_imitation.zip
                phase1/
                    model_phase1.zip
                phase2/
                    model_phase2.zip
                phase3/
                    model_final.zip
                evaluation/
                    evolution_performances.png
```

## Bonnes pratiques

- **Exécuter les cellules dans l'ordre** pour éviter des runs fragmentées
- **Ne pas réexécuter** la cellule MLOps en plein milieu d'un entraînement (sinon une nouvelle run sera créée)
- **Versionner** les hyperparamètres importants en les modifiant dans les cellules avant de lancer
