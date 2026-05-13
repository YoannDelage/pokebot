# Pokebot — Un sparring partner pour apprendre Pokémon à un niveau stratégique

> "Et si je pouvais apprendre à jouer de manière stratégique à un jeu considéré comme enfantin — mais qui ne l'est en fait pas du tout — en entraînant un modèle de machine learning à jouer lui aussi, pour avoir un sparring partner ?"

C'est le point de départ de ce projet.

---

## L'histoire derrière le projet

Quand j'ai commencé à m'intéresser sérieusement aux **combats Randoms** sur Pokémon Showdown, j'ai vite réalisé qu'il y avait deux mondes :

- **Le jeu que la plupart des gens connaissent** : un RPG mignon où on choisit son starter, on bat des champions d'arène, on complete sa collection de monstres.
- **Le jeu compétitif** : un duel d'échecs à six pièces où chaque tour combine **prédiction adverse, gestion des ressources, calcul de probabilités, type de matchups, lecture du méta-game et bluff**.

J'ai voulu progresser. Mais assez vite j'ai vu que s'entraîner uniquement contre des humains a deux limites :

1. **A mon niveau, les joueurs n'ont pas le niveau pour que je puisse apprendre** — et ceux qui l'ont me détruisent en 2 minutes !
2. **Je ne peux pas rejouer la même situation à la demande** — chaque combat est une expérience unique et tu n'as pas de "training mode" comme dans un jeu de combat.

D'où l'idée : **construire un bot capable de jouer décemment**, qui me serve de **sparring partner toujours disponible**. Et tant qu'à faire, en profiter pour explorer un terrain de jeu fascinant pour un projet de machine learning — parce qu'au-delà de l'aspect personnel, Pokémon est un cas d'école remarquable de **prise de décision sous incertitude** avec un espace d'état immense (voir abyssal).

---

## Pourquoi Pokémon est beaucoup plus complexe qu'il n'y paraît

Avant de plonger dans la technique, posons les bases.

### Un combat Pokémon en chiffres

| Élément | Ordre de grandeur |
|---|---|
| Pokémon existants (Gen 9) | ~1 000 |
| Moves différents | ~900 |
| Talents (abilities) | ~300 |
| Objets tenus | ~100 |
| Types (et combinaisons) | 18 types, ~170 combinaisons |
| Conditions de terrain | 5 météos, 5 champs, hazards multiples |

À chaque tour, le joueur fait face à un choix **parmi 9 actions valides en moyenne** (4 moves + 5 switches), mais **chaque choix dépend de centaines de paramètres** : type de l'adversaire, ses boosts, son statut, l'état de ton équipe, les hazards (effets de status entre autres) posés, les screens actifs, etc.

### Les vraies sources de complexité

1. **Information incomplète.** On ne connait ni l'équipe complète de l'adversaire, ni ses objets, ni ses talents (sauf s'ils ont été révélés). On joue dans un brouillard de guerre permanent.
2. **Récompense différée.** Une bonne décision au tour 3 peut ne payer qu'au tour 25. Le signal d'apprentissage est très bruité.
3. **Combinatoire explosive.** L'espace d'état est gigantesque ; aucune table ne peut le représenter explicitement.
4. **Méta-game qui bouge.** Les sets compétitifs changent à chaque patch de Showdown (qui reflete les patchs officiels).
5. **Aléa structurel.** Les coups critiques, les ratés à 10%, les paralysies à 25% : la variance fait pleinement partie du jeu.

C'est pour ça que les pros de Showdown ne sont pas juste "des gens qui aiment Pokémon" — ce sont des joueurs qui ont intégré des modèles mentaux comparables à ceux du poker ou des échecs.

---

## Le projet : pourquoi la Gen 1

**Pourquoi se concentrer sur la Gen 1 ?** Parce que c'est l'environnement le plus **fermé** que propose la franchise :

- Pas de talents (ce système n'existe pas encore)
- Pas d'objets tenus
- Pas de méga-évolution, ni Z-moves, ni Dynamax, ni Téracristal
- 151 Pokémons 'seulement', 165 moves
- Mécaniques de combat **simples et déterministes** (à l'exception de la variance native du jeu, communement appelée HAXX)

Sur la Gen 1, **toute l'information stratégiquement utile est observable** par l'IA. C'est le contexte idéal pour entraîner un modèle de reinforcement learning et obtenir des résultats lisibles : si le bot perd, c'est forcément parce que sa prise de décision est mauvaise, pas parce qu'il manque d'informations.

### Note honnête sur la Gen 9

Un MVP d'adaptation à la **Gen 9 Random Battle** a été tenté (en s'appuyant sur le même socle PPO, avec un espace d'observation porté à 252 dimensions et des priors statistiques sur les sets). Après quelques sessions d'entraînement et un parcours du ladder pour évaluer le comportement réel du bot, j'ai préféré ne **pas inclure cette branche dans le dépôt**.

La raison est simple : la Gen 9 introduit trop de variables **intentionnellement cachées** par le jeu — talents, objets tenus, type Téracristal, méta-game qui bouge à chaque patch — pour qu'un bot puisse y jouer sérieusement sans soit **inférer** ces variables, soit dépenser un budget de calcul significativement plus large que celui d'un projet individuel.

J'assume donc cette délimitation : le dépôt présente **uniquement** le bot Gen 1, qui est complet, fonctionnel et a atteint l'objectif initial — me servir de sparring partner.

---

## Résultats observés (Gen 1)

Le bot Gen 1 est évalué après chaque phase du curriculum sur 30 à 100 combats contre deux adversaires de référence : **Random** (baseline minimale) et **MaxBasePower** (heuristique qui joue toujours le coup le plus puissant — c'est un adversaire fort mais prévisible).

Les courbes complètes sont dans `evolution_performances.png` (généré par le notebook) et tous les runs sont historisés dans `mlruns/` (consultables via `mlflow ui`).

| Phase | Steps | Winrate vs Random | Winrate vs MaxBasePower |
|---|---|---|---|
| Phase 0 — Imitation (BC sur replays ELO ≥ 1200) | ~5 k | ~50 % | ~20 % |
| Phase 1 — vs Random | 20 k | ~85 % | ~30 % |
| Phase 2 — vs MaxBasePower | 200 k | ~95 % | ~60 % |
| Phase 3 — Self-Play | 1 M | ~98 % | ~75 % |

> Ces valeurs sont indicatives et peuvent varier selon la seed et la durée d'entraînement. Le notebook MLflow conserve la trace exacte de chaque run.

L'intérêt de la décomposition par phase est de **visualiser la progression** : on voit clairement le saut entre la phase 1 (le bot apprend les regles du jeu — attaquer plutôt que rien faire par exeple) et la phase 2 (il commence à comprendre les type matchups et les switches), puis la consolidation en self-play.

---

## Architecture technique

Le projet combine trois couches qu'on retrouve communes aux deux branches :

```
┌──────────────────────────────────────────────────────────┐
│  COUCHE 1 — HEURISTIQUE                                  │
│  Action masking (moves x0, sans PP, setup suicidaire).   │
│  Override sur KO sûr / immunité totale / switch forcé.   │
└─────────────────────┬────────────────────────────────────┘
                      │
┌─────────────────────▼────────────────────────────────────┐
│  COUCHE 2 — FEATURES                                     │
│  Observation enrichie : matchups, menaces, vitesse,      │
│  ressources d'équipe, hazards, météo, priors statistiques│
└─────────────────────┬────────────────────────────────────┘
                      │
┌─────────────────────▼────────────────────────────────────┐
│  COUCHE 3 — REWARD SHAPING                               │
│  Signal d'apprentissage guidé : KO, dégâts, switch       │
│  défensif, setup au bon moment, capitalisation status... │
└──────────────────────────────────────────────────────────┘
```

Détail complet dans [`logique de pensee pour jouer un tour pokebot.md`](./logique%20de%20pensee%20pour%20jouer%20un%20tour%20pokebot.md).

### Stack technique

- **`poke-env`** — interface Python pour Pokémon Showdown (PettingZoo)
- **`Stable-Baselines3` + `sb3-contrib`** — algorithme **MaskablePPO** (PPO avec action masking)
- **`Gymnasium`** — wrapper pour adapter PettingZoo → SB3 single-agent
- **`PyTorch`** — backend du réseau de neurones
- **`MLflow`** — suivi des hyperparamètres, métriques et artefacts
- **Pokémon Showdown (serveur Node.js local)** — moteur de combat utilisé pour le projet (et par beaucoup de joueurs competitifs)]

### Le modèle choisi — MaskablePPO

**PPO (Proximal Policy Optimization)** est un algorithme de Reinforcement Learning publié par OpenAI en 2017. C'est aujourd'hui un des modèles les plus utilisés en RL, parce qu'il offre un bon compromis entre **stabilité d'entraînement**, **simplicité d'implémentation** et **performance**. Il appartient à la famille des *policy gradient methods* : le réseau apprend directement une **politique** (une distribution de probabilités sur les actions à prendre) plutôt que d'estimer la valeur de chaque état individuel.

L'idée centrale de PPO est de **limiter la taille des mises à jour de la politique** d'une itération à l'autre (le "proximal" dans le nom). Si on laissait le modèle changer trop brutalement à chaque mise à jour, il pourrait "oublier" ce qu'il vient d'apprendre. PPO impose une contrainte qui garantit que la nouvelle politique reste proche de l'ancienne, ce qui rend l'entraînement beaucoup plus stable que des méthodes plus anciennes comme REINFORCE ou A3C.

**MaskablePPO** (fourni par `sb3-contrib`) est une extension qui ajoute l'**action masking** : avant chaque décision, on fournit au modèle un masque booléen indiquant quelles actions sont **légales** dans l'état courant. Le réseau ne peut alors choisir que parmi celles-ci.

#### Alternatives écartées

| Algorithme | Pourquoi écarté |
|---|---|
| **DQN** | Mal adapté aux espaces d'actions partiellement valides ; le Q-learning ne gère pas naturellement l'action masking et nécessite beaucoup d'astuces pour fonctionner. |
| **A3C / A2C** | Moins stable que PPO, et le gain de parallélisme natif est annulé par le coût du multi-threading sur la simulation Showdown. |
| **SAC** | Conçu pour les actions continues, pas adapté à notre espace discret (move/switch). |
| **MuZero / AlphaZero** | Idéaux pour des jeux à information parfaite (échecs, Go). Inapplicables ici : Pokémon est à information **partielle** (équipe adverse cachée), et l'apprentissage par MCTS suppose de pouvoir simuler le futur — ce qui est impossible quand on ne connait pas l'état complet. |

### Curriculum d'entraînement

Avant de savoir courir un 100m en moins de 10 secondes, meme Usain Bolt a du apprendre a se servir de ses jambes, a marcher, puis courir, pour finalement savoir comment courir plus vite !
Plutôt que d'entraîner le bot directement contre l'adversaire le plus dur, on **enchaîne des paliers de difficulté croissante** :

```
Phase 0  ─►  Phase 1  ─►  Phase 2  ─►  Phase 3
Imitation    vs Random    vs MaxPower   Self-Play
(replays     (warm-up)    (matchups)    (raffinement)
 humains)
```

Cette approche évite que le bot se bloque dans des minimas locaux trop tôt, et permet une analyse phase par phase de ce qui marche ou non.

---

## Structure du dépôt

```
pokebot/
├── README.md                         ← ce fichier
├── MLOPS_README.md                   ← doc MLflow détaillée
├── requirements.txt                  ← dépendances Python
├── logique de pensee ... .md         ← doc d'architecture conceptuelle
├── .gitignore
│
├── notebooks/
│   └── pokebot.ipynb                 ← pipeline complet (phases 0→3 + éval + démo)
│
├── models/                           ← checkpoints SB3 (.zip)   [ignoré par git]
│   ├── model_imitation.zip
│   ├── model_phase1.zip
│   ├── model_phase2.zip
│   └── model_final.zip
│
├── scripts/                          ← bots de test, utilitaires
│   ├── random_bot.py
│   └── test.py
│
├── mlruns/                           ← historique MLflow         [ignoré par git]
├── evolution_performances.png        ← graphique des winrates
│
└── pokemon-showdown/                 ← serveur de simulation (Node.js)
```

> Les dossiers `models/` et `mlruns/` sont exclus du dépôt git (volumineux). Ils sont régénérés en exécutant le notebook.

---

## Mode d'emploi

### 1. Prérequis

- **Python ≥ 3.10**
- **Node.js ≥ 18** (pour le serveur Pokémon Showdown)
- Un terminal de type bash, PowerShell ou WSL

### 2. Installation

```bash
git clone <url-du-repo> pokebot
cd pokebot

pip install -r requirements.txt

cd pokemon-showdown
npm install
cd ..
```

### 3. Lancer le serveur Showdown local

Dans un terminal dédié :

```bash
cd pokemon-showdown
node pokemon-showdown start --no-security
```

Le serveur écoute sur `http://localhost:8000`. Ouvrir cette URL dans un navigateur pour **regarder les combats en direct** pendant l'entraînement et la démo. (regarder les combats permet souvent de se rendre compte dérreur flagrantes pour pouvoir les corriger ensuite)

### 4. Entraîner le bot Gen 1

Ouvrir le notebook :

```bash
jupyter notebook notebooks/pokebot.ipynb
```

Puis exécuter les cellules dans l'ordre :
1. Imports et définition de l'environnement
2. Wrapper Gymnasium + factory des envs parallèles
3. Configuration MLflow
4. Phase 0 (imitation)
5. Phase 1 (vs Random)
6. Phase 2 (vs MaxBasePower)
7. Phase 3 (Self-Play) — la plus longue (~quelques heures)
8. Évaluation comparée + graphique
9. Démo : 100 matchs RL vs MaxBasePower, visualisable sur localhost:8000

### 5. Consulter les résultats MLflow

Dans un autre terminal, depuis la racine du projet :

```bash
mlflow ui
```

Puis ouvrir `http://localhost:5000` pour explorer les runs, comparer les hyperparamètres et télécharger les checkpoints.

---

## Ce que le projet m'a appris

Au-delà du code et de la documentation, ce projet a confirmé deux choses :

1. **Pokémon n'est pas qu'un jeu d'enfant.** Construire un programme qui joue *décemment* en Gen 1 demande déjà un curriculum learning sérieux, du reward shaping fin et plusieurs millions de steps. Penser que les vrais joueurs Showdown gagnent à pile ou face, c'est passer à côté d'un univers stratégique aussi profond que le jeu de Go pour amateurs.
2. **Toutes les frontières ne sont pas franchissables avec un budget personnel.** La Gen 9 m'a appris à reconnaître quand un problème dépasse les moyens disponibles — ce n'est clairement pas a voir comme un échec, c'est plutot une délimitation honnête des limites actuelles de mes moyens et de mon temps.

Le bot Gen 1 me sert maintenant exactement comme prévu : **un sparring partner toujours disponible**, qui ne se vexe pas quand je perds 10 fois d'affilée et qui m'aide à comprendre *pourquoi* je perds.

A defaut d'etre devenu un stratege hors pair (parce que je perds encore **beaucoup**), j'ai pris enormement de plaisir a construire ce projet, c'est déjà une belle victoire.

