# Logique de pensée pour jouer un tour — PokéBot Hybride Heuristique + RL

---

## Vue d'ensemble de l'architecture

```
┌─────────────────────────────────────────────────────┐
│                   COUCHE 1 : HEURISTIQUE            │
│         (règles dures, décisions non-négociables)   │
└────────────────────┬────────────────────────────────┘
                     │ filtre / masque les actions
┌────────────────────▼────────────────────────────────┐
│                   COUCHE 2 : FEATURES               │
│       (observations enrichies pour le PPO)          │
└────────────────────┬────────────────────────────────┘
                     │ input du réseau de neurones
┌────────────────────▼────────────────────────────────┐
│                   COUCHE 3 : REWARD SHAPING         │
│     (signal d'apprentissage guidé par la strat)     │
└─────────────────────────────────────────────────────┘
```

---

## COUCHE 1 — Heuristique (Action Masking + Rule-Based Filters)

L'idée : **le PPO ne choisit que parmi les actions valides stratégiquement**. On ne lui laisse pas explorer des moves objectivement stupides.

### 1.1 — Action Masking obligatoire (erreurs catastrophiques)

Ces règles bloquent des actions avant même que le PPO ne les considère :

```python
def compute_action_mask(battle: Battle) -> np.ndarray:
    mask = np.ones(len(battle.available_moves) + len(battle.available_switches))

    moves = list(battle.available_moves.values())
    switches = list(battle.available_switches.values())

    for i, move in enumerate(moves):

        # Bloquer les moves x0 (immunité type)
        multiplier = battle.opponent_active_pokemon.damage_multiplier(move)
        if multiplier == 0:
            mask[i] = 0

        # Bloquer les moves sans PP
        if move.current_pp == 0:
            mask[i] = 0

        # Bloquer setup si on est en danger de KO ce tour
        if is_in_ko_range(battle) and move.category == MoveCategory.STATUS:
            mask[i] = 0

    for j, switch in enumerate(switches):

        # Bloquer switch si ça rentre dans un move x2 ou x4 évident
        if will_die_to_hazards(switch, battle):
            mask[len(moves) + j] = 0.3  # pas 0, juste pénalisé
            # (parfois on est forcé de switcher quand même)

    return mask
```

### 1.2 — Heuristiques prioritaires (override du PPO si condition remplie)

Ces règles **court-circuitent** le PPO et agissent directement :

```python
def heuristic_override(battle: Battle) -> Optional[BattleOrder]:

    # RÈGLE 1 — KO disponible → le prendre (presque toujours)
    for move in battle.available_moves.values():
        if can_ko_opponent(move, battle):
            # Exception : si le switch-in probable est encore pire
            if not incoming_threatens_team(battle):
                return battle.create_order(move)

    # RÈGLE 2 — Move prioritaire si on se fait KO sinon
    if is_in_ko_range(battle):
        for move in battle.available_moves.values():
            if move.priority > 0:
                multiplier = battle.opponent_active_pokemon.damage_multiplier(move)
                if multiplier >= 1:
                    return battle.create_order(move)

    # RÈGLE 3 — Switch forcé si immunité complète adverse (x0 sur tous nos moves)
    if all_moves_blocked(battle):
        best_switch = get_best_switch(battle)
        if best_switch:
            return battle.create_order(best_switch)

    # Sinon → laisser le PPO décider
    return None
```

### 1.3 — Heuristique de switch (scoring)

Quand le PPO envisage un switch, on peut lui fournir un score pré-calculé :

```python
def score_switch(pokemon: Pokemon, battle: Battle) -> float:
    score = 0.0
    opp = battle.opponent_active_pokemon

    # Résistance au move probable adverse
    for move in get_probable_moves(opp):
        dmg_mult = pokemon.damage_multiplier(move)
        if dmg_mult == 0:   score += 3.0   # immunité
        elif dmg_mult < 1:  score += 1.5   # résistance
        elif dmg_mult > 1:  score -= 2.0   # faiblesse

    # Notre capacité à menacer l'adversaire
    for move in pokemon.moves.values():
        mult = opp.damage_multiplier(move)
        if mult >= 2:       score += 2.0
        elif mult >= 1:     score += 0.5

    # Coût des hazards à l'entrée
    hazard_dmg = estimate_hazard_damage(pokemon, battle)
    score -= hazard_dmg / pokemon.max_hp * 5

    # HP restants
    score += (pokemon.current_hp_fraction) * 1.0

    return score
```

---

## COUCHE 2 — Feature Engineering (Observations pour le PPO)

Chaque élément du framework stratégique se traduit en **feature numérique** dans l'observation.

### 2.1 — Features de vitesse relative

```python
# Speed tier relatif
speed_ratio = our_speed / opp_speed  # >1 = on est plus rapide
speed_ratio_clipped = np.clip(speed_ratio, 0.2, 5.0)

# Trick Room actif → inverser
if battle.fields.get(Field.TRICK_ROOM):
    speed_ratio_clipped = 1 / speed_ratio_clipped

features["speed_advantage"] = speed_ratio_clipped
features["we_are_faster"] = float(speed_ratio > 1)
```

### 2.2 — Features de menace immédiate

```python
# Dégâts entrants estimés (best move adverse probable)
max_incoming_dmg = estimate_max_incoming_damage(battle)
features["incoming_damage_fraction"] = max_incoming_dmg / our_hp

# Sommes-nous dans la kill range ?
features["in_ko_range"] = float(max_incoming_dmg >= our_current_hp)

# Pouvons-nous KO l'adversaire ?
features["can_ko_opponent"] = float(can_ko_opponent_any_move(battle))

# 2HKO possible ce tour ?
features["can_2hko"] = float(best_damage_fraction(battle) >= 0.5)
```

### 2.3 — Features de type et matchup

```python
# Pour chaque move disponible (jusqu'à 4)
for i, move in enumerate(padded_moves):
    features[f"move_{i}_type_multiplier"] = opp.damage_multiplier(move)
    features[f"move_{i}_is_stab"] = float(move.type in our_pokemon.types)
    features[f"move_{i}_estimated_damage"] = estimate_damage(move, battle) / opp.max_hp
    features[f"move_{i}_priority"] = move.priority / 5  # normalisé

# Pour chaque switch possible (jusqu'à 5)
for j, switch in enumerate(padded_switches):
    features[f"switch_{j}_score"] = score_switch(switch, battle)
    features[f"switch_{j}_hp_fraction"] = switch.current_hp_fraction
    features[f"switch_{j}_has_status"] = float(switch.status is not None)
```

### 2.4 — Features de terrain et météo

```python
# Météo (normalisée)
weather_map = {Weather.SUNNYDAY: 0, Weather.RAINDANCE: 1,
               Weather.SANDSTORM: 2, Weather.HAIL: 3, Weather.NONE: 4}
features["weather"] = weather_map.get(battle.weather, 4) / 4

# Terrain (one-hot)
field_vec = np.zeros(5)
for field in battle.fields:
    if field in TERRAIN_FIELDS:
        field_vec[TERRAIN_INDEX[field]] = 1
features["terrain"] = field_vec

# Screens adverses
features["opp_light_screen"] = float(SideCondition.LIGHTSCREEN in battle.opponent_side_conditions)
features["opp_reflect"] = float(SideCondition.REFLECT in battle.opponent_side_conditions)

# Hazards our side (danger au switch)
features["our_stealth_rock"] = float(SideCondition.STEALTH_ROCK in battle.side_conditions)
features["our_spikes_layers"] = battle.side_conditions.get(SideCondition.SPIKES, 0) / 3
```

### 2.5 — Features de statuts et boosts

```python
# Nos boosts (normalisés entre -1 et 1)
boosts = battle.active_pokemon.boosts
for stat in ["atk", "def", "spa", "spd", "spe"]:
    features[f"our_boost_{stat}"] = boosts.get(stat, 0) / 6

# Boosts adverses visibles
opp_boosts = battle.opponent_active_pokemon.boosts
for stat in ["atk", "def", "spa", "spd", "spe"]:
    features[f"opp_boost_{stat}"] = opp_boosts.get(stat, 0) / 6

# Statuts
STATUS_MAP = {Status.BRN: -0.5, Status.PAR: -0.3, Status.PSN: -0.2,
              Status.TOX: -0.4, Status.SLP: -0.6, Status.FRZ: -0.7}
features["our_status_penalty"] = STATUS_MAP.get(battle.active_pokemon.status, 0)
features["opp_status_penalty"] = STATUS_MAP.get(battle.opponent_active_pokemon.status, 0)
```

### 2.6 — Features de win condition

```python
# Score d'équipe global (ressources restantes)
our_alive = sum(1 for p in battle.team.values() if not p.fainted)
opp_alive = sum(1 for p in battle.opponent_team.values() if not p.fainted)

features["our_team_hp_total"] = sum(
    p.current_hp_fraction for p in battle.team.values() if not p.fainted
) / 6

features["opp_team_hp_total"] = sum(
    p.current_hp_fraction for p in battle.opponent_team.values() if not p.fainted
) / 6

features["pokemon_advantage"] = (our_alive - opp_alive) / 6
features["hp_advantage"] = features["our_team_hp_total"] - features["opp_team_hp_total"]

# Terastal disponible
features["our_tera_available"] = float(battle.can_tera is not None)
features["opp_tera_used"] = float(battle.opponent_active_pokemon.terastallized)
```

---

## COUCHE 3 — Reward Shaping (Signal d'apprentissage stratégique, récompense du modele)

Le reward final = `reward_victoire + Σ reward_intermédiaires`

### 3.1 — Structure récompense

```python
def compute_shaped_reward(battle: Battle, prev_battle: Battle) -> float:
    reward = 0.0

    # ── REWARD TERMINAL ──────────────────────────────────────
    if battle.finished:
        if battle.won:
            reward += 10.0
        else:
            reward -= 10.0
        return reward

    # ── REWARDS POSITIFS ─────────────────────────────────────

    # KO infligé
    opp_fainted_now = count_fainted(battle.opponent_team)
    opp_fainted_before = count_fainted(prev_battle.opponent_team)
    reward += (opp_fainted_now - opp_fainted_before) * 2.0

    # Dégâts infligés (proportionnels)
    dmg_dealt = estimate_damage_dealt(battle, prev_battle)
    reward += dmg_dealt * 1.5

    # Statut infligé à l'adversaire
    if (battle.opponent_active_pokemon.status is not None and
        prev_battle.opponent_active_pokemon.status is None):
        reward += 0.5

    # Setup réussi (boost obtenu wipe instant > build momentum)
    our_boost_delta = sum_boosts(battle.active_pokemon) - sum_boosts(prev_battle.active_pokemon)
    if our_boost_delta > 0 and not was_punished(battle, prev_battle):
        reward += our_boost_delta * 0.3

    # Avantage de terrain (hazards posés, screens actifs)
    reward += hazard_advantage_delta(battle, prev_battle) * 0.4

    # ── MALUS ────────────────────────────────────────────────

    # KO subi
    our_fainted_now = count_fainted(battle.team)
    our_fainted_before = count_fainted(prev_battle.team)
    reward -= (our_fainted_now - our_fainted_before) * 2.0

    # Dégâts reçus
    dmg_taken = estimate_damage_taken(battle, prev_battle)
    reward -= dmg_taken * 1.0

    # Statut reçu
    if (battle.active_pokemon.status is not None and
        prev_battle.active_pokemon.status is None):
        reward -= 0.4

    # Move à 0% d'efficacité joué (x0) → grosse pénalité
    if played_zero_effectiveness_move(battle, prev_battle):
        reward -= 1.5

    # Switch inutile dans un move super efficace
    if switched_into_super_effective(battle, prev_battle):
        reward -= 0.8

    # ── REWARDS STRATÉGIQUES AVANCÉS ─────────────────────────

    # KO alors qu'on aurait pu ne pas le faire → pénalité
    if ko_was_available_but_not_taken(battle, prev_battle):
        reward -= 0.5

    # Avantage de vitesse maintenu
    if maintained_speed_advantage(battle, prev_battle):
        reward += 0.1

    return reward
```

### 3.2 — Reward de momentum

```python
def momentum_reward(battle, prev_battle) -> float:

    current_advantage = (
        features["hp_advantage"] * 2.0 +
        features["pokemon_advantage"] * 3.0 +
        our_total_boost_advantage(battle) * 0.5
    )

    prev_advantage = compute_advantage(prev_battle)

    return (current_advantage - prev_advantage) * 0.3
```

---

## 🔗 Intégration dans le Player poke-env

```python
class HybridPPOPlayer(Gen9EnvSinglePlayer):

    def calc_reward(self, last_battle, current_battle) -> float:
        return compute_shaped_reward(current_battle, last_battle)

    def embed_battle(self, battle: Battle) -> np.ndarray:
        features = {}

        # Toutes les features des étapes 0-5
        features.update(speed_features(battle))
        features.update(threat_features(battle))
        features.update(move_features(battle))
        features.update(terrain_features(battle))
        features.update(status_boost_features(battle))
        features.update(wincon_features(battle))

        return np.array(list(features.values()), dtype=np.float32)

    def action_masks(self) -> np.ndarray:
        return compute_action_mask(self.current_battle)

    def choose_move(self, battle: Battle) -> BattleOrder:

        # COUCHE 1 : Heuristique override ?
        heuristic_move = heuristic_override(battle)
        if heuristic_move is not None:
            return heuristic_move

        # COUCHE 2+3 : PPO avec observations enrichies + action masking
        return super().choose_move(battle)
```

---

## Ordre de priorité des couches au runtime

```
Tour du bot
    │
    ▼
[HEURISTIQUE] ──── KO dispo ?  ──── OUI ──► Jouer le KO
                       │ NON
               En danger KO + prioritaire ? ── OUI ──► Move prioritaire
                       │ NON
               Tous moves bloqués ? ────────── OUI ──► Meilleur switch
                       │ NON
                       ▼
              [ACTION MASK appliqué]
              (retire les moves x0, sans PP, setups suicidaires)
                       │
                       ▼
              [PPO] choisit parmi les actions valides
              (guidé par les features enrichies)
                       │
                       ▼
              [REWARD] calculé après le tour
              (signal shaped pour renforcer la stratégie)
```
