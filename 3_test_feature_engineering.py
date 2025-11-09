"""
Test Feature Engineering Script
Applies EXACT SAME feature engineering to test data as training
NO DATA LEAKAGE - doesn't use any training statistics
Creates test_features.csv with 58 features (same as training)
"""

import json
import os
import pandas as pd
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

N_JOBS = 17

print("="*80)
print("TEST FEATURE ENGINEERING")
print("="*80)
print(f"Configuration:")
print(f"  CPU cores: {N_JOBS}")
print(f"  Goal: Extract rich timeline + static features (train-mirrored)")
print(f"  NO data leakage - independent of training data")
print("="*80)

# ============================================================================
# FEATURE FLAGS (Ablation toggles) - must match training
# Load from feature_flags.json if present; otherwise enable all
# ============================================================================
DEFAULT_FEATURE_FLAGS = {
    "early_advantage": True,
    "early_status": True,
    "hp_ahead_share": True,
    "hp_slopes": True,
    "hp_area": True,
    "stab": True,
    "initiative": True,
    "speed_edge_plus": True,
    "team_type_max": True,
    "extremes": True,
    "first_status": True,
    "immobilized": True,
    "switch_faints": True,
    "move_mix": True,
    "type_enhance": True,
}

FEATURE_FLAGS = DEFAULT_FEATURE_FLAGS.copy()
try:
    if os.path.exists('feature_flags.json'):
        with open('feature_flags.json', 'r') as f:
            loaded = json.load(f)
            if isinstance(loaded, dict):
                FEATURE_FLAGS.update({k: bool(v) for k, v in loaded.items() if k in FEATURE_FLAGS})
except Exception:
    pass

# ============================================================================
# TYPE EFFECTIVENESS CHART (Pokemon Gen 1) - SAME AS TRAINING
# ============================================================================
TYPE_CHART = {
    'normal': {'rock': 0.5, 'ghost': 0},
    'fire': {'fire': 0.5, 'water': 0.5, 'grass': 2, 'ice': 2, 'bug': 2, 'rock': 0.5, 'dragon': 0.5},
    'water': {'fire': 2, 'water': 0.5, 'grass': 0.5, 'ground': 2, 'rock': 2, 'dragon': 0.5},
    'electric': {'water': 2, 'electric': 0.5, 'grass': 0.5, 'ground': 0, 'flying': 2, 'dragon': 0.5},
    'grass': {'fire': 0.5, 'water': 2, 'grass': 0.5, 'poison': 0.5, 'ground': 2, 'flying': 0.5, 'bug': 0.5, 'rock': 2, 'dragon': 0.5},
    'ice': {'fire': 0.5, 'water': 0.5, 'grass': 2, 'ice': 0.5, 'ground': 2, 'flying': 2, 'dragon': 2},
    'fighting': {'normal': 2, 'ice': 2, 'poison': 0.5, 'flying': 0.5, 'psychic': 0.5, 'bug': 0.5, 'rock': 2, 'ghost': 0},
    'poison': {'grass': 2, 'poison': 0.5, 'ground': 0.5, 'bug': 2, 'rock': 0.5, 'ghost': 0.5},
    'ground': {'fire': 2, 'electric': 2, 'grass': 0.5, 'poison': 2, 'flying': 0, 'bug': 0.5, 'rock': 2},
    'flying': {'electric': 0.5, 'grass': 2, 'fighting': 2, 'bug': 2, 'rock': 0.5},
    'psychic': {'fighting': 2, 'poison': 2, 'psychic': 0.5},
    'bug': {'fire': 0.5, 'grass': 2, 'fighting': 0.5, 'poison': 2, 'flying': 0.5, 'psychic': 2, 'ghost': 0.5},
    'rock': {'fire': 2, 'ice': 2, 'fighting': 0.5, 'ground': 0.5, 'flying': 2, 'bug': 2},
    'ghost': {'normal': 0, 'psychic': 0, 'ghost': 2},
    'dragon': {'dragon': 2},
}

def get_type_effectiveness(attacker_type, defender_types):
    """Calculate type effectiveness multiplier"""
    if attacker_type == 'notype' or not attacker_type:
        return 1.0
    
    multiplier = 1.0
    for def_type in defender_types:
        if def_type == 'notype' or not def_type:
            continue
        if attacker_type.lower() in TYPE_CHART:
            multiplier *= TYPE_CHART[attacker_type.lower()].get(def_type.lower(), 1.0)
    return multiplier

# ============================================================================
# FEATURE EXTRACTION FUNCTION - IDENTICAL TO TRAINING
# ============================================================================
def extract_features(battle):
    """
    Extract comprehensive features from a single battle
    MUST produce same 58 features as training script!
    """
    features = {}
    
    try:
        p1_team = battle['p1_team_details']
        p2_lead = battle['p2_lead_details']
        timeline = battle['battle_timeline']
        
        # ===== BASIC TEAM STATS =====
        features['p1_mean_hp'] = np.mean([p['base_hp'] for p in p1_team])
        features['p1_mean_atk'] = np.mean([p['base_atk'] for p in p1_team])
        features['p1_mean_def'] = np.mean([p['base_def'] for p in p1_team])
        features['p1_mean_spa'] = np.mean([p['base_spa'] for p in p1_team])
        features['p1_mean_spe'] = np.mean([p['base_spe'] for p in p1_team])
        
        # Stat variance (team diversity)
        features['p1_std_hp'] = np.std([p['base_hp'] for p in p1_team])
        features['p1_std_spe'] = np.std([p['base_spe'] for p in p1_team])
        
        # Min/Max stats
        features['p1_max_spe'] = max([p['base_spe'] for p in p1_team])
        features['p1_min_hp'] = min([p['base_hp'] for p in p1_team])
        
        # ===== OPPONENT LEAD STATS =====
        features['p2_lead_hp'] = p2_lead['base_hp']
        features['p2_lead_atk'] = p2_lead['base_atk']
        features['p2_lead_def'] = p2_lead['base_def']
        features['p2_lead_spa'] = p2_lead['base_spa']
        features['p2_lead_spe'] = p2_lead['base_spe']
        
        # Defensive bulk (interaction features)
        features['p2_lead_bulk'] = p2_lead['base_def'] * p2_lead['base_hp'] / 100
        features['p2_lead_special_bulk'] = p2_lead['base_spa'] * p2_lead['base_hp'] / 100
        
        # ===== LEAD POKEMON FEATURES =====
        p1_lead = p1_team[0]
        features['p1_lead_hp'] = p1_lead['base_hp']
        features['p1_lead_spe'] = p1_lead['base_spe']
        features['p1_lead_spa'] = p1_lead['base_spa']
        features['p1_lead_atk'] = p1_lead['base_atk']
        
        # ===== SPEED ADVANTAGE (VERY IMPORTANT!) =====
        features['speed_advantage'] = p1_lead['base_spe'] - p2_lead['base_spe']
        features['speed_advantage_binary'] = 1 if features['speed_advantage'] > 0 else 0
        features['speed_tier'] = sum([p['base_spe'] > p2_lead['base_spe'] for p in p1_team])
        if FEATURE_FLAGS.get('speed_edge_plus', True):
            features['max_speed_edge'] = max([p['base_spe'] - p2_lead['base_spe'] for p in p1_team])
            features['speed_tier_10'] = sum([p['base_spe'] - p2_lead['base_spe'] > 10 for p in p1_team])
            features['speed_tier_20'] = sum([p['base_spe'] - p2_lead['base_spe'] > 20 for p in p1_team])
        
        # ===== TYPE COVERAGE =====
        p1_types = [t for p in p1_team for t in p['types'] if t != 'notype']
        features['team_type_diversity'] = len(set(p1_types))
        
        # Type advantage for lead matchup
        p1_lead_types = [t for t in p1_lead['types'] if t != 'notype']
        p2_lead_types = [t for t in p2_lead['types'] if t != 'notype']
        
        type_advantages = []
        for p1_type in p1_lead_types:
            type_advantages.append(get_type_effectiveness(p1_type, p2_lead_types))
        features['p1_lead_type_advantage'] = max(type_advantages) if type_advantages else 1.0
        if FEATURE_FLAGS.get('team_type_max', True):
            team_all_types = [t for p in p1_team for t in p.get('types', []) if t and t != 'notype']
            team_max_mult = 1.0
            for t in team_all_types:
                team_max_mult = max(team_max_mult, get_type_effectiveness(t, p2_lead_types))
            features['team_max_type_mult'] = team_max_mult
        
        # ===== BATTLE TIMELINE FEATURES (THE GOLD MINE!) =====
        if len(timeline) > 0:
            # Turn 1 features
            turn1 = timeline[0]
            features['turn1_p1_hp'] = turn1.get('p1_pokemon_state', {}).get('hp_pct', 0)
            features['turn1_p2_hp'] = turn1.get('p2_pokemon_state', {}).get('hp_pct', 0)
            features['turn1_hp_diff'] = features['turn1_p1_hp'] - features['turn1_p2_hp']
            
            # First move details
            if turn1.get('p1_move_details'):
                move = turn1['p1_move_details']
                features['first_move_power'] = move.get('base_power', 0)
                features['first_move_priority'] = move.get('priority', 0)
                features['first_move_category'] = 1 if move.get('category') == 'PHYSICAL' else (2 if move.get('category') == 'SPECIAL' else 0)
            else:
                features['first_move_power'] = 0
                features['first_move_priority'] = 0
                features['first_move_category'] = 0
            
            # Who moved first turn 1
            p1_moved_first = turn1.get('p1_move_details') is not None and turn1.get('p2_move_details') is None
            features['p1_moved_first'] = 1 if p1_moved_first else 0
            if FEATURE_FLAGS.get('initiative', True):
                p1_first_t3 = 0
                p2_first_t3 = 0
                p1_first_t5 = 0
                p2_first_t5 = 0
                upto = min(5, len(timeline))
                for i in range(upto):
                    t = timeline[i]
                    p1m = t.get('p1_move_details')
                    p2m = t.get('p2_move_details')
                    if p1m is not None and p2m is None:
                        if i < 3:
                            p1_first_t3 += 1
                        p1_first_t5 += 1
                    if p2m is not None and p1m is None:
                        if i < 3:
                            p2_first_t3 += 1
                        p2_first_t5 += 1
                features['p1_moved_first_t3'] = p1_first_t3
                features['p1_moved_first_t5'] = p1_first_t5
                features['initiative_net_t5'] = p1_first_t5 - p2_first_t5
        
        # Early game momentum (turns 3, 5, 10)
        for turn_num in [2, 4, 9]:  # Index 2, 4, 9 = Turn 3, 5, 10
            if len(timeline) > turn_num:
                turn = timeline[turn_num]
                p1_hp = turn.get('p1_pokemon_state', {}).get('hp_pct', 0)
                p2_hp = turn.get('p2_pokemon_state', {}).get('hp_pct', 0)
                features[f'turn{turn_num+1}_hp_diff'] = p1_hp - p2_hp
                features[f'turn{turn_num+1}_p1_hp'] = p1_hp
                features[f'turn{turn_num+1}_p2_hp'] = p2_hp
            else:
                features[f'turn{turn_num+1}_hp_diff'] = 0
                features[f'turn{turn_num+1}_p1_hp'] = 0
                features[f'turn{turn_num+1}_p2_hp'] = 0

        # Status condition features (CRITICAL - highest importance!)
        status_counts = {'par': 0, 'slp': 0, 'frz': 0, 'psn': 0, 'brn': 0, 'tox': 0}
        p1_status_counts = {'par': 0, 'slp': 0, 'frz': 0, 'psn': 0, 'brn': 0}
        
        for turn in timeline:
            p2_status = turn.get('p2_pokemon_state', {}).get('status', 'nostatus')
            if p2_status in status_counts:
                status_counts[p2_status] += 1
            
            p1_status = turn.get('p1_pokemon_state', {}).get('status', 'nostatus')
            if p1_status in p1_status_counts:
                p1_status_counts[p1_status] += 1
        
        for status in status_counts:
            features[f'p2_turns_{status}'] = status_counts[status]
        for status in p1_status_counts:
            features[f'p1_turns_{status}'] = p1_status_counts[status]

        # ===== Additional timeline-derived features (mirror training) =====
        p1_hp_series = []
        p2_hp_series = []
        p1_status_series = []
        p2_status_series = []
        p1_moves = []
        p2_moves = []
        p1_names = []
        p2_names = []
        for t in timeline:
            p1s = t.get('p1_pokemon_state', {})
            p2s = t.get('p2_pokemon_state', {})
            p1_hp_series.append(p1s.get('hp_pct', 0) or 0)
            p2_hp_series.append(p2s.get('hp_pct', 0) or 0)
            p1_status_series.append(p1s.get('status', 'nostatus') or 'nostatus')
            p2_status_series.append(p2s.get('status', 'nostatus') or 'nostatus')
            p1_names.append(p1s.get('name', '') or '')
            p2_names.append(p2s.get('name', '') or '')
            m1 = t.get('p1_move_details') or {}
            m2 = t.get('p2_move_details') or {}
            p1_moves.append((m1.get('category'), m1.get('base_power', 0) or 0, m1.get('priority', 0) or 0))
            p2_moves.append((m2.get('category'), m2.get('base_power', 0) or 0, m2.get('priority', 0) or 0))

        # STAB usage (P1 exact, P2 approx for lead)
        if FEATURE_FLAGS.get('stab', True):
            name_to_types = {p.get('name', ''): [t for t in p.get('types', []) if t and t != 'notype'] for p in p1_team}
            p1_stab_total = 0
            p1_stab_t5 = 0
            total_p1_moves = 0
            for i, t in enumerate(timeline):
                m1 = t.get('p1_move_details') or {}
                if m1:
                    total_p1_moves += 1
                    mtype = (m1.get('type') or '').lower()
                    p1n = (t.get('p1_pokemon_state', {}) or {}).get('name', '')
                    ptypes = name_to_types.get(p1n, [])
                    if mtype and any(mtype == ty.lower() for ty in ptypes):
                        p1_stab_total += 1
                        if i < 5:
                            p1_stab_t5 += 1
            features['p1_stab_turns_t5'] = p1_stab_t5
            features['p1_stab_turns'] = p1_stab_total
            features['p1_stab_share'] = (p1_stab_total / total_p1_moves) if total_p1_moves else 0.0
            p2_lead_name = p2_lead.get('name', '')
            p2_stab_lead = 0
            p2_stab_lead_t5 = 0
            for i, t in enumerate(timeline):
                m2 = t.get('p2_move_details') or {}
                if m2:
                    mtype = (m2.get('type') or '').lower()
                    p2n = (t.get('p2_pokemon_state', {}) or {}).get('name', '')
                    if p2n == p2_lead_name and mtype and any(mtype == ty.lower() for ty in p2_lead_types):
                        p2_stab_lead += 1
                        if i < 5:
                            p2_stab_lead_t5 += 1
            features['p2_stab_lead_turns_t5'] = p2_stab_lead_t5
            features['p2_stab_lead_turns'] = p2_stab_lead

        T = len(p1_hp_series)
        hp_diff_series = [(p1_hp_series[i] - p2_hp_series[i]) for i in range(T)] if T else []

        def sum_damage(arr, upto):
            s = 0.0
            for i in range(1, min(upto, len(arr))):
                drop = (arr[i-1] - arr[i])
                if drop > 0:
                    s += drop
            return s

        if FEATURE_FLAGS.get('early_advantage', True):
            p1_damage_taken_t5 = sum_damage(p1_hp_series, 5)
            p1_damage_taken_t10 = sum_damage(p1_hp_series, 10)
            p2_damage_taken_t5 = sum_damage(p2_hp_series, 5)
            p2_damage_taken_t10 = sum_damage(p2_hp_series, 10)
            features['p1_damage_dealt_t5'] = p2_damage_taken_t5
            features['p1_damage_dealt_t10'] = p2_damage_taken_t10
            features['p1_damage_taken_t5'] = p1_damage_taken_t5
            features['p1_damage_taken_t10'] = p1_damage_taken_t10
            features['damage_diff_t5'] = (p2_damage_taken_t5 - p1_damage_taken_t5)
            features['damage_diff_t10'] = (p2_damage_taken_t10 - p1_damage_taken_t10)

        if FEATURE_FLAGS.get('early_status', True):
            for k in ['slp', 'par', 'frz']:
                features[f'p2_turns_{k}_t5'] = sum(1 for i in range(min(5, T)) if p2_status_series[i] == k)
                features[f'p2_turns_{k}_t10'] = sum(1 for i in range(min(10, T)) if p2_status_series[i] == k)
                features[f'p1_turns_{k}_t5'] = sum(1 for i in range(min(5, T)) if p1_status_series[i] == k)
                features[f'p1_turns_{k}_t10'] = sum(1 for i in range(min(10, T)) if p1_status_series[i] == k)

        if FEATURE_FLAGS.get('hp_ahead_share', True):
            if T > 0:
                features['hp_ahead_share_t10'] = float(sum(1 for i in range(min(10, T)) if hp_diff_series[i] > 0)) / float(min(10, T))
                features['hp_ahead_share_t20'] = float(sum(1 for i in range(min(20, T)) if hp_diff_series[i] > 0)) / float(min(20, T))
            else:
                features['hp_ahead_share_t10'] = 0.0
                features['hp_ahead_share_t20'] = 0.0

        if FEATURE_FLAGS.get('extremes', True):
            if hp_diff_series:
                max_val = max(hp_diff_series)
                min_val = min(hp_diff_series)
                features['max_hp_diff'] = max_val
                features['min_hp_diff'] = min_val
                features['argmax_hp_diff'] = (hp_diff_series.index(max_val) + 1)
                features['argmin_hp_diff'] = (hp_diff_series.index(min_val) + 1)
            else:
                features['max_hp_diff'] = 0.0
                features['min_hp_diff'] = 0.0
                features['argmax_hp_diff'] = 0
                features['argmin_hp_diff'] = 0

        # HP diff slopes over windows
        if FEATURE_FLAGS.get('hp_slopes', True):
            def slope_on_window(arr, i0, i1):
                n = len(arr)
                i1 = min(i1, n-1)
                i0 = max(i0, 0)
                if i1 - i0 + 1 >= 2 and n > 1 and i1 >= i0:
                    xs = np.arange(i0, i1+1)
                    ys = np.array(arr[i0:i1+1])
                    try:
                        m, b = np.polyfit(xs, ys, 1)
                        return float(m)
                    except Exception:
                        return 0.0
                return 0.0
            features['hp_diff_slope_1_5'] = slope_on_window(hp_diff_series, 0, 4)
            features['hp_diff_slope_6_10'] = slope_on_window(hp_diff_series, 5, 9)
            features['hp_diff_slope_11_20'] = slope_on_window(hp_diff_series, 10, 19)

        # HP diff area (integrated advantage/disadvantage)
        if FEATURE_FLAGS.get('hp_area', True):
            if hp_diff_series:
                pos_area = float(sum(max(d, 0.0) for d in hp_diff_series))
                neg_area = float(sum(min(d, 0.0) for d in hp_diff_series))
                features['hp_advantage_area_pos'] = pos_area
                features['hp_advantage_area_neg'] = neg_area
                features['hp_advantage_area_abs'] = pos_area - neg_area
            else:
                features['hp_advantage_area_pos'] = 0.0
                features['hp_advantage_area_neg'] = 0.0
                features['hp_advantage_area_abs'] = 0.0

        if FEATURE_FLAGS.get('first_status', True):
            p2_first_turn = 0
            p2_first_type = 'none'
            for i, s in enumerate(p2_status_series):
                if s and s != 'nostatus':
                    p2_first_turn = i + 1
                    p2_first_type = s
                    break
            features['p2_first_status_turn'] = p2_first_turn if p2_first_turn else 0
            status_code_map = {'slp': 1, 'frz': 2, 'par': 3, 'tox': 4, 'psn': 5, 'brn': 6}
            features['p2_first_status_type_code'] = status_code_map.get(p2_first_type, 0)

        if FEATURE_FLAGS.get('immobilized', True):
            p1_immob = 0
            p2_immob = 0
            for i in range(T):
                m1_cat = p1_moves[i][0]
                m2_cat = p2_moves[i][0]
                if p1_status_series[i] in ['slp', 'frz'] and not m1_cat:
                    p1_immob += 1
                if p2_status_series[i] in ['slp', 'frz'] and not m2_cat:
                    p2_immob += 1
            features['p1_immobilized_turns'] = p1_immob
            features['p2_immobilized_turns'] = p2_immob

        if FEATURE_FLAGS.get('switch_faints', True):
            features['p2_pokemon_used'] = len({n for n in p2_names if n})
            def count_faints(hp_series, name_series):
                faints = 0
                for i in range(1, len(hp_series)):
                    if (hp_series[i-1] > 0 and hp_series[i] == 0) or (name_series[i] and name_series[i] != name_series[i-1] and hp_series[i-1] == 0):
                        faints += 1
                return faints
            features['p1_faints'] = count_faints(p1_hp_series, p1_names)
            features['p2_faints'] = count_faints(p2_hp_series, p2_names)

        if FEATURE_FLAGS.get('move_mix', True):
            def move_stats(moves):
                total = sum(1 for c, _, _ in moves if c)
                status = sum(1 for c, _, _ in moves if c == 'STATUS')
                phys_spec = [(bp) for c, bp, _ in moves if c in ('PHYSICAL', 'SPECIAL')]
                prio = sum(1 for _, _, pr in moves if pr and pr > 0)
                share_status = (status / total) if total else 0.0
                mean_bp = (float(np.mean(phys_spec)) if phys_spec else 0.0)
                return share_status, prio, mean_bp
            p1_share_status, p1_prio, p1_mean_bp = move_stats(p1_moves)
            p2_share_status, p2_prio, p2_mean_bp = move_stats(p2_moves)
            features['p1_share_status_moves'] = p1_share_status
            features['p2_share_status_moves'] = p2_share_status
            features['p1_priority_moves'] = p1_prio
            features['p2_priority_moves'] = p2_prio
            features['p1_mean_base_power'] = p1_mean_bp
            features['p2_mean_base_power'] = p2_mean_bp

        # Type enhancements
        if FEATURE_FLAGS.get('type_enhance', True):
            team_se_count = 0
            for p in p1_team:
                for t1 in [tt for tt in p.get('types', []) if tt and tt != 'notype']:
                    mult = get_type_effectiveness(t1, p2_lead_types)
                    if mult >= 2.0:
                        team_se_count += 1
                        break
            features['team_super_effective_count'] = team_se_count
            features['team_has_any_2x'] = 1 if team_se_count > 0 else 0

            max_def_risk = 1.0
            for turn in timeline:
                m2 = turn.get('p2_move_details') or {}
                m2_type = (m2.get('type') or '').lower()
                if m2_type:
                    mult = get_type_effectiveness(m2_type, [t for t in p1_lead['types'] if t != 'notype'])
                    if mult > max_def_risk:
                        max_def_risk = mult
            features['p1_lead_defensive_risk_max'] = max_def_risk
        
        # Stat boost accumulation (turn 10 and 20)
        for turn_idx in [9, 19]:  # Turn 10, 20
            if len(timeline) > turn_idx:
                turn = timeline[turn_idx]
                p1_boosts = turn.get('p1_pokemon_state', {}).get('boosts', {})
                p2_boosts = turn.get('p2_pokemon_state', {}).get('boosts', {})
                
                features[f'p1_total_boosts_turn{turn_idx+1}'] = sum(p1_boosts.values()) if p1_boosts else 0
                features[f'p2_total_boosts_turn{turn_idx+1}'] = sum(p2_boosts.values()) if p2_boosts else 0
            else:
                features[f'p1_total_boosts_turn{turn_idx+1}'] = 0
                features[f'p2_total_boosts_turn{turn_idx+1}'] = 0
        
        # Switch count (strategy indicator)
        p1_pokemon_seen = set()
        for turn in timeline:
            p1_name = turn.get('p1_pokemon_state', {}).get('name', '')
            if p1_name:
                p1_pokemon_seen.add(p1_name)
        features['p1_pokemon_used'] = len(p1_pokemon_seen)
        
        # Momentum shift (HP differential change)
        if len(timeline) >= 15:
            hp_diff_turn5 = features.get('turn5_hp_diff', 0)
            turn15 = timeline[14]
            hp_diff_turn15 = turn15.get('p1_pokemon_state', {}).get('hp_pct', 0) - turn15.get('p2_pokemon_state', {}).get('hp_pct', 0)
            features['momentum_turn5_15'] = hp_diff_turn15 - hp_diff_turn5
        else:
            features['momentum_turn5_15'] = 0
        
        # ===== BATTLE ID (NO TARGET - THIS IS TEST DATA!) =====
        features['battle_id'] = battle['battle_id']
        
    except Exception as e:
        print(f"Error processing battle {battle.get('battle_id', 'unknown')}: {e}")
        # Return default features on error
        features = {f'feature_{i}': 0 for i in range(58)}
        features['battle_id'] = battle.get('battle_id', -1)
    
    return features

# ============================================================================
# MAIN PROCESSING
# ============================================================================
def main():
    print(f"\n[1/3] Loading test data...")
    
    test_data = []
    test_file = 'fds-pokemon-battles-prediction-2025/test.jsonl'
    
    try:
        with open(test_file, 'r') as f:
            for line in f:
                test_data.append(json.loads(line))
        
print(f"Loaded {len(test_data):,} test battles")
    
    except FileNotFoundError:
        print(f"ERROR: Could not find test file at '{test_file}'")
        print(f"  Please ensure the dataset folder 'fds-pokemon-battles-prediction-2025/' exists at repo root.")
        return None
    
    # Extract features in parallel
    print(f"\n[2/3] Extracting features using {N_JOBS} cores...")
    print(f"  This will take ~5-10 seconds...")
    
    with Pool(N_JOBS) as pool:
        features_list = list(tqdm(
            pool.imap(extract_features, test_data),
            total=len(test_data),
            desc="Processing battles"
        ))
    
    # Create DataFrame
    print(f"\n[3/3] Creating DataFrame and saving...")
    test_df = pd.DataFrame(features_list)
    
    # Ensure no NaN values
    test_df = test_df.fillna(0)
    
    # Optionally align to training feature names if available
    aligned = False
    try:
        with open('feature_names.json', 'r') as f:
            train_feature_names = json.load(f)
        # Ensure we have all train columns; fill missing with 0 and drop extras
        cols_to_use = [c for c in train_feature_names if c in test_df.columns]
        missing = [c for c in train_feature_names if c not in test_df.columns]
        extras = [c for c in test_df.columns if c not in train_feature_names + ['battle_id']]
        for m in missing:
            test_df[m] = 0
        test_df = test_df[['battle_id'] + train_feature_names]
        aligned = True
        if missing:
            print(f"\n  Note: Added {len(missing)} missing columns from training (filled with 0).")
        if extras:
            print(f"  Note: Dropped {len(extras)} extra columns not in training features.")
    except Exception:
        pass

    # Report feature count
    n_features = len(test_df.columns) - 1  # Exclude battle_id
    print(f"\n  Feature Summary:")
    print(f"    Total columns:   {len(test_df.columns)}")
    print(f"    Features:        {n_features}")
    print(f"    battle_id:       1")
    print(f"    Aligned to train: {'YES' if aligned else 'NO'}")

    print("\n  Feature flags in use:")
    for k in sorted(FEATURE_FLAGS.keys()):
        print(f"    {k}: {FEATURE_FLAGS[k]}")
    
    # Data quality checks
    null_count = test_df.isnull().sum().sum()
    inf_count = np.isinf(test_df.values).sum()
    print(f"\n  Data Quality:")
    print(f"    Null values: {null_count:6} {'OK' if null_count==0 else 'NOT OK'}")
    print(f"    Inf values:  {inf_count:6} {'OK' if inf_count==0 else 'NOT OK'}")
    
    # Save to CSV
    output_file = 'test_features.csv'
    test_df.to_csv(output_file, index=False)
    
    print(f"\n{'='*80}")
    print("TEST FEATURE ENGINEERING COMPLETE!")
    print(f"{'='*80}")
    print(f"Output file:     {output_file}")
    print(f"Total samples:   {len(test_df):,}")
    print(f"Total features:  {n_features}")
    print(f"File size:       {test_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    print(f"\nFeature Summary (first 15):")
    print(f"  {test_df.columns.tolist()[:15]}")
    
    print(f"\nSample Statistics:")
    print(test_df.describe().iloc[:, :5].to_string())
    
    print(f"\n{'='*80}")
    print("READY FOR INFERENCE!")
    print(f"{'='*80}")
    print(f"Next step: python 4_inference.py")
    print(f"{'='*80}\n")
    
    return test_df

if __name__ == "__main__":
    df = main()
