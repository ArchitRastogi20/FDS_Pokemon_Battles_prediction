"""
Test Feature Engineering Script
Applies EXACT SAME feature engineering to test data as training
NO DATA LEAKAGE - doesn't use any training statistics
Creates test_features.csv with 58 features (same as training)
"""

import json
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
print(f"  Goal: Extract 58 features from test battles")
print(f"  NO data leakage - independent of training data")
print("="*80)

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
    test_file = 'test.jsonl'
    
    try:
        with open(test_file, 'r') as f:
            for line in f:
                test_data.append(json.loads(line))
        
        print(f"✓ Loaded {len(test_data):,} test battles")
    
    except FileNotFoundError:
        print(f"✗ ERROR: Could not find test file at '{test_file}'")
        print(f"  Please ensure competition data is in '../input/fds-pokemon-battles-prediction-2025/'")
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
    
    # Verify feature count
    n_features = len(test_df.columns) - 1  # Exclude battle_id
    print(f"\n  Feature Verification:")
    print(f"    Total columns:   {len(test_df.columns)}")
    print(f"    Features:        {n_features}")
    print(f"    battle_id:       1")
    print(f"    Expected:        58 features + 1 battle_id = 59 columns")
    
    if n_features != 58:
        print(f"\n  ⚠ WARNING: Expected 58 features but got {n_features}!")
        print(f"  This may cause issues during inference!")
    else:
        print(f"\n  ✓ Feature count matches training! (58 features)")
    
    # Data quality checks
    null_count = test_df.isnull().sum().sum()
    inf_count = np.isinf(test_df.values).sum()
    print(f"\n  Data Quality:")
    print(f"    Null values: {null_count:6} {'✓' if null_count==0 else '✗'}")
    print(f"    Inf values:  {inf_count:6} {'✓' if inf_count==0 else '✗'}")
    
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