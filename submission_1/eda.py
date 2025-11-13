import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from multiprocessing import Pool, cpu_count
from functools import partial
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Use 17 cores
N_JOBS = 17

print("="*70)
print("POKEMON BATTLE PREDICTION - PARALLEL EDA (TRAINING DATA ONLY)")
print(f"Using {N_JOBS} CPU cores")
print("="*70)

# ============================================================================
# 1. LOAD TRAINING DATA (Sequential - I/O bound)
# ============================================================================
print("\n[1/8] Loading training data...")
train_data = []
train_file = 'train.jsonl'

with open(train_file, 'r') as f:
    for line in f:
        train_data.append(json.loads(line))

print(f"✓ Loaded {len(train_data)} battles")

# ============================================================================
# PARALLEL HELPER FUNCTIONS
# ============================================================================

def extract_battle_length(battle):
    """Extract battle length"""
    return len(battle['battle_timeline'])

def extract_p1_lead(battle):
    """Extract P1 lead Pokemon"""
    return battle['p1_team_details'][0]['name']

def extract_p2_lead(battle):
    """Extract P2 lead Pokemon"""
    return battle['p2_lead_details']['name']

def extract_p1_types(battle):
    """Extract all types from P1 team"""
    types = []
    for pokemon in battle['p1_team_details']:
        types.extend([t for t in pokemon['types'] if t != 'notype'])
    return types

def extract_team_stats(battle):
    """Extract team stats and outcome"""
    team_stats = {
        'hp': np.mean([p['base_hp'] for p in battle['p1_team_details']]),
        'atk': np.mean([p['base_atk'] for p in battle['p1_team_details']]),
        'def': np.mean([p['base_def'] for p in battle['p1_team_details']]),
        'spa': np.mean([p['base_spa'] for p in battle['p1_team_details']]),
        'spd': np.mean([p['base_spd'] for p in battle['p1_team_details']]),
        'spe': np.mean([p['base_spe'] for p in battle['p1_team_details']]),
        'won': battle['player_won']
    }
    return team_stats

def extract_speed_advantage(battle):
    """Calculate speed advantage"""
    p1_lead_speed = battle['p1_team_details'][0]['base_spe']
    p2_lead_speed = battle['p2_lead_details']['base_spe']
    return {
        'speed_diff': p1_lead_speed - p2_lead_speed,
        'player_won': battle['player_won']
    }

def extract_first_turn_move(battle):
    """Extract first turn move info"""
    if battle['battle_timeline']:
        first_turn = battle['battle_timeline'][0]
        if first_turn.get('p1_move_details'):
            return {
                'move': first_turn['p1_move_details']['name'],
                'category': first_turn['p1_move_details']['category'],
                'player_won': battle['player_won']
            }
    return None

def extract_status_conditions(battle):
    """Extract all status conditions inflicted"""
    statuses = []
    for turn in battle['battle_timeline']:
        if turn.get('p2_pokemon_state'):
            status = turn['p2_pokemon_state'].get('status', 'nostatus')
            if status not in ['nostatus', '']:
                statuses.append(status)
    return statuses

def extract_simple_features(battle):
    """Extract simple features for correlation analysis"""
    p1_team = battle['p1_team_details']
    p2_lead = battle['p2_lead_details']
    
    features = {
        'p1_mean_hp': np.mean([p['base_hp'] for p in p1_team]),
        'p1_mean_atk': np.mean([p['base_atk'] for p in p1_team]),
        'p1_mean_def': np.mean([p['base_def'] for p in p1_team]),
        'p1_mean_spa': np.mean([p['base_spa'] for p in p1_team]),
        'p1_mean_spd': np.mean([p['base_spd'] for p in p1_team]),
        'p1_mean_spe': np.mean([p['base_spe'] for p in p1_team]),
        'p2_lead_hp': p2_lead['base_hp'],
        'p2_lead_atk': p2_lead['base_atk'],
        'p2_lead_def': p2_lead['base_def'],
        'p2_lead_spa': p2_lead['base_spa'],
        'p2_lead_spd': p2_lead['base_spd'],
        'p2_lead_spe': p2_lead['base_spe'],
        'player_won': int(battle['player_won'])
    }
    return features

# ============================================================================
# 2. TARGET VARIABLE ANALYSIS
# ============================================================================
print("\n[2/8] Analyzing target variable...")
wins = sum(1 for b in train_data if b['player_won'])
losses = len(train_data) - wins

print(f"\nTarget Distribution:")
print(f"  Player Wins:   {wins:,} ({wins/len(train_data)*100:.2f}%)")
print(f"  Player Losses: {losses:,} ({losses/len(train_data)*100:.2f}%)")
print(f"  Balance Ratio: {wins/losses:.3f}")

# ============================================================================
# 3. BASIC STATISTICS (PARALLEL)
# ============================================================================
print("\n[3/8] Calculating battle length statistics (parallel)...")

with Pool(N_JOBS) as pool:
    battle_lengths = pool.map(extract_battle_length, train_data)

print(f"\nBattle Length Statistics:")
print(f"  Mean turns:   {np.mean(battle_lengths):.2f}")
print(f"  Median turns: {np.median(battle_lengths):.2f}")
print(f"  Min turns:    {np.min(battle_lengths)}")
print(f"  Max turns:    {np.max(battle_lengths)}")
print(f"  Std dev:      {np.std(battle_lengths):.2f}")

# ============================================================================
# 4. POKEMON SPECIES ANALYSIS (PARALLEL)
# ============================================================================
print("\n[4/8] Analyzing Pokemon species (parallel)...")

with Pool(N_JOBS) as pool:
    p1_leads = pool.map(extract_p1_lead, train_data)
    p2_leads = pool.map(extract_p2_lead, train_data)

p1_lead_counter = Counter(p1_leads)
p2_lead_counter = Counter(p2_leads)

print(f"\nTop 10 P1 Lead Pokemon:")
for pokemon, count in p1_lead_counter.most_common(10):
    print(f"  {pokemon:15} - {count:4} battles ({count/len(train_data)*100:.1f}%)")

print(f"\nTop 10 P2 Lead Pokemon:")
for pokemon, count in p2_lead_counter.most_common(10):
    print(f"  {pokemon:15} - {count:4} battles ({count/len(train_data)*100:.1f}%)")

# ============================================================================
# 5. TYPE ANALYSIS (PARALLEL)
# ============================================================================
print("\n[5/8] Analyzing Pokemon types (parallel)...")

with Pool(N_JOBS) as pool:
    all_type_lists = pool.map(extract_p1_types, train_data)

all_p1_types = [t for sublist in all_type_lists for t in sublist]
type_counter = Counter(all_p1_types)

print(f"\nTop 10 Types in P1 Teams:")
for type_name, count in type_counter.most_common(10):
    print(f"  {type_name:12} - {count:5} occurrences")

# ============================================================================
# 6. STAT ANALYSIS (PARALLEL)
# ============================================================================
print("\n[6/8] Analyzing base stats (parallel)...")

with Pool(N_JOBS) as pool:
    all_team_stats = pool.map(extract_team_stats, train_data)

win_stats = {'hp': [], 'atk': [], 'def': [], 'spa': [], 'spd': [], 'spe': []}
loss_stats = {'hp': [], 'atk': [], 'def': [], 'spa': [], 'spd': [], 'spe': []}

for stats in all_team_stats:
    stats_dict = win_stats if stats['won'] else loss_stats
    for stat in ['hp', 'atk', 'def', 'spa', 'spd', 'spe']:
        stats_dict[stat].append(stats[stat])

print(f"\nAverage Team Stats (P1):")
print(f"{'Stat':<6} {'Wins':>10} {'Losses':>10} {'Difference':>12}")
print("-" * 42)
for stat in ['hp', 'atk', 'def', 'spa', 'spd', 'spe']:
    win_avg = np.mean(win_stats[stat])
    loss_avg = np.mean(loss_stats[stat])
    diff = win_avg - loss_avg
    print(f"{stat.upper():<6} {win_avg:10.2f} {loss_avg:10.2f} {diff:+12.2f}")

# ============================================================================
# 7. SPEED ADVANTAGE ANALYSIS (PARALLEL)
# ============================================================================
print("\n[7/8] Analyzing speed advantage (parallel)...")

with Pool(N_JOBS) as pool:
    speed_advantages = pool.map(extract_speed_advantage, train_data)

speed_df = pd.DataFrame(speed_advantages)
print(f"\nSpeed Advantage Impact:")
print(f"  Win rate when faster:  {speed_df[speed_df['speed_diff'] > 0]['player_won'].mean()*100:.2f}%")
print(f"  Win rate when slower:  {speed_df[speed_df['speed_diff'] < 0]['player_won'].mean()*100:.2f}%")
print(f"  Win rate when tied:    {speed_df[speed_df['speed_diff'] == 0]['player_won'].mean()*100:.2f}%")

# ============================================================================
# 8. BATTLE TIMELINE FEATURES (PARALLEL)
# ============================================================================
print("\n[8/8] Analyzing battle timeline patterns (parallel)...")

# First turn moves
with Pool(N_JOBS) as pool:
    first_turn_moves = pool.map(extract_first_turn_move, train_data)
    first_turn_moves = [m for m in first_turn_moves if m is not None]

if first_turn_moves:
    ft_df = pd.DataFrame(first_turn_moves)
    move_counter = Counter([m['move'] for m in first_turn_moves])
    
    print(f"\nTop 10 First Turn Moves (P1):")
    for move, count in move_counter.most_common(10):
        move_data = ft_df[ft_df['move'] == move]
        win_rate = move_data['player_won'].mean() * 100
        print(f"  {move:20} - {count:4} uses (Win rate: {win_rate:.1f}%)")

# Status conditions
with Pool(N_JOBS) as pool:
    all_status_lists = pool.map(extract_status_conditions, train_data)

all_statuses = [s for sublist in all_status_lists for s in sublist]
status_counts = Counter(all_statuses)

print(f"\nStatus Conditions Inflicted on Opponent:")
for status, count in sorted(status_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"  {status.upper():5} - {count:6} occurrences")

print("\n" + "="*70)
print("EDA COMPLETE - Ready for Feature Engineering!")
print("="*70)

# ============================================================================
# CORRELATION ANALYSIS (PARALLEL)
# ============================================================================
print("\n\n[BONUS] Creating correlation heatmap (parallel)...")

with Pool(N_JOBS) as pool:
    simple_features = pool.map(extract_simple_features, train_data)

corr_df = pd.DataFrame(simple_features)

# Calculate correlation matrix
corr_matrix = corr_df.corr()

# Plot heatmap
plt.figure(figsize=(14, 10))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Feature Correlation Heatmap (Baseline Features)', fontsize=16, pad=20)
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=150, bbox_inches='tight')
print("✓ Saved correlation heatmap to 'correlation_heatmap.png'")

# Identify highly correlated features
print("\nHighly Correlated Feature Pairs (|r| > 0.7, excluding target):")
high_corr = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if abs(corr_matrix.iloc[i, j]) > 0.7:
            col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
            if col1 != 'player_won' and col2 != 'player_won':
                high_corr.append((col1, col2, corr_matrix.iloc[i, j]))

if high_corr:
    for feat1, feat2, corr_val in sorted(high_corr, key=lambda x: abs(x[2]), reverse=True):
        print(f"  {feat1:15} <-> {feat2:15}: {corr_val:+.3f}")
else:
    print("  No highly correlated pairs found!")

# Correlation with target
print("\nFeature Correlation with Target (player_won):")
target_corr = corr_matrix['player_won'].drop('player_won').sort_values(ascending=False)
for feat, corr_val in target_corr.items():
    print(f"  {feat:15}: {corr_val:+.4f}")

print("\n✓ Parallel EDA script complete! Review results and decide next steps.")
print(f"✓ Processing time significantly reduced using {N_JOBS} cores!")