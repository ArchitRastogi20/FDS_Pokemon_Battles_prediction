"""
FINAL INFERENCE SCRIPT - Pokemon Battle Prediction
Generates submission.csv using trained models
Ensemble of 10 fold models for maximum performance
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import json
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURATION
# ============================================================================
CONFIG = {
    'use_ensemble': True,  # Use 10-fold ensemble (RECOMMENDED)
    'n_folds': 10,
    'gpu_id': 0,
    'threshold': 0.5,  # Classification threshold
}



# ============================================================================
# STEP 0:  Run Inference
# ============================================================================
def run_inference(test_features_path, output_path, use_ensemble=True, n_folds=10, threshold=0.5):
    """
    Generate predictions using trained models.
    
    Args:
        test_features_path: Path to test_features.csv
        output_path: Where to save submission.csv
        use_ensemble: Use 10-fold ensemble vs single final model
        n_folds: Number of folds if using ensemble
        threshold: Classification threshold
        
    Returns:
        tuple: (submission DataFrame, probabilities DataFrame)
    """
    import pandas as pd
    import numpy as np
    import xgboost as xgb
    from datetime import datetime
    
    print("="*80)
    print("POKEMON BATTLE PREDICTION - FINAL INFERENCE")
    print("="*80)
    
    # STEP 1: LOAD TEST FEATURES
    print(f"\n[1/4] Loading test features from: {test_features_path}")
    test_df = pd.read_csv(test_features_path)
    print(f"✓ Loaded {len(test_df):,} test samples")
    
    battle_ids = test_df['battle_id'].values
    X_test = test_df.drop(['battle_id'], axis=1)
    print(f"  Features: {X_test.shape[1]}")
    
    # STEP 2: LOAD MODELS
    print(f"\n[2/4] Loading trained models...")
    models = []
    model_names = []
    
    if use_ensemble:
        print(f"  Loading {n_folds} fold models for ensemble...")
        for i in range(1, n_folds + 1):
            model_path = f'model_fold_{i}.json'
            try:
                model = xgb.Booster()
                model.load_model(model_path)
                models.append(model)
                model_names.append(f"Fold {i}")
                print(f"    ✓ Loaded {model_path}")
            except FileNotFoundError:
                print(f"    ✗ Warning: {model_path} not found")
        
        if len(models) == 0:
            print("\n  ⚠ No fold models found! Falling back to final model...")
            use_ensemble = False
    
    if not use_ensemble or len(models) == 0:
        model = xgb.Booster()
        model.load_model('model_final.json')
        models = [model]
        model_names = ["Final Model"]
        print(f"    ✓ Loaded model_final.json")
    
    print(f"\n  Total models loaded: {len(models)}")
    
    # STEP 3: GENERATE PREDICTIONS
    print(f"\n[3/4] Generating predictions...")
    dtest = xgb.DMatrix(X_test)
    
    if len(models) > 1:
        all_predictions = [model.predict(dtest) for model in models]
        predictions_proba = np.mean(all_predictions, axis=0)
    else:
        predictions_proba = models[0].predict(dtest)
    
    predictions_binary = (predictions_proba >= threshold).astype(int)
    
    total = len(predictions_binary)
    wins = predictions_binary.sum()
    losses = total - wins
    
    print(f"\n  Prediction Distribution:")
    print(f"    Predicted Wins:   {wins:5,} ({wins/total*100:5.2f}%)")
    print(f"    Predicted Losses: {losses:5,} ({losses/total*100:5.2f}%)")
    
    # STEP 4: CREATE SUBMISSION
    print(f"\n[4/4] Creating submission file...")
    submission_df = pd.DataFrame({
        'battle_id': battle_ids,
        'player_won': predictions_binary
    })
    submission_df = submission_df.sort_values('battle_id').reset_index(drop=True)
    submission_df.to_csv(output_path, index=False)
    print(f"  ✓ Saved submission to '{output_path}'")
    
    # Save probabilities
    proba_df = pd.DataFrame({
        'battle_id': battle_ids,
        'probability': predictions_proba,
        'prediction': predictions_binary
    }).sort_values('battle_id').reset_index(drop=True)
    
    proba_file = output_path.replace('submission.csv', 'test_probabilities.csv')
    proba_df.to_csv(proba_file, index=False)
    print(f"  ✓ Saved probabilities to '{proba_file}'")
    
    # VALIDATION CHECKS
    print(f"\n{'='*80}")
    print("VALIDATION CHECKS")
    print(f"{'='*80}")
    
    checks = [
        ("Duplicate battle_ids", submission_df['battle_id'].duplicated().sum() == 0),
        ("Missing values", submission_df.isnull().sum().sum() == 0),
        ("Valid predictions", set(submission_df['player_won'].unique()).issubset({0, 1})),
        ("Correct row count", len(submission_df) == len(battle_ids)),
        ("Correct columns", list(submission_df.columns) == ['battle_id', 'player_won'])
    ]
    
    for i, (check_name, passed) in enumerate(checks, 1):
        status = '✓ PASS' if passed else '✗ FAIL'
        print(f"  [{i}/5] {check_name}: {status}")
    
    checks_passed = sum(1 for _, passed in checks if passed)
    print(f"\n  Overall: {checks_passed}/5 checks passed")
    
    print(f"\n{'='*80}")
    print("✓ INFERENCE COMPLETE!")
    print(f"{'='*80}")
    
    return submission_df, proba_df

if __name__ == "__main__":
    # Use default config when running as script
    submission, probabilities = run_inference(
        test_features_path='test_features.csv',
        output_path='submission.csv',
        use_ensemble=CONFIG['use_ensemble'],
        n_folds=CONFIG['n_folds']
    )