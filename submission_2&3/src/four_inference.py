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
def main():
    print("="*80)
    print("POKEMON BATTLE PREDICTION - FINAL INFERENCE")
    print("="*80)

    # ============================================================================
    # CONFIGURATION
    # ============================================================================
    CONFIG = {
        'use_ensemble': True,  # Use K-fold ensemble (RECOMMENDED)
        'n_folds': 10,
        'gpu_id': 0,
        'threshold': 0.5,  # Default threshold (overridden by best_threshold.json if present)
    }

    print(f"\nConfiguration:")
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")
    print("="*80)

    # ============================================================================
    # STEP 1: LOAD TEST FEATURES
    # ============================================================================
    print(f"\n[1/4] Loading test features...")

    try:
        test_df = pd.read_csv('submission_2/model_config_and_feature_selection/test_features.csv')
        print(f"Loaded {len(test_df):,} test samples with {len(test_df.columns)} columns")
        
        # Separate features and battle_id
        battle_ids = test_df['battle_id'].values
        X_test = test_df.drop(['battle_id'], axis=1)
        
        print(f"  Features: {X_test.shape[1]}")
        print(f"  Battle IDs range: {battle_ids.min()} to {battle_ids.max()}")
        
        # Verify data quality
        null_count = X_test.isnull().sum().sum()
        inf_count = np.isinf(X_test.values).sum()
        print(f"  Data quality: Nulls={null_count} {'OK' if null_count==0 else 'NOT OK'} | Infs={inf_count} {'OK' if inf_count==0 else 'NOT OK'}")
        
    except FileNotFoundError:
        print("ERROR: test_features.csv not found!")
        print("  Please run: python 3_test_feature_engineering.py first")
        exit(1)

    # ============================================================================
    # STEP 2: LOAD TRAINED MODELS
    # ============================================================================
    print(f"\n[2/4] Loading trained models...")

    models = []
    model_names = []

    if CONFIG['use_ensemble']:
        print(f"  Loading {CONFIG['n_folds']} fold models for ensemble...")
        
        for i in range(1, CONFIG['n_folds'] + 1):
            model_path = f'model_fold_{i}.json'
            try:
                model = xgb.Booster()
                model.load_model(model_path)
                models.append(model)
                model_names.append(f"Fold {i}")
                print(f"    Loaded {model_path}")
            except FileNotFoundError:
                print(f"    Warning: {model_path} not found, skipping...")
        
        if len(models) == 0:
            print("\n  No fold models found. Falling back to final model.")
            CONFIG['use_ensemble'] = False
        else:
            print(f"  Successfully loaded {len(models)}/{CONFIG['n_folds']} models")

    if not CONFIG['use_ensemble'] or len(models) == 0:
        print("  Loading final model...")
        try:
            model = xgb.Booster()
            model.load_model('submission_2/model_config_and_feature_selection/model_final.json')
            models = [model]
            model_names = ["Final Model"]
            print(f"    Loaded model_final.json")
        except FileNotFoundError:
            print("    ERROR: model_final.json not found!")
            print("  Please run: python 2_train_model.py first")
            exit(1)

    print(f"\n  Total models loaded: {len(models)}")
    print(f"  Ensemble mode: {'ON' if len(models) > 1 else 'OFF'}")

    # ============================================================================
    # STEP 3: GENERATE PREDICTIONS
    # ============================================================================
    print(f"\n[3/4] Generating predictions...")
    print("-"*80)

    # Create DMatrix
    dtest = xgb.DMatrix(X_test)

    # Try loading optimal threshold (prefer CV mean, fallback to holdout)
    loaded_threshold = False
    try:
        with open('submission_2/model_config_and_feature_selection/best_threshold_cv.json', 'r') as f:
            thcv = json.load(f)
            if isinstance(thcv, dict) and 'threshold_mean' in thcv:
                CONFIG['threshold'] = float(thcv['threshold_mean'])
                print(f"  Threshold (CV-mean): {CONFIG['threshold']:.4f}")
                loaded_threshold = True
    except Exception:
        pass
    if not loaded_threshold:
        try:
            with open('submission_2/model_config_and_feature_selection/best_threshold.json', 'r') as f:
                th = json.load(f)
                if isinstance(th, dict) and 'threshold' in th:
                    CONFIG['threshold'] = float(th['threshold'])
                    print(f"  Threshold (holdout): {CONFIG['threshold']:.4f}")
        except Exception:
            pass

    if len(models) > 1:
        print(f"  Using ensemble of {len(models)} models...")
        
        # Prepare AUC-based weights if fold_scores.csv exists
        weights = None
        try:
            fs = pd.read_csv('submission_2/model_config_and_feature_selection/fold_scores.csv')
            aucs = fs.sort_values('fold')['auc'].values[:len(models)]
            w = np.maximum(aucs, 1e-6)
            weights = w / w.sum()
            print(f"  AUC-weighted ensemble enabled")
        except Exception:
            pass

        # Collect predictions from all models
        all_predictions = []
        for i, (model, name) in enumerate(zip(models, model_names), 1):
            preds = model.predict(dtest)
            all_predictions.append(preds)
            print(f"    {name:12} - Mean prob: {preds.mean():.4f} | Std: {preds.std():.4f}")
        
        # Average predictions (ensemble)
        if weights is not None and len(weights) == len(all_predictions):
            predictions_proba = np.average(np.vstack(all_predictions), axis=0, weights=weights)
        else:
            predictions_proba = np.mean(all_predictions, axis=0)
        predictions_std = np.std(all_predictions, axis=0)
        
        print(f"\n  Ensemble Statistics:")
        print(f"    Mean probability:     {predictions_proba.mean():.4f}")
        print(f"    Std of probabilities: {predictions_proba.std():.4f}")
        print(f"    Mean disagreement:    {predictions_std.mean():.4f}")
        print(f"    Max disagreement:     {predictions_std.max():.4f}")
        
    else:
        print(f"  Using single model...")
        model = models[0]
        predictions_proba = model.predict(dtest)
        predictions_std = np.zeros_like(predictions_proba)
        print(f"    Mean probability: {predictions_proba.mean():.4f}")

    # Convert probabilities to binary predictions
    predictions_binary = (predictions_proba >= CONFIG['threshold']).astype(int)

    print(f"\n  Prediction Distribution:")
    total = len(predictions_binary)
    wins = predictions_binary.sum()
    losses = total - wins
    print(f"    Predicted Wins:   {wins:5,} ({wins/total*100:5.2f}%)")
    print(f"    Predicted Losses: {losses:5,} ({losses/total*100:5.2f}%)")

    # Confidence analysis
    high_conf = np.sum((predictions_proba < 0.2) | (predictions_proba > 0.8))
    med_conf = np.sum((predictions_proba >= 0.2) & (predictions_proba < 0.4) | 
                    (predictions_proba >= 0.6) & (predictions_proba <= 0.8))
    low_conf = np.sum((predictions_proba >= 0.4) & (predictions_proba <= 0.6))

    print(f"\n  Confidence Analysis:")
    print(f"    High (p<0.2 or p>0.8):    {high_conf:5,} ({high_conf/total*100:5.2f}%)")
    print(f"    Medium (0.2-0.4, 0.6-0.8): {med_conf:5,} ({med_conf/total*100:5.2f}%)")
    print(f"    Low (0.4-0.6):            {low_conf:5,} ({low_conf/total*100:5.2f}%)")

    # ============================================================================
    # STEP 4: CREATE SUBMISSION FILE
    # ============================================================================
    print(f"\n[4/4] Creating submission file...")
    print("-"*80)

    submission_df = pd.DataFrame({
        'battle_id': battle_ids,
        'player_won': predictions_binary
    })

    # Ensure correct data types
    submission_df['battle_id'] = submission_df['battle_id'].astype(int)
    submission_df['player_won'] = submission_df['player_won'].astype(int)

    # Sort by battle_id
    submission_df = submission_df.sort_values('battle_id').reset_index(drop=True)

    # Save to CSV
    output_file = 'submission_2/submission_csv/submission.csv'
    submission_df.to_csv(output_file, index=False)

    print(f"  Saved submission to '{output_file}'")
    print(f"\n  Submission Preview (first 10 rows):")
    print(submission_df.head(10).to_string(index=False))
    print(f"\n  Submission Preview (last 10 rows):")
    print(submission_df.tail(10).to_string(index=False))

    # ============================================================================
    # VALIDATION CHECKS
    # ============================================================================
    print(f"\n{'='*80}")
    print("VALIDATION CHECKS")
    print(f"{'='*80}")

    checks_passed = 0
    total_checks = 6

    # Check 1: Duplicates
    duplicates = submission_df['battle_id'].duplicated().sum()
    status = 'PASS' if duplicates == 0 else 'FAIL'
    print(f"  [1/6] Duplicate battle_ids: {duplicates:6} {status}")
    if duplicates == 0: checks_passed += 1

    # Check 2: Missing values
    missing = submission_df.isnull().sum().sum()
    status = 'PASS' if missing == 0 else 'FAIL'
    print(f"  [2/6] Missing values:       {missing:6} {status}")
    if missing == 0: checks_passed += 1

    # Check 3: Battle ID range
    expected_min = 0
    expected_max = len(battle_ids) - 1
    actual_min = submission_df['battle_id'].min()
    actual_max = submission_df['battle_id'].max()
    correct_range = (actual_min == expected_min and actual_max == expected_max)
    status = 'PASS' if correct_range else 'FAIL'
    print(f"  [3/6] Battle ID range:      {actual_min:6} to {actual_max:6} {status}")
    if correct_range: checks_passed += 1

    # Check 4: Prediction values
    unique_preds = set(submission_df['player_won'].unique())
    correct_values = (unique_preds.issubset({0, 1}))
    status = 'PASS' if correct_values else 'FAIL'
    print(f"  [4/6] Valid predictions:    {sorted(unique_preds)} {status}")
    if correct_values: checks_passed += 1

    # Check 5: Row count
    correct_count = len(submission_df) == len(battle_ids)
    status = 'PASS' if correct_count else 'FAIL'
    print(f"  [5/6] Row count:            {len(submission_df):6} {status}")
    if correct_count: checks_passed += 1

    # Check 6: Column names
    expected_cols = ['battle_id', 'player_won']
    correct_cols = list(submission_df.columns) == expected_cols
    status = 'PASS' if correct_cols else 'FAIL'
    print(f"  [6/6] Column names:         {list(submission_df.columns)} {status}")
    if correct_cols: checks_passed += 1

    print(f"\n  Overall: {checks_passed}/{total_checks} checks passed")

    # ============================================================================
    # SAVE ADDITIONAL FILES
    # ============================================================================
    print(f"\n{'='*80}")
    print("SAVING ADDITIONAL FILES")
    print(f"{'='*80}")

    # Save probabilities
    proba_df = pd.DataFrame({
        'battle_id': battle_ids,
        'probability': predictions_proba,
        'prediction': predictions_binary
    })
    if len(models) > 1:
        proba_df['std'] = predictions_std

    proba_df = proba_df.sort_values('battle_id').reset_index(drop=True)
    proba_file = 'submission_2/model_config_and_feature_selection/test_probabilities.csv'
    proba_df.to_csv(proba_file, index=False)
    print(f"  Saved probabilities to '{proba_file}'")

    # Save inference summary
    summary = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'test_samples': len(battle_ids),
        'models_used': len(models),
        'model_names': model_names,
        'ensemble': len(models) > 1,
        'threshold': CONFIG['threshold'],
        'predictions': {
            'wins': int(wins),
            'losses': int(losses),
            'win_rate': float(wins/total)
        },
        'confidence': {
            'high': int(high_conf),
            'medium': int(med_conf),
            'low': int(low_conf)
        },
        'statistics': {
            'mean_probability': float(predictions_proba.mean()),
            'std_probability': float(predictions_proba.std()),
            'min_probability': float(predictions_proba.min()),
            'max_probability': float(predictions_proba.max())
        },
        'validation_checks': {
            'passed': checks_passed,
            'total': total_checks
        }
    }

    summary_file = 'submission_2/model_config_and_feature_selection/inference_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=4)
    print(f"  Saved inference summary to '{summary_file}'")

    # ============================================================================
    # FINAL SUMMARY
    # ============================================================================
    print(f"\n{'='*80}")
    print("INFERENCE COMPLETE")
    print(f"{'='*80}")
    print(f"\nSubmission Details:")
    print(f"  Output file:        {output_file}")
    print(f"  Total predictions:  {len(predictions_binary):,}")
    print(f"  Predicted win rate: {predictions_binary.mean()*100:.2f}%")
    print(f"  Mean probability:   {predictions_proba.mean():.4f}")
    print(f"  Models used:        {len(models)} {'(ensemble)' if len(models) > 1 else '(single)'}")
    print(f"  Validation checks:  {checks_passed}/{total_checks} passed")

    if checks_passed == total_checks:
        print(f"\nAll validation checks passed.")
        print(f"submission.csv is ready for upload.")
    else:
        print(f"\nWarning: Some validation checks failed.")
        print(f"Please review the issues before submitting.")

    print(f"\nAdditional files created:")
    print(f"  - {output_file} - Competition submission file")
    print(f"  - {proba_file} - Prediction probabilities")
    print(f"  - {summary_file} - Inference metadata")

    print(f"\n{'='*80}")


    # Show most confident predictions
    print("Most Confident Predictions (10 highest/lowest probabilities):")
    print("\nTop 10 Most Confident WINS (highest probabilities):")
    confident_wins = proba_df.nlargest(10, 'probability')[['battle_id', 'probability', 'prediction']]
    print(confident_wins.to_string(index=False))

    print("\nTop 10 Most Confident LOSSES (lowest probabilities):")
    confident_losses = proba_df.nsmallest(10, 'probability')[['battle_id', 'probability', 'prediction']]
    print(confident_losses.to_string(index=False))

    print("\nMost Uncertain Predictions (closest to 0.5):")
    proba_df['uncertainty'] = np.abs(proba_df['probability'] - 0.5)
    uncertain = proba_df.nsmallest(10, 'uncertainty')[['battle_id', 'probability', 'prediction']]
    print(uncertain.to_string(index=False))
    return "done"


if __name__ == "__main__":
    main()