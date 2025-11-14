import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, classification_report, confusion_matrix
import optuna
from optuna.integration import XGBoostPruningCallback
import wandb
import warnings
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
CONFIG = {
    'n_folds': 10,
    'test_size': 0.15,
    'random_state': 42,
    'n_trials': 100,  # Optuna hyperparameter search trials
    'early_stopping_rounds': 100,
    'gpu_id': 0,
    'n_jobs': 17,  # For CPU-bound operations
    'verbose_eval': 50,
}

print("="*80)
print("POKEMON BATTLE PREDICTION - ULTIMATE TRAINING PIPELINE")
print("="*80)
print(f"Configuration:")
for key, value in CONFIG.items():
    print(f"  {key}: {value}")
print("="*80)

# ============================================================================
# WANDB INITIALIZATION
# ============================================================================
def init_wandb():
    """Initialize Weights & Biases"""
    print("\n[WANDB] Initializing Weights & Biases...")
    
    try:
        wandb.init(
            project="pokemon-battle-prediction",
            name=f"xgboost-gpu-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            config=CONFIG,
            tags=["xgboost", "gpu", "v100", "optuna", "k-fold-10"],
            notes="Ultimate training with 58 engineered features"
        )
        print("✓ W&B initialized successfully")
        return True
    except Exception as e:
        print(f"⚠ W&B initialization failed: {e}")
        print("⚠ Continuing without W&B tracking...")
        return False

# ============================================================================
# DATA LOADING
# ============================================================================
def load_data():
    """Load preprocessed features"""
    print(f"\n{'='*80}")
    print("[1/7] LOADING TRAINING DATA")
    print(f"{'='*80}")
    
    df = pd.read_csv('train_features.csv')
    print(f"✓ Loaded {len(df):,} samples with {len(df.columns)} columns")
    
    # Separate features and target
    X = df.drop(['battle_id', 'player_won'], axis=1)
    y = df['player_won']
    battle_ids = df['battle_id']
    
    print(f"\n  Features: {X.shape[1]}")
    print(f"  Target balance: Wins={y.sum():,} ({y.mean()*100:.2f}%) | Losses={len(y)-y.sum():,} ({(1-y.mean())*100:.2f}%)")
    print(f"  Feature list (first 15): {X.columns.tolist()[:15]}")
    
    # Check for any issues
    null_counts = X.isnull().sum().sum()
    inf_counts = np.isinf(X.values).sum()
    print(f"\n  Data quality checks:")
    print(f"    Null values: {null_counts} {'✓' if null_counts == 0 else '✗'}")
    print(f"    Inf values: {inf_counts} {'✓' if inf_counts == 0 else '✗'}")
    
    return X, y, battle_ids, X.columns.tolist()

# ============================================================================
# TRAIN-VAL SPLIT
# ============================================================================
def create_train_val_split(X, y, battle_ids):
    """Create stratified train and validation sets"""
    print(f"\n{'='*80}")
    print("[2/7] CREATING TRAIN/VALIDATION SPLIT")
    print(f"{'='*80}")
    
    X_train, X_val, y_train, y_val, ids_train, ids_val = train_test_split(
        X, y, battle_ids,
        test_size=CONFIG['test_size'],
        stratify=y,
        random_state=CONFIG['random_state']
    )
    
    print(f"  Train set: {len(X_train):,} samples | Wins: {y_train.sum():,} ({y_train.mean()*100:.2f}%)")
    print(f"  Val set:   {len(X_val):,} samples | Wins: {y_val.sum():,} ({y_val.mean()*100:.2f}%)")
    
    return X_train, X_val, y_train, y_val

# ============================================================================
# OPTUNA OBJECTIVE
# ============================================================================
def objective(trial, X_train, y_train, use_wandb):
    """Optuna objective function with 5-fold CV"""
    
    # Define search space
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'tree_method': 'gpu_hist',
        'gpu_id': CONFIG['gpu_id'],
        'predictor': 'gpu_predictor',
        
        # Hyperparameters to tune
        'max_depth': trial.suggest_int('max_depth', 4, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 30),
        'gamma': trial.suggest_float('gamma', 0.0, 5.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10.0),
        'scale_pos_weight': 1.0,  # Balanced dataset
    }
    
    # 5-Fold CV for speed during optimization
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=CONFIG['random_state'])
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_fold_train = X_train.iloc[train_idx]
        y_fold_train = y_train.iloc[train_idx]
        X_fold_val = X_train.iloc[val_idx]
        y_fold_val = y_train.iloc[val_idx]
        
        dtrain = xgb.DMatrix(X_fold_train, label=y_fold_train)
        dval = xgb.DMatrix(X_fold_val, label=y_fold_val)
        
        # Train with early stopping and pruning
        pruning_callback = XGBoostPruningCallback(trial, f'validation-auc')
        
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=1000,
            evals=[(dval, 'validation')],
            early_stopping_rounds=50,
            verbose_eval=False,
            callbacks=[pruning_callback]
        )
        
        preds = model.predict(dval)
        auc = roc_auc_score(y_fold_val, preds)
        cv_scores.append(auc)
        
        trial.report(auc, fold)
        
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    mean_auc = np.mean(cv_scores)
    
    # Log to wandb if available
    if use_wandb:
        wandb.log({
            f"optuna_trial_{trial.number}_mean_auc": mean_auc,
            f"optuna_trial_{trial.number}_std_auc": np.std(cv_scores)
        })
    
    return mean_auc

# ============================================================================
# HYPERPARAMETER OPTIMIZATION
# ============================================================================
def optimize_hyperparameters(X_train, y_train, use_wandb):
    """Run Optuna hyperparameter optimization"""
    print(f"\n{'='*80}")
    print(f"[3/7] HYPERPARAMETER OPTIMIZATION - {CONFIG['n_trials']} TRIALS")
    print(f"{'='*80}")
    print("This may take 15-30 minutes depending on your GPU...")
    
    study = optuna.create_study(
        direction='maximize',
        study_name='xgboost-pokemon-battle',
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
        sampler=optuna.samplers.TPESampler(seed=CONFIG['random_state'])
    )
    
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, use_wandb),
        n_trials=CONFIG['n_trials'],
        n_jobs=1,
        show_progress_bar=True,
        timeout=None
    )
    
    print(f"\n✓ Optimization complete!")
    print(f"  Best trial: #{study.best_trial.number}")
    print(f"  Best AUC: {study.best_value:.6f}")
    print(f"\n  Best hyperparameters:")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")
    
    if use_wandb:
        wandb.log({
            "best_optuna_auc": study.best_value,
            "best_trial_number": study.best_trial.number
        })
        wandb.config.update(study.best_params, allow_val_change=True)
    
    return study.best_params

# ============================================================================
# K-FOLD CROSS VALIDATION
# ============================================================================
def train_with_kfold(X_train, y_train, best_params, use_wandb):
    """Train with 10-Fold CV using best parameters"""
    print(f"\n{'='*80}")
    print(f"[4/7] K-FOLD CROSS VALIDATION - {CONFIG['n_folds']} FOLDS")
    print(f"{'='*80}")
    
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'tree_method': 'gpu_hist',
        'gpu_id': CONFIG['gpu_id'],
        'predictor': 'gpu_predictor',
        **best_params
    }
    
    skf = StratifiedKFold(n_splits=CONFIG['n_folds'], shuffle=True, random_state=CONFIG['random_state'])
    
    fold_scores = []
    oof_predictions = np.zeros(len(X_train))
    oof_probabilities = np.zeros(len(X_train))
    models = []
    feature_importance_list = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
        print(f"\n{'─'*80}")
        print(f"Fold {fold}/{CONFIG['n_folds']}")
        print(f"{'─'*80}")
        
        X_fold_train = X_train.iloc[train_idx]
        y_fold_train = y_train.iloc[train_idx]
        X_fold_val = X_train.iloc[val_idx]
        y_fold_val = y_train.iloc[val_idx]
        
        dtrain = xgb.DMatrix(X_fold_train, label=y_fold_train)
        dval = xgb.DMatrix(X_fold_val, label=y_fold_val)
        
        # Train
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=2000,
            evals=[(dtrain, 'train'), (dval, 'validation')],
            early_stopping_rounds=CONFIG['early_stopping_rounds'],
            verbose_eval=CONFIG['verbose_eval']
        )
        
        # Predictions
        val_preds_proba = model.predict(dval)
        val_preds_binary = (val_preds_proba > 0.5).astype(int)
        
        oof_predictions[val_idx] = val_preds_binary
        oof_probabilities[val_idx] = val_preds_proba
        
        # Metrics
        fold_auc = roc_auc_score(y_fold_val, val_preds_proba)
        fold_acc = accuracy_score(y_fold_val, val_preds_binary)
        fold_logloss = log_loss(y_fold_val, val_preds_proba)
        
        fold_scores.append({
            'fold': fold,
            'auc': fold_auc,
            'accuracy': fold_acc,
            'logloss': fold_logloss,
            'best_iteration': model.best_iteration
        })
        
        print(f"\nFold {fold} Results:")
        print(f"  AUC:      {fold_auc:.6f}")
        print(f"  Accuracy: {fold_acc:.6f}")
        print(f"  LogLoss:  {fold_logloss:.6f}")
        print(f"  Best iteration: {model.best_iteration}")
        
        # Feature importance
        importance = model.get_score(importance_type='gain')
        feature_importance_list.append(importance)
        
        if use_wandb:
            wandb.log({
                f'fold_{fold}_auc': fold_auc,
                f'fold_{fold}_accuracy': fold_acc,
                f'fold_{fold}_logloss': fold_logloss,
                f'fold_{fold}_best_iter': model.best_iteration
            })
        
        models.append(model)
    
    # Overall OOF performance
    oof_auc = roc_auc_score(y_train, oof_probabilities)
    oof_acc = accuracy_score(y_train, oof_predictions)
    oof_logloss = log_loss(y_train, oof_probabilities)
    
    print(f"\n{'='*80}")
    print(f"K-FOLD CROSS VALIDATION RESULTS")
    print(f"{'='*80}")
    print(f"Per-Fold Statistics:")
    print(f"  Mean AUC:      {np.mean([s['auc'] for s in fold_scores]):.6f} ± {np.std([s['auc'] for s in fold_scores]):.6f}")
    print(f"  Mean Accuracy: {np.mean([s['accuracy'] for s in fold_scores]):.6f} ± {np.std([s['accuracy'] for s in fold_scores]):.6f}")
    print(f"  Mean LogLoss:  {np.mean([s['logloss'] for s in fold_scores]):.6f} ± {np.std([s['logloss'] for s in fold_scores]):.6f}")
    print(f"\nOut-of-Fold Performance:")
    print(f"  OOF AUC:      {oof_auc:.6f}")
    print(f"  OOF Accuracy: {oof_acc:.6f}")
    print(f"  OOF LogLoss:  {oof_logloss:.6f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_train, oof_predictions)
    print(f"\nConfusion Matrix:")
    print(f"  TN: {cm[0,0]:5,} | FP: {cm[0,1]:5,}")
    print(f"  FN: {cm[1,0]:5,} | TP: {cm[1,1]:5,}")
    
    if use_wandb:
        wandb.log({
            'cv_mean_auc': np.mean([s['auc'] for s in fold_scores]),
            'cv_std_auc': np.std([s['auc'] for s in fold_scores]),
            'cv_mean_accuracy': np.mean([s['accuracy'] for s in fold_scores]),
            'oof_auc': oof_auc,
            'oof_accuracy': oof_acc,
            'oof_logloss': oof_logloss
        })
    
    return models, fold_scores, oof_predictions, oof_probabilities, feature_importance_list

# ============================================================================
# FINAL MODEL ON VALIDATION SET
# ============================================================================
def train_final_model(X_train, y_train, X_val, y_val, best_params, use_wandb):
    """Train final model and evaluate on holdout validation set"""
    print(f"\n{'='*80}")
    print("[5/7] TRAINING FINAL MODEL ON FULL TRAIN SET")
    print(f"{'='*80}")
    
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'tree_method': 'gpu_hist',
        'gpu_id': CONFIG['gpu_id'],
        'predictor': 'gpu_predictor',
        **best_params
    }
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    # Train
    final_model = xgb.train(
        params,
        dtrain,
        num_boost_round=2000,
        evals=[(dtrain, 'train'), (dval, 'validation')],
        early_stopping_rounds=CONFIG['early_stopping_rounds'],
        verbose_eval=CONFIG['verbose_eval']
    )
    
    # Evaluate
    val_preds_proba = final_model.predict(dval)
    val_preds_binary = (val_preds_proba > 0.5).astype(int)
    
    val_auc = roc_auc_score(y_val, val_preds_proba)
    val_acc = accuracy_score(y_val, val_preds_binary)
    val_logloss = log_loss(y_val, val_preds_proba)
    
    print(f"\nValidation Set Performance:")
    print(f"  AUC:      {val_auc:.6f}")
    print(f"  Accuracy: {val_acc:.6f}")
    print(f"  LogLoss:  {val_logloss:.6f}")
    print(f"  Best iteration: {final_model.best_iteration}")
    
    print(f"\nDetailed Classification Report:")
    print(classification_report(y_val, val_preds_binary, target_names=['Loss', 'Win'], digits=4))
    
    # Confusion matrix
    cm = confusion_matrix(y_val, val_preds_binary)
    print(f"\nValidation Confusion Matrix:")
    print(f"  TN: {cm[0,0]:4} | FP: {cm[0,1]:4}")
    print(f"  FN: {cm[1,0]:4} | TP: {cm[1,1]:4}")
    
    if use_wandb:
        wandb.log({
            'val_auc': val_auc,
            'val_accuracy': val_acc,
            'val_logloss': val_logloss,
            'val_best_iteration': final_model.best_iteration
        })
    
    return final_model, val_auc, val_acc

# ============================================================================
# FEATURE IMPORTANCE ANALYSIS
# ============================================================================
def analyze_feature_importance(feature_importance_list, feature_names, use_wandb):
    """Aggregate and analyze feature importance across folds"""
    print(f"\n{'='*80}")
    print("[6/7] FEATURE IMPORTANCE ANALYSIS")
    print(f"{'='*80}")
    
    # Aggregate importance across folds
    all_features = set()
    for importance_dict in feature_importance_list:
        all_features.update(importance_dict.keys())
    
    feature_importance_agg = {}
    for feature in all_features:
        importances = [imp_dict.get(feature, 0) for imp_dict in feature_importance_list]
        feature_importance_agg[feature] = {
            'mean': np.mean(importances),
            'std': np.std(importances)
        }
    
    # Create DataFrame
    importance_df = pd.DataFrame([
        {
            'feature': feature,
            'importance_mean': stats['mean'],
            'importance_std': stats['std']
        }
        for feature, stats in feature_importance_agg.items()
    ]).sort_values('importance_mean', ascending=False)
    
    print(f"\nTop 20 Most Important Features:")
    print(importance_df.head(20).to_string(index=False))
    
    # Save to CSV
    importance_df.to_csv('feature_importance.csv', index=False)
    print(f"\n✓ Saved full feature importance to 'feature_importance.csv'")
    
    if use_wandb:
        wandb.log({"feature_importance_top30": wandb.Table(dataframe=importance_df.head(30))})
    
    return importance_df

# ============================================================================
# SAVE ARTIFACTS
# ============================================================================
def save_artifacts(models, final_model, best_params, fold_scores, feature_names, use_wandb):
    """Save all trained models and metadata"""
    print(f"\n{'='*80}")
    print("[7/7] SAVING MODELS AND ARTIFACTS")
    print(f"{'='*80}")
    
    # Save fold models
    for i, model in enumerate(models, 1):
        filename = f'model_fold_{i}.json'
        model.save_model(filename)
        print(f"  ✓ Saved {filename}")
    
    # Save final model
    final_model.save_model('model_final.json')
    print(f"  ✓ Saved model_final.json")
    
    # Save best parameters
    with open('best_params.json', 'w') as f:
        json.dump(best_params, f, indent=4)
    print(f"  ✓ Saved best_params.json")
    
    # Save fold scores
    fold_df = pd.DataFrame(fold_scores)
    fold_df.to_csv('fold_scores.csv', index=False)
    print(f"  ✓ Saved fold_scores.csv")
    
    # Save feature names
    with open('feature_names.json', 'w') as f:
        json.dump(feature_names, f, indent=4)
    print(f"  ✓ Saved feature_names.json")
    
    # Save training config
    with open('training_config.json', 'w') as f:
        json.dump(CONFIG, f, indent=4)
    print(f"  ✓ Saved training_config.json")
    
    if use_wandb:
        wandb.save('model_final.json')
        wandb.save('best_params.json')
        wandb.save('fold_scores.csv')
        wandb.save('feature_importance.csv')
        print(f"  ✓ Uploaded artifacts to W&B")

# ============================================================================
# MAIN PIPELINE
# ============================================================================
def main():
    start_time = datetime.now()
    
    # Initialize W&B
    use_wandb = init_wandb()
    
    # Load data
    X, y, battle_ids, feature_names = load_data()
    
    # Train-val split
    X_train, X_val, y_train, y_val = create_train_val_split(X, y, battle_ids)
    
    # Hyperparameter optimization
    best_params = optimize_hyperparameters(X_train, y_train, use_wandb)
    
    # K-Fold cross validation
    models, fold_scores, oof_preds, oof_proba, feature_importance_list = train_with_kfold(
        X_train, y_train, best_params, use_wandb
    )
    
    # Final model
    final_model, val_auc, val_acc = train_final_model(
        X_train, y_train, X_val, y_val, best_params, use_wandb
    )
    
    # Feature importance
    importance_df = analyze_feature_importance(feature_importance_list, feature_names, use_wandb)
    
    # Save everything
    save_artifacts(models, final_model, best_params, fold_scores, feature_names, use_wandb)
    
    # Final summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    print(f"\n{'='*80}")
    print("🎉 TRAINING PIPELINE COMPLETE! 🎉")
    print(f"{'='*80}")
    print(f"Training duration: {duration}")
    print(f"\nFinal Metrics:")
    print(f"  Cross-Validation AUC:  {np.mean([s['auc'] for s in fold_scores]):.6f} ± {np.std([s['auc'] for s in fold_scores]):.6f}")
    print(f"  Validation AUC:        {val_auc:.6f}")
    print(f"  Validation Accuracy:   {val_acc:.6f}")
    print(f"\nArtifacts saved:")
    print(f"  • {CONFIG['n_folds']} fold models + 1 final model")
    print(f"  • best_params.json")
    print(f"  • fold_scores.csv")
    print(f"  • feature_importance.csv")
    print(f"  • training_config.json")
    
    if use_wandb:
        print(f"\n📊 Check your W&B dashboard for detailed visualizations!")
        wandb.finish()
    
    print(f"\n{'='*80}")
    print("✓ Ready for inference! Run: python 4_inference.py")
    print(f"{'='*80}\n")

def run_training(train_features_path='train_features.csv', 
                 n_folds=10, 
                 n_trials=100, 
                 random_state=42,
                 use_gpu=True,
                 gpu_id=0,
                 n_jobs=17):
    """
    Train XGBoost models with hyperparameter optimization.
    
    Args:
        train_features_path: Path to train_features.csv
        n_folds: Number of cross-validation folds
        n_trials: Optuna optimization trials
        random_state: Random seed
        use_gpu: Enable GPU acceleration
        gpu_id: GPU device ID
        n_jobs: CPU cores for parallel processing
    """
    # Update global configuration
    global CONFIG
    CONFIG = {
        'n_folds': n_folds,
        'test_size': 0.15,
        'random_state': random_state,
        'n_trials': n_trials,
        'early_stopping_rounds': 100,
        'gpu_id': gpu_id if use_gpu else -1,
        'n_jobs': n_jobs,
        'verbose_eval': 50
    }
    
    # Run the main training function
    main()

if __name__ == "__main__":
    main()