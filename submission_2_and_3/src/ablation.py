"""
Ablation runner for feature engineering toggles.

Runs quick XGBoost cross-validation for multiple feature flag settings
to estimate AUC deltas without the full training pipeline.

Usage examples:

  # Baseline + single-off ablations (disable one group at a time)
  python ablation.py --folds 5 --use-gpu

  # Only a subset of flags
  python ablation.py --flags early_advantage early_status extremes

  # Custom experiments from a JSON file
  python ablation.py --grid grid.json

Notes:
- Reads/writes feature_flags.json and restores it after finishing.
- Uses best_params.json if present; otherwise defaults.
- Uses feature_engineering.py to regenerate train_features.csv per run.
"""

import argparse
import json
import os
import sys
import time
import shutil
import subprocess
from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

try:
    import xgboost as xgb
except Exception as e:
    print("ERROR: xgboost is required to run ablation.")
    print("Install: pip install xgboost")
    sys.exit(1)


DEFAULT_FLAGS = {
    "early_advantage": True,
    "early_status": True,
    "hp_ahead_share": True,
    "extremes": True,
    "first_status": True,
    "immobilized": True,
    "switch_faints": True,
    "move_mix": True,
    "type_enhance": True,
}


def load_best_params():
    default = {
        'max_depth': 7,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'colsample_bylevel': 0.8,
        'min_child_weight': 10,
        'gamma': 1.0,
        'reg_alpha': 1.0,
        'reg_lambda': 1.0,
    }
    try:
        with open('best_params.json', 'r') as f:
            loaded = json.load(f)
        default.update(loaded)
    except Exception:
        pass
    return default


def write_flags(flags: dict, path='feature_flags.json'):
    with open(path, 'w') as f:
        json.dump(flags, f, indent=2)


def run_feature_engineering(python_exec=sys.executable):
    print("[Ablation] Running feature_engineering.py ...")
    subprocess.run([python_exec, 'feature_engineering.py'], check=True)


def xgb_cv_auc(train_csv: str, folds: int, use_gpu: bool, max_rounds: int, esr: int, seed: int, max_samples: int = 0):
    df = pd.read_csv(train_csv)
    X = df.drop(['battle_id', 'player_won'], axis=1)
    y = df['player_won']
    if max_samples and max_samples < len(df):
        df_idx = np.random.RandomState(seed).choice(len(df), size=max_samples, replace=False)
        X = X.iloc[df_idx]
        y = y.iloc[df_idx]

    params = load_best_params()
    booster_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'tree_method': 'gpu_hist' if use_gpu else 'hist',
        'predictor': 'gpu_predictor' if use_gpu else 'auto',
        **params,
    }

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scores = []
    for fold, (tr, va) in enumerate(skf.split(X, y), 1):
        dtrain = xgb.DMatrix(X.iloc[tr], label=y.iloc[tr])
        dval = xgb.DMatrix(X.iloc[va], label=y.iloc[va])
        model = xgb.train(
            booster_params,
            dtrain,
            num_boost_round=max_rounds,
            evals=[(dval, 'val')],
            early_stopping_rounds=esr,
            verbose_eval=False,
        )
        preds = model.predict(dval)
        auc = roc_auc_score(y.iloc[va], preds)
        scores.append(auc)
        print(f"  Fold {fold}/{folds}: AUC={auc:.6f}, best_iter={model.best_iteration}")

    return float(np.mean(scores)), float(np.std(scores))


def single_off_experiments(flags_subset=None):
    base = deepcopy(DEFAULT_FLAGS)
    if flags_subset:
        # Only include listed flags; others remain as in base
        base = {k: v for k, v in base.items() if k in flags_subset}
        # But we still need a full set for writing; fill missing with defaults
        full_base = deepcopy(DEFAULT_FLAGS)
        full_base.update(base)
        base = full_base

    exps = []
    # Baseline
    exps.append(("baseline_all_on", deepcopy(base)))
    # Each single flag off
    for k in sorted(DEFAULT_FLAGS.keys()):
        if flags_subset and k not in flags_subset:
            continue
        f = deepcopy(base)
        f[k] = False
        exps.append((f"minus_{k}", f))
    return exps


def load_grid(path):
    with open(path, 'r') as f:
        data = json.load(f)
    # Expect a list of {name: str, flags: {..}}
    exps = []
    if isinstance(data, list):
        for i, item in enumerate(data):
            name = item.get('name', f'exp_{i}')
            flags = deepcopy(DEFAULT_FLAGS)
            flags.update(item.get('flags', {}))
            exps.append((name, flags))
    else:
        raise ValueError('Grid JSON must be a list of {name, flags} objects')
    return exps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--use-gpu', action='store_true')
    ap.add_argument('--max-rounds', type=int, default=1000)
    ap.add_argument('--esr', type=int, default=50, help='early_stopping_rounds')
    ap.add_argument('--max-samples', type=int, default=0, help='optional subsample for speed')
    ap.add_argument('--flags', nargs='*', help='limit to these flags for single-off runs')
    ap.add_argument('--grid', type=str, default='', help='path to JSON grid spec')
    ap.add_argument('--python', type=str, default=sys.executable, help='python executable to run feature script')
    args = ap.parse_args()

    # Backup current feature_flags.json if exists
    backup_path = None
    flags_path = 'feature_flags.json'
    if os.path.exists(flags_path):
        backup_path = flags_path + '.backup'
        shutil.copyfile(flags_path, backup_path)
        print(f"[Ablation] Backed up existing feature_flags.json -> {backup_path}")

    # Build experiments
    if args.grid:
        experiments = load_grid(args.grid)
    else:
        experiments = single_off_experiments(args.flags)

    results = []
    start_all = time.time()
    for name, flags in experiments:
        print("=" * 80)
        print(f"[Ablation] Experiment: {name}")
        print("Flags:")
        for k in sorted(flags.keys()):
            print(f"  {k}: {flags[k]}")

        # Write flags and regenerate features
        write_flags(flags)
        t0 = time.time()
        run_feature_engineering(args.python)

        # Quick CV
        mean_auc, std_auc = xgb_cv_auc(
            train_csv='train_features.csv',
            folds=args.folds,
            use_gpu=args.use_gpu,
            max_rounds=args.max_rounds,
            esr=args.esr,
            seed=args.seed,
            max_samples=args.max_samples,
        )

        duration = time.time() - t0
        # Count features
        try:
            df = pd.read_csv('train_features.csv', nrows=1)
            n_features = df.shape[1] - 2  # exclude battle_id, player_won
        except Exception:
            n_features = -1

        results.append({
            'name': name,
            'flags': json.dumps(flags, sort_keys=True),
            'folds': args.folds,
            'use_gpu': args.use_gpu,
            'mean_auc': mean_auc,
            'std_auc': std_auc,
            'n_features': n_features,
            'duration_sec': round(duration, 2),
        })
        print(f"[Ablation] {name}: AUC={mean_auc:.6f} ± {std_auc:.6f} | features={n_features} | time={duration:.1f}s")

    total_duration = time.time() - start_all

    # Restore flags
    if backup_path and os.path.exists(backup_path):
        shutil.move(backup_path, flags_path)
        print(f"[Ablation] Restored original feature_flags.json from backup")
    else:
        # Leaving the last used flags in place if no backup
        pass

    # Save results
    out_csv = 'ablation_results.csv'
    pd.DataFrame(results).to_csv(out_csv, index=False)
    print("=" * 80)
    print(f"Ablation complete. Ran {len(results)} experiments in {total_duration:.1f}s")
    print(f"Results saved to: {out_csv}")
    # Pretty print summary
    try:
        import tabulate  # optional
        print(tabulate.tabulate(pd.DataFrame(results), headers='keys', tablefmt='github', floatfmt='.6f'))
    except Exception:
        print(pd.DataFrame(results).to_string(index=False))


if __name__ == '__main__':
    main()

