# Pokemon Battle Prediction

**What this is**

* Predicts if Player 1 wins a Pokemon battle using the full 30-turn timeline and team info.
* Binary classification. Metric: **AUC-ROC**.

---

## Quick results

* **Test AUC:** 0.78
* **Cross-val AUC:** 0.8464 ± 0.0122
* **Val AUC:** 0.8666

---

## Data (fast)

* 10,000 training battles (JSONL), 5,000 test battles (JSONL).
* Each battle = 30 turns.
* Player1: full team of 6 visible. Player2: only lead visible.

---

## Main idea

* Baseline (team averages) is weak. Timeline and status effects matter most.
* Extract 58 features per battle (static + timeline + status + strategy).
* Train XGBoost with heavy regularization and Optuna tuning. Ensemble 10 fold models.

---

## Top features (why model works)

* **Sleep and Freeze counts** (opponent sleep = #1).
* **Number of Pokemon used** (switching vs sweep strategy).
* **Speed advantage** and early HP differences (turns 1–5).
* Status conditions together explain most predictive power.

---

## Model & training

* **Model:** XGBoost (GPU, `gpu_hist`).
* **Validation:** 10-fold CV + 15% holdout val.
* **Hyperparam tuning:** Optuna (100 trials).
* **Ensemble:** Average predictions from 10 CV models.
* Total training ~35–50 minutes on GPU.

---

## Inference pipeline

1. Extract the **same 58 features** from `test.jsonl` (parallel).
2. Load 10 XGBoost models.
3. Average probabilistic predictions.
4. Save `submission.csv` (`battle_id,player_won`).

---