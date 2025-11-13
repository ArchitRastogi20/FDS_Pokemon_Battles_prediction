.PHONY: features train-quick train-full test-features predict ablate pipeline-quick pipeline-full help

# Configurable variables (override like: make pipeline-quick USE_GPU=1 FOLDS=5)
PY ?= python
USE_GPU ?= 0
FOLDS ?= 5
MAX_ROUNDS ?= 1200
ESR ?= 50
SEED ?= 42
MAX_SAMPLES ?= 0
# Space-separated subset of flags (optional): early_advantage early_status hp_ahead_share extremes first_status immobilized switch_faints move_mix type_enhance
FLAGS ?=

ABLAT_GPU_FLAG := $(if $(filter 1,$(USE_GPU)),--use-gpu,)
ABLAT_FLAGS_OPT := $(if $(strip $(FLAGS)),--flags $(FLAGS),)
ABLAT_SAMPLES_OPT := $(if $(filter 0,$(MAX_SAMPLES)),,--max-samples $(MAX_SAMPLES))

help:
	@echo "Targets:"
	@echo "  features         - Build training features"
	@echo "  train-quick      - No Optuna, $(FOLDS) folds, $(MAX_ROUNDS) rounds, ESR=$(ESR)"
	@echo "  train-full       - Full training with Optuna (as configured)"
	@echo "  test-features    - Build test features"
	@echo "  predict          - Run inference to produce submission.csv"
	@echo "  ablate           - Run ablation CV over feature flags"
	@echo "  pipeline-quick   - features -> train-quick -> test-features -> predict"
	@echo "  pipeline-full    - features -> train-full  -> test-features -> predict"
	@echo "Variables: PY, USE_GPU(0/1), FOLDS, MAX_ROUNDS, ESR, SEED, MAX_SAMPLES, FLAGS"

features:
	$(PY) feature_engineering.py

train-quick:
	$(PY) train.py --optuna-trials 0 --folds $(FOLDS) --max-rounds $(MAX_ROUNDS) --early-stopping-rounds $(ESR) --no-wandb

train-full:
	$(PY) train.py

test-features:
	$(PY) 3_test_feature_engineering.py

predict:
	$(PY) 4_inference.py

ablate:
	$(PY) ablation.py --folds $(FOLDS) $(ABLAT_GPU_FLAG) --max-rounds $(MAX_ROUNDS) --esr $(ESR) --seed $(SEED) $(ABLAT_SAMPLES_OPT) $(ABLAT_FLAGS_OPT)

pipeline-quick: features train-quick test-features predict

pipeline-full: features train-full test-features predict

