# Pokémon Battle Outcome Prediction – README

This project implements a complete machine-learning pipeline to predict the outcome of Pokémon battles using detailed battle logs and engineered features. The pipeline is organized into four sequential scripts: training feature engineering, model training, test feature engineering, and final inference. The goal is to transform raw battle data into meaningful representations, train robust models, and generate accurate predictions for submission.

---

## 1. Training Feature Engineering (`one_feature_engineering.py`)
This script reads the raw **training battles JSON** and converts each battle into a structured set of engineered features.  
The feature engineering process captures a wide range of battle dynamics, including:

- Team-level statistics  
- Lead matchup indicators  
- Speed and type advantages  
- Turn-by-turn HP changes  
- Status effects and inflicted conditions  
- STAB usage and damage patterns  
- Switch–faint momentum and timeline behavior  

These features are designed to encode both strategic and temporal elements of each battle, producing a dataset suitable for model training.

---

## 2. Model Training (`two_train.py`)
This script builds and evaluates the predictive model using the engineered training features.  
It includes:

- Loading and validating engineered features  
- Train/validation splitting  
- Optional hyperparameter tuning with **Optuna**  
- **10-fold cross-validation** to train multiple fold models  
- Computation and saving of optimal probability thresholds  
- Training of the final consolidated XGBoost model  

All fold models, thresholds, and the final model are saved for use during inference.

---

## 3. Test Feature Engineering (`three_test_feature_engineering.py`)
This script applies the **same feature-engineering logic** used for the training data to the test battles JSON.  
It ensures that the test features match the exact structure, ordering, and transformations used during training.  
This consistency is crucial for ensuring that the trained models can reliably infer outcomes on unseen battles.

---

## 4. Final Inference (`four_inference.py`)
This script generates the final predictions used for submission. It:

- Loads the engineered test features  
- Applies all 10 fold models  
- Averages the prediction probabilities  
- Applies the learned thresholds to convert probabilities into binary outcomes  
- Performs sanity checks (missing values, invalid ranges, duplicates)  
- Produces the final `submission.csv` file

The inference step ensures that predictions are stable, validated, and aligned with the formats required for evaluation.

---

## Exploratory Data Analysis (EDA)
Prior to building the pipeline, an exploratory analysis was conducted to understand dataset characteristics such as:

- Battle length distributions  
- Common Pokémon species and types  
- HP and damage trends across turns  
- Speed advantage patterns  
- Frequency of status effects  
- Early-game momentum indicators  
- Correlations between base stats and win rates  

These insights informed the selection and design of the features used in the model.

---

## Submission 2 & 3 Results (Hyperparameter Tuning Impact)
While the overall pipeline remained identical, Submissions 2 and 3 introduced **Optuna-based hyperparameter tuning** during training.

- **Submission 2:** Used a narrower search space and achieved **82.9% accuracy**  
- **Submission 3:** Expanded and optimized the search space, improving performance to **83.2% accuracy**

This demonstrates that even when the feature engineering and structure remain unchanged, targeted hyperparameter optimization can yield measurable improvements in predictive accuracy.

