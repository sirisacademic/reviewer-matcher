# train_final_lasso.py

import os
import pandas as pd
import numpy as np
import json

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import resample
from sklearn.feature_selection import SelectKBest, SelectFromModel, f_classif, f_regression, chi2, mutual_info_classif
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from modules.utils import save_scaler, rank_experts
from modules.model_training import train_model, save_model
from modules.evaluate import evaluate_model

from config import *

print('========== Configuration ========')
print(f'TRAIN_EVAL_DATA_PATH={TRAIN_EVAL_DATA_PATH}')
print(f'TRAIN_EVAL_DATA_FILE={TRAIN_EVAL_DATA_FILE}')
print(f'SET_FEATURES={SET_FEATURES}')
print(f'FEATURE_COLUMNS[SET_FEATURES]={FEATURE_COLUMNS[SET_FEATURES]}')
print(f'FEATURE_SELECTION={FEATURE_SELECTION}')
print(f'COLUMN_TASK_GOLD={COLUMN_TASK_GOLD}')
print(f'ANNOTATION_POSITIVE={ANNOTATION_POSITIVE}')
print(f'ANNOTATION_NEGATIVE={ANNOTATION_NEGATIVE}')
print(f'STANDARIZE_DATA={STANDARIZE_DATA}')
print(f'MODEL_FINAL_SAVE_PATH={MODEL_FINAL_SAVE_PATH}')
print(f'OUTPUT_PREDICTIONS_FINAL_PATH={OUTPUT_PREDICTIONS_FINAL_PATH}')
print(f'USE_SUBSET_FINAL_ASSIGNMENTS={USE_SUBSET_FINAL_ASSIGNMENTS}')
print(f'BALANCE_CLASSES={BALANCE_CLASSES}')
print(f'PREFIX={PREFIX}')
print('=================================')

prefix = f'{PREFIX}_subset_top_reviewers_' if USE_SUBSET_FINAL_ASSIGNMENTS else PREFIX

# Class used for feature selection using correlations
class CorrelationFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, k=10):
        self.k = k
        self.selected_features = None

    def fit(self, X, y):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        y = pd.Series(y)
        correlations = X.corrwith(y)
        self.selected_features = correlations.abs().sort_values(ascending=False).head(self.k).index.tolist()
        return self

    def transform(self, X):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        return X[self.selected_features]

    def get_support(self):
        return self.selected_features

# Pipeline definition
if FEATURE_SELECTION == 'mi':
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('feature_selection', SelectKBest(mutual_info_classif)),
        ('classifier', LogisticRegression(penalty="l1", solver="liblinear", max_iter=5000, random_state=42))
    ])
elif FEATURE_SELECTION == 'fc':
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('feature_selection', SelectKBest(f_classif)),
        ('classifier', LogisticRegression(penalty="l1", solver="liblinear", max_iter=5000, random_state=42))
    ])
elif FEATURE_SELECTION == 'chi2':
    pipeline = Pipeline([
        ('scaler', MinMaxScaler()),
        ('feature_selection', SelectKBest(chi2)),
        ('classifier', LogisticRegression(penalty="l1", solver="liblinear", max_iter=5000, random_state=42))
    ])
elif FEATURE_SELECTION == 'rf':
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('feature_selection', SelectFromModel(
            RandomForestClassifier(random_state=42),
            max_features=None
        )),
        ('classifier', LogisticRegression(penalty="l1", solver="liblinear", max_iter=5000, random_state=42))
    ])
elif FEATURE_SELECTION == 'corr':
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('feature_selection', CorrelationFeatureSelector()),
        ('classifier', LogisticRegression(penalty="l1", solver="liblinear", max_iter=5000, random_state=42))
    ])
else:
    print("*** NOT PERFORMING FEATURE SELECTION ***")
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(penalty="l1", solver="liblinear", max_iter=5000, random_state=42))
    ])

# Hyper-parameter grid
PARAM_GRID = {
    "classifier__C": [0.01, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 3, 10, 15, 20],
    "classifier__class_weight": [
        {0: 1, 1: 1},
        {0: 1, 1: 1.25},
        {0: 1, 1: 1.5},
        {0: 1, 1: 1.75},
        {0: 1, 1: 2},
        {0: 1, 1: 3},
        {0: 0.5, 1: 1}
    ]
}

if FEATURE_SELECTION == 'rf':
    PARAM_GRID["feature_selection__estimator__n_estimators"] = [100, 200]
    PARAM_GRID["feature_selection__max_features"] = [5, 10, 15, 20]
    PARAM_GRID["feature_selection__estimator__min_samples_split"] = [2, 5]
    PARAM_GRID["feature_selection__estimator__min_samples_leaf"] = [1, 2]
elif FEATURE_SELECTION:
    PARAM_GRID["feature_selection__k"] = [5, 10, 15, 20]

SCORING_METRIC_CV = "roc_auc"

# Define cross-validation grid search
grid_search = GridSearchCV(
    pipeline,
    param_grid=PARAM_GRID,
    scoring=SCORING_METRIC_CV,
    cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
)

print(f"SCORING_METRIC_CV={SCORING_METRIC_CV}")

# Load all data.
data_train_path = f"{TRAIN_EVAL_DATA_PATH}/{prefix}{TRAIN_EVAL_DATA_FILE}"
print(f'Reading training data from {data_train_path}')
data = pd.read_csv(data_train_path, sep='\t')

# Prepare all data for initial training.
X_initial = data[FEATURE_COLUMNS[SET_FEATURES]]
y_initial = data[COLUMN_TASK_GOLD].apply(lambda x: 1 if x == ANNOTATION_POSITIVE else 0)

# Initial feature selection/hyper-parameter optimization.
grid_search.fit(X_initial, y_initial)

# Best parameters and selected features for initial model.
best_pipeline_initial = grid_search.best_estimator_
best_params_initial = grid_search.best_params_

if 'feature_selection' in best_pipeline_initial.named_steps:
    selected_features_mask = best_pipeline_initial.named_steps['feature_selection'].get_support()
    selected_features = X_initial.columns[selected_features_mask]
else:
    selected_features = FEATURE_COLUMNS[SET_FEATURES]

# Predict probabilities on all data.
data["Predicted_Label"] = best_pipeline_initial.predict(X_initial)
data["Probability_Positive"] = best_pipeline_initial.predict_proba(X_initial)[:, 1]

# Define filtering thresholds.
BOUNDARY_PREDICTIONS = 0.5
ADJUSTMENT_FACTOR = 0.2
LOWER_THRESHOLD = BOUNDARY_PREDICTIONS - ADJUSTMENT_FACTOR
HIGHER_THRESHOLD = BOUNDARY_PREDICTIONS + ADJUSTMENT_FACTOR

# Filter data using thresholds.
exclude_conditions = (
    (
        (data["Annotation"] == ANNOTATION_POSITIVE) &
        (data["Probability_Positive"] <= LOWER_THRESHOLD)
    ) |
    (
        (data["Annotation"] == ANNOTATION_NEGATIVE) &
        (data["Probability_Positive"] >= HIGHER_THRESHOLD)
    )
)
filtered_data = data[~exclude_conditions]

print(f"Original samples = {len(data)}, Filtered samples = {len(filtered_data)}")

# Prepare filtered data for final training.
X_filtered = filtered_data[FEATURE_COLUMNS[SET_FEATURES]]
y_filtered = filtered_data[COLUMN_TASK_GOLD].apply(lambda x: 1 if x == ANNOTATION_POSITIVE else 0)

# Final feature selection/hyper-parameter optimization with filtered data.
grid_search.fit(X_filtered, y_filtered)

# Get best parameters and model for filtered data
best_pipeline_filtered = grid_search.best_estimator_
best_params_filtered = grid_search.best_params_

if 'feature_selection' in best_pipeline_filtered.named_steps:
    selected_features_mask = best_pipeline_filtered.named_steps['feature_selection'].get_support()
    selected_features = X_filtered.columns[selected_features_mask]
else:
    selected_features = FEATURE_COLUMNS[SET_FEATURES]

excluded_features = set(FEATURE_COLUMNS[SET_FEATURES]) - set(selected_features)

print(f"Number original features: {len(excluded_features)+len(selected_features)}")
print(f"Number selected features: {len(selected_features)}")
print(f"Selected features: {selected_features}")
print(f"Excluded features: {excluded_features}")

print(f"Best parameters for final model: {best_params_filtered}")

# Save final model
os.makedirs(MODEL_FINAL_SAVE_PATH, exist_ok=True)
model_file = f"{MODEL_FINAL_SAVE_PATH}/final_model_{prefix}{SET_FEATURES}.pkl"
save_model(best_pipeline_filtered, model_file)
print(f"Final model saved to {model_file}")

# Add final predictions to data
data["Final_Predicted_Label"] = best_pipeline_filtered.predict(X_initial)
data["Final_Probability_Positive"] = best_pipeline_filtered.predict_proba(X_initial)[:, 1]

# Save final predictions
os.makedirs(OUTPUT_PREDICTIONS_FINAL_PATH, exist_ok=True)
predictions_file = f"{OUTPUT_PREDICTIONS_FINAL_PATH}/predictions_{prefix}{SET_FEATURES}.tsv"
data.to_csv(predictions_file, sep='\t', index=False)
print(f"Final predictions saved to {predictions_file}")

# Calculate and print metrics on complete dataset
metrics = evaluate_model(best_pipeline_filtered, X_initial, y_initial)
print("\nFinal model performance metrics:")
print(f"AUC: {metrics['AUC']:.3f}")
print(f"Accuracy: {metrics['Accuracy']:.3f}")
print(f"F1: {metrics['F1']:.3f}")
print(f"Precision: {metrics['Precision']:.3f}")
print(f"Recall: {metrics['Recall']:.3f}")

# Save metrics
metrics = {k: float(v) if isinstance(v, (np.float32, np.float64)) else v 
          for k, v in metrics.items() if k != "probs"}
os.makedirs(OUTPUT_METRICS_PATH, exist_ok=True)
metrics_file = f"{OUTPUT_METRICS_PATH}/final_model_metrics_{prefix}{SET_FEATURES}.json"
with open(metrics_file, "w") as f:
    json.dump(metrics, f, indent=4)
print(f"Final metrics saved to {metrics_file}")
