# train_eval_lasso-years-filtered_annotations.py

import os
import pandas as pd
import numpy as np
import json

from sklearn.model_selection import train_test_split, GridSearchCV
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
print(f'EVALUATION_YEAR={EVALUATION_YEAR}')
print(f'TRAIN_EVAL_DATA_PATH={TRAIN_EVAL_DATA_PATH}')
print(f'TRAIN_EVAL_DATA_FILE={TRAIN_EVAL_DATA_FILE}')
print(f'SET_FEATURES={SET_FEATURES}')
print(f'FEATURE_COLUMNS[SET_FEATURES]={FEATURE_COLUMNS[SET_FEATURES]}')
print(f'FEATURE_SELECTION={FEATURE_SELECTION}')
print(f'COLUMN_TASK_GOLD={COLUMN_TASK_GOLD}')
print(f'ANNOTATION_POSITIVE={ANNOTATION_POSITIVE}')
print(f'ANNOTATION_NEGATIVE={ANNOTATION_NEGATIVE}')
print(f'TEST_SIZE_PERC={TEST_SIZE_PERC}')
print(f'STANDARIZE_DATA={STANDARIZE_DATA}')
print(f'MODEL_EVAL_SAVE_PATH={MODEL_EVAL_SAVE_PATH}')
print(f'OUTPUT_METRICS_PATH={OUTPUT_METRICS_PATH}')
print(f'OUTPUT_PREDICTIONS_EVAL_PATH={OUTPUT_PREDICTIONS_EVAL_PATH}')
print(f'USE_SUBSET_FINAL_ASSIGNMENTS={USE_SUBSET_FINAL_ASSIGNMENTS}')
print(f'BALANCE_CLASSES={BALANCE_CLASSES}')
print(f'PREFIX={PREFIX}')
print('=================================')

prefix = f'{PREFIX}_subset_top_reviewers_' if USE_SUBSET_FINAL_ASSIGNMENTS else PREFIX

# Class used for feature selection using correlations.
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

# Pipeline definition.
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
    # Pipeline for chi2 needs MinMaxScaler to ensure non-negative values
    pipeline = Pipeline([
        ('scaler', MinMaxScaler()),  # Changed from StandardScaler to MinMaxScaler
        ('feature_selection', SelectKBest(chi2)),  # Changed from f_classif to chi2
        ('classifier', LogisticRegression(penalty="l1", solver="liblinear", max_iter=5000, random_state=42))
    ])
elif FEATURE_SELECTION == 'rf':
    # Pipeline with RandomForest-based feature selection
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('feature_selection', SelectFromModel(
            RandomForestClassifier(random_state=42),
            max_features=None  # This will be tuned via grid search
        )),
        ('classifier', LogisticRegression(penalty="l1", solver="liblinear", max_iter=5000, random_state=42))
    ])
elif FEATURE_SELECTION == 'corr': 
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('feature_selection', CorrelationFeatureSelector()),
        ('classifier', LogisticRegression(penalty="l1", solver="liblinear", max_iter=5000, random_state=42))
    ])
# We do not perform feature selection.
else:
    print("*** NOT PERFORMING FEATURE SELECTION ***")
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(penalty="l1", solver="liblinear", max_iter=5000, random_state=42))
    ])

# Hyper-parameter grid.
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
else:
    pass

SCORING_METRIC_CV = "roc_auc"

# Define cross-validation grid search.
grid_search = GridSearchCV(
    pipeline,
    param_grid=PARAM_GRID,
    scoring=SCORING_METRIC_CV,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
)

print(f"SCORING_METRIC_CV={SCORING_METRIC_CV}")

# Load data.
data_train_eval_path = f"{TRAIN_EVAL_DATA_PATH}/{prefix}{TRAIN_EVAL_DATA_FILE}"
print(f'Reading training/test data from {data_train_eval_path}')
data = pd.read_csv(data_train_eval_path, sep='\t')
data['Year'] = data['Project_ID'].str.extract(r'^(\d{4})').astype(int)

# Exclude evaluation year for filtering
eval_year_data = data[data['Year'] == EVALUATION_YEAR].copy()
non_eval_year_data = data[data['Year'] != EVALUATION_YEAR].copy()

# Train model on non-EVALUATION_YEAR data for initial predictions
X_train_initial = non_eval_year_data[FEATURE_COLUMNS[SET_FEATURES]]
y_train_initial = non_eval_year_data[COLUMN_TASK_GOLD].apply(lambda x: 1 if x == ANNOTATION_POSITIVE else 0)

# Initial feature selection/hyper-parameter optimization.
grid_search.fit(X_train_initial, y_train_initial)

# Best parameters and selected features for initial model.
best_pipeline_initial = grid_search.best_estimator_
best_params_initial = grid_search.best_params_

if 'feature_selection' in best_pipeline_initial.named_steps:
    selected_features_mask = best_pipeline_initial.named_steps['feature_selection'].get_support()
    selected_features = X_train_initial.columns[selected_features_mask]
else:
    selected_features = FEATURE_COLUMNS[SET_FEATURES]

# Predict probabilities on non-EVALUATION_YEAR data
non_eval_year_data["Predicted_Label"] = best_pipeline_initial.predict(X_train_initial)
non_eval_year_data["Probability_Positive"] = best_pipeline_initial.predict_proba(X_train_initial)[:, 1]

# Calculate total examples across all years
total_examples = len(non_eval_year_data)

DYNAMIC_FILTERING = False
BOUNDARY_PREDICTIONS = 0.5
ADJUSTMENT_FACTOR = 0.2

# Default fixed thresholds to remove mis-annotated/noisy instances.
DEFAULT_LOWER_THRESHOLD = BOUNDARY_PREDICTIONS - ADJUSTMENT_FACTOR
DEFAULT_HIGHER_THRESHOLD = BOUNDARY_PREDICTIONS + ADJUSTMENT_FACTOR

if DYNAMIC_FILTERING:
    DECISION_BOUNDARY_MISCLASSIFICATION = 0.1  # Threshold for identifying noisy years based on misclassification rate.
    # Compute weighted misclassification rates per year
    weighted_misclassification_rates = {}
    for year in non_eval_year_data["Year"].unique():
        year_data = non_eval_year_data[non_eval_year_data["Year"] == year]
        # Identify misclassified examples
        misclassified = (
            ((year_data["Annotation"] == ANNOTATION_POSITIVE) & (year_data["Predicted_Label"] == 0)) |
            ((year_data["Annotation"] == ANNOTATION_NEGATIVE) & (year_data["Predicted_Label"] == 1))
        )
        # Calculate weighted misrate
        weighted_misclassification_rates[year] = misclassified.sum() / total_examples
    # Calculate the nth percentile of misclassification rates
    mis_rate_values = list(weighted_misclassification_rates.values())
    percentile_n = np.percentile(mis_rate_values, 90)
    # Define hybrid thresholds based on weighted misclassification rates.
    year_thresholds = {}
    for year, weighted_mis_rate in weighted_misclassification_rates.items():
        # Use dynamic threshold for noisy years.
        if weighted_mis_rate > min(DECISION_BOUNDARY_MISCLASSIFICATION, percentile_n):
            # Use weighted thresholds for noisy years.
            weighted_lower = max(DEFAULT_LOWER_THRESHOLD, BOUNDARY_PREDICTIONS - ADJUSTMENT_FACTOR * weighted_mis_rate)
            weighted_higher = min(DEFAULT_HIGHER_THRESHOLD, BOUNDARY_PREDICTIONS + ADJUSTMENT_FACTOR * weighted_mis_rate)
            year_thresholds[year] = {"LOWER_THRESHOLD": weighted_lower, "HIGHER_THRESHOLD": weighted_higher}
        else:
            # Use fixed thresholds for other years.
            year_thresholds[year] = {"LOWER_THRESHOLD": DEFAULT_LOWER_THRESHOLD, "HIGHER_THRESHOLD": DEFAULT_HIGHER_THRESHOLD}
else:
    # Apply fixed thresholds for all years in a consistent way.
    year_thresholds = {
        year: {"LOWER_THRESHOLD": DEFAULT_LOWER_THRESHOLD, "HIGHER_THRESHOLD": DEFAULT_HIGHER_THRESHOLD}
        for year in non_eval_year_data["Year"].unique()
    }

# Print thresholds.
print("Weighted thresholds by year")
for year, thresholds in year_thresholds.items():
    print(f"Year {year}: LOWER_THRESHOLD={year_thresholds[year]['LOWER_THRESHOLD']}, HIGHER_THRESHOLD={year_thresholds[year]['HIGHER_THRESHOLD']}")

# Filter data using year-specific thresholds
filtered_train_data = pd.DataFrame()
for year in non_eval_year_data["Year"].unique():
    year_data = non_eval_year_data[non_eval_year_data["Year"] == year]
    original_samples = len(year_data)
    thresholds = year_thresholds[year]
    exclude_conditions = (
        (
            (year_data["Annotation"] == ANNOTATION_POSITIVE) &
            (year_data["Probability_Positive"] <= thresholds["LOWER_THRESHOLD"])
        ) |
        (
            (year_data["Annotation"] == ANNOTATION_NEGATIVE) &
            (year_data["Probability_Positive"] >= thresholds["HIGHER_THRESHOLD"])
        )
    )
    filtered_year_data = year_data[~exclude_conditions]
    filtered_samples = len(filtered_year_data)
    # Print sample counts for the year
    print(f"Year {year}: Original samples = {original_samples}, Filtered samples = {filtered_samples}")
    filtered_train_data = pd.concat([filtered_train_data, filtered_year_data])

# Prepare training data after filtering
#X_train_filtered = filtered_train_data[selected_features]
X_train_filtered = filtered_train_data[FEATURE_COLUMNS[SET_FEATURES]]  # Use all features
y_train_filtered = filtered_train_data[COLUMN_TASK_GOLD].apply(lambda x: 1 if x == ANNOTATION_POSITIVE else 0)

# Feature selection/hyper-parameter optimization with filtered data.
grid_search.fit(X_train_filtered, y_train_filtered)

# Best parameters for filtered data.
best_pipeline_filtered = grid_search.best_estimator_
best_params_filtered = grid_search.best_params_

if 'feature_selection' in best_pipeline_filtered.named_steps:
    selected_features_mask = best_pipeline_filtered.named_steps['feature_selection'].get_support()
    selected_features = X_train_filtered.columns[selected_features_mask]
else:
    selected_features = FEATURE_COLUMNS[SET_FEATURES]

excluded_features = set(FEATURE_COLUMNS[SET_FEATURES]) - set(selected_features)

print(f"Number original features: {len(excluded_features)+len(selected_features)}")
print(f"Number selected features: {len(selected_features)}")
print(f"Selected features: {selected_features}")
print(f"Excluded features: {excluded_features}")

print(f"Best parameters for final model after filtering: {best_params_filtered}")

# Evaluate on evaluation year data.
#X_test_eval_year = eval_year_data[selected_features]
X_test_eval_year = eval_year_data[FEATURE_COLUMNS[SET_FEATURES]]  # Use all features instead of selected_features
y_test_eval_year = eval_year_data[COLUMN_TASK_GOLD].apply(lambda x: 1 if x == ANNOTATION_POSITIVE else 0)

# Use the best pipeline for evaluation.
evaluation_metrics_eval_year = evaluate_model(best_pipeline_filtered, X_test_eval_year, y_test_eval_year)

# Add predictions to evaluation year data
eval_year_data["Predicted_Label"] = best_pipeline_filtered.predict(X_test_eval_year)
eval_year_data["Probability_Positive"] = evaluation_metrics_eval_year["probs"]

# Save predictions for year 2023
output_predictions_file = f"{OUTPUT_PREDICTIONS_EVAL_PATH}/predictions_year_{EVALUATION_YEAR}_{prefix}{SET_FEATURES}_filtered.tsv"
os.makedirs(OUTPUT_PREDICTIONS_EVAL_PATH, exist_ok=True)
eval_year_data.to_csv(output_predictions_file, sep='\t', index=False)
print(f"Predictions for year {EVALUATION_YEAR} saved to {output_predictions_file}")

# Print final evaluation metrics for Year 2023
print(f"Evaluation metrics for year {EVALUATION_YEAR}:")
print(f"AUC: {evaluation_metrics_eval_year['AUC']:.3f}")
print(f"Accuracy: {evaluation_metrics_eval_year['Accuracy']:.3f}")
print(f"F1: {evaluation_metrics_eval_year['F1']:.3f}")
print(f"Precision: {evaluation_metrics_eval_year['Precision']:.3f}")
print(f"Recall: {evaluation_metrics_eval_year['Recall']:.3f}")

# Save final metrics to JSON
results = {
    "Year {EVALUATION_YEAR} metrics": {
        k: float(v) if isinstance(v, (np.float32, np.float64)) else v
        for k, v in evaluation_metrics_eval_year.items() if k != "probs"
    }
}
os.makedirs(OUTPUT_METRICS_PATH, exist_ok=True)
results_file = f"{OUTPUT_METRICS_PATH}/final_results_year_{EVALUATION_YEAR}_{prefix}{SET_FEATURES}_filtered.json"
with open(results_file, "w") as f:
    json.dump(results, f, indent=4)
print(f"Final metrics for year {EVALUATION_YEAR} saved to {results_file}")

