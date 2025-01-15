# File: make_final_predictions.py

import os
import pandas as pd

from modules.model_training import load_model
from modules.utils import rank_experts
from config import (
    MODEL_NAME,
    TRAIN_EVAL_DATA_PATH,
    EXPERT_PROJECT_SCORES_FILE,
    OUTPUT_PREDICTIONS_FINAL_PATH,
    PREDICTIONS_ALL_PAIRS_FILE,
    MODEL_FINAL_SAVE_PATH,
    FEATURE_COLUMNS,
    SET_FEATURES,
    USE_SUBSET_FINAL_ASSIGNMENTS,
    PREFIX
)

def make_predictions(model_path, input_file_features, output_file_predictions):
    # Load the input data (e.g., feature matrix for expert-project pairs).
    print(f"Reading features for expert-project pairs from file {input_file_features}")
    input_data = pd.read_csv(input_file_features, sep='\t')
    X_input = input_data[FEATURE_COLUMNS[SET_FEATURES]]
    # Load the trained model pipeline.
    model = load_model(model_path)
    print(f'Making predictions for {len(X_input)} pairs with model {model_path}.')
    # Predict probabilities using the pipeline (includes scaling and selection).
    input_data["Predicted_Prob"] = model.predict_proba(X_input)[:, 1]
    # Get rankings for each project.
    input_data = rank_experts(input_data, input_data["Predicted_Prob"], "Predicted_Prob")
    # Save predictions to a file.
    input_data.to_csv(output_file_predictions, sep="\t", index=False)
    print(f"Predictions saved to {output_file_predictions}")

if __name__ == "__main__":
    prefix = f'{PREFIX}_subset_top_reviewers_' if USE_SUBSET_FINAL_ASSIGNMENTS else PREFIX
    model_path = f"{MODEL_FINAL_SAVE_PATH}/final_model_{prefix}{SET_FEATURES}.pkl"
    features_path = f"{TRAIN_EVAL_DATA_PATH}/{EXPERT_PROJECT_SCORES_FILE}"
    output_predictions_path = f"{OUTPUT_PREDICTIONS_FINAL_PATH}/{MODEL_NAME}_{prefix}{SET_FEATURES}_{PREDICTIONS_ALL_PAIRS_FILE}"
    os.makedirs(OUTPUT_PREDICTIONS_FINAL_PATH, exist_ok=True)
    make_predictions(model_path, features_path, output_predictions_path)

