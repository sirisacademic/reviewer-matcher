import os
import pandas as pd
import numpy as np
import joblib

class ExpertRanker:
    def __init__(self, config_manager):
        """
        Initialize the ExpertRanker with configuration settings.
        """
        # Column mappings.
        self.project_cols = config_manager.get('OUTPUT_COLUMNS_PROJECTS')
        self.expert_cols = config_manager.get('OUTPUT_COLUMNS_EXPERTS')
        # Input columns.
        self.expert_id_input_col = config_manager.get('EXPERT_ID_INPUT_COLUMN', 'ID')
        self.project_id_input_col = config_manager.get('PROJECT_ID_INPUT_COLUMN', 'ID')
        # Output columns.
        self.expert_id_output_col = self.expert_cols[self.expert_id_input_col]
        self.project_id_output_col = self.project_cols[self.project_id_input_col]
        self.predicted_prob_col = config_manager.get('PREDICTED_PROB_COLUMN', 'Predicted_Prob')
        self.predicted_prob_rank_col = config_manager.get('PREDICTED_PROB_RANK_COLUMN', 'Predicted_Prob_Rank')
        # Features.
        self.feature_columns = config_manager.get('FEATURE_COLUMNS')
        self.set_features = config_manager.get('SET_FEATURES')
        # Model.
        self.model_path = config_manager.get('MODEL_PATH')
        self._load_model()

    def _load_model(self):
        """Load the pre-trained model pipeline."""
        try:
            self.model = joblib.load(self.model_path)
            print(f"Model pipeline loaded successfully from {self.model_path}")
        except Exception as e:
            raise RuntimeError(f"Error loading model pipeline: {str(e)}")

    def generate_predictions(self, features_df):
        """
        Generate predictions and rankings for expert-project pairs.
        Args:
            features_df: DataFrame containing features and Expert_ID, Project_ID columns
        Returns:
            DataFrame with predictions and rankings
        """
        try:
            # Select features for prediction
            X_input = features_df[self.feature_columns[self.set_features]].copy()
            print(f'Making predictions for {len(X_input)} expert-project pairs')
            # Generate predictions using the pipeline
            features_df[self.predicted_prob_col] = self.model.predict_proba(X_input)[:, 1]
            # Generate rankings based on probabilities
            features_df[self.predicted_prob_rank_col] = features_df.groupby(self.project_id_output_col)[self.predicted_prob_col].rank(
                ascending=False, 
                method="min"
            )
            # Select only required columns
            output = features_df[[self.expert_id_output_col, self.project_id_output_col, self.predicted_prob_col, self.predicted_prob_rank_col]]
            # Sort by project ID and probability
            output = output.sort_values(
                [self.project_id_output_col, self.predicted_prob_col],
                ascending=[True, False]
            )
            return output
        except Exception as e:
            raise RuntimeError(f"Error generating predictions: {str(e)}")


