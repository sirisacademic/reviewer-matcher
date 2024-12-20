# File: feature_generator.py

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class FeatureGenerator:
    def __init__(self, config_manager):
        """Initialize the FeatureGenerator with feature groups and PCA options."""
        self.pca_variance_threshold = config_manager.get('PCA_VARIANCE_THRESHOLD', 0.9)
        # TODO: Unify handling of input output columns !!!!
        # Input columns pre-computed scores.
        self.feature_groups = config_manager.get('FEATURE_GROUPS')
        self.expert_id_col = config_manager.get('EXPERT_ID_COLUMN', 'Expert_ID')
        self.project_id_col = config_manager.get('PROJECT_ID_COLUMN', 'Project_ID')
        # Input columns experts data.
        self.expert_id_experts_col = config_manager.get('COLUMN_EXPERT_ID', 'ID')
        self.seniority_publications_column = config_manager.get('COLUMN_SENIORITY_PUBLICATIONS', 'SENIORITY_PUBLICATIONS')
        self.seniority_reviewer_column = config_manager.get('COLUMN_SENIORITY_REVIEWER', 'SENIORITY_REVIEWER')
        # Output columns experts data.
        self.output_column_expert_seniority_publications = config_manager.get('OUTPUT_COLUMN_EXPERT_SENIORITY_PUBLICATIONS', 'Expert_Seniority_Publications')
        self.output_column_expert_seniority_reviewer = config_manager.get('OUTPUT_COLUMN_EXPERT_SENIORITY_REVIEWER', 'Expert_Seniority_Reviewer')
        self.scaler = StandardScaler()

    def _unify_id_types(self, score_dataframes):
        """Unify the data types of the ID columns across DataFrames."""
        for df in score_dataframes:
            for col in [self.expert_id_col, self.project_id_col]:
                if col in df.columns:
                    df[col] = df[col].astype(int)

    def generate_features(self, experts, score_dataframes):
        """Generate features by normalizing, applying PCA, and creating aggregated ranks."""
        # Unify ID types across DataFrames
        self._unify_id_types(score_dataframes)
        # Merge all DataFrames on unified Expert_ID and Project_ID
        features = score_dataframes[0]
        for df in score_dataframes[1:]:
            features = pd.merge(features, df, on=[self.expert_id_col, self.project_id_col], how='inner')
        # Add experts' seniority features.
        # First make sure there are no missing values.
        experts_seniority_columns = [self.seniority_publications_column, self.seniority_reviewer_column]
        experts[experts_seniority_columns] = experts[experts_seniority_columns].apply(pd.to_numeric, errors='coerce')
        experts[experts_seniority_columns] = experts[experts_seniority_columns].fillna(experts[experts_seniority_columns].mean())
        features = features.merge(
            experts[[self.expert_id_experts_col, self.seniority_publications_column, self.seniority_reviewer_column]],
            how='left',
            left_on=self.expert_id_col,
            right_on=self.expert_id_experts_col
        ).drop(columns=[self.expert_id_experts_col])  # Drop 'ID' immediately after the merge
        # Rename the columns in the "features" dataframe
        features.rename(
            columns={
                self.seniority_publications_column: self.output_column_expert_seniority_publications,
                self.seniority_reviewer_column: self.output_column_expert_seniority_reviewer
            },
            inplace=True
        )
        # Get total of unique experts to normalize rankings.
        total_experts = features[self.expert_id_col].nunique()
        # Normalize numeric columns except Expert_ID and Project_ID
        numeric_columns = features.select_dtypes(include='number').columns.difference([self.expert_id_col, self.project_id_col])
        # Fill missing values with the mean of the column
        features[numeric_columns] = features[numeric_columns].fillna(features[numeric_columns].mean())
        # Normalize numeric columns except Expert_ID and Project_ID
        features[numeric_columns] = self.scaler.fit_transform(features[numeric_columns])
        # Apply PCA to feature groups
        for group_name, group_features in self.feature_groups.items():
            if any(feature in features.columns for feature in group_features):
                # Extract the feature group subset
                valid_features = [f for f in group_features if f in features.columns]
                # Perform PCA
                pca = PCA(n_components=self.pca_variance_threshold)
                X_pca = pca.fit_transform(features[valid_features])
                # Add PCA components to the main dataframe
                pca_columns = [f'{group_name}_PCA_{i+1}' for i in range(X_pca.shape[1])]
                pca_df = pd.DataFrame(X_pca, columns=pca_columns, index=features.index)
                features = pd.concat([features, pca_df], axis=1)
        # Compute average and ranks of PCA-transformed columns
        pca_score_columns = [col for col in features.columns if '_PCA_' in col]
        features['PCA_Average'] = features[pca_score_columns].mean(axis=1)
        features['PCA_Rank'] = (
            features.groupby(self.project_id_col)['PCA_Average']
            .rank(ascending=False, method='min', na_option='bottom')
        )
        features['PCA_Relative_Rank'] = features['PCA_Rank'] / total_experts
        # Compute average and ranks of all numeric columns (excluding IDs)
        features['All_Columns_Average'] = features[numeric_columns].mean(axis=1)
        features['All_Columns_Rank'] = (
            features.groupby(self.project_id_col)['All_Columns_Average'].rank(ascending=False, method='min', na_option='bottom')
        )
        features['All_Columns_Relative_Rank'] = features['All_Columns_Rank'] / total_experts
        return features

