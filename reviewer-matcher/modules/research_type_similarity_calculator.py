import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class ResearchTypeSimilarityCalculator:
    def __init__(self, config_manager):
        """
        Initialize the ResearchTypeSimilarityCalculator.
        """
        # retrieve weight and priority configurations for similarity calculations
        self.weight_exact_match = config_manager.get('WEIGHT_EXACT_MATCH')
        self.weight_partial_match = config_manager.get('WEIGHT_PARTIAL_MATCH')
        self.weight_basic_research_priority = config_manager.get('WEIGHT_BASIC_RESEARCH_PRIORITY')
        self.weight_related_match = config_manager.get('WEIGHT_RELATED_MATCH')

        # retrieve configuration settings for filtering and related research types
        self.related_types = config_manager.get('RELATED_TYPES')
        self.exclude_types = config_manager.get('EXCLUDE_TYPES')
        self.priority_type = config_manager.get('PRIORITY_TYPE')

        # set file paths for input and output data
        self.file_path_content_similarity = config_manager.get('FILE_PATH_EXPERT_PROJECT_CONTENT_SIMILARITY_SCORES')
        self.file_path_mesh_scores = config_manager.get('FILE_PATH_EXPERT_PROJECT_MESH_SCORES')
        self.file_path_jaccard_scores = config_manager.get('FILE_PATH_EXPERT_PROJECT_JACCARD_SIMILARITY')
        self.file_path_combined_scores = config_manager.get('FILE_PATH_COMBINED_SCORES')

        # define column mappings and normalization settings
        self.columns_projects = config_manager.get('COLUMNS_PROJECTS')
        self.columns_experts = config_manager.get('COLUMNS_EXPERTS')
        self.columns_to_normalize = config_manager.get('COLUMNS_TO_NORMALIZE')

    def compute_similarity(self, expert_types, project_types):
        """
        Compute the similarity score between expert and project research types.
        """
        # split the research types into sets and exclude unwanted types
        expert_set = set(expert_types.split('|')) - self.exclude_types
        project_set = set(project_types.split('|')) - self.exclude_types

        # calculate partial match score based on common types
        common_types = expert_set.intersection(project_set)
        score = len(common_types) * self.weight_partial_match

        # add extra weight for exact matches between expert and project types
        if expert_set == project_set:
            score += self.weight_exact_match

        # prioritize similarity for a specific research type if defined
        if self.priority_type in expert_set and self.priority_type in project_set:
            score += self.weight_basic_research_priority

        # calculate additional score for related types based on overlap
        for project_type in project_set:
            if project_type in self.related_types:
                related_match_count = len(expert_set.intersection(self.related_types[project_type]))
                score += related_match_count * self.weight_related_match

        return score

    def load_data(self):
        """
        Load data from specified file paths and return as dataframes.
        """
        # load content similarity, MeSH scores, and Jaccard similarity data from files
        df_content_similarity_scores = pd.read_pickle(self.file_path_content_similarity)
        df_mesh_scores = pd.read_pickle(self.file_path_mesh_scores)
        df_jaccard_similarity_scores = pd.read_pickle(self.file_path_jaccard_scores)
        return df_content_similarity_scores, df_mesh_scores, df_jaccard_similarity_scores

    def preprocess_data(self, df_projects, df_experts, df_content_similarity_scores, df_mesh_scores, df_jaccard_similarity_scores):
        """
        Preprocess and ensure uniformity across dataframes.
        """
        # rename project and expert columns based on the defined mappings
        df_projects = df_projects[list(self.columns_projects.keys())].rename(columns=self.columns_projects)
        df_experts = df_experts[list(self.columns_experts.keys())].rename(columns=self.columns_experts)

        # ensure consistent data types and formatting for ids and names in projects and experts
        df_projects['Project_ID'] = df_projects['Project_ID'].astype(str)
        df_experts['Expert_ID'] = df_experts['Expert_ID'].astype(str)
        df_experts['Expert_Full_Name'] = df_experts['Expert_Full_Name'].str.strip()

        # ensure consistent data types for project and expert ids in similarity score dataframes
        for df in [df_content_similarity_scores, df_mesh_scores, df_jaccard_similarity_scores]:
            df['Project_ID'] = df['Project_ID'].astype(str)
            df['Expert_ID'] = df['Expert_ID'].astype(str)

        return df_projects, df_experts, df_content_similarity_scores, df_mesh_scores, df_jaccard_similarity_scores

    def combine_data(self, df_projects, df_experts, df_content_similarity_scores, df_mesh_scores, df_jaccard_similarity_scores):
        """
        Combine all similarity scores and add metadata, return combined dataframe.
        """
        # merge Jaccard similarity scores with MeSH scores
        df_combined = pd.merge(
            df_jaccard_similarity_scores, df_mesh_scores, on=['Expert_ID', 'Project_ID'], how='inner'
        )
        # merge the result with content similarity scores
        df_combined = pd.merge(
            df_combined, df_content_similarity_scores, on=['Expert_ID', 'Project_ID'], how='inner'
        )

        # clean up column names by removing redundant prefixes
        new_column_names = {
            col: col.replace('Expert_', '', 1)
            for col in df_combined.columns if col.startswith('Expert_') and col != 'Expert_ID'
        }
        df_combined.rename(columns=new_column_names, inplace=True)

        # merge the combined data with project and expert metadata
        df_combined = pd.merge(df_combined, df_projects, on='Project_ID', how='left')
        df_combined = pd.merge(df_combined, df_experts, on='Expert_ID', how='left')

        return df_combined

    def calculate_similarity_scores(self, df_combined):
        """
        Calculate research type similarity scores for each expert-project pair.
        """
        # apply similarity calculation function to each row in the combined dataframe
        df_combined['Research_Type_Similarity_Score'] = df_combined.apply(
            lambda row: self.compute_similarity(row['Research_Types'], row['Project_Research_Types']),
            axis=1
        )
        return df_combined

    def normalize_columns(self, df_combined):
        """
        Normalize specified columns in the combined dataframe.
        """
        # scale specified columns to a range of 0 to 1 for consistency
        scaler = MinMaxScaler()
        df_combined[self.columns_to_normalize] = scaler.fit_transform(df_combined[self.columns_to_normalize])
        return df_combined

    def reorder_columns(self, df_combined):
        """
        Reorder columns in the dataframe and return the updated dataframe.
        """
        # define a fixed order for columns and rearrange the dataframe
        column_order = [
            # expert-specific columns
            'Expert_ID', 'Expert_Full_Name', 'Expert_Gender', 'Expert_Research_Types', 'Expert_Seniority',
            'Expert_Experience_Reviewer', 'Expert_Experience_Panel', 'Expert_Number_Publications', 'Expert_Number_Citations',
            # project-specific columns
            'Project_ID', 'Project_Title', 'Project_Research_Types',
            # similarity scores (Jaccard, Content, MeSH)
            'Research_Type_Similarity_Score',
            'Research_Areas_Jaccard_Similarity', 'Research_Approaches_Jaccard_Similarity',
            'Topic_Similarity_Max', 'Topic_Similarity_Avg',
            'Objectives_Max_Similarity_Max', 'Objectives_Max_Similarity_Avg',
            'Objectives_Avg_Similarity_Max', 'Objectives_Avg_Similarity_Avg',
            'Methods_Max_Similarity_Max', 'Methods_Max_Similarity_Avg',
            'Methods_Avg_Similarity_Max', 'Methods_Avg_Similarity_Avg',
            'Methods_Max_Similarity_Weighted_Max', 'Methods_Max_Similarity_Weighted_Avg',
            'Methods_Avg_Similarity_Weighted_Max', 'Methods_Avg_Similarity_Weighted_Avg',
            'MeSH_Semantic_Coverage_Score', 'MeSH_Max_Similarity_Max',
            'MeSH_Max_Similarity_Avg', 'MeSH_Avg_Similarity_Max',
            'MeSH_Avg_Similarity_Avg', 'MeSH_Max_Similarity_Weighted_Max',
            'MeSH_Max_Similarity_Weighted_Avg', 'MeSH_Avg_Similarity_Weighted_Max',
            'MeSH_Avg_Similarity_Weighted_Avg'
        ]
        return df_combined[column_order]

    def save_combined_data(self, df_combined):
        """
        Save the combined similarity scores to a file.
        """
        # export the final combined dataframe to a tab-separated file
        df_combined.to_csv(self.file_path_combined_scores, sep='\t', index=False)
