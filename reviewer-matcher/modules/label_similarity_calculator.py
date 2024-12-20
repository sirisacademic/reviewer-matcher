# File: label_similarity_calculator.py

import pandas as pd
from tqdm import tqdm
from utils.functions_similarity import (
    convert_to_list,
    calculate_jaccard_similarity,
    calculate_dice_similarity,
    one_hot_encode,
    calculate_overlap_coefficient
)

class LabelSimilarityCalculator:
    def __init__(self, config_manager):
        """Initialize the LabelSimilarityCalculator with configuration and required settings."""
        self.separator_output = config_manager.get('SEPARATOR_VALUES_OUTPUT', '|')
        self.col_id_project = config_manager.get('ID_COLUMN_NAME', 'ID')
        self.col_id_expert = config_manager.get('ID_COLUMN_NAME', 'ID')
        self.research_areas_column = config_manager.get('RESEARCH_AREAS_COLUMN', 'RESEARCH_AREAS')
        self.research_approaches_column = config_manager.get('RESEARCH_APPROACHES_COLUMN', 'RESEARCH_APPROACHES')

    def compute_similarity(self, experts_data, projects_data):
    
        """Compute various similarity scores between experts and projects."""
        # Prepare unique terms.
        all_research_areas = self._prepare_unique_terms(experts_data, projects_data, self.research_areas_column)
        all_research_approaches = self._prepare_unique_terms(experts_data, projects_data, self.research_approaches_column)
        # One-hot encode data.
        experts_encoded = {
            'areas': one_hot_encode(experts_data, self.research_areas_column, all_research_areas, self.separator_output),
            'approaches': one_hot_encode(experts_data, self.research_approaches_column, all_research_approaches, self.separator_output),
        }
        projects_encoded = {
            'areas': one_hot_encode(projects_data, self.research_areas_column, all_research_areas, self.separator_output),
            'approaches': one_hot_encode(projects_data, self.research_approaches_column, all_research_approaches, self.separator_output),
        }
        # Calculate similarity.
        similarity_scores = []
        total_iterations = len(experts_data) * len(projects_data)
        with tqdm(total=total_iterations, desc="Calculating similarity scores") as pbar:
            for expert_index, expert_row in experts_data.iterrows():
                for project_index, project_row in projects_data.iterrows():
                    pbar.update(1)
                    # Convert research areas/approaches to sets.
                    expert_areas = set(convert_to_list(expert_row[self.research_areas_column], self.separator_output))
                    project_areas = set(convert_to_list(project_row[self.research_areas_column], self.separator_output))
                    expert_approaches = set(convert_to_list(expert_row[self.research_approaches_column], self.separator_output))
                    project_approaches = set(convert_to_list(project_row[self.research_approaches_column], self.separator_output))
                    # Jaccard and Dice.
                    area_jaccard = calculate_jaccard_similarity(
                        experts_encoded['areas'][expert_index],
                        projects_encoded['areas'][project_index]
                    )
                    approach_jaccard = calculate_jaccard_similarity(
                        experts_encoded['approaches'][expert_index],
                        projects_encoded['approaches'][project_index]
                    )
                    area_dice = calculate_dice_similarity(
                        experts_encoded['areas'][expert_index],
                        projects_encoded['areas'][project_index]
                    )
                    approach_dice = calculate_dice_similarity(
                        experts_encoded['approaches'][expert_index],
                        projects_encoded['approaches'][project_index]
                    )
                    # Overlap coefficient.
                    area_overlap = calculate_overlap_coefficient(expert_areas, project_areas)
                    approach_overlap = calculate_overlap_coefficient(expert_approaches, project_approaches)
                    # Append scores.
                    # TODO: Get output column names for experts/projects from configuration files!!!
                    similarity_scores.append({
                        'Expert_ID': expert_row[self.col_id_expert],
                        'Project_ID': project_row[self.col_id_project],
                        'Research_Areas_Jaccard_Similarity': area_jaccard,
                        'Research_Areas_Dice_Similarity': area_dice,
                        'Research_Areas_Overlap_Coefficient': area_overlap,
                        'Research_Approaches_Jaccard_Similarity': approach_jaccard,
                        'Research_Approaches_Dice_Similarity': approach_dice,
                        'Research_Approaches_Overlap_Coefficient': approach_overlap,
                    })
        # Convert results to DataFrame
        expert_project_scores = pd.DataFrame(similarity_scores)
        # Return similarity scores
        return expert_project_scores

    def _prepare_unique_terms(self, experts_data, projects_data, column):
        """Collect all unique terms from the given column across experts and projects datasets."""
        combined_terms = pd.concat([experts_data[column], projects_data[column]]).dropna()
        unique_terms = set(
            term.strip() for terms in combined_terms.apply(lambda x: convert_to_list(x, self.separator_output))
            for term in terms
        )
        return list(unique_terms)

