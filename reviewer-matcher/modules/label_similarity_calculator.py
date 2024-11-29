import pandas as pd
import abbreviations
import re
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import jaccard_score

from tqdm import tqdm

class LabelSimilarityCalculator:
    def __init__(self, config_manager):
        # Configurations read from config file handled by config_manager.
        self.separator_output = config_manager.get('SEPARATOR_VALUES_OUTPUT', '|')
        self.OUTPUT_FILE_EXPERT_PROJECT_JACCARD_SIMILARITY  = config_manager.get('OUTPUT_FILE_EXPERT_PROJECT_JACCARD_SIMILARITY')
        # self.expert_publications = experts
        # self.projects = projects
        
    def convert_to_list(self, column_value):
        '''
        Convert a column of strings to lists, removing empty or whitespace-only entries
        '''

        if pd.isna(column_value) or column_value == '':
            return []
        return [item.strip() for item in column_value.split(self.separator_output) if item.strip() != '']

    def compute_similarity(self, experts_data, projects_data):
        '''Compute overlap in expert-project research areas and approaches based on Jaccard similarity.'''

        # Extract all unique research areas and approaches
        all_research_areas = set()
        all_research_approaches = set()

        for col in ['RESEARCH_AREAS', 'RESEARCH_APPROACHES']:
            set_terms = all_research_areas if col == 'RESEARCH_AREAS' else all_research_approaches
            set_terms.update(
                term.strip() for terms in pd.concat([self.experts_data[col], self.projects_data[col]]).dropna().apply(lambda x: x.split(self.separator_output))
                for term in terms
                if term.strip()
            )

        all_research_areas = list(all_research_areas)
        all_research_approaches = list(all_research_approaches)

        # One-Hot encoding for experts and projects

        # Initialize one-hot encoder
        mlb_areas = MultiLabelBinarizer(classes=all_research_areas)
        mlb_approaches = MultiLabelBinarizer(classes=all_research_approaches)

        # Convert and one-hot encode experts and projects' research areas
        expert_areas_one_hot = mlb_areas.fit_transform(self.experts_data['RESEARCH_AREAS'].apply(lambda x: self.convert_to_list(x)))
        project_areas_one_hot = mlb_areas.transform(self.projects_data['RESEARCH_AREAS'].apply(lambda x: self.convert_to_list(x)))

        # Convert and one-hot encode experts and projects' for research approaches
        expert_approaches_one_hot = mlb_approaches.fit_transform(self.experts_data['RESEARCH_APPROACHES'].apply(lambda x: self.convert_to_list(x)))
        project_approaches_one_hot = mlb_approaches.transform(self.projects_data['RESEARCH_APPROACHES'].apply(lambda x: self.convert_to_list(x)))

        # Calculate Jaccard similarity

        # Initialize list to store similarity results
        expert_project_jaccard_similarity_scores = []

        # Iterate over all expert-project pairs
        total_iterations = len(self.experts_data) * len(self.projects_data)

        with tqdm(total=total_iterations, desc="Calculating Jaccard similarity scores") as pbar:
            for expert_index, expert_row in self.experts_data.iterrows():
                for project_index, project_row in self.projects_data.iterrows():
                    pbar.update(1)

                    # Calculate Jaccard Similarity for Research Areas
                    area_similarity = jaccard_score(
                        expert_areas_one_hot[expert_index],
                        project_areas_one_hot[project_index],
                        average='binary'
                    )

                    # Calculate Jaccard Similarity for Research Approaches
                    approach_similarity = jaccard_score(
                        expert_approaches_one_hot[expert_index],
                        project_approaches_one_hot[project_index],
                        average='binary'
                    )

                    # Store the similarity scores
                    expert_project_jaccard_similarity_scores.append({
                        'Expert_ID': expert_row['ID'],
                        'Project_ID': project_row['ID'],
                        'Research_Areas_Jaccard_Similarity': area_similarity,
                        'Research_Approaches_Jaccard_Similarity': approach_similarity,
                    })

        # Convert to DataFrame
        df_jaccard_similarity_scores = pd.DataFrame(expert_project_jaccard_similarity_scores)

        # Save the similarity scores to a file for further analysis
        df_jaccard_similarity_scores.to_pickle(self.OUTPUT_FILE_EXPERT_PROJECT_JACCARD_SIMILARITY)
        print(f'Expert-project Jaccard similarity scores saved to {self.OUTPUT_FILE_EXPERT_PROJECT_JACCARD_SIMILARITY}')

        return df_jaccard_similarity_scores

