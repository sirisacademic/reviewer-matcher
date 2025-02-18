import pandas as pd

class ResearchTypeSimilarityCalculator:
    def __init__(self, config_manager):
        """
        Initialize the ResearchTypeSimilarityCalculator.
        """
        # General settings.
        self.separator_output = config_manager.get('SEPARATOR_VALUES_OUTPUT', '|')
        # Retrieve weight and priority configurations for similarity calculations
        self.weight_exact_match = config_manager.get('WEIGHT_EXACT_MATCH', 2.0)
        self.weight_partial_match = config_manager.get('WEIGHT_PARTIAL_MATCH', 1.0)
        self.weight_basic_research_priority = config_manager.get('WEIGHT_BASIC_RESEARCH_PRIORITY', 3.0)
        self.weight_related_match = config_manager.get('WEIGHT_RELATED_MATCH', 1.5)
        # Retrieve configuration settings for filtering and related research types
        self.related_types = {
            k.lower(): {v.lower() for v in values} for k, values in config_manager.get('RELATED_TYPES').items()
        }
        self.exclude_types = {t.lower() for t in config_manager.get('EXCLUDE_TYPES')}
        self.priority_type = config_manager.get('PRIORITY_TYPE', 'basic').lower()
        # Column mappings
        self.output_columns_experts = config_manager.get('OUTPUT_COLUMNS_EXPERTS')
        self.output_columns_projects = config_manager.get('OUTPUT_COLUMNS_PROJECTS') 
        # TODO: Unify the way in which input and output column names are included in configuration files !
        # Input column names.
        self.input_expert_id_column = 'ID'
        self.input_expert_research_type_column = 'RESEARCH_TYPES'
        self.input_project_id_column = 'ID'
        self.input_project_research_type_column = 'RESEARCH_TYPE'
        # Output column names.
        self.output_expert_id_column = self.output_columns_experts[self.input_expert_id_column]
        self.output_project_id_column = self.output_columns_projects[self.input_project_id_column]
        self.output_expert_research_type_column = self.output_columns_experts[self.input_expert_research_type_column]
        self.output_project_research_type_column = self.output_columns_projects[self.input_project_research_type_column]
        self.output_column_research_type_similarity = config_manager.get('OUTPUT_COLUMN_RESEARCH_TYPE_SIMILARITY', 'Research_Type_Similarity_Score')
       
    def compute_similarity(self, experts, projects):
        """Compute research type similarity scores for all expert-project pairs."""
        # Select and rename expert columns.
        experts = experts[
            [self.input_expert_id_column, self.input_expert_research_type_column]
        ].rename(columns={
            self.input_expert_id_column: self.output_expert_id_column,
            self.input_expert_research_type_column: self.output_expert_research_type_column
        })
        print(experts.columns)
        print(experts.head(1).values)
        # Select and rename project columns.
        projects = projects[
            [self.input_project_id_column, self.input_project_research_type_column]
        ].rename(columns={
            self.input_project_id_column: self.output_project_id_column,
            self.input_project_research_type_column: self.output_project_research_type_column
        })
        print(projects.columns)
        print(projects.head(1).values)
        # Create a Cartesian product of experts and projects with their research types.
        expert_project_research_types = experts.merge(projects, how='cross')
        # Apply similarity computation for each pair.
        expert_project_research_types[self.output_column_research_type_similarity] = expert_project_research_types.apply(
            lambda row: self._compute_pair_similarity(
                row[self.output_expert_research_type_column],
                row[self.output_project_research_type_column]
            ),
            axis=1
        )
        # Return the experts and projects with their similarity.
        return expert_project_research_types

    def _compute_pair_similarity(self, expert_types, project_types):
        """Compute the similarity score between expert and project research types."""
        expert_set = (
            set(map(str.lower, expert_types.split(self.separator_output))) - self.exclude_types
            if pd.notna(expert_types) and expert_types.strip() else set()
        )
        project_set = (
            set(map(str.lower, project_types.split(self.separator_output))) - self.exclude_types
            if pd.notna(project_types) and project_types.strip() else set()
        )
        # Calculate partial match score based on common types
        common_types = expert_set.intersection(project_set)
        score = len(common_types) * self.weight_partial_match
        # Add extra weight for exact matches between expert and project types
        if expert_set == project_set:
            score += self.weight_exact_match
        # Prioritize similarity for a specific research type if defined
        if self.priority_type in expert_set and self.priority_type in project_set:
            score += self.weight_basic_research_priority
        # Calculate additional score for related types based on overlap
        for project_type in project_set:
            if project_type in self.related_types:
                related_match_count = len(expert_set.intersection(self.related_types[project_type]))
                score += related_match_count * self.weight_related_match
        return score

