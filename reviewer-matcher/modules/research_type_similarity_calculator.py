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
