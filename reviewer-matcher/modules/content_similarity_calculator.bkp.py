# File: content_similarity_calculator.py

import pandas as pd
from sentence_transformers import SentenceTransformer, util
from utils.functions_similarity import (
    convert_to_list,
    cluster_items,
    compute_list_similarity,
    compute_specificity_weight,
    process_publication_project_pairs,
    aggregate_expert_scores
)

class ContentSimilarityCalculator:
    def __init__(self, config_manager):
        """Initialize the ContentSimilarityCalculator with configuration and thresholds."""
        text_similarity_model = config_manager.get('TEXT_SIMILARITY_MODEL', 'FremyCompany/BioLORD-2023')
        self.distance_threshold_clusters = config_manager.get('DISTANCE_THRESHOLD_CLUSTERS', 0.2)
        self.research_topic_column = config_manager.get('RESEARCH_TOPIC_COLUMN', 'RESEARCH_TOPIC')
        self.methods_specific_column = config_manager.get('METHODS_SPECIFIC_COLUMN', 'METHODS_SPECIFIC')
        self.methods_column = config_manager.get('METHODS_COLUMN', 'METHODS')
        self.objectives_column = config_manager.get('OBJECTIVES_COLUMN', 'OBJECTIVES')
        self.separator_output = config_manager.get('SEPARATOR_VALUES_OUTPUT', '|')
        # ID columns.
        self.col_id_project = config_manager.get('ID_COLUMN_NAME', 'ID')
        self.col_id_pub_publication = config_manager.get('COLUMN_PUB_PUBLICATION_ID', 'PUB_ID')
        self.col_id_pub_expert = config_manager.get('COLUMN_PUB_EXPERT_ID', 'EXPERT_ID')
        # Initialize model
        self.model = SentenceTransformer(text_similarity_model)
        # Initialize cluster-related properties for methods to give less weight to very frequent methods.
        self.method_to_cluster = None
        self.cluster_counts = None

    def compute_similarity(self, publications, projects):
        """Compute similarity scores between publication content and project content."""
        # Ensure clusters are computed.
        if (
            self.method_to_cluster is None or self.cluster_counts is None or
            self.method_specific_to_cluster is None or self.method_specific_cluster_counts is None
        ):
            self._compute_clusters(publications)
        # Process publication-project pairs.
        df_publication_scores = process_publication_project_pairs(publications, projects, self._process_pub_project_pair)
        # Aggregate scores.
        agg_funcs = {
            'Topic_Similarity': ['max', 'avg'],
            'Objectives_Max_Similarity': ['max', 'avg'],
            'Objectives_Avg_Similarity': ['max', 'avg'],
            'Methods_Specific_Max_Similarity': ['max', 'avg'],
            'Methods_Specific_Avg_Similarity': ['max', 'avg'],
            'Methods_Max_Similarity': ['max', 'avg'],
            'Methods_Avg_Similarity': ['max', 'avg'],
            'Methods_Max_Similarity_Weighted': ['max', 'avg'],
            'Methods_Avg_Similarity_Weighted': ['max', 'avg']
        }
        # TODO: Get output column names for experts/projects from configuration files!!!
        expert_project_scores = aggregate_expert_scores(df_publication_scores, agg_funcs, expert_id_col='Expert_ID', project_id_col='Project_ID')
        return expert_project_scores

    def _compute_clusters(self, publications):
        """Compute clusters for METHODS from publication content."""
        # Methods clusters.
        all_methods = [
            method.strip()
            for methods in publications[self.methods_column]
            for method in convert_to_list(methods, self.separator_output)
        ]
        self.method_to_cluster, self.cluster_counts, _ = cluster_items(
            self.model, all_methods, self.distance_threshold_clusters
        )
        
    def _process_pub_project_pair(self, pub_row, project_row):
        """Process a single publication-project pair to compute similarity scores."""
        # Topic similarity.
        if pub_row[self.research_topic_column] and project_row[self.research_topic_column]:
            topic_similarity = util.pytorch_cos_sim(
                self.model.encode(pub_row[self.research_topic_column], convert_to_tensor=True, show_progress_bar=False),
                self.model.encode(project_row[self.research_topic_column], convert_to_tensor=True, show_progress_bar=False)
            ).cpu().item()
        else:
            topic_similarity = 0
        # Objectives similarity.
        objectives_avg_sim, objectives_max_sim = compute_list_similarity(
            self.model,
            convert_to_list(pub_row[self.objectives_column], self.separator_output),
            convert_to_list(project_row[self.objectives_column], self.separator_output)
        )
        # Methods specific similarity - not weighted.
        methods_specific_avg_sim, methods_specific_max_sim = compute_list_similarity(
            self.model,
            convert_to_list(pub_row[self.methods_specific_column], self.separator_output),
            convert_to_list(project_row[self.methods_specific_column], self.separator_output)
        )
        # All methods similarity.
        pub_methods = convert_to_list(pub_row[self.methods_column], self.separator_output)
        proj_methods = convert_to_list(project_row[self.methods_column], self.separator_output)
        methods_avg_sim, methods_max_sim = compute_list_similarity(self.model, pub_methods, proj_methods)
        # All methods similarity - weighted inversely proportionally to their frequency (based on clusters of similar methods).
        methods_weight = compute_specificity_weight(pub_methods, self.method_to_cluster, self.cluster_counts)
        methods_avg_sim_weighted = methods_avg_sim * methods_weight
        methods_max_sim_weighted = methods_max_sim * methods_weight
        # TODO: Get output column names for experts/projects from configuration files!!!
        return {
            'Pub_ID': pub_row[self.col_id_pub_publication],
            'Expert_ID': pub_row[self.col_id_pub_expert],
            'Project_ID': project_row[self.col_id_project],
            'Topic_Similarity': topic_similarity,
            'Objectives_Avg_Similarity': objectives_avg_sim,
            'Objectives_Max_Similarity': objectives_max_sim,
            'Methods_Specific_Avg_Similarity': methods_specific_avg_sim,
            'Methods_Specific_Max_Similarity': methods_specific_max_sim,
            'Methods_Avg_Similarity': methods_avg_sim,
            'Methods_Max_Similarity': methods_max_sim,
            'Methods_Avg_Similarity_Weighted': methods_avg_sim_weighted,
            'Methods_Max_Similarity_Weighted': methods_max_sim_weighted
        }




