# File: mesh_similarity_calculator.py

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

class MeSHSimilarityCalculator:
    def __init__(self, config_manager):
        """Initialize the MeSHSimilarityCalculator with configuration and thresholds."""
        self.separator_output = config_manager.get('SEPARATOR_VALUES_OUTPUT', '|')
        self.distance_threshold_clusters = config_manager.get('DISTANCE_THRESHOLD_CLUSTERS', 0.2)
        self.similarity_threshold_terms = config_manager.get('SIMILARITY_THRESHOLD_TERMS', 0.6)
        text_similarity_model = config_manager.get('TEXT_SIMILARITY_MODEL', 'FremyCompany/BioLORD-2023')
        self.include_experts_mesh_coverage_score = config_manager.get('INCLUDE_EXPERTS_MESH_COVERAGE_SCORE', False)
        self.mesh_combined_output_column = config_manager.get('MESH_COMBINED_OUTPUT_COLUMN', 'MESH_EXTRACTED')
        # ID columns.
        self.col_id_project = config_manager.get('ID_COLUMN_NAME', 'ID')
        self.col_id_pub_publication = config_manager.get('COLUMN_PUB_PUBLICATION_ID', 'PUB_ID')
        self.col_id_pub_expert = config_manager.get('COLUMN_PUB_EXPERT_ID', 'EXPERT_ID')
        # Initialize model.
        self.model = SentenceTransformer(text_similarity_model)
        # Initialize cluster-related properties.
        self.mesh_to_cluster = None
        self.cluster_counts = None

    def compute_similarity(self, publications, projects):
        """Compute MeSH similarity scores between expert publications and project proposals."""
        #### !!!!!!!!!!!!!!!!!!!! TEST !!!!!!!!!!!!!!!!
        #projects = projects[projects[self.col_id_project]==228]
        #publications = publications[publications[self.col_id_pub_expert]<=50]
        ####################################################################
        # Ensure clusters are computed
        if self.mesh_to_cluster is None or self.cluster_counts is None:
            self._compute_clusters(publications, projects)
        # Process publication-project pairs.
        publication_project_scores = process_publication_project_pairs(publications, projects, self._process_pub_project_pair)
        # Define aggregation functions.
        agg_funcs = {
            'MeSH_Max_Similarity': ['max', 'avg'],
            'MeSH_Avg_Similarity': ['max', 'avg'],
            'MeSH_Max_Similarity_Weighted': ['max', 'avg'],
            'MeSH_Avg_Similarity_Weighted': ['max', 'avg']
        }
        # Aggregate scores to expert-project level.
        # TODO: Get output column names for experts/projects from configuration files!!!
        expert_project_scores = aggregate_expert_scores(publication_project_scores, agg_funcs, expert_id_col='Expert_ID', project_id_col='Project_ID')
        # Optionally compute mesh_semantic_coverage_score.
        if self.include_experts_mesh_coverage_score:
            # Generate a dictionary of unique MeSH terms for each expert.
            expert_terms = (
                publications.dropna(subset=[self.mesh_combined_output_column])
                .groupby(self.col_id_pub_expert)[self.mesh_combined_output_column]
                .apply(lambda terms: list(set(term for term_list in terms for term in convert_to_list(term_list, self.separator_output))))
                .to_dict()
            )
            # Generate a dictionary of MeSH terms for each project.
            project_terms = {
                row[self.col_id_project]: convert_to_list(row[self.mesh_combined_output_column], self.separator_output)
                for _, row in projects.iterrows()
            }
            # Precompute embeddings for experts and projects.
            expert_embeddings_dict = self._precompute_embeddings(expert_terms)
            # Precompute embeddings for projects
            project_embeddings_dict = self._precompute_embeddings(project_terms)
            # Add expert-project semantic coverage scores.
            expert_project_scores = self._add_mesh_semantic_coverage_scores(expert_project_scores, expert_embeddings_dict, project_embeddings_dict)
        # Return results
        return expert_project_scores
       
    def _compute_clusters(self, publications, projects):
        print('Generating MeSH term clusters.')
        all_mesh_terms = [
            term.strip()
            for mesh_terms in pd.concat([publications[self.mesh_combined_output_column], projects[self.mesh_combined_output_column]])
            for term in convert_to_list(mesh_terms, self.separator_output)
        ]
        all_mesh_terms = list(set(all_mesh_terms))  # Deduplicate terms
        print(f'Number of unique MeSH terms: {len(all_mesh_terms)}')
        self.mesh_to_cluster, self.cluster_counts, _ = cluster_items(
            model=self.model,
            all_items=all_mesh_terms,
            distance_threshold=self.distance_threshold_clusters
        )
        #print(f"Clusters: {self.mesh_to_cluster}")  # Debug print
        #print(f"Cluster counts: {self.cluster_counts}")  # Debug print


    def _calculate_semantic_coverage_score(self, expert_embeddings, proposal_embeddings):
        #print('Calculating expert-project semantic coverage.')
        if expert_embeddings is None or proposal_embeddings is None:
            return 0
        cosine_scores = util.pytorch_cos_sim(expert_embeddings, proposal_embeddings).cpu().numpy()
        #print(f"Cosine similarity scores: {cosine_scores}")  # Debug print
        covered_terms_count = sum(
            any(cosine_scores[i, j] >= self.similarity_threshold_terms for i in range(cosine_scores.shape[0]))
            for j in range(cosine_scores.shape[1])
        )
        #print(f"Similarity threshold: {self.similarity_threshold_terms}")  # Debug print
        return covered_terms_count / cosine_scores.shape[1] if cosine_scores.shape[1] > 0 else 0


    def _precompute_embeddings(self, terms_dict):
        print('Precomputing embeddings.')
        embeddings_dict = {}
        for id_, terms_list in terms_dict.items():
            if terms_list:
                embeddings = self.model.encode(terms_list, convert_to_tensor=True, show_progress_bar=False)
                embeddings_dict[id_] = (terms_list, embeddings)
                #print(f"Embeddings for {id_}: {embeddings}")  # Debug print
            else:
                embeddings_dict[id_] = ([], None)
        return embeddings_dict


    def _process_pub_project_pair(self, pub_row, project_row):
        pub_mesh_terms = convert_to_list(pub_row[self.mesh_combined_output_column], self.separator_output)
        proj_mesh_terms = convert_to_list(project_row[self.mesh_combined_output_column], self.separator_output)
        #print(f"Publication MeSH terms: {pub_mesh_terms}")  # Debug print
        #print(f"Project MeSH terms: {proj_mesh_terms}")  # Debug print
        avg_sim, max_sim = compute_list_similarity(self.model, pub_mesh_terms, proj_mesh_terms)
        specificity_weight = compute_specificity_weight(pub_mesh_terms, self.mesh_to_cluster, self.cluster_counts)
        return {
            'Pub_ID': pub_row[self.col_id_pub_publication],
            'Expert_ID': pub_row[self.col_id_pub_expert],
            'Project_ID': project_row[self.col_id_project],
            'MeSH_Avg_Similarity': avg_sim,
            'MeSH_Max_Similarity': max_sim,
            'MeSH_Avg_Similarity_Weighted': avg_sim * specificity_weight,
            'MeSH_Max_Similarity_Weighted': max_sim * specificity_weight
        }

    def _add_mesh_semantic_coverage_scores(self, expert_project_scores, expert_embeddings_dict, project_embeddings_dict):
        """Compute mesh_semantic_coverage_score for each expert-project pair using precomputed embeddings."""
        # Collect scores in a list
        coverage_scores = []
        for idx, row in expert_project_scores.iterrows():
            expert_id = row['Expert_ID']
            project_id = row['Project_ID']
            # Retrieve precomputed embeddings
            _, expert_embeddings = expert_embeddings_dict.get(expert_id, (None, None))
            _, project_embeddings = project_embeddings_dict.get(project_id, (None, None))
            # Calculate semantic coverage score
            mesh_semantic_coverage_score = self._calculate_semantic_coverage_score(expert_embeddings, project_embeddings)
            # Append the score to the list
            coverage_scores.append(mesh_semantic_coverage_score)
        # Update the DataFrame in bulk
        expert_project_scores['Expert_MeSH_Semantic_Coverage_Score'] = coverage_scores
        return expert_project_scores

        
        


