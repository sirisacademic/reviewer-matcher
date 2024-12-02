import pandas as pd
import numpy as np
import abbreviations
from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import AgglomerativeClustering
from collections import defaultdict
from tqdm import tqdm

class ContentSimilarityCalculator:
    def __init__(self, config_manager, DISTANCE_THRESHOLD_CLUSTERS = 0.2, MODEL_NAME = 'FremyCompany/BioLORD-2023'):
        # Configurations read from config file handled by config_manager.
        self.separator_output = config_manager.get('SEPARATOR_VALUES_OUTPUT', '|')
        self.OUTPUT_FILE_EXPERT_PROJECT_SEMANTIC_SIMILARITY  = config_manager.get('OUTPUT_FILE_EXPERT_PROJECT_SEMANTIC_SIMILARITY')
        #self.expert_publications = expert_publications
        #self.projects = projects
        self.DISTANCE_THRESHOLD_CLUSTERS = DISTANCE_THRESHOLD_CLUSTERS
        self.MODEL_NAME = MODEL_NAME
        self.model = SentenceTransformer(self.MODEL_NAME)


    # Convert a column of strings to lists, removing empty or whitespace-only entries
    def convert_to_list(self, column_value):
    #--------------------------------
        if pd.isna(column_value) or column_value == '':
            return []
        return [item.strip() for item in column_value.split(self.separator_output) if item.strip() != '']

    # Compute semantic similarity between two lists of phrases.
    # This function takes two lists of phrases, computes their embeddings using the provided model, and calculates the semantic similarity between the lists.
    # It finds the maximum similarity for each item in the second list against all items in the first list, then returns the maximum and average of these maximum similarities.
    def compute_list_similarity(self, model, list1, list2):
    #-----------------------------------------------
        if not list1 or not list2:
            return 0, 0
        embeddings1 = model.encode(list1, convert_to_tensor=True)
        embeddings2 = model.encode(list2, convert_to_tensor=True)
        cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2).cpu().numpy()
        avg_similarity = np.mean([max(cosine_scores[:, j]) for j in range(len(list2))])
        max_similarity = np.max(cosine_scores) if cosine_scores.size > 0 else 0
        return avg_similarity, max_similarity

    # Function to compute specificity weight based on semantic clusters of methods.
    def compute_semantic_specificity_weight(self, methods_list, method_to_cluster, cluster_counts):
    #---------------------------------------------------------------------------------------
        if not methods_list:
            return 0
        weights = []
        for method in methods_list:
            cluster = method_to_cluster.get(method)
            if cluster is not None:
                cluster_frequency = cluster_counts[cluster]
                if cluster_frequency > 0:
                    # Inverse frequency as the weight for specificity
                    weights.append(1.0 / cluster_frequency)
        return np.mean(weights) if weights else 0

    # Using Agglomerative Clustering to group methods based on their embeddings
    def cluster_methods(self, method_embeddings, all_methods, distance_threshold=0.3):
    #--------------------------------------------------------------------------
        clustering_model = AgglomerativeClustering(n_clusters=None, distance_threshold=distance_threshold, metric='cosine', linkage='average')
        method_labels = clustering_model.fit_predict(method_embeddings.cpu().numpy())
        # Return a dictionary to store cluster labels and their corresponding methods
        clusters_dict = defaultdict(list)
        for method, label in zip(all_methods, method_labels):
            clusters_dict[label].append(method)
        return method_labels, clusters_dict

    # Create clusters and assign methods to clusters
    def create_and_assign_clusters(self, all_methods, distance_threshold=0.3):
    #------------------------------------------------------------------
        # Generate embeddings
        method_embeddings = self.model.encode(all_methods, convert_to_tensor=True)
        # Perform clustering
        method_labels, clusters_dict = self.cluster_methods(method_embeddings, all_methods, distance_threshold=distance_threshold)
        # Assign methods to clusters
        method_to_cluster = dict(zip(all_methods, method_labels))
        # Compute cluster-based frequency
        cluster_counts = defaultdict(int)
        for cluster_label in method_labels:
            cluster_counts[cluster_label] += 1
        return method_to_cluster, cluster_counts, clusters_dict


    def compute_similarity(self, publication_content, project_content):

        ### Generate clusters of methods, which are then used to compute the methods' specificity.
        # !!!! SET THRESHOLD !!!!!

        # Note: clusters_dict is only needed if we want to display the clusters to debug/set the threshold.
        # Lower distance ---> more similar methods in cluster.
        all_methods = [
            method.strip() for methods in publication_content['METHODS_SPECIFIC'].apply(lambda x: x.split(self.separator_output))
            for method in methods if method.strip() != ''
        ]
        method_to_cluster, cluster_counts, clusters_dict = self.create_and_assign_clusters(all_methods, distance_threshold=self.DISTANCE_THRESHOLD_CLUSTERS)
        
        # Create clusters for MeSH terms
        print('Generating clusters of MeSH terms...')

        # Initialize a list to store cluster information
        cluster_data = []

        # Iterate over clusters to prepare data for DataFrame
        for cluster_label, methods in clusters_dict.items():
            for method in methods:
                cluster_data.append({
                    'Cluster_Label': cluster_label,
                    'Method': method,
                    'Number_of_Methods_in_Cluster': len(methods)
                })

        # Create a DataFrame from the cluster data
        df_clusters = pd.DataFrame(cluster_data)

        # Compute indicators for each publication-proposal pair
        publication_project_scores = []

        # Compute iterations to keep track of progress.
        total_iterations = len(project_content) * len(publication_content)

        with tqdm(total=total_iterations, desc="Processing publication-progress pairs") as pbar:

            for _, project_row in project_content.iterrows():
                for _, pub_row in publication_content.iterrows():

                    # Update progress bar.
                    pbar.update(1)

                    # RESEARCH_TOPIC Indicator
                    if pub_row['RESEARCH_TOPIC'] and project_row['RESEARCH_TOPIC']:
                        topic_similarity = util.pytorch_cos_sim(
                            self.model.encode(pub_row['RESEARCH_TOPIC'], convert_to_tensor=True),
                            self.model.encode(project_row['RESEARCH_TOPIC'], convert_to_tensor=True)
                        ).item()
                    else:
                        topic_similarity = 0

                    # OBJECTIVES Indicators
                    # objectives_avg_similarity:
                    # - This gives an overall measure of alignment between all the objectives of the publication and the proposal.
                    # - It helps quantify the general overlap between them.
                    # objectives_max_similarity:
                    # This value captures the strongest match between any of the objectives from the publication and the proposal.
                    # This can be useful if we want to ensure that at least one objective aligns very closely, even if the others don't.
                    objectives_avg_similarity, objectives_max_similarity = self.compute_list_similarity(
                        self.model,
                        self.convert_to_list(pub_row['OBJECTIVES']),
                        self.convert_to_list(project_row['OBJECTIVES'])
                    )

                    # METHODS_SPECIFIC Indicator (coverage and specificity)
                    # methods_avg_similarity / methods_specific_avg_similarity:
                    # - This provides an overall view of how well the methods used in the publication align with those proposed in the project.
                    # - Multiplying this value by the specificity weight allows to account for how unique the methods are, giving more importance to experts using rare methods.
                    # methods_max_similarity / methods_specific_max_similarity:
                    # - This captures whether there is a particularly strong alignment between a method used in the publication and one proposed in the project.
                    # - Multiplying by the specificity weight ensures that rare methods are given more importance.
                    pub_methods = self.convert_to_list(pub_row['METHODS_SPECIFIC'])
                    proj_methods = self.convert_to_list(project_row['METHODS_SPECIFIC'])
                    methods_avg_similarity, methods_max_similarity = self.compute_list_similarity(
                        self.model,
                        pub_methods,
                        proj_methods
                    )
                    specificity_weight = self.compute_semantic_specificity_weight(pub_methods, method_to_cluster, cluster_counts)
                    methods_avg_similarity_weighted = methods_avg_similarity * specificity_weight
                    methods_max_similarity_weighted = methods_max_similarity * specificity_weight

                    # Store results
                    publication_project_scores.append({
                        'PMID': pub_row['PMID'],
                        'Expert_ID': pub_row['ID'],
                        'Project_ID': project_row['ID'],
                        'Topic_Similarity': topic_similarity if topic_similarity != 0 else None,
                        'Objectives_Avg_Similarity': objectives_avg_similarity if objectives_avg_similarity != 0 else None,
                        'Objectives_Max_Similarity': objectives_max_similarity if objectives_max_similarity != 0 else None,
                        'Methods_Avg_Similarity': methods_avg_similarity if methods_avg_similarity != 0 else None,
                        'Methods_Max_Similarity': methods_max_similarity if methods_max_similarity != 0 else None,
                        'Methods_Avg_Similarity_Weighted': methods_avg_similarity_weighted if methods_avg_similarity_weighted != 0 else None,
                        'Methods_Max_Similarity_Weighted': methods_max_similarity_weighted if methods_max_similarity_weighted != 0 else None
                })

        # Convert publication-level results to DataFrame
        df_publication_project_scores = pd.DataFrame(publication_project_scores)

        # Aggregate publication-level scores to expert-level scores
        expert_project_scores = []

        # Group by Expert and Project
        grouped = df_publication_project_scores.groupby(['Expert_ID', 'Project_ID'])

        # Get the total number of expert-project pairs for progress tracking
        total_groups = len(grouped)

        for (expert_id, project_id), group in tqdm(grouped, total=total_groups, desc="Processing expert-project pairs"):

            # Handle missing values in the group DataFrame before aggregation by creating a copy
            group_filled = group.fillna(0)

            # Filter out rows where the value is 0 for averages
            filtered_topic_similarity = group_filled.loc[group_filled['Topic_Similarity'] != 0, 'Topic_Similarity']
            filtered_objectives_max_similarity = group_filled.loc[group_filled['Objectives_Max_Similarity'] != 0, 'Objectives_Max_Similarity']
            filtered_objectives_avg_similarity = group_filled.loc[group_filled['Objectives_Avg_Similarity'] != 0, 'Objectives_Avg_Similarity']
            filtered_methods_max_similarity = group_filled.loc[group_filled['Methods_Max_Similarity'] != 0, 'Methods_Max_Similarity']
            filtered_methods_avg_similarity = group_filled.loc[group_filled['Methods_Avg_Similarity'] != 0, 'Methods_Avg_Similarity']
            filtered_methods_max_similarity_weighted = group_filled.loc[group_filled['Methods_Max_Similarity_Weighted'] != 0, 'Methods_Max_Similarity_Weighted']
            filtered_methods_avg_similarity_weighted = group_filled.loc[group_filled['Methods_Avg_Similarity_Weighted'] != 0, 'Methods_Avg_Similarity_Weighted']

            # Store expert-level results
            expert_project_scores.append({
                'Expert_ID': expert_id,
                'Project_ID': project_id,
                'Expert_Topic_Similarity_Max': group_filled['Topic_Similarity'].max(),
                'Expert_Topic_Similarity_Avg': filtered_topic_similarity.mean() if not filtered_topic_similarity.empty else 0,
                'Expert_Objectives_Max_Similarity_Max': group_filled['Objectives_Max_Similarity'].max(),
                'Expert_Objectives_Max_Similarity_Avg': filtered_objectives_max_similarity.mean() if not filtered_objectives_max_similarity.empty else 0,
                'Expert_Objectives_Avg_Similarity_Max': group_filled['Objectives_Avg_Similarity'].max(),
                'Expert_Objectives_Avg_Similarity_Avg': filtered_objectives_avg_similarity.mean() if not filtered_objectives_avg_similarity.empty else 0,
                'Expert_Methods_Max_Similarity_Max': group_filled['Methods_Max_Similarity'].max(),
                'Expert_Methods_Max_Similarity_Avg': filtered_methods_max_similarity.mean() if not filtered_methods_max_similarity.empty else 0,
                'Expert_Methods_Avg_Similarity_Max': group_filled['Methods_Avg_Similarity'].max(),
                'Expert_Methods_Avg_Similarity_Avg': filtered_methods_avg_similarity.mean() if not filtered_methods_avg_similarity.empty else 0,
                'Expert_Methods_Max_Similarity_Weighted_Max': group_filled['Methods_Max_Similarity_Weighted'].max(),
                'Expert_Methods_Max_Similarity_Weighted_Avg': filtered_methods_max_similarity_weighted.mean() if not filtered_methods_max_similarity_weighted.empty else 0,
                'Expert_Methods_Avg_Similarity_Weighted_Max': group_filled['Methods_Avg_Similarity_Weighted'].max(),
                'Expert_Methods_Avg_Similarity_Weighted_Avg': filtered_methods_avg_similarity_weighted.mean() if not filtered_methods_avg_similarity_weighted.empty else 0
            })

        # Convert expert-level results to DataFrame
        df_expert_project_scores = pd.DataFrame(expert_project_scores)

        # Save the similarity scores to a file for further analysis
        df_expert_project_scores.to_pickle(self.OUTPUT_FILE_EXPERT_PROJECT_SEMANTIC_SIMILARITY)
        print(f'Expert-project semantic similarity scores saved to {self.OUTPUT_FILE_EXPERT_PROJECT_SEMANTIC_SIMILARITY}')

        return df_expert_project_scores





