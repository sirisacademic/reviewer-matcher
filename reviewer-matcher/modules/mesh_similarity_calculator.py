import pandas as pd
import numpy as np
import abbreviations
from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import AgglomerativeClustering
from collections import defaultdict
from tqdm import tqdm

class MeSHSimilarityCalculator:
    def __init__(self, config_manager, DISTANCE_THRESHOLD_CLUSTERS = 0.2, SIMILARITY_THRESHOLD_TERMS = 0.6, MODEL_NAME = 'FremyCompany/BioLORD-2023'):
        # Configurations read from config file handled by config_manager.
        self.separator_output = config_manager.get('SEPARATOR_VALUES_OUTPUT', '|')
        self.OUTPUT_FILE_EXPERT_PROJECT_MESH_SIMILARITY  = config_manager.get('OUTPUT_FILE_EXPERT_PROJECT_MESH_SIMILARITY')
        #self.expert_publications = expert_publications
        #self.projects = projects
        self.DISTANCE_THRESHOLD_CLUSTERS = DISTANCE_THRESHOLD_CLUSTERS
        self.SIMILARITY_THRESHOLD_TERMS = SIMILARITY_THRESHOLD_TERMS
        self.MODEL_NAME = MODEL_NAME
        self.model = SentenceTransformer(self.MODEL_NAME)

    ## !! TODO: Unify functions used in more than one notebook !!

    # Function to compute embeddings for a list of MeSH terms.
    def get_embeddings(self, model, mesh_terms):
    #-----------------------------
        terms_list = [term.strip() for term in mesh_terms.split(self.separator_output) if term.strip()]
        if not terms_list:
            return [], None
        embeddings = model.encode(terms_list, convert_to_tensor=True)
        return terms_list, embeddings

    # Compute semantic similarity between two lists of phrases.
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

    # Function to compute specificity weight based on semantic clusters of MeSH terms
    def compute_mesh_specificity_weight(self, mesh_terms_list, mesh_to_cluster, cluster_counts):
    #------------------------------------------------------------------------------------
        if not mesh_terms_list:
            return 0
        weights = []
        for term in mesh_terms_list:
            cluster = mesh_to_cluster.get(term)
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

    # Function to calculate coverage diversity score with semantic similarity
    # to consider giving more weight to experts with multiple publications that cover different aspects of the proposal
    def calculate_semantic_coverage_score(self, model, expert_mesh, proposal_mesh, similarity_threshold=0.7):
    #-------------------------------------------------------------------------------------------------
        # Get embeddings for expert and proposal MeSH terms
        expert_terms, expert_embeddings = self.get_embeddings(model, expert_mesh)
        proposal_terms, proposal_embeddings = self.get_embeddings(model, proposal_mesh)
        if expert_embeddings is None or proposal_embeddings is None:
            return 0  # Return 0 if embeddings are missing
        # Compute pairwise cosine similarities
        cosine_scores = util.pytorch_cos_sim(expert_embeddings, proposal_embeddings).cpu().numpy()
        # Count the number of terms in the proposal that have a semantic match in the expert's publications
        covered_terms_count = 0
        for j in range(cosine_scores.shape[1]):  # Iterate over proposal terms
            if any(cosine_scores[i, j] >= similarity_threshold for i in range(cosine_scores.shape[0])):
                covered_terms_count += 1
        # Calculate coverage score
        coverage_score = covered_terms_count / len(proposal_terms) if len(proposal_terms) > 0 else 0
        return coverage_score

    # Function used to debug and adjust the threshold.
    def display_similarity_matrix(expert_terms, proposal_terms, cosine_scores):
    #-------------------------------------------------------------------------
        # Convert cosine_scores into a pandas DataFrame for better visualization
        similarity_df = pd.DataFrame(cosine_scores, index=expert_terms, columns=proposal_terms)
        # Display the DataFrame in Colab
        from IPython.display import display
        display(similarity_df)

    def compute_expert_project_similarity(self, publication_data, project_data):
        
        # Extract all unique MeSH terms from publications and projects
        all_mesh_terms = [
            term.strip() for mesh_terms in pd.concat([publication_data['MESH_EXTRACTED'], project_data['MESH_EXTRACTED']]).apply(lambda x: x.split(self.separator_output))
            for term in mesh_terms if term.strip() != ''
        ]

        # Create clusters for MeSH terms
        # Lower distance ---> more similar terms in cluster.
        mesh_to_cluster, cluster_counts, clusters_dict = self.create_and_assign_clusters(all_mesh_terms, distance_threshold=self.DISTANCE_THRESHOLD_CLUSTERS)

        # DEBUG
        for cluster_label, mesh_terms in clusters_dict.items():
            print(f"Cluster {cluster_label} (Number of MeSH terms: {len(mesh_terms)}):")
            #for mesh_term in mesh_terms:
            #    print(f"  - {mesh_term}")
            #print()

        # Create clusters for MeSH terms
        print('Generating clusters of MeSH terms...')

        # Initialize a list to store cluster information
        cluster_data = []

        # Iterate over clusters to prepare data for DataFrame
        for cluster_label, mesh_terms in clusters_dict.items():
            for mesh_term in mesh_terms:
                cluster_data.append({
                    'Cluster_Label': cluster_label,
                    'MeSH_Term': mesh_term,
                    'Number_of_MeSH_Terms_in_Cluster': len(mesh_terms)
                })

        # Create a DataFrame from the cluster data
        df_clusters = pd.DataFrame(cluster_data)

        # Step 1: Compute Publication-Project Scores
        publication_project_scores = []

        # Compute iterations to keep track of progress.
        total_iterations = len(project_data) * len(publication_data)

        with tqdm(total=total_iterations, desc="Processing publication-progress pairs") as pbar:

            for _, project_row in project_data.iterrows():
                for _, pub_row in publication_data.iterrows():

                    # Update progress bar.
                    pbar.update(1)

                    # Extract MeSH terms as lists
                    pub_mesh_terms = [term.strip() for term in pub_row['MESH_EXTRACTED'].split(self.separator_output) if term.strip()]
                    proj_mesh_terms = [term.strip() for term in project_row['MESH_EXTRACTED'].split(self.separator_output) if term.strip()]

                    # Compute similarity between publication and project
                    avg_similarity, max_similarity = self.compute_list_similarity(self.model, pub_mesh_terms, proj_mesh_terms)

                    # Compute specificity weight based on MeSH clusters
                    specificity_weight = self.compute_mesh_specificity_weight(pub_mesh_terms, mesh_to_cluster, cluster_counts)

                    # Adjust similarity by specificity weight
                    avg_similarity_weighted = avg_similarity * specificity_weight
                    max_similarity_weighted = max_similarity * specificity_weight

                    # Store the publication-project level scores
                    publication_project_scores.append({
                        'PMID': pub_row['PMID'],
                        'Expert_ID': pub_row['ID'],
                        'Project_ID': project_row['ID'],
                        'Pub_MeSH_Avg_Similarity': avg_similarity if avg_similarity != 0 else None,
                        'Pub_MeSH_Max_Similarity': max_similarity if max_similarity != 0 else None,
                        'Pub_MeSH_Avg_Similarity_Weighted': avg_similarity_weighted if avg_similarity_weighted != 0 else None,
                        'Pub_MeSH_Max_Similarity_Weighted': max_similarity_weighted if max_similarity_weighted != 0 else None
                    })

        # Convert to DataFrame
        df_publication_scores = pd.DataFrame(publication_project_scores)

        # Step 2: Aggregate to Compute Expert-Project Scores
        expert_project_scores = []

        # Group publications by Expert_ID and Project_ID
        grouped = df_publication_scores.groupby(['Expert_ID', 'Project_ID'])

        # Get the total number of expert-project pairs for progress tracking
        total_groups = len(grouped)

        for (expert_id, project_id), group in tqdm(grouped, total=total_groups, desc="Processing expert-project pairs"):
            # Combine all MeSH terms for an expert to calculate the diversity score
            all_expert_mesh_terms = self.separator_output.join(publication_data[publication_data['ID'] == expert_id]['MESH_EXTRACTED'].unique())
            project_mesh_terms = project_data[project_data['ID'] == project_id]['MESH_EXTRACTED'].iloc[0]

            # Calculate semantic coverage score
            mesh_semantic_coverage_score = self.calculate_semantic_coverage_score(self.model, all_expert_mesh_terms, project_mesh_terms, similarity_threshold=self.SIMILARITY_THRESHOLD_TERMS)

            # Handle missing values in the group DataFrame before aggregation by creating a copy
            group_filled = group.fillna(0)

            # Filter out rows where the similarity is zero before calculating averages
            filtered_max_similarity_weighted = group_filled.loc[group_filled['Pub_MeSH_Max_Similarity_Weighted'] != 0, 'Pub_MeSH_Max_Similarity_Weighted']
            filtered_avg_similarity_weighted = group_filled.loc[group_filled['Pub_MeSH_Avg_Similarity_Weighted'] != 0, 'Pub_MeSH_Avg_Similarity_Weighted']
            filtered_max_similarity = group_filled.loc[group_filled['Pub_MeSH_Max_Similarity'] != 0, 'Pub_MeSH_Max_Similarity']
            filtered_avg_similarity = group_filled.loc[group_filled['Pub_MeSH_Avg_Similarity'] != 0, 'Pub_MeSH_Avg_Similarity']

            # Store expert-level results
            expert_project_scores.append({
                'Expert_ID': expert_id,
                'Project_ID': project_id,
                'Expert_MeSH_Semantic_Coverage_Score': mesh_semantic_coverage_score,
                'Expert_MeSH_Max_Similarity_Max': group_filled['Pub_MeSH_Max_Similarity'].max(),
                'Expert_MeSH_Max_Similarity_Avg': filtered_max_similarity.mean() if not filtered_max_similarity.empty else 0,
                'Expert_MeSH_Avg_Similarity_Max': group_filled['Pub_MeSH_Avg_Similarity'].max(),
                'Expert_MeSH_Avg_Similarity_Avg': filtered_avg_similarity.mean() if not filtered_avg_similarity.empty else 0,
                'Expert_MeSH_Max_Similarity_Weighted_Max': group_filled['Pub_MeSH_Max_Similarity_Weighted'].max(),
                'Expert_MeSH_Max_Similarity_Weighted_Avg': filtered_max_similarity_weighted.mean() if not filtered_max_similarity_weighted.empty else 0,
                'Expert_MeSH_Avg_Similarity_Weighted_Max': group_filled['Pub_MeSH_Avg_Similarity_Weighted'].max(),
                'Expert_MeSH_Avg_Similarity_Weighted_Avg': filtered_avg_similarity_weighted.mean() if not filtered_avg_similarity_weighted.empty else 0
            })

        # Convert to DataFrame
        df_expert_scores = pd.DataFrame(expert_project_scores)

        df_expert_scores.to_pickle(self.OUTPUT_FILE_EXPERT_PROJECT_MESH_SIMILARITY)
        print(f'Expert-project Jaccard similarity scores saved to {self.OUTPUT_FILE_EXPERT_PROJECT_MESH_SIMILARITY}')

        return df_expert_scores

