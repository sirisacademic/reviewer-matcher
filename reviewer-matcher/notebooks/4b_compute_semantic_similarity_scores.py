# -*- coding: utf-8 -*-
"""4b-compute_semantic_similarity_scores.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1oT0MJOW5-sqBiT1TrvgtZTSvlrcpAeH0
"""

!pip install -q transformers sentence-transformers

# TODO: Modify files before saving in the notebook that generates the data and saves the Pickle files so this is not necessary.
!pip install -q abbreviations

import pandas as pd
import numpy as np
import abbreviations

from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import AgglomerativeClustering
from collections import defaultdict
from google.auth import default
from google.colab import auth
from google.colab import drive
from tqdm import tqdm

from google.colab import drive
drive.mount('/content/drive')

### Input/output paths.

# Specific call folder (used to retrieve the configuration, URLs, etc).
CALL_NAME = '2021-Salut Mental'

# Bath path for all the sample data.
BASE_PATH = '/content/drive/MyDrive/1_Current_projects_SIRIS/2024AQUAS-ReviewerMatcher'

# Code path.
CODE_PATH = f'{BASE_PATH}/Implementation/Notebooks'

# Data path.
DATA_PATH = f'{BASE_PATH}/Implementation/Data'

# Input files.
INPUT_FILE_PATH_PROJECTS = f'{DATA_PATH}/{CALL_NAME}/projects.pkl'
INPUT_FILE_PATH_PUBLICATIONS = f'{DATA_PATH}/{CALL_NAME}/expert_publications.pkl'

# Output files.
OUTPUT_FILE_PUBLICATIONS_PROJECTS = f'{DATA_PATH}/{CALL_NAME}/scores/publications_projects_content_similarity_scores.pkl'
OUTPUT_FILE_EXPERTS_PROJECTS = f'{DATA_PATH}/{CALL_NAME}/scores/expert_projects_content_similarity_scores.pkl'
OUTPUT_FILE_CLUSTERS = f'{DATA_PATH}/{CALL_NAME}/clusters_methods.tsv'

SAVE_PUBLICATIONS_PROJECTS = False
SAVE_EXPERTS_PROJECTS = False
SAVE_CLUSTERS = False

### Settings

SEPARATOR_VALUES = '|'

# Semantic similarity thresholds.
DISTANCE_THRESHOLD_CLUSTERS = 0.2

### Model used for semantic similarity computation.

MODEL_NAME = 'FremyCompany/BioLORD-2023'

### Test size if we want to process a subset for testing. Set as 0 to ignore.

TEST_SIZE_PROJECTS = 0
TEST_SIZE_PUBLICATIONS = 10

# Load data

"""
df_publications = pd.DataFrame({
    'ID': [1, 1, 2],
    'RESEARCH_TOPIC': ['Auditory frequency discrimination in developmental dyslexia',
                       'Long-term neuropsychological outcomes and neurochemistry in children with cholestatic liver disease',
                       'Oxidative stress and neuroinflammation in schizophrenia'],
    'OBJECTIVES': ['Evaluate the cumulative evidence for group differences|Explore the impact of moderator variables',
                   'Differentiate the long-term cognitive effects|Assess the impact of disease duration',
                   'Investigate the mechanisms linking oxidative stress|Explore the impact of redox dysregulation'],
    'METHODS_SPECIFIC': ['Moderator variable analyses|Behavioral and cognitive assessments',
                         'Psychometric assessments|Magnetic resonance spectroscopy',
                         'Oxidative stress measurement|Microglia activation assessment']
})

df_projects = pd.DataFrame({
    'ID': ['P1'],
    'RESEARCH_TOPIC': ['Oxidative stress and neuroinflammation mechanisms'],
    'OBJECTIVES': ['Investigate the mechanisms linking oxidative stress with neuroinflammatory processes|Identify potential biomarkers'],
    'METHODS_SPECIFIC': ['Oxidative stress measurement|Microglia activation assessment']
})
"""

df_projects = pd.read_pickle(INPUT_FILE_PATH_PROJECTS).fillna('')
df_publications = pd.read_pickle(INPUT_FILE_PATH_PUBLICATIONS).fillna('')

if TEST_SIZE_PROJECTS:
  df_projects = df_projects.head(TEST_SIZE_PROJECTS)

if TEST_SIZE_PUBLICATIONS:
  df_publications = df_publications.head(TEST_SIZE_PUBLICATIONS)

## !! TODO: Unify functions used in more than one notebook !!

# Convert a column of strings to lists, removing empty or whitespace-only entries
def convert_to_list(column_value):
#--------------------------------
    if pd.isna(column_value) or column_value == '':
        return []
    return [item.strip() for item in column_value.split(SEPARATOR_VALUES) if item.strip() != '']

# Compute semantic similarity between two lists of phrases.
# This function takes two lists of phrases, computes their embeddings using the provided model, and calculates the semantic similarity between the lists.
# It finds the maximum similarity for each item in the second list against all items in the first list, then returns the maximum and average of these maximum similarities.
def compute_list_similarity(model, list1, list2):
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
def compute_semantic_specificity_weight(methods_list, method_to_cluster, cluster_counts):
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
def cluster_methods(method_embeddings, all_methods, distance_threshold=0.3):
#--------------------------------------------------------------------------
    clustering_model = AgglomerativeClustering(n_clusters=None, distance_threshold=distance_threshold, metric='cosine', linkage='average')
    method_labels = clustering_model.fit_predict(method_embeddings.cpu().numpy())
    # Return a dictionary to store cluster labels and their corresponding methods
    clusters_dict = defaultdict(list)
    for method, label in zip(all_methods, method_labels):
        clusters_dict[label].append(method)
    return method_labels, clusters_dict

# Create clusters and assign methods to clusters
def create_and_assign_clusters(all_methods, distance_threshold=0.3):
#------------------------------------------------------------------
    # Generate embeddings
    method_embeddings = model.encode(all_methods, convert_to_tensor=True)
    # Perform clustering
    method_labels, clusters_dict = cluster_methods(method_embeddings, all_methods, distance_threshold=distance_threshold)
    # Assign methods to clusters
    method_to_cluster = dict(zip(all_methods, method_labels))
    # Compute cluster-based frequency
    cluster_counts = defaultdict(int)
    for cluster_label in method_labels:
        cluster_counts[cluster_label] += 1
    return method_to_cluster, cluster_counts, clusters_dict

# Load model.
model = SentenceTransformer(MODEL_NAME)

### Generate clusters of methods, which are then used to compute the methods' specificity.
# !!!! SET THRESHOLD !!!!!

# Note: clusters_dict is only needed if we want to display the clusters to debug/set the threshold.
# Lower distance ---> more similar methods in cluster.
all_methods = [
    method.strip() for methods in df_publications['METHODS_SPECIFIC'].apply(lambda x: x.split(SEPARATOR_VALUES))
    for method in methods if method.strip() != ''
]
method_to_cluster, cluster_counts, clusters_dict = create_and_assign_clusters(all_methods, distance_threshold=DISTANCE_THRESHOLD_CLUSTERS)

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

# Save the DataFrame to a CSV file

if SAVE_CLUSTERS:
  df_clusters.to_csv(OUTPUT_FILE_CLUSTERS, sep='\t', index=False)
  print(f'Clusters data saved to {OUTPUT_FILE_CLUSTERS}')

# Compute indicators for each publication-proposal pair
publication_project_scores = []

# Compute iterations to keep track of progress.
total_iterations = len(df_projects) * len(df_publications)

with tqdm(total=total_iterations, desc="Processing publication-progress pairs") as pbar:

  for _, project_row in df_projects.iterrows():
      for _, pub_row in df_publications.iterrows():

          # Update progress bar.
          pbar.update(1)

          # RESEARCH_TOPIC Indicator
          if pub_row['RESEARCH_TOPIC'] and project_row['RESEARCH_TOPIC']:
              topic_similarity = util.pytorch_cos_sim(
                  model.encode(pub_row['RESEARCH_TOPIC'], convert_to_tensor=True),
                  model.encode(project_row['RESEARCH_TOPIC'], convert_to_tensor=True)
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
          objectives_avg_similarity, objectives_max_similarity = compute_list_similarity(
              model,
              convert_to_list(pub_row['OBJECTIVES']),
              convert_to_list(project_row['OBJECTIVES'])
          )

          # METHODS_SPECIFIC Indicator (coverage and specificity)
          # methods_avg_similarity / methods_specific_avg_similarity:
          # - This provides an overall view of how well the methods used in the publication align with those proposed in the project.
          # - Multiplying this value by the specificity weight allows to account for how unique the methods are, giving more importance to experts using rare methods.
          # methods_max_similarity / methods_specific_max_similarity:
          # - This captures whether there is a particularly strong alignment between a method used in the publication and one proposed in the project.
          # - Multiplying by the specificity weight ensures that rare methods are given more importance.
          pub_methods = convert_to_list(pub_row['METHODS_SPECIFIC'])
          proj_methods = convert_to_list(project_row['METHODS_SPECIFIC'])
          methods_avg_similarity, methods_max_similarity = compute_list_similarity(
              model,
              pub_methods,
              proj_methods
          )
          specificity_weight = compute_semantic_specificity_weight(pub_methods, method_to_cluster, cluster_counts)
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

# Save publication-level scores for analysis, etc.

if SAVE_PUBLICATIONS_PROJECTS:
  df_publication_project_scores.to_pickle(OUTPUT_FILE_PUBLICATIONS_PROJECTS)
  print(f'Saved publication-project scores to file {OUTPUT_FILE_PUBLICATIONS_PROJECTS}')

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

# Save expert-level scores.

if SAVE_EXPERTS_PROJECTS:
  df_expert_project_scores.to_pickle(OUTPUT_FILE_EXPERTS_PROJECTS)
  print(f'Saved expert-project scores to file {OUTPUT_FILE_EXPERTS_PROJECTS}')

df_expert_project_scores.head(10)