# File: functions_similarity.py

import sys
import numpy as np
import pandas as pd
import torch

from collections import defaultdict
from tqdm import tqdm
from sentence_transformers import util
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import jaccard_score

def one_hot_encode(data, column, unique_terms, separator='|'):
    """
    One-hot encode the terms in a specified column using the given list of unique terms.
    """
    mlb = MultiLabelBinarizer(classes=unique_terms)
    encoded_data = mlb.fit_transform(data[column].apply(lambda x: convert_to_list(x, separator)))
    return encoded_data
  
def calculate_jaccard_similarity(encoded1, encoded2):
    """
    Calculate the Jaccard similarity score between two encoded vectors.
    """
    if encoded1.sum() == 0 and encoded2.sum() == 0:
        return 0
    return jaccard_score(encoded1, encoded2, average='binary')

def calculate_dice_similarity(encoded1, encoded2):
    """
    Calculate the Dice similarity score between two encoded vectors.
    """
    intersection = (encoded1 & encoded2).sum()
    total_sum = encoded1.sum() + encoded2.sum()
    if total_sum == 0:
        return 0
    return (2 * intersection) / total_sum

def calculate_overlap_coefficient(set1, set2):
    """
    Calculate the Overlap Coefficient between two sets.
    """
    intersection = len(set1 & set2)
    smaller_set_size = min(len(set1), len(set2))
    return intersection / smaller_set_size if smaller_set_size > 0 else 0

def compute_list_similarity(model, list1, list2, batch_size=100):
    """Compute semantic similarity between two lists of phrases using batching to manage memory. Returns average and maximum similarity scores."""
    if not list1 or not list2:
        return 0, 0
    avg_similarities = []
    max_similarity = 0
    embeddings1 = model.encode(list1, convert_to_tensor=True, show_progress_bar=False)
    embeddings2 = model.encode(list2, convert_to_tensor=True, show_progress_bar=False)
    for i in range(0, len(embeddings1), batch_size):
        batch1 = embeddings1[i:i + batch_size]
        cosine_scores = util.pytorch_cos_sim(batch1, embeddings2).cpu().numpy()
        avg_similarities.append(np.mean([max(cosine_scores[:, j]) for j in range(len(list2))]))
        max_similarity = max(max_similarity, np.max(cosine_scores))
    avg_similarity = np.mean(avg_similarities)
    return avg_similarity, max_similarity

def compute_specificity_weight(items_list, item_to_cluster, cluster_counts):
    """
    Compute specificity weight based on semantic clusters.
    Works for both MeSH terms and methods.
    """
    if not items_list:
        return 0
    weights = []
    for item in items_list:
        cluster = item_to_cluster.get(item)
        if cluster is not None:
            cluster_frequency = cluster_counts[cluster]
            if cluster_frequency > 0:
                weights.append(1.0 / cluster_frequency)
    return np.mean(weights) if weights else 0

def cluster_items(model, all_items, distance_threshold=0.3, batch_size=1000):
    """Perform clustering based on embeddings generated in batches."""
    print('Generating embeddings in batches.')
    item_embeddings = []
    # Generate embeddings in batches
    for i in range(0, len(all_items), batch_size):
        batch = all_items[i:i + batch_size]
        batch_embeddings = model.encode(batch, convert_to_tensor=True, show_progress_bar=False)
        item_embeddings.append(batch_embeddings)
    # Concatenate all batched embeddings
    item_embeddings = torch.cat(item_embeddings, dim=0)
    # Perform clustering with precomputed embeddings.
    clustering_model = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric='cosine',
        linkage='average'
    )
    item_labels = clustering_model.fit_predict(item_embeddings.cpu().numpy())
    # Create mappings and counts
    clusters_dict = defaultdict(list)
    for item, label in zip(all_items, item_labels):
        clusters_dict[label].append(item)
    item_to_cluster = dict(zip(all_items, item_labels))
    cluster_counts = defaultdict(int)
    for label in item_labels:
        cluster_counts[label] += 1
    return item_to_cluster, cluster_counts, clusters_dict

def convert_to_list(text, separator='|'):
    """
    Convert a string of separator-delimited items into a list,
    removing empty or whitespace-only entries.
    """
    if pd.isna(text) or text == '':
        return []
    return [item.strip() for item in text.split(separator) if item.strip() != '']

def process_publication_project_pairs(publications, projects, process_row_func):
    scores = []
    with tqdm(total=len(projects), desc="Processing publication-project pairs", file=sys.stdout) as pbar:
        for _, project_row in projects.iterrows():
            for _, pub_row in publications.iterrows():
                score_dict = process_row_func(pub_row, project_row)
                scores.append(score_dict)
            pbar.update(1)
    return pd.DataFrame(scores)

def aggregate_expert_scores(publication_scores, agg_funcs, expert_id_col='Expert_ID', project_id_col='Project_ID'):
    """Aggregate publication-level scores to expert-level scores using provided aggregation functions."""
    expert_scores = []
    grouped = publication_scores.groupby([project_id_col])
    with tqdm(total=len(grouped), desc="Aggregating expert-project scores", file=sys.stdout) as project_pbar:
        for project_id, project_group in grouped:
            if isinstance(project_id, tuple):
                project_id = project_id[0]
            for expert_id, group in project_group.groupby(expert_id_col):
                group_filled = group.fillna(0)
                score_dict = {expert_id_col: expert_id, project_id_col: project_id}
                for col, funcs in agg_funcs.items():
                    for func in funcs:
                        filtered_values = group_filled.loc[group_filled[col] != 0, col]
                        if func == 'max':
                            score_dict[f'Expert_{col}_Max'] = group_filled[col].max()
                        elif func == 'avg':
                            score_dict[f'Expert_{col}_Avg'] = filtered_values.mean() if not filtered_values.empty else 0
                expert_scores.append(score_dict)
            project_pbar.update(1)  # Update progress after processing all experts for a project
    return pd.DataFrame(expert_scores)


