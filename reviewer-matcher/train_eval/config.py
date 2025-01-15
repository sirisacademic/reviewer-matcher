# config.py

# List of calls to include in training/evaluation.
CALLS = [
    '2018-Cancer',
    '2019-Minoritaries',
    '2021-Salut_Mental',
    '2022-Salut_Cardiovascular',
    '2023-Salut_Sexual_i_Reproductiva'
]

# Data used only for evaluation - excluded from training when doing evaluation (but used to train final model).
EVALUATION_YEAR = 2023

# Paths dynamically created for each call.
CALL_PATHS = {
    call: {
        "data": f"../calls/{call}/data",
        "scores": f"../calls/{call}/scores"
    }
    for call in CALLS
}

# MODEL NAME.
#MODEL_NAME = 'xgboost'
MODEL_NAME = 'lasso'

# Standardize data before training/predicting - only if the data was not standardized when generated.
STANDARIZE_DATA = True

# Use only final assignments (3) as positive to train/evaluate (instead of the 8 candidates assigned as potential candidates).
USE_SUBSET_FINAL_ASSIGNMENTS = False

# Whether to balance class weights during training.
BALANCE_CLASSES = False

# Model and output paths.
TRAIN_EVAL_DATA_PATH = 'data'
MODEL_EVAL_SAVE_PATH = 'models/eval'
MODEL_FINAL_SAVE_PATH = 'models/final'
OUTPUT_TRAIN_EVAL_PATH = 'output_train_eval'
OUTPUT_PREDICTIONS_FINAL_PATH = 'predictions/final'
OUTPUT_PREDICTIONS_EVAL_PATH = 'predictions/eval'
OUTPUT_METRICS_PATH = 'results'

# Input file names.
EXPERTS_FILE = 'experts.pkl'
MANUAL_ANNOTATIONS_FILE = 'annotations.tsv'
MANUAL_ASSIGNMENTS_FILE = 'manual_assignments.tsv'
EXPERT_PROJECT_SCORES_FILE = 'expert_project_features.tsv'

# Generated files.
FINAL_ANNOTATIONS_FILE = 'final_annotations_with_ids.tsv'
TRAIN_EVAL_DATA_FILE = 'train_eval_data.tsv'
PREDICTIONS_ALL_PAIRS_FILE = 'predictions_all_pairs.tsv'

# These are used for evaluation of manual assignments not used in training.
MANUAL_ASSIGNMENTS_NOT_TRAINING_FILE = 'manual_assignments_not_used_in_training.tsv'
PREDICTIONS_EVALUATION_FILE = 'predictions_for_ranking_evaluation.tsv'

# Column and valid values for considered task.
COLUMN_TASK_GOLD = 'Annotation'
ANNOTATION_POSITIVE = 'Adequate'
ANNOTATION_NEGATIVE = 'Inadequate'

# Settings for train/test splits.
# When only final assignments are used there are fewer examples available for stratified sampling for the test set.
if USE_SUBSET_FINAL_ASSIGNMENTS:
    TEST_SIZE_PERC = 0.20
else:
    TEST_SIZE_PERC = 0.20

# Subset of features to use when training/making predictions (see below).
SET_FEATURES = 'all_features'

# Method for feature selection. Values: 'corr', 'mi', 'fc', 'chi2', 'rf'.
# Simple correlation (seems to perform better in this context), or mutual information.
# Leave empty to skip feature selection (and use all features of SET_FEATURES).
FEATURE_SELECTION = 'fc'

# Additional prefix. Leave empty if no specific prefix should be added to data/models.
PREFIX = ''

# !!!!!! manual_corr_features still work best !!!!!

# Feature columns
FEATURE_COLUMNS = {
    'feature_selection': [
        'Expert_Topic_Similarity_Max',
        'Expert_Topic_Similarity_Avg',
        'Expert_Objectives_Max_Similarity_Max',
        'Expert_Objectives_Max_Similarity_Avg',
        'Expert_Objectives_Avg_Similarity_Max',
        'Expert_Objectives_Avg_Similarity_Avg',
        'Expert_Methods_Max_Similarity_Max',
        'Expert_Methods_Max_Similarity_Avg',
        'Expert_Methods_Avg_Similarity_Max',
        'Expert_Methods_Avg_Similarity_Avg',
        'Expert_MeSH_Max_Similarity_Avg',
        'Expert_MeSH_Avg_Similarity_Max',
        'Expert_MeSH_Avg_Similarity_Avg',
        'Research_Areas_Jaccard_Similarity',
        'Research_Areas_Dice_Similarity',
        'Research_Areas_Overlap_Coefficient',
        'Research_Approaches_Jaccard_Similarity',
        'Research_Approaches_Dice_Similarity',
        'Research_Approaches_Overlap_Coefficient',
        'All_Columns_Average'
    ],
    'pub_content_general_features': [
        'Expert_Topic_Similarity_Max',
        'Expert_Topic_Similarity_Avg',
        'Expert_Objectives_Max_Similarity_Max',
        'Expert_Objectives_Max_Similarity_Avg',
        'Expert_Objectives_Avg_Similarity_Max',
        'Expert_Objectives_Avg_Similarity_Avg',
        'Expert_Methods_Max_Similarity_Max',
        'Expert_Methods_Max_Similarity_Avg',
        'Expert_Methods_Avg_Similarity_Max',
        'Expert_Methods_Avg_Similarity_Avg',
        'Expert_Methods_Max_Similarity_Weighted_Max',
        'Expert_Methods_Max_Similarity_Weighted_Avg',
        'Expert_Methods_Avg_Similarity_Weighted_Max',
        'Expert_Methods_Avg_Similarity_Weighted_Avg'
    ],
    'pub_content_features': [
        'Expert_Topic_Similarity_Max',
        'Expert_Topic_Similarity_Avg',
        'Expert_Objectives_Max_Similarity_Max',
        'Expert_Objectives_Max_Similarity_Avg',
        'Expert_Objectives_Avg_Similarity_Max',
        'Expert_Objectives_Avg_Similarity_Avg',
        'Expert_Methods_Specific_Max_Similarity_Max',
        'Expert_Methods_Specific_Max_Similarity_Avg',
        'Expert_Methods_Specific_Avg_Similarity_Max',
        'Expert_Methods_Specific_Avg_Similarity_Avg',
        'Expert_Methods_Max_Similarity_Max',
        'Expert_Methods_Max_Similarity_Avg',
        'Expert_Methods_Avg_Similarity_Max',
        'Expert_Methods_Avg_Similarity_Avg',
        'Expert_Methods_Max_Similarity_Weighted_Max',
        'Expert_Methods_Max_Similarity_Weighted_Avg',
        'Expert_Methods_Avg_Similarity_Weighted_Max',
        'Expert_Methods_Avg_Similarity_Weighted_Avg'
    ],
    'pub_mesh_features': [
        'Expert_MeSH_Max_Similarity_Max',
        'Expert_MeSH_Max_Similarity_Avg',
        'Expert_MeSH_Avg_Similarity_Max',
        'Expert_MeSH_Avg_Similarity_Avg',
        'Expert_MeSH_Max_Similarity_Weighted_Max',
        'Expert_MeSH_Max_Similarity_Weighted_Avg',
        'Expert_MeSH_Avg_Similarity_Weighted_Max',
        'Expert_MeSH_Avg_Similarity_Weighted_Avg',
        'Expert_MeSH_Semantic_Coverage_Score'
    ],
    'expert_features': [
        'Research_Areas_Jaccard_Similarity',
        'Research_Areas_Dice_Similarity',
        'Research_Areas_Overlap_Coefficient',
        'Research_Approaches_Jaccard_Similarity',
        'Research_Approaches_Dice_Similarity',
        'Research_Approaches_Overlap_Coefficient'
    ],
    'global_features': [
        'Research_Type_Similarity_Score',
        'All_Columns_Average'
    ],
    'pca_features': [
        'MeSH_PCA_1',
        'MeSH_PCA_2',
        'MeSH_PCA_3',
        'Topic_PCA_1',
        'Objectives_PCA_1',
        'Methods_Specific_PCA_1',
        'Methods_PCA_1',
        'Methods_PCA_2',
        'Methods_PCA_3'
    ]
}

# All features.
FEATURE_COLUMNS['all_features'] = (
    FEATURE_COLUMNS['pub_content_features'] + 
    FEATURE_COLUMNS['pub_mesh_features'] + 
    FEATURE_COLUMNS['expert_features'] +
    FEATURE_COLUMNS['global_features']
)

# All features with PCA features.
FEATURE_COLUMNS['all_features_plus_pca'] = FEATURE_COLUMNS['all_features'] + FEATURE_COLUMNS['pca_features']

# All features with Gender.
FEATURE_COLUMNS['all_features_plus_gender'] = FEATURE_COLUMNS['all_features'] + ['Gender']

# Define ablation study feature sets
FEATURE_COLUMNS['ablation_pub_content'] = list(
    set(FEATURE_COLUMNS['all_features']) - set(FEATURE_COLUMNS['pub_content_features'])
)

FEATURE_COLUMNS['ablation_pub_content_general_features'] = list(
    set(FEATURE_COLUMNS['all_features']) - set(FEATURE_COLUMNS['pub_content_general_features'])
)

FEATURE_COLUMNS['ablation_pub_mesh'] = list(
    set(FEATURE_COLUMNS['all_features']) - set(FEATURE_COLUMNS['pub_mesh_features'])
)

FEATURE_COLUMNS['ablation_expert'] = list(
    set(FEATURE_COLUMNS['all_features']) - set(FEATURE_COLUMNS['expert_features'])
)


