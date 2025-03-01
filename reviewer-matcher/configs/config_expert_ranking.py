# Output file features.
FILE_EXPERT_PROJECT_FEATURES = 'expert_project_features.tsv'

# Output file predictions.
# Should contain expert and project IDs, gender, maximum number of projects for each expert - with default value if the column is not
FILE_EXPERT_PROJECT_PREDICTIONS = 'expert_project_predictions.tsv'

# Output expert assignment.
FILE_EXPERT_PROJECT_ASSIGNMENTS = 'expert_project_assignments.tsv'

# Potential projects to be assigned for each expert.
FILE_EXPERT_PROJECT_POTENTIAL_ASSIGNMENTS = "expert_project_potential_assignments.tsv"

# !!!! TODO: Unify handling of column names accorss configuration files !!!!

# Input column names (in original data obtained from spreadsheets).
EXPERT_ID_INPUT_COLUMN = 'ID'
EXPERT_GENDER_INPUT_COLUMN = 'GENDER'
EXPERT_MAX_PROPOSED_PROJECTS_INPUT_COLUMN = 'MAX_PROJECTS_REVIEW'
EXPERT_NAME_INPUT_COLUMN = 'FULL_NAME'
EXPERT_RESEARCH_TYPES_INPUT_COLUMN = 'RESEARCH_TYPES'
EXPERT_RESEARCH_APPROACHES_INPUT_COLUMN = 'RESEARCH_APPROACHES'

PROJECT_ID_INPUT_COLUMN = 'ID'
PROJECT_TITLE_INPUT_COLUMN = 'TITLE'
PROJECT_APPROACH_TYPE_INPUT_COLUMN = 'APPROACH_TYPE'
PROJECT_RESEARCH_TYPES_INPUT_COLUMN = 'RESEARCH_TYPE'
PROJECT_RESEARCH_APPROACHES_INPUT_COLUMN = 'RESEARCH_APPROACHES'

MIN_PROBABILITY_THRESHOLD = 0.5

# Output column names (see other definitions in config_similarity_scores.py)
PREDICTED_PROB_COLUMN = "Predicted_Prob"
PREDICTED_PROB_RANK_COLUMN = "Predicted_Prob_Rank"

EXPERT_GENDER_VALUE_WOMEN = 'female'
EXPERT_GENDER_VALUE_MEN = 'male'

# Pre-trained model to predict expert-project adequacy.
MODEL_PATH = 'model/final_model_all_features.pkl'

# Assignment criteria
NUM_PROPOSED_EXPERTS = 3
NUM_ALTERNATIVE_EXPERTS = 20
MIN_WOMEN_PROPOSED = 1
MIN_WOMEN_ALTERNATIVE = 2

# Min. number of projects an expert should be assigned to (if possible)
MIN_PROJECTS_PER_EXPERT = 2

# Max. default projects to which the expert can be assigned as "proposed"
MAX_DEFAULT_PROPOSED_PROJECTS_PER_EXPERT = 5

# Max. total projects to which the expert can be assigned.
MAX_DEFAULT_TOTAL_PROJECTS_PER_EXPERT = 20

# Compute PCA scores.
COMPUTE_PCA_SCORES = False

# We are not standarizing here because we use data from several years so we better standarize it if needed afterwards.
STANDARIZE_NUMERIC_DATA = False

# Feature groups for PCA-dimensionality reduction.
FEATURE_GROUPS = {
    'MeSH': [
        'Expert_MeSH_Max_Similarity_Max',
        'Expert_MeSH_Max_Similarity_Avg',
        'Expert_MeSH_Avg_Similarity_Max',
        'Expert_MeSH_Avg_Similarity_Avg',
        'Expert_MeSH_Max_Similarity_Weighted_Max',
        'Expert_MeSH_Max_Similarity_Weighted_Avg',
        'Expert_MeSH_Avg_Similarity_Weighted_Max',
        'Expert_MeSH_Avg_Similarity_Weighted_Avg'
    ],
    'Topic': [
        'Expert_Topic_Similarity_Max',
        'Expert_Topic_Similarity_Avg'
    ],
    'Objectives': [
        'Expert_Objectives_Max_Similarity_Max',
        'Expert_Objectives_Max_Similarity_Avg',
        'Expert_Objectives_Avg_Similarity_Max',
        'Expert_Objectives_Avg_Similarity_Avg'
    ],
    'Methods_Specific': [
       'Expert_Methods_Specific_Max_Similarity_Max',
       'Expert_Methods_Specific_Max_Similarity_Avg',
       'Expert_Methods_Specific_Avg_Similarity_Max',
       'Expert_Methods_Specific_Avg_Similarity_Avg'
    ],
    'Methods': [
        'Expert_Methods_Max_Similarity_Max',
        'Expert_Methods_Max_Similarity_Avg',
        'Expert_Methods_Avg_Similarity_Max',
        'Expert_Methods_Avg_Similarity_Avg',
        'Expert_Methods_Max_Similarity_Weighted_Max',
        'Expert_Methods_Max_Similarity_Weighted_Avg',
        'Expert_Methods_Avg_Similarity_Weighted_Max',
        'Expert_Methods_Avg_Similarity_Weighted_Avg'
    ]
}



