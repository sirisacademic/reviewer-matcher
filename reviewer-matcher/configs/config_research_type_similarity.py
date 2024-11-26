# Call to be processed (used to retrieve the configuration, URLs, etc).
CALL = '2021-Salut_Mental'

# Directory with call-specific data.
CALL_PATH = f'calls/{CALL}'

# Call-specific path.
DATA_PATH = f'{CALL_PATH}/data'

# Input files.
FILE_PATH_PROJECTS = f'{DATA_PATH}/{CALL_PATH}/projects.pkl'
FILE_PATH_EXPERTS = f'{DATA_PATH}/{CALL_PATH}/experts.pkl'

FILE_PATH_EXPERT_PROJECT_CONTENT_SIMILARITY_SCORES = f'{DATA_PATH}/{CALL_PATH}/scores/expert_projects_content_similarity_scores.pkl'
FILE_PATH_EXPERT_PROJECT_MESH_SCORES = f'{DATA_PATH}/{CALL_PATH}/scores/expert_projects_mesh_scores.pkl'
FILE_PATH_EXPERT_PROJECT_JACCARD_SIMILARITY = f'{DATA_PATH}/{CALL_PATH}/scores/expert_project_jaccard_similarity_scores.pkl'

# Output file.
FILE_PATH_COMBINED_SCORES = f'{DATA_PATH}/{CALL_PATH}/combined_similarity_scores.tsv'

# Settings for matching research types.
WEIGHT_EXACT_MATCH = 2.0
WEIGHT_PARTIAL_MATCH = 1.0
WEIGHT_BASIC_RESEARCH_PRIORITY = 3.0
WEIGHT_RELATED_MATCH = 1.5

RELATED_TYPES = {
    'clinical': {'clinical', 'applied_clinical'},
    'applied_clinical': {'clinical', 'applied_clinical'},
    'basic': {'basic'},
    'epidemiology': {'epidemiology'}
}

EXCLUDE_TYPES = {'translational'}

PRIORITY_TYPE = 'basic'

# Column mappings for projects and experts.
COLUMNS_PROJECTS = {
    'ID': 'Project_ID',
    'TITLE': 'Project_Title',
    'RESEARCH_TYPE': 'Project_Research_Types'
}

COLUMNS_EXPERTS = {
    'ID': 'Expert_ID',
    'FULL_NAME': 'Expert_Full_Name',
    'GENDER': 'Expert_Gender',
    'RESEARCH_TYPES': 'Expert_Research_Types',
    'SENIORITY': 'Expert_Seniority',
    'EXPERIENCE_REVIEWER': 'Expert_Experience_Reviewer',
    'EXPERIENCE_PANEL': 'Expert_Experience_Panel',
    'NUMBER_PUBLICATIONS': 'Expert_Number_Publications',
    'NUMBER_CITATIONS': 'Expert_Number_Citations'
}

# Columns to normalize.
COLUMNS_TO_NORMALIZE = ['Research_Type_Similarity_Score', 'Expert_Seniority']
