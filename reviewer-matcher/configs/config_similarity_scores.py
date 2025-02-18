# File: config_similarity_scores.py

# Output files similarity scores - paths relative to SCORES_PATH defined in config_general.
FILE_EXPERT_PROJECT_RESEARCH_TYPE_SIMILARITY = 'expert_project_research_type_similarity_scores.pkl'
FILE_EXPERT_PROJECT_CONTENT_SIMILARITY = 'expert_project_content_similarity_scores.pkl'
FILE_EXPERT_PROJECT_MESH_SIMILARITY = 'expert_project_mesh_similarity_scores.pkl'
FILE_EXPERT_PROJECT_LABEL_SIMILARITY = 'expert_project_label_similarity_scores.pkl'

# Text similarity settings - used form MeSH and content similarity computations.
TEXT_SIMILARITY_MODEL = 'FremyCompany/BioLORD-2023'
DISTANCE_THRESHOLD_CLUSTERS = 0.2
SIMILARITY_THRESHOLD_TERMS = 0.6

# Whether to compute and include coverage score for MeSH terms in projects for each expert based on their publications.
INCLUDE_EXPERTS_MESH_COVERAGE_SCORE = True

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

# !!! TODO: Reorganize the definition of input/output columns across configuration files !!!

# Column mappings for projects and experts.
OUTPUT_COLUMNS_PROJECTS = {
    'ID': 'Project_ID',
    'TITLE': 'Project_Title',
    'APPROACH_TYPE': 'Project_Type',
    'RESEARCH_TYPE': 'Project_Research_Types',
    'RESEARCH_APPROACHES': 'Project_Research_Approaches'
}

OUTPUT_COLUMNS_EXPERTS = {
    'ID': 'Expert_ID',
    'FULL_NAME': 'Expert_Full_Name',
    'GENDER': 'Expert_Gender',
    'RESEARCH_TYPES': 'Expert_Research_Types',
    'RESEARCH_APPROACHES': 'Expert_Research_Approaches',
    'SENIORITY_PUBLICATIONS': 'Expert_Seniority_Publications',
    'SENIORITY_REVIEWER': 'Expert_Seniority_Reviewer',
    'EXPERIENCE_REVIEWER': 'Expert_Experience_Reviewer',
    'EXPERIENCE_PANEL': 'Expert_Experience_Panel',
    'NUMBER_PUBLICATIONS': 'Expert_Number_Publications',
    'NUMBER_CITATIONS': 'Expert_Number_Citations',
    'MAX_PROJECTS_REVIEW': 'Expert_Max_Projects_Review'
}

OUTPUT_COLUMN_RESEARCH_TYPE_SIMILARITY = 'Research_Type_Similarity_Score'
OUTPUT_COLUMN_EXPERT_SENIORITY_PUBLICATIONS = 'Expert_Seniority_Publications'
OUTPUT_COLUMN_EXPERT_SENIORITY_REVIEWER = 'Expert_Seniority_Reviewer'

# Columns to normalize. !!!!! VER !!!!!
COLUMNS_TO_NORMALIZE = [
  OUTPUT_COLUMN_RESEARCH_TYPE_SIMILARITY,
  OUTPUT_COLUMN_EXPERT_SENIORITY_PUBLICATIONS,
  OUTPUT_COLUMN_EXPERT_SENIORITY_REVIEWER
]




