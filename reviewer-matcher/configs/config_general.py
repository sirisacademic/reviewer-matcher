### config_general.py
### Input paths and names.

### Calls
#2015-Diabetis_i_Obesitat
#2016-Ictus
#2017-Infeccioses
#2018-Cancer
#2019-Minoritaries
#2021-Salut_Mental
#2022-Salut_Cardiovascular
#2023-Salut_Sexual_i_Reproductiva

# Run in TEST_MODE
TEST_MODE = False
TEST_NUMBER = 10

# Call to be processed (used to retrieve the configuration, URLs, etc).
CALL = '2023-Salut_Sexual_i_Reproductiva'

# Directory with call-specific data.
CALL_PATH = f'calls/{CALL}'

# Call-specific  path.
DATA_PATH = f'{CALL_PATH}/data'

# Mappings path.
MAPPINGS_PATH = f'{CALL_PATH}/mappings'

# Output scores path.
SCORES_PATH = f'{CALL_PATH}/scores'

# Output path used for predicted ranks and assignments.
ASSIGNMENTS_PATH = f'{CALL_PATH}/predicted_assignments'

# JSON file with column mappings for projects.
# Relative to MAPPINGS_PATH !
MAPPINGS_PROJECTS = 'mappings_projects.json'

# JSON file with column mappings for reserchers.
# Relative to MAPPINGS_PATH !
MAPPINGS_EXPERTS = 'mappings_experts.json'

# Ouptut file projects.
# Relative to DATA_PATH !
FILE_NAME_PROJECTS = 'projects.pkl'

# Output file experts.
# Relative to DATA_PATH !
FILE_NAME_EXPERTS = 'experts.pkl'

# Combined projects.
FILE_PATH_PROJECTS_ALL_YEARS = 'calls/all_years/projects.pkl'

# Separator for values in generated files.
SEPARATOR_VALUES_OUTPUT = '|'

# ID Column name to add if it does not exist.
ID_COLUMN_NAME = 'ID'

# HuggingFace token.
HF_TOKEN = '[HF TOKEN HERE]'

SPACY_MODEL = 'en_core_web_sm'

MESH_MODEL = 'Wellcome/WellcomeBertMesh'

# MeSH Labeler Configuration.
# Terms to exclude from MeSH labeling
MESH_EXCLUDE_TERMS = [
  'Animals',
  'Humans',
  'Female',
  'Male',
  'Adult',
  'Young Adult',
  'Infant',
  'Child',
  'Child, Preschool',
  'Adolescent',
  'Aged',
  'Middle Aged',
  'Anatomy',
  'Organism',
  'Cell',
  'Tissue',
  'Organ',
  'Disease'
]  
MESH_THRESHOLD = 0.6  # Probability threshold for MeSH term inclusion

# Excluding the title because for some reason it generates a long list of unrelated MeSH terms in some cases.
# Input column configurations for MeSH tagging.
MESH_INPUT_COLUMNS_PROJECTS = {
    'ABSTRACT': 'string',
    'RESEARCH_TOPIC': 'string',
    'OBJECTIVES': 'string',
    'METHODS': 'list'
}

MESH_INPUT_COLUMNS_PUBLICATIONS = {
    'ABSTRACT': 'string',
    'RESEARCH_TOPIC': 'string',
    'OBJECTIVES': 'string',
    'METHODS': 'list'
}

MESH_COMBINED_OUTPUT_COLUMN = 'MESH_EXTRACTED'

RESEARCH_TOPIC_COLUMN = 'RESEARCH_TOPIC'
METHODS_SPECIFIC_COLUMN = 'METHODS_SPECIFIC'
METHODS_COLUMN = 'METHODS'
OBJECTIVES_COLUMN = 'OBJECTIVES'

