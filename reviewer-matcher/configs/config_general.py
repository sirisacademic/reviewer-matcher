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

# Call to be processed (used to retrieve the configuration, URLs, etc).
CALL = '2021-Salut_Mental'

# Directory with call-specific data.
CALL_PATH = f'calls/{CALL}'

# Call-specific  path.
DATA_PATH = f'{CALL_PATH}/data'

# Mappings path.
MAPPINGS_PATH = f'{CALL_PATH}/mappings'

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

# Output file expert publications.
# Relative to DATA_PATH !
FILE_NAME_PUBLICATIONS = 'expert_publications.pkl'

# Combined projects.
FILE_PATH_PROJECTS_ALL_YEARS = 'calls/all_years/projects.pkl'

# Separator for values in generated files.
SEPARATOR_VALUES_OUTPUT = '|'

# ID Column name to add if it does not exist.
ID_COLUMN_NAME = 'ID'

# HuggingFace token.
HF_TOKEN = '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'

# MeSH Labeler Configuration.
MESH_EXCLUDE_TERMS = ['Animals', 'Humans', 'Female', 'Male']  # Terms to exclude from MeSH labeling
MESH_THRESHOLD = 0.6  # Probability threshold for MeSH term inclusion

# Input column configurations for MeSH tagging.
MESH_INPUT_COLUMNS_PROJECTS = {
    'TITLE': 'string',
    'ABSTRACT': 'string',
    'RESEARCH_TOPIC': 'string',
    'OBJECTIVES': 'string',
    'METHODS': 'list'
}

MESH_INPUT_COLUMNS_PUBLICATIONS = {
    'TITLE_PUBMED': 'string',
    'ABSTRACT_PUBMED': 'string',
    'RESEARCH_TOPIC': 'string',
    'OBJECTIVES': 'string',
    'METHODS': 'list'
}

# OpenAlex API Key and Base URL.

# OPENALEX_API_KEY = ...
OPENALEX_BASE_URL = "https://api.openalex.org"

# Setting of values for low, middle and high seniority.

SENIORITY_UNDETERMINED = 0
SENIORITY_LOW = 1
SENIORITY_MIDDLE = 2
SENIORITY_HIGH = 3

NUM_PUBS_TOP_PERC_SENIORITY = 30
NUM_CITATIONS_TOP_PERC_SENIORITY = 30