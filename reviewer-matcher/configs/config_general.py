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

# Combined projects 
FILE_PATH_PROJECTS_ALL_YEARS = 'calls/all_years/projects.pkl'

# Separator for values in generated files.
SEPARATOR_VALUES_OUTPUT = '|'

# HuggingFace token.
HF_TOKEN = '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'

### Columns of the experts' dataframe containing pre-defined values for research areas (topics) and approaches in the experts' spreadsheet.
# These values are used to tag each project indicating whether it can be associated with these research areas or approaches.
# Only values actually ocurring are considered (pre-defined values that are not associated with any expert are not considered).
EXTRACT_VALUES_EXPERTS_COLUMNS = [
    'RESEARCH_AREAS',
    'RESEARCH_APPROACHES'
]


