### LLM model.

# Define model usage type: 'local' or 'external'
USE_EXTERNAL_LLM_MODEL = True  # Change to use local model.

# Model/parameters for generative model used to extract data for projects.
if USE_EXTERNAL_LLM_MODEL:
  MODEL_NAME_GENERATIVE = 'google/gemma-2-27b-it'
  MAX_NEW_TOKENS = 3000
  MAX_NUMBER_TOPICS_PER_PROMPT = 20
  MAX_RETRIES = 5
  RETRY_DELAY = 2
  REQUEST_JUSTIFICATION = False
  TEMPERATURE_CLASSIFICATION = 0.0
  TEMPERATURE_GENERATION = 0.5
else:
  MODEL_NAME_GENERATIVE = 'microsoft/Phi-3.5-mini-instruct'
  MAX_NEW_TOKENS = 750
  MAX_NUMBER_TOPICS_PER_PROMPT = 5
  REQUEST_JUSTIFICATION = True
  MAX_RETRIES = 3
  RETRY_DELAY = 2
  TEMPERATURE_CLASSIFICATION = 0.0
  TEMPERATURE_GENERATION = 0.2

MODELS_JSON_RESPONSE = [
  'meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo',
  'meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo',
  'mistralai/Mixtral-8x7B-Instruct-v0.1',
  'mistralai/Mistral-7B-Instruct-v0.1'
]

# Add external model specific settings. Now using Together.AI for external models, but it can be changed to any OpenAI-compliant provider.
EXTERNAL_MODEL_URL = 'https://api.together.xyz/v1/chat/completions'
EXTERNAL_MODEL_API_KEY = 'c9bd0e95399a1bf9defad8ae8ab1b37db4c706b9e13a2e3e04bc33dac363845e'

### Whether to extract fine-grained methods in one step.
# If False, methods are first extracted and then classified into "standard" or "specific" in a second step.

# It does not work well to try to do everything in one step with the currently used model, so we are first identifying methods and then classifying them.
ONE_STEP_FINE_GRAINED_METHODS = False

### Columns used for content extracted by means of generative model.

COLUMNS_FINE_GRAINED_METHODS = ['METHODS_STANDARD', 'METHODS_SPECIFIC']
EXTRACTED_CONTENT_COLUMNS = ['RESEARCH_TOPIC', 'OBJECTIVES']

if ONE_STEP_FINE_GRAINED_METHODS:
  EXTRACTED_CONTENT_COLUMNS += COLUMNS_FINE_GRAINED_METHODS
else:
  EXTRACTED_CONTENT_COLUMNS += ['METHODS']
  
### Prompts to use for generative model.
# !!! Relative to CODE_PATH defined in config_general.py !!!

PROMPTS_DIR = 'prompts'

# Input columns for title, abstract.
COLUMN_TITLE = 'TITLE'
COLUMN_ABSTRACT = 'ABSTRACT'

### Columns of the experts' dataframe containing pre-defined values for research areas (topics) and approaches in the experts' spreadsheet.
# These values are used to tag each project indicating whether it can be associated with these research areas or approaches.
# Only values actually ocurring are considered (pre-defined values that are not associated with any expert are not considered).
COLUMN_RESEARCH_AREAS = 'RESEARCH_AREAS'
COLUMN_RESEARCH_APPROACHES = 'RESEARCH_APPROACHES'

# Prompt files.
if REQUEST_JUSTIFICATION:
  FILE_PROMPT_TOPICS = 'prompt_label_topics_justification.txt'
  FILE_PROMPT_APPROACHES = 'prompt_label_approaches_justification.txt'
else:
  FILE_PROMPT_TOPICS = 'prompt_label_topics.txt'
  FILE_PROMPT_APPROACHES = 'prompt_label_approaches.txt'

FILE_PROMPT_METHODS = 'prompt_classification_methods.txt'

if ONE_STEP_FINE_GRAINED_METHODS:
  FILE_PROMPT_CONTENT = 'prompt_extract_fine_grained_contents_sections.txt'
else:
  FILE_PROMPT_CONTENT = 'prompt_extract_contents_sections.txt'

### Default responses to validate the responses of generative model.
# These structures are used as default responses when the model fails to provide a well-structured response.

DEFAULT_RESPONSE_METHODS = {
    'methods_standard': [],
    'methods_specific': []
}

if ONE_STEP_FINE_GRAINED_METHODS:
  DEFAULT_RESPONSE_CONTENT = {
      'research_topic': '',
      'objectives': [],
      'methods_standard': [],
      'methods_specific': []
    }
else:
  DEFAULT_RESPONSE_CONTENT = {
      'research_topic': '',
      'objectives': [],
      'methods': []
  }


  
