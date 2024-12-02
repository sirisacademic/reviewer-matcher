import pandas as pd
from tqdm import tqdm
from abbreviations import schwartz_hearst
from utils.functions_read_data import flatten_lists
from .llm_handler import LLMHandler
### TODO: Include as methods.
from utils.functions_llm import extract_content, expand_abbreviations, value_as_list, generate_research_summary_schema

class ContentSummarizer(LLMHandler):
    def __init__(self, config_manager):
        # Initialize LLM with classification task type, as used in ResearchLabeler
        super().__init__(config_manager)
        # Configurations read from config file handled by config_manager.
        self.separator_output = config_manager.get('SEPARATOR_VALUES_OUTPUT', '|')
        self.max_retries = config_manager.get('MAX_RETRIES', 5)
        self.retry_delay = config_manager.get('RETRY_DELAY', 2)
        self.json_response = config_manager.get('JSON_RESPONSE', False)
        # Prompts specific to content extraction.
        self.prompt_content_file = config_manager.get('FILE_PROMPT_CONTENT')
        self.prompt_classification_methods = config_manager.get('FILE_PROMPT_METHODS')
        self.default_response_content = config_manager.get('DEFAULT_RESPONSE_CONTENT', {
            'research_topic': '',
            'objectives': [],
            'methods': []
        })
        self.default_response_methods = config_manager.get('DEFAULT_RESPONSE_METHODS', {'methods_standard': [], 'methods_specific': []})
        # Column names from the configuration.
        self.input_cols = {
            'title': config_manager.get('COLUMN_TITLE', 'TITLE'),
            'abstract': config_manager.get('COLUMN_ABSTRACT', 'ABSTRACT'),
            'methods': config_manager.get('COLUMN_METHODS', 'METHODS'),
        }
        # Output columns.
        self.output_cols = {
            'extracted_content': config_manager.get('EXTRACTED_CONTENT_COLUMNS', ['RESEARCH_TOPIC', 'OBJECTIVES']),
            'fine_grained_methods': config_manager.get('COLUMNS_FINE_GRAINED_METHODS', ['METHODS_STANDARD', 'METHODS_SPECIFIC']),
        }
        # Columns related to abbreviations and expanded content.
        self.abbreviation_cols = {
            'abbreviations': 'ABBREVIATIONS'
        }
        # If methods extraction is needed in two steps, expand further.
        self.one_step_fine_grained_methods = config_manager.get('ONE_STEP_FINE_GRAINED_METHODS', False)
        # Set schema for JSON response if supported by the model. 
        if self.config_manager.get('JSON_RESPONSE'):
            response_format = {
                'schema': generate_research_summary_schema(),
                'type': 'json_object'
            }
            self.generation_args['parameters'] = {'json_mode': True}
            self.generation_args['response_format'] = response_format
        # Set temperature for generation.
        self.generation_args['temperature'] = config_manager.get('TEMPERATURE_GENERATION', 0.2)
        
    def summarize_content(self, df):
        """Extract content and expand abbreviations for projects in the dataframe."""
        tqdm.pandas()
        # Get abbreviations using Schwartz-Hearst method
        df[self.abbreviation_cols['abbreviations']] = df.apply(
            lambda row: schwartz_hearst.extract_abbreviation_definition_pairs(
                doc_text=f'{row[self.input_cols["title"]]} {row[self.input_cols["abstract"]]}'
            ),
            axis=1
        )
        # Get summary sentences with abbreviations expanded.
        df[self.output_cols['extracted_content']] = df.progress_apply(
            lambda row: pd.Series(
                expand_abbreviations(
                    extract_content(
                        self.pipeline_generation,
                        self.generation_args,
                        self._load_prompt(self.prompt_content_file).format(
                            title=row[self.input_cols['title']], 
                            abstract=row[self.input_cols['abstract']]
                        ),
                        self.default_response_content,
                        max_retries=self.max_retries,
                        retry_delay=self.retry_delay
                    ),
                    row[self.abbreviation_cols['abbreviations']]
                )
            ),
            axis=1
        )
        # If methods extraction is needed in two steps, expand further.
        if not self.one_step_fine_grained_methods:
            df[self.output_cols['fine_grained_methods']] = df.progress_apply(
                lambda row: pd.Series(
                    expand_abbreviations(
                        extract_content(
                            self.pipeline_generation,
                            self.generation_args,
                            self._load_prompt(self.prompt_classification_methods).format(
                                title=row[self.input_cols['title']], 
                                methods='- ' + '\n- '.join(value_as_list(row[self.input_cols['methods']], self.separator_output))
                            ),
                            self.default_response_methods,
                            max_retries=self.max_retries,
                            retry_delay=self.retry_delay
                        ),
                        row[self.abbreviation_cols['abbreviations']]
                    )
                ),
                axis=1
            )
        # Flatten lists in the dataframe (if required)
        df = flatten_lists(df, self.separator_output)
        return df

