import pandas as pd
from tqdm import tqdm
from .llm_handler import LLMHandler
### TODO: Refactor code and include functions as methods.
from utils.functions_llm import label_by_topic, extract_and_combine_responses

class ResearchLabeler(LLMHandler):
    def __init__(self, config_manager):
        # Initialize LLM.
        super().__init__(config_manager)
        # Configurations read from config file handled by config_manager.
        self.separator_output = config_manager.get('SEPARATOR_VALUES_OUTPUT', '|')
        self.max_topics = config_manager.get('MAX_NUMBER_TOPICS_PER_PROMPT', 5)
        self.max_retries = config_manager.get('MAX_RETRIES', 5)
        self.retry_delay = config_manager.get('RETRY_DELAY', 2)
        self.json_response = config_manager.get('JSON_RESPONSE', False)
        self.input_cols = {
            'title': config_manager.get('COLUMN_TITLE'),
            'abstract': config_manager.get('COLUMN_ABSTRACT'),
        }
        # Set temperature for classification.
        self.generation_args['temperature'] = config_manager.get('TEMPERATURE_CLASSIFICATION', 0.0)

    def label_topics(self, df, topics, prompt_file, output_column):
        """
        Label projects with a list of topics and combine responses.
        Args:
            df: DataFrame containing project data (with TITLE and ABSTRACT columns).
            topics: List of topics to label projects with.
            prompt_type: The type of prompt to use (e.g., 'RESEARCH_AREAS', 'RESEARCH_APPROACHES').
            model_pipeline: The LLM pipeline to use (local or external).
            separator: Separator for combining responses in the final output.
        Returns:
            pd.Series: Combined labeled topics for each project.
        """
        tqdm.pandas()
        # Load prompt.
        prompt = self._load_prompt(prompt_file)
        # Get responses.
        research_labels = df.progress_apply(lambda row:
            pd.Series(
                label_by_topic(
                    self.pipeline_generation,
                    self.generation_args,
                    prompt,
                    title=row[self.input_cols['title']],
                    abstract=row[self.input_cols['abstract']],
                    topics=topics,
                    max_topics=self.max_topics,
                    max_retries=self.max_retries,
                    retry_delay=self.retry_delay,
                    json_response=self.json_response
                )
            ),
            axis=1
        )
        # Combine output and add column to dataframe.
        df[output_column] = research_labels.progress_apply(
            lambda row: extract_and_combine_responses(row, topics, self.separator_output),
            axis=1
        )
        return df

