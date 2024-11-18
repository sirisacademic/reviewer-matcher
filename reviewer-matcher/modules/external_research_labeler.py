from .research_labeler import ResearchLabeler

class ExternalResearchLabeler(ResearchLabeler):
    def __init__(self, config_manager):
        # DEBUG
        print('Running external LLM...')
        super().__init__(config_manager)
        self.generation_args = {
              'model': config_manager.get('MODEL_NAME_GENERATIVE'),
              'external_model_url': config_manager.get('EXTERNAL_MODEL_URL'),
              'api_key': config_manager.get('EXTERNAL_MODEL_API_KEY'),
              'max_tokens': config_manager.get('MAX_NEW_TOKENS', 3000),
              'temperature': config_manager.get('TEMPERATURE_CLASSIFICATION', 0.0),
              'top_p': 1.0,
              'echo': False
            }

    
    
