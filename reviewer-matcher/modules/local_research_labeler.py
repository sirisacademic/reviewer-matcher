from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from .research_labeler import ResearchLabeler

class LocalResearchLabeler(ResearchLabeler):
    def __init__(self, config_manager):
        # DEBUG
        print('Running local LLM...')
        super().__init__(config_manager)
        self.local_model_name = config_manager.get('MODEL_NAME_GENERATIVE')
        self.generation_args = {
            'max_new_tokens': config_manager.get('MAX_NEW_TOKENS', 750),
            'return_full_text': False,
            'temperature': config_manager.get('TEMPERATURE_CLASSIFICATION', 0.0),
            'do_sample': False,
          }
        self.pipeline_generation = self._get_model_pipeline()

    def _get_model_pipeline(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.local_model_name,
            device_map='auto',
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(self.local_model_name)
        return pipeline('text-generation', model=model, tokenizer=tokenizer)

