import os
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

class LLMHandler:
    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.pipeline_generation = None
        self.generation_args = None
        self.prompts_dir = config_manager.get('PROMPTS_DIR')
        self.use_external = config_manager.get('USE_EXTERNAL_LLM_MODEL', False)
        # Initialize the appropriate model based on whether we're using an external or local LLM
        self._initialize_model()

    def _initialize_model(self):
        """Initializes the model pipeline (local or external)."""
        if self.use_external:
            print('Running external LLM...')
            # Initialize parameters used in external model.
            self.generation_args = {
                'model': self.config_manager.get('MODEL_NAME_GENERATIVE'),
                'external_model_url': self.config_manager.get('EXTERNAL_MODEL_URL'),
                'api_key': self.config_manager.get('EXTERNAL_MODEL_API_KEY'),
                'max_tokens': self.config_manager.get('MAX_NEW_TOKENS', 3000),
                'top_p': 1.0,
                'echo': False
            }
        else:
            print('Running local LLM...')
            # Load the local model using Transformers.
            self.local_model_name = self.config_manager.get('MODEL_NAME_GENERATIVE')
            self.generation_args = {
                'max_new_tokens': self.config_manager.get('MAX_NEW_TOKENS', 750),
                'return_full_text': False,
                'do_sample': False
            }
            model = AutoModelForCausalLM.from_pretrained(
                self.local_model_name,
                device_map='auto',
                torch_dtype='auto',
                trust_remote_code=True
            )
            tokenizer = AutoTokenizer.from_pretrained(self.local_model_name)
            self.pipeline_generation = pipeline('text-generation', model=model, tokenizer=tokenizer)

    def _load_prompt(self, prompt_file):
        """Load prompts from files."""
        prompt = None
        with open(os.path.join(self.prompts_dir, prompt_file)) as f:
            prompt = f.read()
        return prompt

