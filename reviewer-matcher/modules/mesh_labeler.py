from utils.bert_mesh import BertMeshConfig, BertMesh
from utils.text_splitter import TextSplitter
from transformers import AutoTokenizer, AutoModel
from collections import defaultdict, Counter
import spacy

# TODO: Generalize to use other sentence splitters.

class MeSHLabeler:
    def __init__(self, config_manager):
        self.text_splitter = TextSplitter()
        self.model_name = config_manager.get('MESH_MODEL')
        self.exclude_terms = config_manager.get('MESH_EXCLUDE_TERMS')
        self.threshold = config_manager.get('MESH_THRESHOLD')
        self.separator = config_manager.get('SEPARATOR_VALUES_OUTPUT')
        self.output_column = config_manager.get('MESH_COMBINED_OUTPUT_COLUMN', 'MESH_EXTRACTED')
        self._init_model()

    def _init_model(self):
        # Load model configuration from the pretrained model.
        config = BertMeshConfig.from_pretrained(self.model_name)
        # Ensure num_labels is set correctly based on id2label.
        num_labels = len(config.id2label)
        config.num_labels = num_labels
        # Load the tokenizer.
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # Initialize the model with the loaded configuration.
        self.model = BertMesh.from_pretrained(self.model_name, config=config)

    def get_labels(self, text, threshold=0.5):
        inputs = self.tokenizer([text], padding='max_length', return_tensors='pt', truncation=True)
        attention_mask = inputs['attention_mask']
        labels = self.model(input_ids=inputs['input_ids'], attention_mask=attention_mask, return_labels=True, threshold=threshold)
        return labels[0]

    def get_mesh_terms(self, text, return_occurrences=True):
        sentences = []
        mesh_terms = []
        mesh_probs = defaultdict(float)
        # We split the text into sentences because the model returns very few terms otherwise.
        if isinstance(text, str):
            sentences = self.text_splitter.split_sentences(text)
        elif isinstance(text, list):
            sentences = text
        # Process sentences to obtain MeSH terms, exlcuding the ones set in the config file.
        for sentence in sentences:
            labels = self.get_labels(sentence, threshold=self.threshold)
            mesh_terms.extend([term for term in labels if term not in self.exclude_terms])
        # Aggregate results.
        if return_occurrences:
            counts = Counter(mesh_terms)
            return {term: {'probability': mesh_probs[term], 'count': counts[term]} for term in counts}
        return list(set(mesh_terms))

    def label_with_mesh(self, data, input_columns):
        """Apply MeSH term tagging for the specified columns in the dataset."""
        for col_name, col_type in input_columns.items():
            if col_name in data.columns:
                data[f'MESH_{col_name}'] = data[col_name].apply(
                    lambda text: self.get_mesh_terms(
                        text.split(self.separator) if col_type == 'list' else text,
                        return_occurrences=False
                    )
                )
        # combine extracted terms into a single column
        combined_columns = [f'MESH_{col}' for col in input_columns.keys()]
        data[self.output_column] = data[combined_columns].apply(
            lambda row: self.separator.join(set().union(*row)), axis=1
        )
        return data
        

