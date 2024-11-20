from transformers import AutoTokenizer, AutoModel
from collections import defaultdict, Counter
import spacy

class MeSHLabeler:
    def __init__(self, model_name='Wellcome/WellcomeBertMesh', spacy_model='en_core_web_sm', exclude_terms=None, threshold=0.6):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.spacy_model = spacy.load(spacy_model, disable=['tagger', 'ner', 'lemmatizer', 'textcat'])
        self.exclude_terms = exclude_terms or ['Animals', 'Humans', 'Female', 'Male']
        self.threshold = threshold
        self.separator = '|'

    def get_mesh_terms(self, text, return_occurrences=True):
        sentences = []
        mesh_terms = []
        mesh_probs = defaultdict(float)

        # sentence splitting
        if isinstance(text, str):
            sentences = [sent.text for sent in self.spacy_model(text).sents]
        elif isinstance(text, list):
            sentences = text

        # process sentences
        for sentence in sentences:
            inputs = self.tokenizer([sentence], padding='max_length', return_tensors="pt", truncation=True)
            outputs = self.model(**inputs)
            probs = outputs.logits.softmax(dim=-1)

            for i, prob in enumerate(probs[0]):
                if prob > self.threshold:
                    term = self.model.config.id2label[i]
                    if term not in self.exclude_terms:
                        mesh_probs[term] = max(mesh_probs[term], prob.item())
                        mesh_terms.append(term)

        # aggregate results
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
        data['MESH_EXTRACTED'] = data[combined_columns].apply(
            lambda row: self.separator.join(set().union(*row)), axis=1
        )
        return data
