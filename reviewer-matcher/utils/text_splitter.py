import spacy

class TextSplitter:
    def __init__(self, model_name='en_core_web_sm'):
        # Load the SpaCy model
        self.nlp = spacy.load(model_name, disable=['tagger', 'ner', 'lemmatizer', 'textcat'])

    def split_sentences(self, text):
        # Process the text using the SpaCy model
        doc = self.nlp(text)
        # Extract sentences from the processed text
        sentences = [sent.text for sent in doc.sents]
        return sentences
        
        
