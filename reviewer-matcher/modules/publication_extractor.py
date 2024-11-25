#from citation_parser import CitationParser
import pandas as pd
import re
import time
from tqdm import tqdm
import os

class PublicationExtractor:
    def __init__(self, config_manager, reviewer_form_path, column = 'State up to FIVE of your most relevant publications in the past five years related to the topic of the call (authors, title, journal, year):'):
        self.citation_parser = CitationParser()
        self.column = column
        self.data = pd.read_excel(reviewer_form_path)
        self.linked_output = None

    # Function to split references
    def split_references(self, input):
        input = input.strip()
        placeholder = "<<<SPLIT>>>"
        text = re.sub(r"\n\s*\n", placeholder, input)

        # Step 2: Split by placeholder first
        references = text.split(placeholder)

        # If splitting by placeholder doesn't yield multiple references, fallback to '\n'
        if len(references) == 1:
            references = text.split('\n')

        if len(references) > 1:
            references = [ref.strip().replace('\n','') for ref in references if ref.strip().replace('\n','')!='']

        # Step 3: Clean up individual references
        cleaned_references = [re.sub(r'^\d+\.\s*|^\d+\)\s*', '', ref).strip().replace('• ','') for ref in references if ref.strip()]
        
        return cleaned_references
    
    def reconstruct_abstract(self, abstractInvertedIndex):
        """Reconstruct an abstract from an abstract_inverted_index object from"""

        # check if NaN
        if pd.isna(abstractInvertedIndex):
            return ""

        word_index = []
        for k,v in abstractInvertedIndex.items():
            for index in v:
                word_index.append([k,index])
        word_index = sorted(word_index, key = lambda x : x[1])
        return ' '.join([w[0] for w in word_index])

    def extract_publication_titles_for_test(self):

        references_list = self.data[self.column].values.tolist()
        # Process each item in the list
        cleaned_references_list = [self.split_references(block) for block in references_list]

        # Print the output
        linked = []
        for idx, item in tqdm(enumerate(cleaned_references_list), total = len(cleaned_references_list), desc="Downloading references from OpenAlex"):
            for ref in item:
                try:
                    citation = self.citation_parser.link_citation(ref,results='advanced')

                    if citation['result']!=None:

                        title = citation['full-publication']['title']
                        abstract = self.reconstruct_abstract(citation['full-publication']['abstract_inverted_index'])
                        mesh = citation['full-publication']['mesh']
                        openalex_id = citation['full-publication']['id']
                        try: 
                            pubmed_id = citation['full-publication']['ids']['pmid'].replace('https://pubmed.ncbi.nlm.nih.gov/','')
                        except:
                            pubmed_id = None

                        linked.append((idx, ref, openalex_id ,pubmed_id, title, abstract, mesh))

                    else:
                        linked.append((idx, ref, None,None, None,None, None))
                except:
                    print(f'error with: {ref}')

        linked_output = pd.DataFrame(linked)
        linked_output.to_csv('local_data/linked_salut_mental.csv',sep=';')

    def extract_publications(self):

        references_list = self.data[self.column][:3].values.tolist()
        authors_list = self.data['Full Name:'].values.tolist()
        # Process each item in the list
        cleaned_references_list = [self.split_references(block) for block in references_list]

        # Print the output
        linked = []
        for idx, item in tqdm(enumerate(cleaned_references_list), total = len(cleaned_references_list), desc="Downloading references from OpenAlex"):
            for ref in item:
                try:
                    citation = self.citation_parser.link_citation(ref,results='advanced')

                    if citation['result']!=None:

                        title = citation['full-publication']['title']
                        abstract = self.reconstruct_abstract(citation['full-publication']['abstract_inverted_index'])
                        mesh = citation['full-publication']['mesh']
                        openalex_id = citation['full-publication']['id']

                        author = authors_list[idx]
                        try: 
                            pubmed_id = citation['full-publication']['ids']['pmid'].replace('https://pubmed.ncbi.nlm.nih.gov/','')
                        except:
                            pubmed_id = None

                        linked.append((idx, author, ref, openalex_id ,pubmed_id, title, abstract, mesh))

                except:
                    print(f'error with: {ref} | {citation}')
            os.system('cls' if os.name == 'nt' else 'clear')

        linked_output = pd.DataFrame(linked, columns = ['ID','FULL_NAME',"REFERENCE", 'OPENALEX_ID','PMID','TITLE_PUBMED','ABSTRACT_PUBMED','MESH_PUBMED'])
        self.linked_output = linked_output
        return linked_output
            
# publication_extractor = PublicationExtractor('config_manager','local_data/Reviewer Form.xlsx')
# publication_extractor.extract_publications()