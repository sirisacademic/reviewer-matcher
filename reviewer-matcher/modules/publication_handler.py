import pandas as pd
import os
from tqdm import tqdm
from utils.functions_publications import split_raw_references

# Needed for OpenAlex.
from utils.citation_parser import CitationParser
from utils.functions_publications import reconstruct_openalex_abstract, format_mesh_terms_pubmed

# Needed for PubMed.
import Levenshtein
from utils.functions_publications import get_pubmed_content

class PublicationHandler:
    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.verbose = config_manager.get('GET_PUBLICATIONS_VERBOSE', True)
        self.default_source = config_manager.get('PUBLICATIONS_SOURCE', 'openalex')
        self._initialize_source = {
            'openalex': self._initialize_openalex,
            'pubmed': self._initialize_pubmed
        }

    def _initialize_openalex(self):
        self.citation_parser = CitationParser()
        print('Initialized OpenAlex')

    def _initialize_pubmed(self):
        print('Initialized PubMed')

    def get_publications_experts(self, experts, source=''):
        if source not in self._initialize_source:
            source = self.default_source
        self._initialize_source[source]()
        return self._get_publications_openalex(experts) if source=='openalex' else self._get_publications_pubmed(experts)

    def _get_input_data(self, experts):
        # Input column names
        col_expert_raw_publications = self.config_manager.get('COLUMN_EXPERT_PUBLICATIONS', 'RAW_PUBLICATIONS')
        col_expert_id = self.config_manager.get('COLUMN_EXPERT_ID', 'ID')
        col_expert_full_name = self.config_manager.get('COLUMN_EXPERT_FULL_NAME', 'FULL_NAME')
        # Output column names
        col_pub_publication_id = self.config_manager.get('COLUMN_PUB_PUBLICATION_ID', 'PUB_ID')
        col_pub_expert_id = self.config_manager.get('COLUMN_PUB_EXPERT_ID', 'EXPERT_ID')
        col_pub_reference = self.config_manager.get('COLUMN_PUB_REFERENCE', 'REFERENCE')
        col_pub_openalex_id = self.config_manager.get('COLUMN_PUB_OPENALEX_ID', 'OPENALEX_ID')
        col_pub_pmid = self.config_manager.get('COLUMN_PUB_PMID', 'PMID')
        col_pub_title = self.config_manager.get('COLUMN_PUB_TITLE', 'TITLE')
        col_pub_abstract = self.config_manager.get('COLUMN_PUB_ABSTRACT', 'ABSTRACT')
        col_pub_mesh_openalex = self.config_manager.get('COLUMN_PUB_MESH_OPENALEX', 'MESH_OPENALEX')
        col_pub_mesh_pubmed = self.config_manager.get('COLUMN_PUB_MESH_PUBMED', 'MESH_PUBMED')
        # List of output columns.
        output_columns = [
            col_pub_publication_id,
            col_pub_expert_id,
            col_expert_full_name,
            col_pub_reference,
            col_pub_openalex_id,
            col_pub_pmid,
            col_pub_title,
            col_pub_abstract,
            col_pub_mesh_openalex,
            col_pub_mesh_pubmed
        ]
        # Get input data from experts' dataframe.
        references_list = experts[col_expert_raw_publications].values.tolist()
        expert_full_names = experts[col_expert_full_name].values.tolist()
        expert_ids = experts[col_expert_id].values.tolist()
        return references_list, expert_full_names, expert_ids, output_columns

    def _process_references(self, references_list):
        # Process each item in the list
        cleaned_references_list = [split_raw_references(raw_references) for raw_references in references_list]
        return cleaned_references_list

    def _get_publications_openalex(self, experts):
        # Retrieve settings.
        pubmed_base_url = self.config_manager.get('PUBMED_BASE_URL')
        openalex_base_url = self.config_manager.get('OPENALEX_BASE_URL')
        # Get clean references from raw data.
        references_list, expert_full_names, expert_ids, output_columns = self._get_input_data(experts)
        cleaned_references_list = self._process_references(references_list)
        # Process references for each expert.
        list_publications = []
        # Separator (used for MeSH terms in PubMed format)
        separator = self.config_manager.get('SEPARATOR_VALUES_OUTPUT', '|')
        for expert_data in tqdm(
            zip(expert_ids, expert_full_names, cleaned_references_list),
            total=len(cleaned_references_list),
            desc='Retrieving references from OpenAlex'
        ):
            expert_id, expert_full_name, references = expert_data
            openalex_ids_author = []
            for reference in references:
                try:
                    citation = self.citation_parser.link_citation(reference, results='advanced')
                    if citation['result'] is not None:
                        title = citation['full-publication']['title']
                        abstract = reconstruct_openalex_abstract(citation['full-publication']['abstract_inverted_index'])
                        mesh_openalex = citation['full-publication']['mesh']
                        mesh_pubmed = format_mesh_terms_pubmed(mesh_openalex, separator)
                        openalex_id = citation['full-publication']['id'].replace(openalex_base_url, '')
                        try: 
                            pubmed_id = citation['full-publication']['ids']['pmid'].replace(pubmed_base_url, '')
                        except:
                            pubmed_id = ''
                        if openalex_id not in openalex_ids_author:
                            openalex_ids_author.append(openalex_id)
                            list_publications.append((
                                f'oa_{openalex_id}',
                                expert_id,
                                expert_full_name,
                                reference,
                                openalex_id,
                                pubmed_id,
                                title,
                                abstract,
                                mesh_openalex,
                                mesh_pubmed)
                              )
                except Exception as e:
                    print(f'Error with expert {expert_id}, reference: {reference} | {e}')
            if not self.verbose:
                os.system('cls' if os.name == 'nt' else 'clear')
        publications = pd.DataFrame(list_publications, columns=output_columns)
        return publications

    def _get_publications_pubmed(self, experts):
        # Retrieve settings.
        pubmed_search_url = self.config_manager.get('PUBMED_SEARCH_URL')
        pubmed_pmid_url = self.config_manager.get('PUBMED_PMID_URL')
        separator = self.config_manager.get('SEPARATOR_VALUES_OUTPUT')
        min_length_title = self.config_manager.get('PUBMED_MIN_LENGTH_TITLE', 5)
        threshold_diff_titles = self.config_manager.get('PUBMED_THRESHOLD_DIFF_TITLES', 10)
        # Get clean references from raw data.
        references_list, expert_full_names, expert_ids, output_columns = self._get_input_data(experts)
        cleaned_references_list = self._process_references(references_list)
        # Process references for each expert.
        list_publications = []
        for expert_data in tqdm(
            zip(expert_ids, expert_full_names, cleaned_references_list),
            total=len(cleaned_references_list),
            desc='Retrieving references from PubMed'
        ):
            expert_id, expert_full_name, references = expert_data
            pmids_author = []
            for reference in references:
                title = reference.strip()
                openalex_id = ''
                mesh_openalex = []
                if len(title) >= min_length_title and title[0].isupper():
                    pubmed_data = get_pubmed_content(expert_full_name, title, pubmed_search_url, pubmed_pmid_url, separator)
                    if pubmed_data:
                        distance_titles = Levenshtein.distance(title.lower(), pubmed_data['title'].lower())
                        pmid = pubmed_data['pmid']
                        if distance_titles < threshold_diff_titles and pmid not in pmids_author:
                            pmids_author.append(pmid)
                            list_publications.append((
                                f'pmid_{pmid}',
                                expert_id,
                                expert_full_name,
                                reference,
                                openalex_id,
                                pmid,
                                pubmed_data['title'],
                                pubmed_data['abstract'],
                                mesh_openalex,
                                pubmed_data['mesh'])
                              )
            if not self.verbose:
                os.system('cls' if os.name == 'nt' else 'clear')
        publications = pd.DataFrame(list_publications, columns=output_columns)
        return publications



