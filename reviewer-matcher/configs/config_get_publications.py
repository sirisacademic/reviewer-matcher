# Default source from where to retrieve publications ('openalex' or 'pubmed').
PUBLICATIONS_SOURCE = 'openalex'

# OpenAlex base url.
OPENALEX_BASE_URL = 'https://openalex.org/'

# PubMed URLs to retrieve expert publications' content.
PUBMED_BASE_URL = 'https://pubmed.ncbi.nlm.nih.gov/'
PUBMED_SEARCH_URL = f'{PUBMED_BASE_URL}?term={{query}}'
PUBMED_PMID_URL = f'{PUBMED_BASE_URL}{{pmid}}/'

# Settings to consider/discard expert publications retrieved from PubMed.
PUBMED_MIN_LENGTH_TITLE = 30
PUBMED_THRESHOLD_DIFF_TITLES = 50

# Model/parameters for NER model used to extract expert publication titles.
MODEL_NER_PUB_TITLES = 'SIRIS-Lab/patstat-citation-parser-ENTITY'

# Input columns.
COLUMN_EXPERT_PUBLICATIONS = 'RAW_PUBLICATIONS'
COLUMN_EXPERT_FULL_NAME = 'FULL_NAME' 
COLUMN_EXPERT_ID = 'ID' 

# Output columns.
COLUMN_PUB_PUBLICATION_ID = 'PUB_ID'
COLUMN_PUB_EXPERT_ID = 'EXPERT_ID'
COLUMN_PUB_REFERENCE = 'REFERENCE'
COLUMN_PUB_OPENALEX_ID = 'OPENALEX_ID'
COLUMN_PUB_PMID = 'PMID'
COLUMN_PUB_TITLE = 'TITLE'
COLUMN_PUB_ABSTRACT = 'ABSTRACT'
COLUMN_PUB_MESH_OPENALEX = 'MESH_OPENALEX'
COLUMN_PUB_MESH_PUBMED = 'MESH_PUBMED'

# Abstracts longer than this will be truncated.
PUB_ABSTRACT_MAX_LENGTH = 5000

# Verbose mode.
GET_PUBLICATIONS_VERBOSE = True


