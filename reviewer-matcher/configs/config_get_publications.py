# PubMed URLs to retrieve expert publications' content.
PUBMED_SEARCH_URL = 'https://pubmed.ncbi.nlm.nih.gov/?term={query}'
PUBMED_PMID_URL = 'https://pubmed.ncbi.nlm.nih.gov/{pmid}/'

# Settings to consider/discard expert publications retrieved from PubMed.
MIN_LENGTH_TITLE = 30
THRESHOLD_DIFF_TITLES = 50

# Model/parameters for NER model used to extract expert publication titles.
MODEL_NER_PUB_TITLES = 'SIRIS-Lab/patstat-citation-parser-ENTITY'

COLUMN_PUBLICATIONS = 'State up to FIVE of your most relevant publications in the past five years related to the topic of the call (authors, title, journal, year):'

COLUMN_REVIEWER_NAME = 'Full Name:' 


