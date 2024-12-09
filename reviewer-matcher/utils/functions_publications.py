# functions_get_publications.py

import requests                    # For making HTTP requests to PubMed
from bs4 import BeautifulSoup      # For parsing HTML content from PubMed
import re                          # For regular expressions used in content parsing
import pandas as pd

"""
# NOT using this now.
#
# Return string with list of expert publication titles previously extracted from the raw list provided in the form.
def extract_publication_titles(pipe, raw_text_publications, separator):
#----------------------------------------------------------
  publications = []
  ner_output = pipe(raw_text_publications)
  for entity in ner_output:
    # Not considering score now.
    if entity['entity_group'] == 'TITLE' and entity['word']:
      publications.append(entity['word'])
  return separator.join(publications)
"""

# Function to split references - alternative to extract_publication_titles
def split_raw_references(raw_references, threshold=100):
#------------------------------------------------------
    """ Splits and cleans raw references from a string. """
    # Normalize newlines
    raw_references = raw_references.strip().replace('\r\n', '\n').replace('\r', '\n')
    # Compile regex patterns
    pattern_list = re.compile(r'^\s*\d+[\.\):]\s*|^\s*\(\d+\)\s*|^\s*[•\-\*]\s*')
    pattern_year_doi = re.compile(r'(\d{4})|doi:')
    # Check if references are indicated as a bullet or numbered list
    is_bullet_list = bool(re.match(pattern_list, raw_references))
    # Replace multiple newlines with a placeholder
    placeholder = "<<<SPLIT>>>"
    text = re.sub(r'\n\s*\n', placeholder, raw_references)
    # Split by placeholder first to isolate references
    references = text.split(placeholder)
    if len(references) == 1:
        references = text.split('\n')
    # Merge parts of references that might have been split by internal newlines
    merged_references = []
    current_reference = ''
    for ref in references:
        ref = ref.strip()
        if (is_bullet_list and not re.match(pattern_list, ref)) or (not re.search(pattern_year_doi, ref) and len(ref.split()) <= 5):
            # Continue the previous reference if it doesn't appear to be a new reference
            current_reference += ' ' + ref
        else:
            # We have found a valid reference or the next part of a reference (e.g., a DOI or year)
            if current_reference:
                merged_references.append(current_reference.strip())
            current_reference = ref
    # Add the last reference
    if current_reference:
        merged_references.append(current_reference.strip())
    # Clean up individual references
    cleaned_references = [
        re.sub(pattern_list, '', ref).strip()
        for ref in merged_references if ref.strip()
    ]
    # Additional pass to handle long references containing \n.
    final_references = []
    for ref in cleaned_references:
        # Split by newlines
        parts = [part.strip() for part in ref.split('\n') if part.strip()]
        if all(len(part) >= threshold and re.search(pattern_year_doi, part.strip()) for part in parts):
            final_references.extend(parts)
        else:
            final_references.append('. '.join(parts))
    return final_references

def reconstruct_openalex_abstract(inverted_index):
#--------------------------------------------------------------
  """Reconstruct an abstract from an abstract inverted index."""
  # check if NaN
  if not inverted_index or pd.isna(inverted_index):
    return ''
  word_index = []
  for k,v in inverted_index.items():
    for index in v:
      word_index.append([k,index])
  word_index = sorted(word_index, key = lambda x : x[1])
  return ' '.join([w[0] for w in word_index])

# Function to format MeSH terms returned by OpenAlex in the same format as returned by PubMed.
def format_mesh_terms_pubmed(mesh_list, separator):
#----------------------------------
    formatted_terms = []
    descriptor_qualifiers = {}
    descriptors_major = {}
    if mesh_list:
        for term in mesh_list:
            descriptor = term['descriptor_name']
            qualifier = term['qualifier_name']
            is_major = term['is_major_topic']
            if is_major:
                descriptors_major[descriptor] = True
            elif descriptor not in descriptors_major or not descriptors_major[descriptor]:
                descriptors_major[descriptor] = False
            if qualifier:
                if is_major:
                    qualifier = f'*{qualifier}'
                if descriptor not in descriptor_qualifiers:
                    descriptor_qualifiers[descriptor] = []
                descriptor_qualifiers[descriptor].append(qualifier)
        # Add desciptors to output.
        for descriptor in descriptors_major:
            if descriptor in descriptor_qualifiers:
                qualifiers_str = '/'.join(descriptor_qualifiers[descriptor])
                descriptor_str = f'{descriptor}/{qualifiers_str}'
            elif descriptors_major[descriptor]:
                descriptor_str = f'*{descriptor}'
            else:
                descriptor_str = descriptor
            formatted_terms.append(descriptor_str)
    return separator.join(formatted_terms)

def get_pubmed_content(author, title, pubmed_search_url, pubmed_pmid_url, separator):
#----------------------------------
  parsed_content = None
  pubmed_url = get_pubmed_url(f'{author}. {title}', pubmed_search_url, pubmed_pmid_url)
  if pubmed_url:
    response = requests.get(f'{pubmed_url}?format=pubmed')
    if response.status_code == 200:
      parsed_content = parse_pubmed_content(response.text, separator)
  return parsed_content

def get_pubmed_url(search_text, pubmed_search_url, pubmed_pmid_url):
#------------------------------
  pubmed_url = None
  formatted_text = search_text.replace(' ', '+')
  url = pubmed_search_url.format(query=formatted_text)
  response = requests.get(url)
  # Check if the final URL is different, meaning a redirection happened.
  if response.history:
    pubmed_url = response.url
  else:
    # See whether more than one results were retrieved.
    # In this case now we are getting the first one.
    soup = BeautifulSoup(response.text, 'html.parser')
    # Extract the <meta> tags with "log_resultcount" and "log_displayeduids"
    displayed_uids_tag = soup.find('meta', {'name': 'log_displayeduids'})
    if displayed_uids_tag:
      pmids = [pmid.strip() for pmid in displayed_uids_tag.get('content').split(',')]
      pubmed_url =  pubmed_pmid_url.format(pmid=pmids[0])
  return pubmed_url

def parse_pubmed_content(pubmed_html, separator):
#--------------------------------
  # Dictionary to store the parsed data
  data = {}
  soup = BeautifulSoup(pubmed_html, 'html.parser')
  pre_tag = soup.find('pre', class_='article-details', id='article-details')
  if pre_tag:
    # Remove newlines and spaces that break up the fields.
    content = pre_tag.get_text().replace('\r', '')
    content = content.replace('\n      ', ' ')
    # Extract PMID
    pmid_match = re.search(r'^PMID\s*-\s+(.*)$', content, re.MULTILINE)
    data['pmid'] = pmid_match.group(1) if pmid_match else ''
    # Extract Title (TI)
    ti_match = re.search(r'^TI\s*-\s+(.*)$', content, re.MULTILINE)
    data['title'] = ti_match.group(1) if ti_match else ''
    # Extract Abstract (AB)
    ab_match = re.search(r'^AB\s*-\s+(.*)$', content, re.MULTILINE)
    data['abstract'] = ab_match.group(1) if ab_match else ''
    # Extract full name of authors (FAU - can have multiple entries)
    fau_matches = re.findall(r'^FAU\s*-\s+(.*)$', content, re.MULTILINE)
    data['authors'] = separator.join(fau_matches) if fau_matches else ''
    # Extract MeSH terms (MH - can have multiple entries)
    mh_matches = re.findall(r'^MH\s*-\s+(.*)$', content, re.MULTILINE)
    data['mesh'] = separator.join(mh_matches) if mh_matches else ''
  return data
  

