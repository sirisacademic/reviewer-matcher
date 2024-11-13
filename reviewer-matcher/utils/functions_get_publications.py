# functions_get_publications.py

import requests                    # For making HTTP requests to PubMed
from bs4 import BeautifulSoup      # For parsing HTML content from PubMed
import re                          # For regular expressions used in content parsing

# Return string with list of expert publication titles extracted from the raw list provided in the form.
def extract_publication_titles(pipe, raw_text_publications, separator):
#----------------------------------------------------------
  publications = []
  ner_output = pipe(raw_text_publications)
  for entity in ner_output:
    # Not considering score now.
    if entity['entity_group'] == 'TITLE' and entity['word']:
      publications.append(entity['word'])
  return separator.join(publications)
  
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
  

