# File: functions_llm.py

import sys
import json
import re
import time
import requests

from tenacity import Retrying, stop_after_attempt, wait_exponential, retry_if_exception_type, before_sleep
from requests.exceptions import RequestException

def expand_abbreviations(data, abbreviations, sections=[]):
#------------------------------------------
  if not sections:
    sections = data.keys()
  for section in sections:
    if section in data:
      if isinstance(data[section], str):
        data[section] = expand_text(data[section], abbreviations)
      elif isinstance(data[section], list):
        data[section] = [expand_text(item, abbreviations) if isinstance(item, str) else item for item in data[section]]
  return data

def expand_text(text, abbreviations):
#------------------------------------
  for abbr, definition in abbreviations.items():
    if definition not in text:
      text = re.sub(r'\b' + re.escape(abbr) + r'\b', definition, text)
  return text

# Function to make sure that a list is processed (as there could be differences when generating the data or reading it from a file).
def value_as_list(value, separator):
#-----------------------
  if isinstance(value, list):
    return value
  else:
    return value.split(separator)

# Generate the response schema for topics/approaches - for models supporting JSON responses.
def generate_schema_topics(topics):
#-----------------------
  # Define the base structure of the schema
  schema = {
    "title": "Topics",
    "type": "object",
    "properties": {},
    "required": topics
  }
  # Add each topic to properties with the required "response" field
  for topic in topics:
    schema["properties"][topic] = {
      "type": "object",
      "properties": {
        "response": {
          "type": "string",
          "enum": ["YES", "NO"],
          "description": "The response for each topic, either 'YES' or 'NO'."
        }
      },
      "required": ["response"]
    }
  return schema

# Generate the response schema for content extraction - for models supporting JSON responses.
def generate_research_summary_schema():
#-----------------------
  """
  Generates a JSON schema for structuring the summary of a research proposal with research topic, objectives, and methods.

  Returns:
  dict: JSON schema defining the required structure for the research summary response.
  """
  schema = {
    "title": "ResearchSummary",
    "type": "object",
    "properties": {
      "research_topic": {
        "type": "string",
        "description": "A concise summary of the research topic."
      },
      "objectives": {
        "type": "array",
        "items": {
          "type": "string",
          "description": "A short, independent sentence summarizing a main research objective."
        },
        "description": "List of key objectives, with one to three sentences summarizing each main objective."
      },
      "methods": {
        "type": "array",
        "items": {
          "type": "string",
          "description": "A concise sentence summarizing a main method, technique, or approach used in the study."
        },
        "description": "List of main methods or approaches, with two to three sentences summarizing each."
      }
    },
    "required": ["research_topic", "objectives", "methods"],
    "description": "Structured summary of a research proposal including research topic, objectives, and methods."
  }
  return schema

# Generate the response schema for methods - for models supporting JSON responses.
def generate_method_classification_schema():
#-----------------------
  """
  Generates a JSON schema for classifying methods in biomedical studies as standard or specific.

  Returns:
  dict: JSON schema defining the required structure for the method classification response.
  """
  schema = {
    "title": "MethodClassification",
    "type": "object",
    "properties": {
      "methods_standard": {
        "type": "array",
        "items": {
          "type": "string",
          "description": "Standard research methods, such as common research designs or statistical analyses."
        },
        "description": "List of general research methods, may be empty if none are classified as standard."
      },
      "methods_specific": {
        "type": "array",
        "items": {
          "type": "string",
          "description": "Specific methods, techniques, tools, or analyses relevant to the study's focus."
        },
        "description": "List of study-specific methods, techniques, tools, or analyses."
      }
    },
    "required": ["methods_standard", "methods_specific"],
    "description": "Classification of methods in biomedical studies as either standard or specific."
  }
  return schema

# Generate the default structure for dynamic expected responses by the generative model
def generate_default_structure_labels(topics):
#---------------------------------------------
  """
  Generates the default expected structure for the topics/approaches prompts.
  Parameters:
  topics (list): A list of topics.
  Returns:
  dict: Expected structure for the JSON output.
  """
  expected_structure = {}
  for topic in topics:
      expected_structure[topic] = {
          'response': 'NO',
          'justification': ''
      }
  return expected_structure

  
# Call LLM and extract the responses.
def get_model_response(pipe, generation_args, prompt, max_retries=3, retry_delay=2):
#-----------------------------------------------------------
  """
  Calls the model based on the chosen type (local or external) and returns the result. Retries on failure.
  For external models pipe is None.
  """
  if pipe:
      return get_local_model_response(pipe, generation_args, prompt, max_retries, retry_delay)
  else:
      return get_external_model_response(generation_args, prompt, max_retries, retry_delay)


def get_local_model_response(pipe, generation_args, prompt, max_retries=3, retry_delay=2):
#-----------------------------------------------------------
  """
  Calls the model with the prompt and returns the result. Retries on failure.

  Parameters:
  prompt (str): The input prompt for the model.
  max_retries (int): Maximum number of retries in case of failure.
  retry_delay (int): Time in seconds to wait before retrying.

  Returns:
  str: The generated text from the model.
  None: If the request fails after retries.
  """
  for attempt in range(max_retries):
    try:
      messages = [
        {'role': 'system', 'content': 'You are a helpful AI assistant.'},
        {'role': 'user', 'content': prompt},
      ]
      output = pipe(messages, **generation_args)
      return output[0]['generated_text']
    except Exception as e:
      print(f'Error calling model: {e}. Retrying (retry attempt {attempt + 1}/{max_retries})', file=sys.stderr)
      time.sleep(retry_delay)
  print('Failed to get a response from the model after multiple attempts.')
  return None

def make_external_request_with_retry(external_model_url, headers, filtered_args, max_retries=5, retry_delay=2, initial_timeout=30):
    def before_sleep_print(retry_state):
        print(f"Retrying in {retry_state.next_action.sleep} seconds...", file=sys.stderr)
    retry_strategy = Retrying(
        stop=stop_after_attempt(max_retries),
        wait=wait_exponential(multiplier=retry_delay, min=retry_delay, max=retry_delay * 4),
        retry=retry_if_exception_type((RequestException, ValueError)),
        before_sleep=before_sleep_print
    )
    for attempt in retry_strategy:
        # Calculate the current timeout value with exponential backoff for each retry
        current_timeout = initial_timeout * (2 ** attempt.retry_state.attempt_number)
        #print(f"Attempt {attempt.retry_state.attempt_number+1}/{max_retries}: Trying with timeout {current_timeout} seconds...", file=sys.stderr)
        #print(f"Request Data: {filtered_args}")
        try:
            # Make the request
            response = requests.post(external_model_url, headers=headers, json=filtered_args, timeout=current_timeout)
            response.raise_for_status()  # Raise an HTTPError for bad responses
            result = response.json()
            # Check if 'choices' is in the response
            if 'choices' in result and isinstance(result['choices'], list):
                return response
            else:
                raise ValueError("Unexpected response format: missing 'choices'.")
        except RequestException as e:
            # Print retry attempt only if there is an error
            print(f"Attempt {attempt.retry_state.attempt_number+1}/{max_retries}: Failed with error: {e}. Retrying with timeout {current_timeout} seconds...", file=sys.stderr)
        except ValueError as e:
            print(f"Attempt {attempt.retry_state.attempt_number+1}/{max_retries}: Failed with error: {e}. Retrying with timeout {current_timeout} seconds...", file=sys.stderr)
    # If the function exits the loop without a successful response
    print("Failed to get a response from the external model after multiple attempts.")
    print("Failed to get a response from the external model after multiple attempts.", file=sys.stderr)
    # Fallback mechanism: Log the problematic data for manual review
    print(f"Problematic data: {filtered_args}", file=sys.stderr)
    return None

# Get response using an externally-hosted LLM.
#----------------------------------------------------------
def get_external_model_response(generation_args, prompt, max_retries=5, retry_delay=2):
  external_model_url = generation_args['external_model_url']
  api_key = generation_args['api_key']
  headers = {
    'Authorization': f'Bearer {api_key}',
    'Content-Type': 'application/json'
  }
  filtered_args = {
    'prompt': prompt,
    'model': generation_args.get('model'),
    'max_tokens': generation_args.get('max_tokens'),
    'temperature': generation_args.get('temperature', 0.0),
    'top_p': generation_args.get('top_p', 1.0),
    'echo': generation_args.get('echo', False)
  }  
  try:
    response = make_external_request_with_retry(external_model_url, headers, filtered_args, max_retries, retry_delay)
    result = response.json()
    # Extract the generated text from the response.
    if 'choices' in result and isinstance(result['choices'], list):
      choice = result['choices'][0]
      if 'text' in choice:
        return choice['text'].strip()
      elif 'message' in choice and 'content' in choice['message']:
        return choice['message']['content'].strip()
      else:
        raise ValueError("Unexpected response structure in 'choices'.")
    else:
      raise ValueError("Unexpected response format: missing 'choices'.")
  except RequestException as e:
    print(f"Failed to get a response after multiple attempts: {e}")
  return None
    
def extract_and_parse_json(input_string, default_response):
#---------------------------------------------------
  """
  Extracts JSON content directly if available, or extracts JSON from a string with code block marks (```json ... ```),
  and parses it into a Python dictionary.

  Parameters:
  input_string (str): The input string possibly containing JSON content or JSON content between code block marks.

  Returns:
  dict: Parsed JSON data as a Python dictionary with appropriate structure.
  """
  # First attempt to decode JSON directly
  try:
    parsed_data = json.loads(input_string)
    # Validate parsed data against expected structure
    validated_data = validate_response(parsed_data, default_response)
    return validated_data
  except json.JSONDecodeError:
    # If direct decoding fails, fall back to parsing within code block markers
    match = re.search(r'```json(.*?)```', input_string, re.DOTALL)
    if match:
      json_content = match.group(1).strip()
      try:
        # Remove trailing commas that might cause JSON errors.
        json_content = re.sub(r',\s*([\]}])', r'\1', json_content)
        parsed_data = json.loads(json_content)
        # Validate parsed data against expected structure
        validated_data = validate_response(parsed_data, default_response)
        return validated_data
      except json.JSONDecodeError as e:
        print(f'Error when decoding string: {input_string}', file=sys.stderr)
        print(f'Error parsing JSON: {e}', file=sys.stderr)
    else:
      print(f'Error when decoding string: {input_string}', file=sys.stderr)
      print('No JSON code block found.', file=sys.stderr)
  # Return the default expected structure with empty values if parsing fails
  return None

# Validate expected response.
def validate_response(parsed_data, default_response):
#---------------------------------------------------
  """
  Validates that the parsed data contains all keys of the expected structure and fills in missing keys.

  Parameters:
  parsed_data (dict): The parsed JSON data.
  default_response (dict): The structure to validate against.

  Returns:
  dict: Parsed data with missing keys filled with default values from the expected structure.
  """
  # Iterate over the keys in the expected structure.
  for key, value in default_response.items():
    # If key is missing or the value type doesn't match, use the expected structure's default value.
    if key not in parsed_data or not isinstance(parsed_data[key], type(value)):
      parsed_data[key] = value
    elif isinstance(value, dict):  # Recursively validate nested dictionaries.
      parsed_data[key] = validate_response(parsed_data[key], value)
  return parsed_data

# We now process topics in smaller chunks because otherwise the model makes mistakes when generating the output.
# This function is used both to label projects with research areas (topics) and research approaches - using different prompts in each case.
def label_by_topic(pipe, generation_args, prompt, title, abstract, topics, max_topics=5, max_retries=5, retry_delay=2, json_response=False):
#-------------------------------------------------------------------------------------------------------------------------------------
  """
  Gets labeled topics by calling the model and parsing the result. Retries on failure.
  Automatically chunks topics into smaller lists if necessary.

  Parameters:
  title (str): The title of the project.
  abstract (str): The abstract of the project.
  topics (list): A list of topics to process.
  max_retries (int): Maximum number of retries in case of failure.
  retry_delay (int): Time in seconds to wait before retrying.

  Returns:
  dict: Combined results of labeled topics across all chunks.
  """
  def chunk_list(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
      yield lst[i:i + n]
  topics_chunks = list(chunk_list(topics, max_topics))
  combined_results = {}
  for topics_chunk in topics_chunks:
    if json_response:
      response_format = {
        'schema': generate_schema_topics(topics),
        'type': 'json_object'
      }
      generation_args['parameters'] = {'json_mode': True}
      generation_args['response_format'] = response_format
    formatted_topics = '\n'.join([f'- {topic.strip()}' for topic in topics_chunk])
    prompt_text = prompt.format(title=title, abstract=abstract, topics=formatted_topics)
    default_structure_labels = generate_default_structure_labels(topics_chunk)
    for attempt in range(max_retries):
      if attempt > 0:
        print(f'Failed to parse response. Retrying... (retry {attempt + 1}/{max_retries})', file=sys.stderr)
      model_response = get_model_response(pipe, generation_args, prompt_text, max_retries, retry_delay)
      #print(model_response)
      parsed_data = extract_and_parse_json(model_response, default_structure_labels)
      if parsed_data is None and attempt == max_retries-1:
        parsed_data = default_structure_labels
        print(f'Returning the default values as the data could not be obtained after {max_retries} attempts for prompt:')
        print(prompt)
      else:
        if parsed_data is not None:
            for key, value in parsed_data.items():
              if key in combined_results:
                combined_results[key].extend(value) if isinstance(value, list) else value
              else:
                combined_results[key] = value
            break
      time.sleep(retry_delay)
  #print(combined_results)
  return combined_results

# Combine responses for research areas and approaches to assign to a single column in projects.
def extract_and_combine_responses(row, columns, separator):
#----------------------------------------------
  values = []
  for column in columns:
    cell = row.get(column, {})
    if isinstance(cell, dict):
      response = cell.get('response', '')
      if response.startswith('YES'):
        values.append(column)
  return separator.join(values)

def extract_content(pipe, generation_args, prompt, default_response, max_retries=5, retry_delay=2):
#-----------------------------------------------------------------------------------------------------------------------
  """
  Extracts and summarizes relevant content from abstracts. Retries on failure.

  Parameters:
  pipe: The model pipeline.
  generation_args (dict): Arguments for text generation.
  prompt (str): The prompt to provide to the model.
  default_response (dict): The default structure to use when parsing fails.

  Returns:
  dict: Parsed JSON data as a Python dictionary with appropriate structure.
  """
  #print(prompt)
  for attempt in range(max_retries):
    model_response = get_model_response(pipe, generation_args, prompt, max_retries, retry_delay)
    #print('------------------- RESPONSE -------------------')
    #print(model_response)
    parsed_data = extract_and_parse_json(model_response, default_response)
    if parsed_data is not None:
      return parsed_data
    print(f'Failed to parse response. Retrying... (retry {attempt + 1}/{max_retries})', file=sys.stderr)
    time.sleep(retry_delay)
  # Return expected structure if all retries fail
  print('Failed to get valid summarized content after multiple attempts for prompt:')
  print(prompt)
  return default_response
  

