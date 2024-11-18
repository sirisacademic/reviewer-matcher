# functions_read_data.py

import os
import json
import re
import string
import pandas as pd

# Function to get settings corresponding to a call.
def get_settings_by_call(json_file_path):
#------------------------------------------------------------------------
  # Open and load the JSON file
  with open(json_file_path, 'r') as f:
      settings_data = json.load(f)
  return settings_data

# Load data from a specified file, supporting multiple formats
def load_file(file_path, file_type=None):
#---------------------------------
  """Load data from a specified file, supporting Excel, TSV, Pickle, and Parquet formats."""
  # Determine file type based on the extension if file_type is not provided
  file_type = file_type or os.path.splitext(file_path)[-1].lower().strip('.')
  if file_type in ['excel', 'xlsx', 'xls']:
    return pd.read_excel(file_path).fillna('')
  elif file_type == 'tsv':
    return pd.read_csv(file_path, sep='\t').fillna('')
  elif file_type in ['pkl', 'pickle']:
    return pd.read_pickle(file_path).fillna('')
  elif file_type == 'parquet':
    return pd.read_parquet(file_path).fillna('')
  else:
    raise ValueError(f'Unsupported file type: {file_type}')

"""
# Version for Google Sheets.
# Convert column letter to index using gspread's utility function.

def column_letter_to_index(letter):
#---------------------------------
  _, col = gspread.utils.a1_to_rowcol(f'{letter}1')
  return col - 1
"""

def column_letter_to_index(letter):
#---------------------------------
  index = 0
  for char in letter:
    index = index * 26 + (ord(char.upper()) - ord('A') + 1)
  return index - 1

# Function used to replace ampersands and other problematic characters in spreadsheet column names.
def clean_header_column(value):
#--------------------------------------
  value = value.replace('&', ' and ')
  return re.sub('\s+', ' ', value).strip(f'\'"{string.whitespace}')

# Function to replace values and apply separators to multi-value columns.
def replace_values(df, column, value_map, input_separator, output_separator):
#---------------------------------------------------------------------------
  if column in df.columns:
    df[column] = df[column].apply(lambda value:
      output_separator.join([value_map.get(item.strip(f'{input_separator}{string.whitespace}'), item)
        for item in value.strip(f'{input_separator}{string.whitespace}').split(input_separator)]))
  return df

# Function to only apply separator replacement for fields without value mapping.
def replace_separators(df, column, input_separator, output_separator):
#--------------------------------------------------------------------
  if column in df.columns:
    df[column] = df[column].apply(lambda value:
      output_separator.join([item.strip(f'{input_separator}{string.whitespace}')
        for item in value.strip(f'{input_separator}{string.whitespace}').split(input_separator)]))
  return df

# Function to flatten lists in dataset and join into a string for all columns
def flatten_lists(df, separator):
#--------------------
  for col in df.columns:
    df[col] = df[col].apply(lambda value: separator.join(map(str, value)) if isinstance(value, list) else value)
  return df
  
# Extract values from cells with a given separator and return them as a list.
def extract_values_column(data, column, separator):
#----------------------------------------
  if column in data:
    column_values = data[column].dropna().apply(lambda x: x.split(separator))
    return list(set(
        value.strip(f'\'"{string.whitespace}')
        for sublist in column_values
        for value in sublist
        if value.strip(f'\'"{string.whitespace}')
    ))
  else:
    return []
    
