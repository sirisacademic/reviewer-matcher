import os
import pandas as pd
from utils.functions_read_data import replace_values, replace_separators, clean_header_column

class DataReader:
    def __init__(self, data_path, config_manager, settings_manager):
        '''Initialize with a settings manager to access configuration settings.'''
        self.input_path = os.path.join(data_path, self.input_file)
        self.separator_output = separator_output
        self.data = None
        # Settings read from JSON file.
        self.input_file = settings_manager.get('input_file')
        self.file_type = settings_manager.get('file_type')
        self.column_mappings = settings_manager.get('column_mappings', {})
        self.multi_column_values = settings_manager.get('multi_column_values', {})
        self.value_mappings = settings_manager.get('values', {})
        self.separator_settings = settings_manager.get('separators', {})
        self.position_header_row = settings_manager.get('header_row', 0)
        self.position_first_data_row = settings_manager.get('first_data_row', 1)

    def load_and_process_data(self):
        '''Load and process projects/expert data with specific transformations.'''
        self.data = self.load_file()
        self.data = self.read_data(self.data)
        self.data = self.extract_multi_column_values(self.data)
        self.data = self.apply_value_mappings(self.data)
        self.data = self.add_identifier_column(self.data)
        return self.data

    def read_data(self, df):
        '''Map columns to standardized names using the settings.'''
        # Map project column names and values.
        column_indices = {key: column_letter_to_index(col_letter) for key, col_letter in self.column_mappings.items()}\
        # Read the data corresponding to the projects in the call.
        header_row = self.data[self.position_header_row]
        data_rows = self.data[self.position_first_data_row:]
        columns = [column_indices_projects[key] for key in self.column_mappings_projects.keys()]
        self.data = pd.DataFrame([[row[idx] for idx in columns] for row in data_rows], columns=self.column_mappings.keys())

    def extract_multi_column_values(self, df):
        '''Combine values from column names across specified column ranges into a single field (e.g. for expert topics/approaches).'''
        header_row = self.data[self.position_header_row]
        for key, value_map in self.multi_column_values.items():
          first_column, second_column = value_map.split(':')
          first_column_index = column_letter_to_index(first_column)
          second_column_index = column_letter_to_index(second_column)
          columns_to_join = range(first_column_index, second_column_index + 1)
          # Get topics/approaches for each expert.
          df[key] = [
              self.separator_output.join(
              [clean_header_column(header_row[idx]) for idx in columns_to_join if str(row[idx]).strip() and str(row[idx]).strip() != '0'])
              for row in data_rows
          ]
        return df

    def apply_value_mappings(self, df):
        '''Replace values in specified columns based on mappings in settings.'''
        # Replace values in relevant columns.
        for key, value_map in self.value_mappings.items():
            input_separator = self.separator_settings.get(key, ';')
            df = replace_values(df, key, value_map, input_separator, self.separator_output)
        # Separator replacements for fields without value mappings.
        for key in set(self.separator_settings.keys()) - set(self.value_mappings.keys()):
            df = replace_separators(df, key, self.separator_settings[key], self.separator_output) 
        return df

    def add_identifier_column(self, df):
        '''Add a sequential ID column if it does not already exist.'''
        if 'ID' not in df.columns:
            df.insert(0, 'ID', range(1, len(df) + 1))
        return df

    def load_file(self):
        '''Load data from a specified file, supporting Excel, TSV, Pickle, and Parquet formats.'''
        # Guess file type if not explicitly provided (in this case, self.file_type is None).
        if not self.file_type:
            extension = os.path.splitext(self.input_file)[-1].lower().strip('.')
            if extension in ['xlsx', 'xls']:
                file_type = 'excel'
            elif extension == 'tsv':
                file_type = 'tsv'
            elif extension == 'pkl':
                file_type = 'pickle'
            elif extension == 'parquet':
                file_type = 'parquet'
            else:
                raise ValueError(f'Unsupported file extension: {extension}')
        # Load data based on file type
        if file_type == 'excel':
            return pd.read_excel(input_path)
        elif file_type == 'tsv':
            return pd.read_csv(input_path, sep='\t')
        elif file_type == 'pickle':
            return pd.read_pickle(input_path)
        elif file_type == 'parquet':
            return pd.read_parquet(input_path)
        else:
            raise ValueError(f'Unsupported file type: {file_type}')
