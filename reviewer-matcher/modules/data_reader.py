### data_reader.py

import os
import pandas as pd
from utils.functions_read_data import load_file, replace_values, replace_separators, clean_header_column, column_letter_to_index

class DataReader:
    def __init__(self, config_manager, settings_manager):
        # Configurations read from config file handled by config_manager.
        self.data_path = config_manager.get('DATA_PATH', '.')
        self.separator_output = config_manager.get('SEPARATOR_VALUES_OUTPUT', '|')
        self.id_column_name = config_manager.get('ID_COLUMN_NAME', 'ID')
        # Settings read from JSON files handled by settings_manager.
        self.input_file = settings_manager.get('input_file')
        self.file_type = settings_manager.get('file_type')
        self.column_mappings = settings_manager.get('column_mappings', {})
        self.multi_column_values = settings_manager.get('multi_column_values', {})
        self.value_mappings = settings_manager.get('values', {})
        self.separator_settings = settings_manager.get('separators', {})
        self.position_header_row = settings_manager.get('header_row', 0)
        self.position_first_data_row = settings_manager.get('first_data_row', 1)
        # Initializations.
        self.input_path = os.path.join(self.data_path, self.input_file)
        self.header = None
        self.data = None

    def load_data(self):
        '''Load and process projects/expert data with specific transformations.'''
        raw_data = load_file(self.input_path, self.file_type)
        self.read_data(raw_data)
        self.extract_multi_column_values(raw_data)
        self.apply_value_mappings()
        self.add_identifier_column()
        return self.data

    def _get_header_values(self, raw_data):
        '''Get a list of column headers or values from a specific row depending on position_first_data_row.'''
        if self.position_header_row == 0:
            # If first_data_row is 0, return the column names.
            return list(raw_data.columns)
        else:
            # Otherwise, get values from the specified row.
            return raw_data.iloc[self.position_header_row].tolist()

    def read_data(self, raw_data):
        '''Map columns to standardized names using the settings.'''
        self.header = self._get_header_values(raw_data)
        data_rows = raw_data.iloc[self.position_first_data_row:].values
        # Map project column names and values.
        column_indices = {key: column_letter_to_index(col_letter) for key, col_letter in self.column_mappings.items()}
        columns = [column_indices[key] for key in self.column_mappings.keys()]
        self.data = pd.DataFrame([[row[idx] for idx in columns] for row in data_rows], columns=self.column_mappings.keys())

    def extract_multi_column_values(self, raw_data):
        '''Combine column headers into a single field for specified column ranges where the cell value is not empty or 0.'''
        for key, value_map in self.multi_column_values.items():
            # Extract the range of columns to join based on the Excel-style notation
            first_column, second_column = value_map.split(':')
            first_column_index = column_letter_to_index(first_column)
            second_column_index = column_letter_to_index(second_column)
            # Select columns to join within the range in raw_data
            columns_to_join = raw_data.iloc[:, first_column_index:second_column_index + 1]
            column_headers = self.header[first_column_index:second_column_index + 1]
            # Concatenate column names for each row where the value is 1
            self.data[key] = columns_to_join.apply(
                lambda row: self.separator_output.join(
                    [clean_header_column(column_headers[idx])
                      for idx, value in enumerate(row) if str(value).strip() and str(value).strip() != '0']
                ),
                axis=1
            )

    def apply_value_mappings(self):
        '''Replace values in specified columns based on mappings in settings.'''
        # Replace values in relevant columns.
        for key, value_map in self.value_mappings.items():
            input_separator = self.separator_settings.get(key, ';') if key not in self.multi_column_values else self.separator_output
            self.data = replace_values(self.data, key, value_map, input_separator, self.separator_output)
        # Separator replacements for fields without value mappings.
        for key in set(self.separator_settings.keys()) - set(self.value_mappings.keys()):
            input_separator = self.separator_settings.get(key, ';') if key not in self.multi_column_values else self.separator_output
            self.data = replace_separators(self.data, key, input_separator, self.separator_output) 

    def add_identifier_column(self):
        '''Add a sequential ID column if it does not already exist.'''
        if self.id_column_name not in self.data.columns:
            self.data.insert(0, self.id_column_name, range(1, len(self.data) + 1))


            

