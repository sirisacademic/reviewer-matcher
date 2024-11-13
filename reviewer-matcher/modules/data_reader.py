import os
import pandas as pd
from utils.functions_read_data import replace_values, parse_multi_value_column

class DataReader:
    def __init__(self, data_path, settings_manager):
        '''Initialize with a settings manager to access configuration settings.'''
        self.settings = settings_manager
        self.input_file = self.settings.get('input_file')
        self.file_type = self.settings.get('file_type')
        self.input_path = os.path.join(data_path, self.input_file)
        self.data = None

    def load_data(self):
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

    def map_columns(self, df):
        '''Map columns to standardized names using the settings.'''
        column_mappings = self.settings.get('column_mappings', {})
        return df.rename(columns=column_mappings)

    def apply_value_mappings(self, df):
        '''Replace values in specified columns based on mappings in settings.'''
        value_mappings = self.settings.get('values', {})
        separator_values_output = '|'
        for column, mapping in value_mappings.items():
            # Use a custom separator if specified in the settings
            input_separator = self.settings.get('separators', {}).get(column, ';')
            df = replace_values(df, column, mapping, input_separator, separator_values_output)
        return df

    def apply_separators(self, df):
        '''Apply custom separators in multi-value fields based on settings.'''
        separator_settings = self.settings.get('separators', {})
        for column, separator in separator_settings.items():
            df = parse_multi_value_column(df, column, separator)
        return df

    def add_identifier_column(self, df):
        '''Add a sequential ID column if it does not already exist.'''
        if 'ID' not in df.columns:
            df.insert(0, 'ID', range(1, len(df) + 1))
        return df

