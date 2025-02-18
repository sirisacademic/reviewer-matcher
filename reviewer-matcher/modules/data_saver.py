### data_saver.py

import os
import pandas as pd
import pickle
from datetime import datetime

class DataSaver:
    def __init__(self, config_manager):
        # Configurations read from config file handled by config_manager.
        self.default_output_dir = config_manager.get('DATA_PATH')
        self.separator = config_manager.get('SEPARATOR_VALUES_OUTPUT')
        self.test_mode = config_manager.get('TEST_MODE', False)
        
    def save_data(self, df, file_name, output_dir=None, file_type=None, add_timestamp=False, verbose=True):
        '''Save a DataFrame to a specified file, supporting Excel, TSV, Pickle, and Parquet formats.'''
        # Get file extension and determine the file path
        extension = os.path.splitext(file_name)[-1].lower().strip('.')
        output_dir = self.default_output_dir if output_dir is None else output_dir
        file_path = self._get_filepath(output_dir, file_name, extension, add_timestamp)
        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        # Determine file type from extension if not provided
        file_type = file_type or {
            'xlsx': 'excel', 'xls': 'excel',
            'tsv': 'tsv', 'pkl': 'pickle',
            'pickle': 'pickle', 'parquet': 'parquet'
        }.get(extension, None)
        if not file_type:
            raise ValueError(f'Unsupported file extension: {extension}')
        # Save the DataFrame based on file type
        save_methods = {
            'excel': lambda: df.to_excel(file_path, index=False),
            'tsv': lambda: df.to_csv(file_path, sep='\t', index=False),
            'pickle': lambda: df.to_pickle(file_path),
            'parquet': lambda: df.to_parquet(file_path, index=False),
        }
        if file_type in save_methods:
            save_methods[file_type]()
        else:
            raise ValueError(f'Unsupported file type: {file_type}')
        if verbose:
            print(f'Data saved to {file_path}')
          
    def _get_filepath(self, output_dir, file_name, extension, add_timestamp):
        '''Generate the full file path, optionally adding a timestamp for uniqueness.'''
        if add_timestamp:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            file_name = f'{file_name}_{timestamp}'
        if not file_name.endswith(extension):
            file_name = f'{file_name}.{extension}'
        # Append `_test` if running in test mode and not already a test file
        if self.test_mode and "_test" not in file_name:
            file_name = file_name.replace(f".{extension}", f"_test.{extension}")
        return os.path.join(output_dir, file_name)

