### data_saver.py

import os
import pandas as pd
import pickle
from datetime import datetime

class DataSaver:
    def __init__(self, config_manager):
        # Configurations read from config file handled by config_manager.
        self.output_dir = config_manager.get('DATA_PATH')
        self.separator = config_manager.get('SEPARATOR_VALUES_OUTPUT')

    def save_data(self, df, file_name, file_type=None, add_timestamp=False, verbose=True):
      extension = os.path.splitext(file_name)[-1].lower().strip('.')
      file_path = self._get_filepath(file_name, extension, add_timestamp)
      # If file type not set, guess it from the extension if present.
      if not file_type:
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
      # Save data based on file type.
      if file_type == 'excel':
          df.to_excel(file_path, index=False)
      elif file_type == 'tsv':
          df.to_csv(file_path, sep='\t', index=False)
      elif file_type == 'pickle':
          df.to_pickle(file_path)
      elif file_type == 'parquet':
          df.to_parquet(file_path, index=False)
      else:
          raise ValueError(f'Unsupported file type: {file_type}')
      if verbose:
          print(f'Data saved to {file_path}')

    def _get_filepath(self, file_name, extension, add_timestamp):
        '''Generate the full file path, optionally adding a timestamp for uniqueness.'''
        if add_timestamp:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'{filename}_{timestamp}'
        if not file_name.endswith(extension):
            file_name = f'{file_name}.{extension}'
        return os.path.join(self.output_dir, file_name)

