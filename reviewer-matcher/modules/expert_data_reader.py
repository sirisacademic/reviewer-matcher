from .data_reader import DataReader

class ExpertDataReader(DataReader):
    def __init__(self, data_path, settings_manager):
        super().__init__(data_path, settings_manager)

    def load_and_process_data(self):
        '''Load and process expert data with specific transformations.'''
        self.data = self.load_data()
        self.data = self.map_columns(self.data)
        self.data = self.apply_multi_column_values(self.data)
        self.data = self.apply_value_mappings(self.data)
        self.data = self.apply_separators(self.data)
        self.data = self.add_identifier_column(self.data)
        return self.data

    def apply_multi_column_values(self, df):
        '''Combine values across specified column ranges into a single field.'''
        multi_column_settings = self.settings.get('multi_column_values', {})
        for key, col_range in multi_column_settings.items():
            columns = df.loc[:, col_range].apply(lambda row: '|'.join(row.dropna().astype(str)), axis=1)
            df[key] = columns
        return df

