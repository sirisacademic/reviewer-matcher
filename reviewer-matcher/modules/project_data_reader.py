from .data_reader import DataReader

class ProjectDataReader(DataReader):
    def __init__(self, data_path, settings_manager):
        super().__init__(data_path, settings_manager)

    def load_and_process_data(self, file_path):
        self.data = self.load_data()
        self.data = self.map_columns(self.data)
        self.data = self.apply_value_mappings(self.data)
        self.data = self.apply_separators(self.data)
        self.data = self.add_identifier_column(self.data)
        return self.data

