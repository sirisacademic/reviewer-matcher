import os
from modules.data_reader import DataReader
from core.config_handler import ConfigManager
from core.settings_manager import SettingsManager
from modules.data_saver import DataSaver
#from modules.llm_handler import LLMHandler

class DataProcessingPipeline:
    def __init__(self, config_module_name):
        # Initialize ConfigManager to retrieve configuration and paths.
        self.config_manager = ConfigManager(config_module_name)
        # Project, expert, publication specific files to be generated / modified by the pipeline.
        self.file_projects_pipeline = self.config_manager.get('FILE_NAME_PROJECTS')
        self.file_experts_pipeline = self.config_manager.get('FILE_NAME_EXPERTS')
        # Initialize settings (input file, mappings) for projects and experts - reading from the corresponding files.
        mappings_path = self.config_manager.get('MAPPINGS_PATH')
        mappings_path_projects = os.path.join(mappings_path, self.config_manager.get('MAPPINGS_PROJECTS'))    
        mappings_path_experts = os.path.join(mappings_path, self.config_manager.get('MAPPINGS_EXPERTS'))    
        # Settings hanlding for projects and experts.
        self.project_settings = SettingsManager(mappings_path_projects)
        self.expert_settings = SettingsManager(mappings_path_experts)
        # Initialize DataSaver.
        self.data_saver = DataSaver(self.config_manager)

    def run_pipeline(self):
        '''Run the full data processing pipeline.'''
        print('Starting pipeline...')

        # Step 1: Load projects (considering mappings, etc).
        project_reader = DataReader(self.config_manager, self.project_settings)
        projects_data = project_reader.load_data()
        print('Projects data loaded and processed.')
        self.data_saver.save_data(projects_data, self.file_projects_pipeline)

        # Step 2: Load experts (considering mappings, etc).
        expert_reader = DataReader(self.config_manager, self.expert_settings)
        experts_data = expert_reader.load_data()
        print('Experts data loaded and processed.')
        self.data_saver.save_data(experts_data, self.file_experts_pipeline)

        print('Pipeline completed successfully.')
        return projects_data, experts_data

