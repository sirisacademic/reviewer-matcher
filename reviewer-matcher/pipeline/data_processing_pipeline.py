import os
from modules.project_data_processor import ProjectDataReader
from modules.expert_data_processor import ExpertDataReader
from core.config_handler import ConfigManager
from core.settings_manager import SettingsManager
from modules.data_saver import DataSaver
#from modules.llm_handler import LLMHandler

class DataProcessingPipeline:
    def __init__(self, config_module_name):
        # Initialize ConfigManager to retrieve configuration and paths.
        self.config_manager = ConfigManager(config_module_name)
        # Initialize settings (input file, mappings) for projects and experts - reading from the corresponding files.
        mappings_path = self.config_manager.get('MAPPINGS_PATH')
        mappings_path_projects = os.path.join(mappings_path, self.config_manager.get('MAPPINGS_PROJECTS'))    
        mappings_path_experts = os.path.join(mappings_path, self.config_manager.get('MAPPINGS_EXPERTS'))    
        self.project_settings = SettingsManager(mappings_path_projects)
        self.expert_settings = SettingsManager(mappings_path_experts)
        # Initialize DataSaver with the output directory.
        self.data_saver = DataSaver(self.config_manager.get('DATA_PATH'))
        # Project, expert, publication specific files to be generated / modified by the pipeline.
        self.file_projects_pipeline = self.config_manager.get('FILE_NAME_PROJECTS')
        self.file_experts_pipeline = self.config_manager.get('FILE_NAME_EXPERTS')

    def load_projects(self):
        '''Load and process project data.'''
        '''The path of the source data file is included as one field in the settings.'''
        project_reader = ProjectDataReader(self.data_path, self.project_settings)
        return project_reader.load_and_process_data()

    def load_experts(self):
        '''Load and process expert data.'''
        expert_reader = ExpertDataReader(self.data_path, self.expert_settings)
        return expert_reader.load_and_process_data()

    def run_pipeline(self):
        '''Run the full data processing pipeline.'''
        print('Starting pipeline...')

        # Step 1: Load projects (considering mappings, etc).
        projects_data = self.load_projects()
        print('Projects data loaded and processed.')
        self.data_saver.save_data(projects_data, self.file_projects_pipeline)

        # Step 2: Load experts (considering mappings, etc).
        experts_data = self.load_experts()
        print('Experts data loaded and processed.')
        self.data_saver.save_data(experts_data, self.file_experts_pipeline)

        print('Pipeline completed successfully.')
        return projects_data, experts_data

