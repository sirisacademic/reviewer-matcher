import os
from modules.data_reader import DataReader
from core.config_handler import ConfigManager
from core.settings_manager import SettingsManager
from modules.data_saver import DataSaver

class DataProcessingPipeline:
    def __init__(self, config_module_name, call=None):
        # Initialize ConfigManager to retrieve configuration and paths.
        self.config_manager = ConfigManager(config_module_name)
        
        # Override CALL and related paths if a call parameter is provided
        if call:
            self.config_manager.set('CALL', call)
            call_path = f'calls/{call}'
            self.config_manager.set('CALL_PATH', call_path)
            self.config_manager.set('DATA_PATH', f'{call_path}/data')
            self.config_manager.set('MAPPINGS_PATH', f'{call_path}/mappings')
        
        # Project, expert, publication specific files to be generated/modified by the pipeline.
        self.file_projects_pipeline = self.config_manager.get('FILE_NAME_PROJECTS')
        self.file_experts_pipeline = self.config_manager.get('FILE_NAME_EXPERTS')
        self.file_publications_pipeline = self.config_manager.get('FILE_NAME_PUBLICATIONS')
        
        # Initialize settings (input file, mappings) for projects and experts.
        mappings_path = self.config_manager.get('MAPPINGS_PATH')
        mappings_path_projects = os.path.join(mappings_path, self.config_manager.get('MAPPINGS_PROJECTS'))    
        mappings_path_experts = os.path.join(mappings_path, self.config_manager.get('MAPPINGS_EXPERTS'))    
        self.project_settings = SettingsManager(mappings_path_projects)
        self.expert_settings = SettingsManager(mappings_path_experts)
        
        # Initialize DataSaver
        self.data_saver = DataSaver(self.config_manager)

    def run_pipeline(self, components=None, exclude=None):
        '''Run the full data processing pipeline with optional components.'''
        components = components or ['projects', 'experts', 'publications']
        exclude = exclude or []
        
        # Adjust components based on exclude list
        components = [comp for comp in components if comp not in exclude]

        print('Starting pipeline...')

        # Run each pipeline step based on specified components
        if 'projects' in components:
            project_reader = DataReader(self.config_manager, self.project_settings)
            projects_data = project_reader.load_data()
            print('Projects data loaded and processed.')
            self.data_saver.save_data(projects_data, self.file_projects_pipeline)
        
        if 'experts' in components:
            expert_reader = DataReader(self.config_manager, self.expert_settings)
            experts_data = expert_reader.load_data()
            print('Experts data loaded and processed.')
            self.data_saver.save_data(experts_data, self.file_experts_pipeline)
        
        # Placeholder for publications processing (if required in future)
        if 'publications' in components:
            print('Publications processing is currently not implemented.')
            # publications_data = your_publications_loading_function()
            # self.data_saver.save_data(publications_data, self.file_publications_pipeline)

        print('Pipeline completed successfully.')

