# File: data_processing_pipeline.py

import os

from core.settings_manager import SettingsManager

from utils.functions_read_data import load_file, extract_values_column

from modules.data_reader import DataReader
from modules.data_saver import DataSaver
from modules.publication_handler import PublicationHandler
from modules.content_summarizer import ContentSummarizer
from modules.research_labeler import ResearchLabeler
from modules.mesh_labeler import MeSHLabeler
from modules.label_similarity_calculator import LabelSimilarityCalculator
from modules.mesh_similarity_calculator import MeSHSimilarityCalculator
from modules.content_similarity_calculator import ContentSimilarityCalculator
from modules.research_type_similarity_calculator import ResearchTypeSimilarityCalculator
# !!!! This is to be integrated into the ExpertProfiler class. !!!!
from modules.expert_seniority_calculator import ExpertSeniorityCalculator
#from modules.expert_ranker import ExpertRanker
#from modules.expert_assigner import ExpertAssigner
#from modules.feature_generator import FeatureGenerator
#from modules.expert_profiler import ExpertProfiler


class DataProcessingPipeline:

    def __init__(self, config_manager, call=None, all_components=None, test_mode=False, test_number=10):
        """
        Initialize pipeline with configuration, settings, and available components.
        """
        # Set test mode and number.
        self.test_mode = test_mode
        self.test_number = test_number
        # Set components to run.
        self.all_components = all_components or []
        # Load configuration manager.
        self.config_manager = config_manager
        # Override TEST_MODE and TEST_NUMBER.
        self.config_manager.set('TEST_MODE', test_mode, namespace='config_general')
        self.config_manager.set('TEST_NUMBER', test_number, namespace='config_general')
        # Override CALL if specified
        if call:
            self._override_call_settings(call)
        # Data for experts, projects, publications.
        self.projects = None
        self.experts = None
        self.publications = None
        # Data for similarity scores.
        self.label_similarity_scores = None
        self.mesh_similarity_scores = None
        self.content_similarity_scores = None
        # Initialize paths
        self.data_path = self.config_manager.get('DATA_PATH')
        self.file_projects_pipeline = self.config_manager.get('FILE_NAME_PROJECTS')
        self.file_experts_pipeline = self.config_manager.get('FILE_NAME_EXPERTS')
        self.file_publications_pipeline = self.config_manager.get('FILE_NAME_PUBLICATIONS')
        self.file_research_type_similarity_scores = config_manager.get('FILE_EXPERT_PROJECT_RESEARCH_TYPE_SIMILARITY')
        self.file_label_similarity_scores  = config_manager.get('FILE_EXPERT_PROJECT_LABEL_SIMILARITY')
        self.file_mesh_similarity_scores = config_manager.get('FILE_EXPERT_PROJECT_MESH_SIMILARITY') 
        self.file_content_similarity_scores = config_manager.get('FILE_EXPERT_PROJECT_CONTENT_SIMILARITY')
        # Initialize settings and modules
        self._initialize_settings()
        self._initialize_modules()
        # Component mapping
        self.component_map = {
            # No need to run the data loading components explicitely as they will be invoked if necessary.
            # Included only for testing purposes.
            'project_data_loading': self._get_projects,
            'expert_data_loading': self._get_experts,
            'publication_data_loading': self._get_publications,
            'project_classification': self._classify_projects,
            'project_summarization': self._summarize_projects,
            'project_mesh_tagging': self._mesh_tag_projects,
            'publication_summarization': self._summarize_publications,
            'publication_mesh_tagging': self._mesh_tag_publications,
            'similarity_computation': self._compute_similarity_scores,
            'expert_ranking': self._rank_experts,
            'expert_assignment': self._assign_experts
        }

    def _override_call_settings(self, call):
        """Override CALL-specific settings."""
        call_path = f'calls/{call}'
        self.config_manager.set('CALL', call, namespace='config_general')
        self.config_manager.set('CALL_PATH', call_path, namespace='config_general')
        self.config_manager.set('DATA_PATH', f'{call_path}/data', namespace='config_general')
        self.config_manager.set('MAPPINGS_PATH', f'{call_path}/mappings', namespace='config_general')
        self.config_manager.set('SCORES_PATH', f'{call_path}/scores', namespace='config_general')

    def _initialize_settings(self):
        """Initialize settings for projects and experts."""
        mappings_path = self.config_manager.get('MAPPINGS_PATH')
        self.project_settings = SettingsManager(
            os.path.join(mappings_path, self.config_manager.get('MAPPINGS_PROJECTS'))
        )
        self.expert_settings = SettingsManager(
            os.path.join(mappings_path, self.config_manager.get('MAPPINGS_EXPERTS'))
        )

    def _initialize_modules(self):
        """Initialize all pipeline modules."""
        self.data_saver = DataSaver(self.config_manager)
        self.publication_handler = PublicationHandler(self.config_manager)
        self.research_labeler = ResearchLabeler(self.config_manager)
        self.content_summarizer = ContentSummarizer(self.config_manager)
        self.mesh_labeler = MeSHLabeler(self.config_manager)
        self.label_similarity_calculator = LabelSimilarityCalculator(self.config_manager)
        self.mesh_similarity_calculator = MeSHSimilarityCalculator(self.config_manager)
        self.content_similarity_calculator = ContentSimilarityCalculator(self.config_manager)
        self.research_type_similarity_calculator = ResearchTypeSimilarityCalculator(self.config_manager)
        # !!!! This is to be integrated into the ExpertProfiler class. !!!!
        self.expert_seniority_calculator = ExpertSeniorityCalculator(self.config_manager)
        #self.expert_ranker = ExpertRanker(self.config_manager)
        #self.expert_assigner = ExpertAssigner(self.config_manager)
        #self.feature_generator = FeatureGenerator(self.config_manager)
        #self.expert_profiler = ExpertProfiler(self.config_manager)

    def _run_component(self, component_name, *args, **kwargs):
        """Run a single pipeline component."""
        if component_name in self.component_map:
            print(f"Running {component_name}...")
            self.component_map[component_name](*args, **kwargs)
        else:
            raise ValueError(f"Component '{component_name}' is not recognized.")

    def run_pipeline(self, components=None, exclude=None):
        """
        Run the data processing pipeline based on the specified components to include or exclude.
        """
        if self.test_mode:
            print('Running pipeline in TEST MODE.')
        else:
            print('Running pipeline in NORMAL MODE.')
        components = components or self.all_components
        exclude = exclude or []
        components = [comp for comp in components if comp not in exclude]
        print('Running the following components:')
        print('\n- '.join(components))
        for component in components:
            self._run_component(component)

    # ======== Process and/or load input data (partially processed or raw) ========

    def _get_projects(self):
        """Retrieve, load and/or preprocess project data."""
        if self.projects is None:
            try:
                projects_path = os.path.join(self.data_path, self.file_projects_pipeline)
                if os.path.exists(projects_path):
                    print('Loading pre-processed project data...')
                    self.projects = load_file(projects_path)
                else:
                    print('Reading project data from source file...')
                    self.projects = DataReader(self.config_manager, self.project_settings).load_data()
                    self.data_saver.save_data(self.projects, self.file_projects_pipeline)
                if self.test_mode:
                    print(f"Processing a subset of the data for test mode: {len(self.projects)} projects reduced to {self.test_number} rows.")
                    self.projects = self.projects.head(self.test_number)
            except Exception as e:
                print(f"Error in _get_projects: {e}")
                raise
        return self.projects  

    def _get_experts(self):
        """Load or preprocess expert data, ensuring expert profile enrichment is performed if not already done."""
        if self.experts is None:
            try:
                experts_path = os.path.join(self.data_path, self.file_experts_pipeline)
                if os.path.exists(experts_path):
                    print('Loading pre-processed expert data...')
                    self.experts = load_file(experts_path)
                else:
                    print('Reading expert data from source file...')
                    self.experts = DataReader(self.config_manager, self.expert_settings).load_data()
                # TODO: Extend this to check / add all expert profile relevant information.
                seniority_column = self.config_manager.get('COLUMN_SENIORITY', 'SENIORITY')
                if seniority_column not in self.experts.columns:
                    print('Enriching expert profiles...')
                    # !!!! This is to be integrated into the ExpertProfiler class. !!!!
                    self.experts = self.expert_seniority_calculator.enrich_with_seniority(self.experts)
                    self.data_saver.save_data(self.experts, self.file_experts_pipeline)
                if self.test_mode:
                    print(f"Processing a subset of the data for test mode: {len(self.experts)} experts reduced to {self.test_number} rows.")
                    self.experts = self.experts.head(self.test_number)
            except Exception as e:
                print(f"Error in _get_experts: {e}")
                raise
        return self.experts

    def _get_publications(self):
        """Load or preprocess publication data."""
        if self.publications is None:
            try:
                publications_path = os.path.join(self.data_path, self.file_publications_pipeline)
                if os.path.exists(publications_path):
                    print('Loading pre-processed publication data...')
                    self.publications = load_file(publications_path)
                else:
                    print('Retrieving publication data from experts...')
                    experts = self._get_experts()
                    self.publications = self.publication_handler.get_publications_experts(experts, source='openalex')
                    self.data_saver.save_data(self.publications, self.file_publications_pipeline)
                # Truncate abstracts too long before processing.
                max_abstract_length = self.config_manager.get('PUB_ABSTRACT_MAX_LENGTH', 5000)
                abstract_col = self.config_manager.get('COLUMN_PUB_ABSTRACT', 'ABSTRACT')
                self.publications[abstract_col] = self.publications[abstract_col].apply(
                    lambda abstract: abstract[:max_abstract_length] if len(abstract) > max_abstract_length else abstract
                )
                if self.test_mode:
                    print(f"Processing a subset of the data for test mode: {len(self.publications)} publications reduced to {self.test_number} rows.")
                    self.publications = self.publications.head(self.test_number)
            except Exception as e:
                print(f"Error in _get_publications: {e}")
                raise
        return self.publications 

    # ======== Module-invoking methods ========
    
    def _classify_projects(self):
        """Classify projects with research areas and approaches."""
        try:
            separator = self.config_manager.get('SEPARATOR_VALUES_OUTPUT', '|')
            projects = self._get_projects()
            experts = self._get_experts()
            print('Classifying projects with research areas...')
            column_name = self.config_manager.get('COLUMN_RESEARCH_AREAS')
            topics = extract_values_column(experts, column_name, separator)
            prompt_file = self.config_manager.get('FILE_PROMPT_TOPICS')
            projects = self.research_labeler.label_topics(projects, topics, prompt_file, column_name)
            self.data_saver.save_data(projects, self.file_projects_pipeline)
            print('Classifying projects with research approaches...')
            column_name = self.config_manager.get('COLUMN_RESEARCH_APPROACHES')
            topics = extract_values_column(experts, column_name, separator)
            prompt_file = self.config_manager.get('FILE_PROMPT_APPROACHES')
            projects = self.research_labeler.label_topics(projects, topics, prompt_file, column_name)
            self.data_saver.save_data(projects, self.file_projects_pipeline)
            # Updating projects in pipeline.
            self.projects = projects
        except Exception as e:
            print(f"Error in _classify_projects: {e}")
            raise

    def _summarize_projects(self):
        """Summarize content (research topic, objectives, methods) from projects."""
        try:
            # Load project data.
            projects = self._get_projects()
            # Summarize content for the projects.
            print("Starting projects summarization...")
            projects = self.content_summarizer.summarize_content(projects)
            # Save the enriched project data.
            self.data_saver.save_data(projects, self.file_projects_pipeline)
            # Updating projects in pipeline.
            self.projects = projects
        except Exception as e:
            print(f"Error in _summarize_projects: {e}")
            raise
      
    def _summarize_publications(self):
        """Summarize content (research topic, objectives, methods) from publications."""
        try:
            # Load publications data
            publications = self._get_publications()
            # Summarize content for the publications
            print("Starting publications summarization...")
            publications = self.content_summarizer.summarize_content(publications)
            # Save the enriched publications data
            self.data_saver.save_data(publications, self.file_publications_pipeline)
            # Updating publications in pipeline.
            self.publications = publications
        except Exception as e:
            print(f"Error in _summarize_publications: {e}")
            raise

    # MeSH tagging for projects and publications.
    # TODO: Use MeSH terms categories to aggregate MeSH terms - different categories would be used to generate different features.
    # TODO: In the case of publications we should consider using the MeSH terms retrieved from PubMed/OpenAlex when available.
    #       Now obtaining MeSH terms for publications by means of the same model used for projects for consistency sake.
    def _mesh_tag_projects(self): # done
        """Tag projects with MeSH terms."""
        try:
            projects = self._get_projects()
            print('Tagging projects with MeSH terms...')
            input_columns = self.config_manager.get('MESH_INPUT_COLUMNS_PROJECTS')
            projects = self.mesh_labeler.label_with_mesh(projects, input_columns)
            self.data_saver.save_data(projects, self.file_projects_pipeline)
            # Updating projects in pipeline.
            self.projects = projects
        except Exception as e:
            print(f"Error in _mesh_tag_projects: {e}")
            raise

    def _mesh_tag_publications(self): # done
        """Tag publications with MeSH terms."""
        try:
            publications = self._get_publications()
            print('Tagging publications with MeSH terms...')
            input_columns = self.config_manager.get('MESH_INPUT_COLUMNS_PUBLICATIONS')
            publications = self.mesh_labeler.label_with_mesh(publications, input_columns)
            self.data_saver.save_data(publications, self.file_publications_pipeline)
            # Updating publications in pipeline.
            self.publications = publications
        except Exception as e:
            print(f"Error in _mesh_tag_publications: {e}")
            raise

    def _compute_similarity_scores(self):
        """Compute similarity scores for experts and projects."""
        try:
            projects = self._get_projects()
            experts = self._get_experts()
            publications = self._get_publications()
            # Compute and save scores.
            scores_output_dir = self.config_manager.get('SCORES_PATH')
            print('Computing expert-project research type similarity scores...')
            self.research_type_similarity_scores = self.research_type_similarity_calculator.compute_similarity(experts, projects)
            self.data_saver.save_data(self.research_type_similarity_scores, self.file_research_type_similarity_scores, output_dir=scores_output_dir)
            print('Computing expert-project research topics/approaches similarity scores...')
            self.label_similarity_scores = self.label_similarity_calculator.compute_similarity(experts, projects)
            self.data_saver.save_data(self.label_similarity_scores, self.file_label_similarity_scores, output_dir=scores_output_dir)
            print('Computing publication-project MeSH similarity scores...')
            self.mesh_similarity_scores = self.mesh_similarity_calculator.compute_similarity(publications, projects)
            self.data_saver.save_data(self.mesh_similarity_scores, self.file_mesh_similarity_scores, output_dir=scores_output_dir)
            print('Computing publication-project content similarity scores...')
            self.content_similarity_scores = self.content_similarity_calculator.compute_similarity(publications, projects)
            self.data_saver.save_data(self.content_similarity_scores, self.file_content_similarity_scores, output_dir=scores_output_dir)
        except Exception as e:
            print(f"Error in _compute_similarity_scores: {e}")
            raise

    def _rank_experts(self):
        """Rank experts based on similarity scores."""
        try:
            print('Ranking experts with respect to projects...')
            label_similarity_scores, mesh_similarity_scores, content_similarity_scores = self._compute_similarity_scores()
            experts = self._get_experts()
            projects = self._get_projects()
            expert_project_rankings = self.expert_ranker.rank_experts(
                experts,
                projects,
                label_similarity_scores,
                mesh_similarity_scores,
                content_similarity_scores
            )
            return expert_project_rankings
        except Exception as e:
            print(f"Error in _rank_experts: {e}")
            raise

    def _assign_experts(self):
        """Assign experts to projects."""
        try:
            print('Assigning experts to projects...')
            expert_project_rankings = self._rank_experts()
            assignments = self.expert_assigner.assign_experts(expert_project_rankings)
            print(f"Assignments: {assignments}")
            return assignments
        except Exception as e:
            print(f"Error in _assign_experts: {e}")
            raise


