import os

from core.settings_manager import SettingsManager

from utils.functions_read_data import load_file, extract_values_column

from modules.data_reader import DataReader
from modules.data_saver import DataSaver
from modules.publication_extractor import PublicationExtractor
from modules.pubmed_retriever import PubMedRetriever
from modules.content_extractor import ContentExtractor
from modules.external_research_labeler import ExternalResearchLabeler
from modules.local_research_labeler import LocalResearchLabeler
from modules.mesh_labeler import MeSHLabeler
from modules.label_similarity_calculator import LabelSimilarityCalculator
from modules.mesh_similarity_calculator import MeSHSimilarityCalculator
from modules.content_similarity_calculator import ContentSimilarityCalculator
from modules.expert_ranker import ExpertRanker
from modules.expert_assigner import ExpertAssigner

class DataProcessingPipeline:

    def __init__(self, config_manager, call=None, all_components=None, test_mode=False, test_number=10):
        """
        Initialize pipeline with configuration, settings, and available components.
        """
        self.test_mode = test_mode
        self.test_number = test_number
        self.all_components = all_components or []
        self.config_manager = config_manager
           
        # Override CALL if specified
        if call:
            self._override_call_settings(call)

        # Initialize paths
        self.data_path = self.config_manager.get('DATA_PATH')
        self.file_projects_pipeline = self.config_manager.get('FILE_NAME_PROJECTS')
        self.file_experts_pipeline = self.config_manager.get('FILE_NAME_EXPERTS')
        self.file_publications_pipeline = self.config_manager.get('FILE_NAME_PUBLICATIONS')

        # Initialize settings and modules
        self._initialize_settings()
        self._initialize_modules()

        # Component mapping
        self.component_map = {
            'project_data_loading': self._load_project_data,
            'expert_data_loading': self._load_expert_data,
            'publication_extraction': self._extract_publications,
            'pubmed_retrieval': self._retrieve_pubmed_data,
            'project_enrichment': self._enrich_projects,
            'publication_enrichment': self._enrich_publications,
            'project_classification': self._classify_projects,
            'project_mesh_tagging': self._mesh_tag_projects,
            'publication_mesh_tagging': self._mesh_tag_publications,
            'similarity_computation': self._compute_similarity,
            'expert_ranking': self._rank_experts,
            'expert_assignment': self._assign_experts
        }

    def _override_call_settings(self, call):
        """Override CALL-specific settings."""
        self.config_manager.set('CALL', call)
        call_path = f'calls/{call}'
        self.config_manager.set('CALL_PATH', call_path)
        self.config_manager.set('DATA_PATH', f'{call_path}/data')
        self.config_manager.set('MAPPINGS_PATH', f'{call_path}/mappings')

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
        self.data_saver = DataSaver(self.config_manager, self.test_mode)
        self.pubmed_retriever = PubMedRetriever(self.config_manager)
        self.publication_extractor = PublicationExtractor(self.config_manager)
        research_labeler_class = (
            ExternalResearchLabeler if self.config_manager.get('USE_EXTERNAL_LLM_MODEL') else LocalResearchLabeler
        )
        self.research_labeler = research_labeler_class(self.config_manager)
        self.content_extractor = ContentExtractor()
        self.mesh_labeler = MeSHLabeler()
        self.label_similarity_calculator = LabelSimilarityCalculator()
        self.mesh_similarity_calculator = MeSHSimilarityCalculator()
        self.content_similarity_calculator = ContentSimilarityCalculator()
        self.expert_ranker = ExpertRanker()
        self.expert_assigner = ExpertAssigner()

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

    # ======== Helper Methods ========

    def _load_project_data(self):
        """Load or preprocess project data."""
        try:
            projects_path = os.path.join(self.data_path, self.file_projects_pipeline)
            if os.path.exists(projects_path):
                print('Loading pre-processed project data...')
                projects = load_file(projects_path)
            else:
                print('Reading project data from source file...')
                projects = DataReader(self.config_manager, self.project_settings).load_data()
                self.data_saver.save_data(projects, self.file_projects_pipeline)
            if self.test_mode:
                print(f"Processing a subset of the data for test mode: {len(projects)} projects reduced to {self.test_number} rows.")
                projects = projects.head(self.test_number)
            return projects
        except Exception as e:
            print(f"Error in _load_project_data: {e}")
            raise

    def _load_expert_data(self):
        """Load or preprocess expert data."""
        try:
            experts_path = os.path.join(self.data_path, self.file_experts_pipeline)
            if os.path.exists(experts_path):
                print('Loading pre-processed expert data...')
                experts = load_file(experts_path)
            else:
                print('Reading expert data from source file...')
                experts = DataReader(self.config_manager, self.expert_settings).load_data()
                self.data_saver.save_data(experts, self.file_experts_pipeline)
            if self.test_mode:
                print(f"Processing a subset of the data for test mode: {len(experts)} experts reduced to {self.test_number} rows.")
                experts = experts.head(self.test_number)
            return experts
        except Exception as e:
            print(f"Error in _load_expert_data: {e}")
            raise

    def _extract_publications(self):
        """Extract publication titles from expert data."""
        try:
            # Ensure expert data is loaded
            experts = self._load_expert_data()  
            print('Extracting publication titles using NER...')
            expert_publication_titles = self.publication_extractor.extract_publication_titles(experts)
            return expert_publication_titles
        except Exception as e:
            print(f"Error in _extract_publications: {e}")
            raise

    def _retrieve_pubmed_data(self):
        """Retrieve data from PubMed."""
        try:
            expert_publication_titles = self._extract_publications()
            print('Retrieving PubMed data...')
            publications = self.pubmed_retriever.fetch_publications(expert_publication_titles)
            self.data_saver.save_data(publications, self.file_publications_pipeline)
            return publications
        except Exception as e:
            print(f"Error in _retrieve_pubmed_data: {e}")
            raise

    def _classify_projects(self):
        """Classify projects with research areas and approaches."""
        try:
            separator = self.config_manager.get('SEPARATOR_VALUES_OUTPUT', '|')
            projects = self._load_project_data()
            experts = self._load_expert_data()
            print('Classifying projects with research areas...')
            column_name = self.config_manager.get('COLUMN_RESEARCH_AREAS')
            topics = extract_values_column(experts, column_name, separator)
            prompt_file = self.config_manager.get('FILE_PROMPT_TOPICS')
            projects = self.research_labeler.label_topics(projects, topics, prompt_file, column_name)
            print('Classifying projects with research approaches...')
            column_name = self.config_manager.get('COLUMN_RESEARCH_APPROACHES')
            topics = extract_values_column(experts, column_name, separator)
            prompt_file = self.config_manager.get('FILE_PROMPT_APPROACHES')
            projects = self.research_labeler.label_topics(projects, topics, prompt_file, column_name)
            self.data_saver.save_data(projects, self.file_projects_pipeline)
            return projects
        except Exception as e:
            print(f"Error in _classify_projects: {e}")
            raise

    ### FUNCTIONS BELOW TO IMPLEMENT UPDATE !!! ###

    def _enrich_projects(self):
        """Enrich project data."""
        try:
            projects = self._load_project_data()
            print('Enriching projects...')
            projects = self.content_extractor.extract_content(projects)
            self.data_saver.save_data(projects, self.file_projects_pipeline)
            return projects
        except Exception as e:
            print(f"Error in _enrich_projects: {e}")
            raise

    def _enrich_publications(self):
        """Enrich publication data."""
        try:
            publications_path = os.path.join(self.data_path, self.file_publications_pipeline)
            if os.path.exists(publications_path):
                print('Loading pre-processed publication data...')
                publications = load_file(publications_path)
            else:
                publications = self._retrieve_pubmed_data()
            if self.test_mode:
                print(f"Processing a subset of the data for test mode: {len(publications)} publications reduced to {self.test_number} rows.")
                publications = publications.head(self.test_number)
            print('Enriching publications...')
            publications = self.content_extractor.extract_content(publications)
            self.data_saver.save_data(publications, self.file_publications_pipeline)
            return publications
        except Exception as e:
            print(f"Error in _enrich_publications: {e}")
            raise

    # MeSH tagging for projects and publications - Labels items with MeSH terms, aggregates them.
    # Involves two steps: 
    # - Tagging MeSH terms by means of the MeSH-tagging model.
    # - Obtaining MeSH term categories and aggregating them.
    # NOTE:
    # In the case of publications we should consider the possibility to tag them or use the MeSH terms retrieved from PubMed
    # and tag only the publications that do not contain MeSH terms.

    def _mesh_tag_projects(self):
        """Tag projects with MeSH terms."""
        try:
            projects = self._load_project_data()
            print('Tagging projects with MeSH terms...')
            mesh_labeler = MeSHLabeler()
            input_columns = {
                'TITLE': 'string',
                'ABSTRACT': 'string',
                'RESEARCH_TOPIC': 'string',
                'OBJECTIVES': 'string',
                'METHODS': 'list'
            }
            projects = mesh_labeler.label_with_mesh(projects, input_columns)
            self.data_saver.save_data(projects, self.file_projects_pipeline)
            return projects
        except Exception as e:
            print(f"Error in _mesh_tag_projects: {e}")
            raise

    def _mesh_tag_publications(self):
        """Tag publications with MeSH terms."""
        try:
            publications = self._enrich_publications()
            print('Tagging publications with MeSH terms...')
            mesh_labeler = MeSHLabeler()
            input_columns = {
                'TITLE_PUBMED': 'string',
                'ABSTRACT_PUBMED': 'string',
                'RESEARCH_TOPIC': 'string',
                'OBJECTIVES': 'string',
                'METHODS': 'list'
            }
            publications = mesh_labeler.label_with_mesh(publications, input_columns)
            self.data_saver.save_data(publications, self.file_publications_pipeline)
            return publications
        except Exception as e:
            print(f"Error in _mesh_tag_publications: {e}")
            raise

    def _compute_similarity(self):
        """Compute similarity scores for experts and projects."""
        try:
            print('Computing similarity scores...')
            projects = self._load_project_data()
            experts = self._load_expert_data()
            publications = self._enrich_publications()
            label_similarity_scores = self.label_similarity_calculator.compute_similarity(experts, projects)
            mesh_similarity_scores = self.mesh_similarity_calculator.compute_expert_project_similarity(publications, projects)
            content_similarity_scores = self.content_similarity_calculator.compute_similarity(publications, projects)
            return label_similarity_scores, mesh_similarity_scores, content_similarity_scores
        except Exception as e:
            print(f"Error in _compute_similarity: {e}")
            raise

    def _rank_experts(self):
        """Rank experts based on similarity scores."""
        try:
            print('Ranking experts with respect to projects...')
            label_similarity_scores, mesh_similarity_scores, content_similarity_scores = self._compute_similarity()
            experts = self._load_expert_data()
            projects = self._load_project_data()
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


