import os

from core.config_handler import ConfigManager
from core.settings_manager import SettingsManager

from utils.functions_read_data import load_file

from modules.data_reader import DataReader
from modules.data_saver import DataSaver
from modules.publication_extractor import PublicationExtractor
from modules.pubmed_retriever import PubMedRetriever
from modules.content_extractor import ContentExtractor
from modules.research_labeler import ResearchLabeler
from modules.mesh_labeler import MeSHLabeler
from modules.label_similarity_calculator import LabelSimilarityCalculator
from modules.mesh_similarity_calculator import MeSHSimilarityCalculator
from modules.content_similarity_calculator import ContentSimilarityCalculator
from modules.expert_ranker import ExpertRanker
from modules.expert_assigner import ExpertAssigner

class DataProcessingPipeline:
    def __init__(self, config_module_name, call=None, all_components=None):
        """
        Initialize pipeline with configuration, settings, and available components.
        """
        self.all_components = all_components or []
        self.config_manager = ConfigManager(config_module_name)
        
        # Override CALL if specified, setting paths based on call name.
        if call:
            self.config_manager.set('CALL', call)
            call_path = f'calls/{call}'
            self.config_manager.set('CALL_PATH', call_path)
            self.config_manager.set('DATA_PATH', f'{call_path}/data')
            self.config_manager.set('MAPPINGS_PATH', f'{call_path}/mappings')

        # Project, expert, publication specific files to be generated/modified by the pipeline.
        self.data_path = self.config_manager.get('DATA_PATH')
        self.file_projects_pipeline = self.config_manager.get('FILE_NAME_PROJECTS')
        self.file_experts_pipeline = self.config_manager.get('FILE_NAME_EXPERTS')
        self.file_publications_pipeline = self.config_manager.get('FILE_NAME_PUBLICATIONS')
        
        # Initialize settings managers for projects and experts.
        # Set paths.
        mappings_path = self.config_manager.get('MAPPINGS_PATH')
        mappings_path_projects = os.path.join(mappings_path, self.config_manager.get('MAPPINGS_PROJECTS'))
        mappings_path_experts = os.path.join(mappings_path, self.config_manager.get('MAPPINGS_EXPERTS'))
        # Initialize settings.
        self.project_settings = SettingsManager(mappings_path_projects)
        self.expert_settings = SettingsManager(mappings_path_experts)
               
        # Initialize data saver module.
        self.data_saver = DataSaver(self.config_manager)

        # Initialize publication extraction module.
        self.publication_extractor = PublicationExtractor(self.config_manager)

        # Initialize data retrieval from external sources
        self.pubmed_retriever = PubMedRetriever(self.config_manager)
        
        # Initialize enrichment modules
        self.content_extractor = ContentExtractor()
        self.research_labeler = ResearchLabeler()
        self.mesh_labeler = MeSHLabeler()

        # Initialize similarity calculation modules
        self.label_similarity_calculator = LabelSimilarityCalculator()
        self.mesh_similarity_calculator = MeSHSimilarityCalculator()
        self.content_similarity_calculator = ContentSimilarityCalculator()

        # Initialize expert ranking and assignment modules
        self.expert_ranker = ExpertRanker()
        self.expert_assigner = ExpertAssigner()

    def run_pipeline(self, components=None, exclude=None):
        """
        Run the data processing pipeline based on the specified components to include or exclude.
        Each component corresponds to a step in the pipeline. Custom processing logic should be implemented
        in each module where "To be implemented" comments are present.
        """

        # Set components to run.
        components = components or self.all_components
        exclude = exclude or []
        components = [comp for comp in components if comp not in exclude]

        # Step: Data loading - Read expert / project data from input files.
        if 'project_data_loading' in components:
            print("Reading project data from source file...")
            projects_data = DataReader(self.config_manager, self.project_settings).load_data()
            self.data_saver.save_data(projects_data, self.file_projects_pipeline)
            print('Project data read from source file and saved.')
        else:
            print("Loading pre-processed project data...")
            projects_path = os.path.join(self.data_path, self.file_projects_pipeline)
            projects_data = load_file(projects_path)
                      
        if 'expert_data_loading' in components:
            experts_data = DataReader(self.config_manager, self.expert_settings).load_data()
            self.data_saver.save_data(experts_data, self.file_experts_pipeline)
            print('Expert data read from source file and saved.')
        else:
            print("Loading pre-processed expert data...")
            experts_path = os.path.join(self.data_path, self.file_experts_pipeline)
            experts_data = load_file(experts_path)

        # Step: Extract publication titles from expert data.
        if 'publication_extraction' in components:
            print("Extracting publication titles using NER...")
            expert_publication_titles = self.publication_extractor.extract_publication_titles(experts_data)

        # Step: Data retrieval from external sources (e.g., PubMed) - To be implemented
        if 'pubmed_retrieval' in components:
            print("Retrieving PubMed data...")
            publications = self.pubmed_retriever.fetch_publications(expert_publication_titles)

        # Step: Enrich projects/publications with content extracted/generated by means of LLM.
        if 'project_enrichment' in components:
            print("Enriching projects...")
            enriched_projects = self.content_extractor.extract_content(projects_data)
        if 'publication_enrichment' in components and 'publications' in locals():
            print("Enriching publications...")
            enriched_publications = self.content_extractor.extract_content(publications)

        # Step: Project classification with research topics/areas - Uses LLM for project-specific topics
        if 'project_classification' in components:
            print("Classifying projects...")
            labeled_projects = self.research_labeler.label_project(enriched_projects)
            
        # Step: MeSH tagging for projects and publications - Labels items with MeSH terms, aggregates them.
        # Involves two steps: 
        # - Tagging MeSH terms by means of the MeSH-tagging model.
        # - Obtaining MeSH term categories and aggregating them.
        # NOTE:
        # In the case of publications we should consider the possibility to tag them or use the MeSH terms retrieved from PubMed
        # and tag only the publications that do not contain MeSH terms.
        if 'project_mesh_tagging' in components:
            print("Tagging projects and publications with MeSH terms...")
            # Note: Ensure mesh_labeler can handle data structure of projects and publications
            projects_with_mesh = self.mesh_labeler.label_with_mesh(labeled_projects)
        if 'publication_mesh_tagging' in components:
            publications_with_mesh = self.mesh_labeler.label_with_mesh(enriched_publications)

        # Step: Similarity computations - Calculates label, MeSH, and content similarity scores
        if 'similarity_computation' in components:
            print("Computing similarity scores...")
            # Label similarity: Calculates based on research areas/approaches
            label_similarity_scores = self.label_similarity_calculator.compute_similarity(experts_data, labeled_projects)
            # MeSH similarity: Calculates between publications and projects, with aggregation for expert-project scores
            mesh_similarity_scores = self.mesh_similarity_calculator.compute_expert_project_similarity(
                publications_with_mesh, projects_with_mesh)
            # Content similarity: Calculates between extracted content from publications and projects
            content_similarity_scores = self.content_similarity_calculator.compute_similarity(
                enriched_publications, enriched_projects)

        # Step: Expert-project ranking based on similarity scores
        if 'expert_ranking' in components:
            print("Ranking experts with respect to projects...")
            expert_project_rankings = self.expert_ranker.rank_experts(
                experts_data,
                projects_data,
                label_similarity_scores,
                mesh_similarity_scores,
                content_similarity_scores
            )

        # Step: Expert-project assignment - Generates assignments based on rankings and constraints
        if 'expert_assignment' in components and 'expert_project_rankings' in locals():
            print("Assigning experts to projects...")
            assignments = self.expert_assigner.assign_experts(expert_project_rankings)
            print("Assignments:", assignments)

