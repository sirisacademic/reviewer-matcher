# File: data_processing_pipeline.py

import os
import pandas as pd
import traceback

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
from modules.feature_generator import FeatureGenerator
from modules.expert_ranker import ExpertRanker
from modules.expert_assigner import ExpertAssigner
#from modules.expert_profiler import ExpertProfiler

class DataProcessingPipeline:

    def __init__(self, config_manager, call=None, all_components=None, test_mode=False, test_number=10, force_recompute=False):
        """
        Initialize pipeline with configuration, settings, and available components.
        """
        self.call = call
        # Set test mode and number.
        self.test_mode = test_mode
        self.test_number = test_number
        # Whether to load existing or force recomputing existing data.
        self.force_recompute = force_recompute
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
        self.research_type_similarity_scores = None
        self.expert_project_features = None
        self.expert_project_predicted_ranks = None
        # Initialize paths
        self.data_path = self.config_manager.get('DATA_PATH')
        self.scores_output_dir = self.config_manager.get('SCORES_PATH')
        self.assignments_output_dir = self.config_manager.get('ASSIGNMENTS_PATH')
        self.mappings_path = self.config_manager.get('MAPPINGS_PATH')
        self.file_projects_pipeline = self.config_manager.get('FILE_NAME_PROJECTS')
        self.file_experts_pipeline = self.config_manager.get('FILE_NAME_EXPERTS')
        self.file_publications_pipeline = self.config_manager.get('FILE_NAME_PUBLICATIONS')
        self.file_research_type_similarity_scores = config_manager.get('FILE_EXPERT_PROJECT_RESEARCH_TYPE_SIMILARITY')
        self.file_label_similarity_scores  = config_manager.get('FILE_EXPERT_PROJECT_LABEL_SIMILARITY')
        self.file_mesh_similarity_scores = config_manager.get('FILE_EXPERT_PROJECT_MESH_SIMILARITY') 
        self.file_content_similarity_scores = config_manager.get('FILE_EXPERT_PROJECT_CONTENT_SIMILARITY')
        self.file_expert_project_features = config_manager.get('FILE_EXPERT_PROJECT_FEATURES')
        self.file_expert_project_predictions = config_manager.get('FILE_EXPERT_PROJECT_PREDICTIONS')
        self.file_expert_project_assignments = config_manager.get('FILE_EXPERT_PROJECT_ASSIGNMENTS')
        # Initialize modules
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
            'similarity_computation': self._get_similarity_scores,
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
        self.feature_generator = FeatureGenerator(self.config_manager)
        self.expert_ranker = ExpertRanker(self.config_manager)
        self.expert_assigner = ExpertAssigner(self.config_manager)
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
        print(f'Processing data for call {self.call}')
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
                    project_settings = SettingsManager(
                        os.path.join(self.config_manager.get('MAPPINGS_PATH'), self.config_manager.get('MAPPINGS_PROJECTS'))
                    )
                    self.projects = DataReader(self.config_manager, project_settings).load_data()
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
                    expert_settings = SettingsManager(
                        os.path.join(self.config_manager.get('MAPPINGS_PATH'), self.config_manager.get('MAPPINGS_EXPERTS'))
                    )
                    self.experts = DataReader(self.config_manager, expert_settings).load_data()
                # TODO: Extend this to check / add all expert profile relevant information.
                print('Adding expert seniority information...')
                seniority_column = self.config_manager.get('COLUMN_SENIORITY_PUBLICATIONS', 'SENIORITY_PUBLICATIONS')
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
                    # The source used to retrieve publications is set in config_get_publications.py.
                    self.publications = self.publication_handler.get_publications_experts(experts)
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
        """
        Summarize content (research topic, objectives, methods) from publications.
        Processes publications in batches of 100 and saves intermediate results.
        Skips already processed batches if their output files exist.
        """
        try:
            # Load publications data
            publications = self._get_publications()
            # Process in batches. TODO: Pass size as parameter.
            batch_size = 100
            # Calculate total number of batches
            total_batches = (len(publications) + batch_size - 1) // batch_size
            print(f"Starting publications summarization... Total batches: {total_batches}")
            processed_publications = []
            for batch_num in range(total_batches):
                batch_file = f'batch_summarization_{batch_num + 1}_{self.file_publications_pipeline}'
                batch_file_path = os.path.join(self.data_path, batch_file)
                # If batch file exists we load it.
                if os.path.exists(batch_file_path):
                    summarized_batch = load_file(batch_file_path)
                    processed_publications.append(summarized_batch)
                    continue
                # Else, get and process the current batch.
                start_idx = batch_num * batch_size
                end_idx = min((batch_num + 1) * batch_size, len(publications))
                current_batch = publications.iloc[start_idx:end_idx].copy()
                print(f"Processing batch {batch_num + 1}/{total_batches}...")
                # Summarize content for the current batch
                summarized_batch = self.content_summarizer.summarize_content(current_batch)
                # Save the current batch using DataSaver's save_data method
                self.data_saver.save_data(summarized_batch, batch_file)
                processed_publications.append(summarized_batch)
                print(f"Batch {batch_num + 1} completed and saved")
            # Combine all processed batches
            if processed_publications:
                # Update publications in pipeline.
                self.publications = pd.concat(processed_publications, ignore_index=True)
                # Save the publications.
                self.data_saver.save_data(self.publications, self.file_publications_pipeline)
        except Exception as e:
            print(f"Error in summarize_publications: {e}")
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
   
    def _get_similarity_scores(self):
        """Compute similarity scores for experts and projects, or load pre-computed scores if available."""
        try:
            # Get project, expert, publications data.
            projects = self._get_projects()
            experts = self._get_experts()
            publications = self._get_publications()
            # Get or compute research type similarity scores.
            research_type_file_path = os.path.join(self.scores_output_dir, self.file_research_type_similarity_scores)
            if os.path.exists(research_type_file_path) and not self.force_recompute:
                print('Loading pre-computed research type similarity scores...')
                self.research_type_similarity_scores = load_file(research_type_file_path)
            else:
                print('Computing expert-project research type similarity scores...')
                self.research_type_similarity_scores = self.research_type_similarity_calculator.compute_similarity(experts, projects)
                self.data_saver.save_data(
                    self.research_type_similarity_scores,
                    self.file_research_type_similarity_scores,
                    output_dir=self.scores_output_dir
                  )
            # Get or compute label similarity scores.
            label_similarity_file_path = os.path.join(self.scores_output_dir, self.file_label_similarity_scores)
            if os.path.exists(label_similarity_file_path) and not self.force_recompute:
                print('Loading pre-computed label similarity scores...')
                self.label_similarity_scores = load_file(label_similarity_file_path)
            else:
                print('Computing expert-project research topics/approaches similarity scores...')
                self.label_similarity_scores = self.label_similarity_calculator.compute_similarity(experts, projects)
                self.data_saver.save_data(
                    self.label_similarity_scores,
                    self.file_label_similarity_scores,
                    output_dir=self.scores_output_dir
                  )
            # Get or compute MeSH similarity scores.
            mesh_similarity_file_path = os.path.join(self.scores_output_dir, self.file_mesh_similarity_scores)
            if os.path.exists(mesh_similarity_file_path) and not self.force_recompute:
                print('Loading pre-computed MeSH similarity scores...')
                self.mesh_similarity_scores = load_file(mesh_similarity_file_path)
            else:
                print('Computing publication-project MeSH similarity scores...')
                self.mesh_similarity_scores = self.mesh_similarity_calculator.compute_similarity(publications, projects)
                self.data_saver.save_data(
                    self.mesh_similarity_scores,
                    self.file_mesh_similarity_scores,
                    output_dir=self.scores_output_dir
                  )
            # Get or compute content similarity scores.
            content_similarity_file_path = os.path.join(self.scores_output_dir, self.file_content_similarity_scores)
            if os.path.exists(content_similarity_file_path) and not self.force_recompute:
                print('Loading pre-computed content similarity scores...')
                self.content_similarity_scores = load_file(content_similarity_file_path)
            else:
                print('Computing publication-project content similarity scores...')
                self.content_similarity_scores = self.content_similarity_calculator.compute_similarity(publications, projects)
                self.data_saver.save_data(
                  self.content_similarity_scores,
                  self.file_content_similarity_scores,
                  output_dir=self.scores_output_dir
                )
        except Exception as e:
            print(f"Error in _get_similarity_scores: {e}")
            raise

    def _rank_experts(self):
        """Rank experts based on similarity scores."""
        try:
            # Get or compute features.
            features_file_path = os.path.join(self.scores_output_dir, self.file_expert_project_features)
            if os.path.exists(features_file_path) and not self.force_recompute:
                print('Loading pre-computed features...')
                self.expert_project_features = load_file(features_file_path)
            else:
                # Retrieve experts data.
                experts = self._get_experts()
                # Retrieve or compute similarity scores.
                self._get_similarity_scores()
                # Score dataframes
                score_dataframes = [
                    self.content_similarity_scores,
                    self.mesh_similarity_scores,
                    self.label_similarity_scores,
                    self.research_type_similarity_scores
                ]
                # Generate and save features used to predict ranks.
                print('Generating features for predictions...')
                self.expert_project_features = self.feature_generator.generate_features(experts, score_dataframes)
                self.data_saver.save_data(
                    self.expert_project_features,
                    self.file_expert_project_features,
                    output_dir=self.scores_output_dir
                  )
            # Get or compute probabilities with pre-trained model and rank expert-project pairs.
            probabilities_file_path = os.path.join(self.assignments_output_dir, self.file_expert_project_predictions)
            if os.path.exists(probabilities_file_path) and not self.force_recompute:
                print('Loading pre-computed probabilities and rankings...')
                self.expert_project_predicted_ranks = load_file(probabilities_file_path)
            else:
                print('Getting predictions and rankings for expert-project pairs...')
                self.expert_project_predicted_ranks = self.expert_ranker.generate_predictions(self.expert_project_features)
                self.data_saver.save_data(
                    self.expert_project_predicted_ranks,
                    self.file_expert_project_predictions,
                    output_dir=self.assignments_output_dir
                  )
        except Exception as e:
            # Print the error message
            print(f"Error in _rank_experts: {e}")
            # Print the full traceback
            traceback.print_exc()
            # Optionally re-raise the exception if needed
            raise

    def _assign_experts(self):
        """Generate final expert-project assignments based on rankings."""
        try:
            # Get the required data
            if self.expert_project_predicted_ranks is None:
                self._rank_experts()  # Ensure rankings are generated/loaded.
            experts = self._get_experts()
            projects = self._get_projects()
            print(f'\nAssigning {len(experts)} experts to {len(projects)} projects...')
            # First analyze potential issues before making assignments
            print('Analyzing potential assignment issues...')
            # Check for expert capacity issues
            capacity_issues = self.expert_assigner.get_expert_capacity_issues(
                self.expert_project_predicted_ranks,
                experts
            )
            if not capacity_issues.empty:
                print(f'Pre-assignment warning: There are {len(capacity_issues)} experts that could be overloaded. Please check the final assignment reports.')
            # Generate assignments
            print('Generating expert-project assignments...')
            self.expert_project_assignments = self.expert_assigner.generate_assignments(
                self.expert_project_predicted_ranks,
                experts,
                projects
            )
            # Save assignments
            self.data_saver.save_data(
                self.expert_project_assignments,
                self.file_expert_project_assignments,
                output_dir=self.assignments_output_dir
            )
            print('\nExpert-project assignments generated and saved successfully.')
            # Analyze final assignment outcomes
            print('\nAnalyzing assignment outcomes...')
            # Get overall statistics
            assignment_stats = self.expert_assigner.get_assignment_stats(self.expert_project_assignments)
            print('Assignment Statistics:')
            print(f"Total projects assigned: {len(self.expert_project_assignments['Project_ID'].unique())}")
            print(f"Gender distribution in proposed experts: {assignment_stats['gender_distribution']['proposed']}")
            print(f"Gender distribution in alternative experts: {assignment_stats['gender_distribution']['alternative']}")
            # Get expert assignment distribution
            expert_distribution = self.expert_assigner.get_expert_assignment_distribution(
                self.expert_project_assignments,
                experts
            )
            # Get comprehensive project status
            project_status = self.expert_assigner.get_project_assignment_status(
                self.expert_project_assignments,
                projects
            )
            # Display summary of incomplete_projects/problematic projects
            flexed_projects = project_status[project_status['All_Requirements_Met'] == False]
            if not flexed_projects.empty:
                print(f'Post-assignment warning: There are {len(flexed_projects)} projects for which the assignment constrains were flexed.')
            
            # Save all reports
            reports = {
                'expert_distribution': expert_distribution if not expert_distribution.empty else None,
                'flexed_projects': flexed_projects if not flexed_projects.empty else None,
            }
            for report_name, df in reports.items():
                if df is not None:
                    filename = f"assignment_issues_{report_name}.tsv"
                    print(f"\nSaving {report_name} report to {filename}")
                    self.data_saver.save_data(
                        df,
                        filename,
                        output_dir=self.assignments_output_dir
                    )
        except Exception as e:
            print(f"Error in _assign_experts: {e}")
            traceback.print_exc()
            raise

