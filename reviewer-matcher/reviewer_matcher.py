# reviewer_matcher.py

from .data_processor import DataProcessor
from .metadata_enricher import MetadataEnricher
from .content_processor import ContentProcessor
from .expert_profiler import ExpertProfiler
from .similarity_calculator import SimilarityCalculator
from .relevance_predictor import RelevancePredictor
from .panel_optimizer import PanelOptimizer

import os
import pandas as pd
import warnings

class ReviewerMatcher:
    def __init__(self):
        self.data_processor = DataProcessor()
        self.metadata_enricher = MetadataEnricher()
        self.content_processor = ContentProcessor()
        self.expert_profiler = ExpertProfiler(
            data_path=os.path.join("..", "examples", "example_data", "reviewers.csv"),
            api_key="d17525f409e675a5c89c428e1aae6871",  # replace with your openalex api key
        )
        self.similarity_calculator = SimilarityCalculator()
        self.relevance_predictor = RelevancePredictor()
        self.panel_optimizer = PanelOptimizer()
    
    def run_matcher(self, project):
        """Main entry point to initiate the matching process for a project."""
        pass
    
    def get_recommendations(self, project):
        """Returns ranked expert recommendations for a project."""
        pass
    
    def generate_report(self, project, experts):
        """Generates a report detailing the expert-project matches."""
        pass

    def profile_experts(self, experts):
        """Generates reviewer information from full name and orcid."""

        # read in example data 
        reviewers = pd.read_csv(self.data_path)

        # enrich data and compute completeness using ORCID numbers and fallback to full names
        completeness_both, reviewers_enriched = self.expert_profiler.compute_completeness_for_method(reviewers, method="both")

        # save enriched data
        save_path = os.path.join("..", "intermediate_data", "reviewer_enrichment")
        file_path = os.path.join(save_path, 'reviewers_enriched.csv')
        reviewers_enriched.to_csv(file_path, index=False)

        # print completeness and enriched data 
        print(f"Reviewer Information Retrieval Success Rate (Using ORCID + Full Name): {completeness_both}%")
        print(reviewers_enriched.head())

        # predict the reviewers' gender 
        reviewers_gender = self.expert_profiler.enrich_data_with_predicted_gender(reviewers_enriched, self.expert_profiler.api_key)

        # save the gender
        save_path = os.path.join("..", "intermediate_data", "reviewer_gender")
        file_path = os.path.join(save_path, 'reviewers_gender.csv')
        reviewers_gender.to_csv(file_path, index=False)

        # print the breakdown of genders
        print("Gender Breakdown:")
        print(reviewers_gender['gender'].value_counts(normalize=True) * 100)

        # directory to save the publications
        save_path = os.path.join("..", "intermediate_data", "reviewer_publications")

        # retrieve publications for each reviewer
        self.expert_profiler.save_publications_to_csv(reviewers_gender, save_path)

        # print average publications per reviwer
        average_publications = self.expert_profiler.calculate_average_publications_per_author(save_path)
        print(f"Average number of publications per reviewer: {average_publications:.2f}")

        # define input and output folders
        input_folder = "../intermediate_data/reviewer_publications"
        output_folder = "../intermediate_data/reviewer_research_phase"

        #  classify research phase
        self.expert_profiler.classify_all_publications_by_research_phase(input_folder, output_folder)

        # combine all data from the research phase classification output folder
        research_phase_data = self.expert_profiler.combine_data(output_folder)

        # print the breakdown of research phases
        print("Research Phase Breakdown:")
        print(research_phase_data['research_phase'].value_counts(normalize=True) * 100)

        # define input and output folders
        input_folder = "../intermediate_data/reviewer_research_phase"
        output_folder = "../intermediate_data/reviewer_domain"

        # classify domain
        self.expert_profiler.classify_all_publications_by_domain(input_folder, output_folder)

        # combine all data from the domain classification output folder
        domain_data = self.expert_profiler.combine_data(output_folder)

        # print the proportion of papers classified as biomedical
        biomedicine_proportion = (domain_data['domain'] == 'Biomedicine').mean() * 100
        print(f"\nProportion of papers in the biomedicine domain: {biomedicine_proportion:.2f}%")

        # define input and output folders
        input_folder = "../intermediate_data/reviewer_domain"
        output_folder = "../intermediate_data/reviewer_mental_health"

        # classify mental health
        warnings.filterwarnings("ignore")
        self.expert_profiler.classify_all_publications_by_mental_health(input_folder, output_folder)

        # combine data to display statistics 
        combined_df = self.expert_profiler.combine_data(output_folder)

        # print the proportion of mental health related papers
        print(f"Total papers: {len(combined_df)}")
        print(f"Mental health-related papers: {combined_df['mental_health'].sum()}")
        print(f"Proportion of mental health-related papers: {combined_df['mental_health'].sum() / len(combined_df)}")

        # define input and output folders
        input_folder = "../intermediate_data/reviewer_mental_health"
        output_folder = "../intermediate_data/ranked_mesh_terms"

        # rank mesh terms across all papers
        ranked_mesh_terms = self.expert_profiler.rank_mesh_terms_across_all(input_folder)

        # save the ranked mesh terms 
        ranked_mesh_terms.to_csv(os.path.join(output_folder, "ranked_mesh_terms.csv"), index=True)

        # print top 25 mesh terms
        print("Top 25 MeSH Terms:")
        print(ranked_mesh_terms.head(25))

        # define input and output folders
        input_folder = "../intermediate_data/reviewer_mental_health"
        output_folder = "../intermediate_data/reviewer_seniority"

        # calculate reviewer seniority
        reviewer_seniority = self.expert_profiler.calculate_reviewer_seniority(input_folder)

        # save reviewer seniority 
        save_path = os.path.join(output_folder, 'reviewers_seniority.csv')
        reviewer_seniority.to_csv(save_path, index=False)

        # print descriptive statistics
        print("\nDescriptive Statistics for Reviewer Seniority:")
        print(reviewer_seniority.describe())

        pass