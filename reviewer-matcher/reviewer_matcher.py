# reviewer_matcher.py

from .data_processor import DataProcessor
from .metadata_enricher import MetadataEnricher
from .content_processor import ContentProcessor
from .expert_profiler import ExpertProfiler
from .similarity_calculator import SimilarityCalculator
from .relevance_predictor import RelevancePredictor
from .panel_optimizer import PanelOptimizer

import os
import sys 
import pandas as pd

class ReviewerMatcher:
    def __init__(self):
        self.data_processor = DataProcessor()
        self.metadata_enricher = MetadataEnricher()
        self.content_processor = ContentProcessor()
        self.expert_profiler = ExpertProfiler(
            data_path=os.path.join("examples", "example_data", "reviewers.csv"),
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

        # directory to save the enriched data 
        save_path = os.path.join("..", "intermediate_data", "reviewer_enrichment"),

        # enrich data and compute completeness using ORCID numbers and fallback to full names
        completeness_both, reviewers_enriched = self.expert_profiler.compute_completeness_for_method(reviewers, method="both")
        file_path = os.path.join(save_path, 'reviewers_enriched.csv')
        reviewers_enriched.to_csv(file_path, index=False)
        print(f"Completeness (ORCID + Full Name): {completeness_both}%")
        print(reviewers_enriched.head())

        # directory to save the publications
        save_path = os.path.join("..", "intermediate_data", "reviewer_publications")

        # iterate over each reviewer and retrieve their publications
        for index, row in reviewers.iterrows():
            full_name = row['Full Name']
            orcid = row['orcid']
    
            author_name, publications = self.expert_profiler.get_author_publications(full_name=full_name, orcid=orcid)
    
            if author_name and publications:
                filename = f"{author_name.replace(' ', '_').lower()}_publications.csv"
                file_path = os.path.join(save_path, filename)
                self.expert_profiler.save_publications_to_csv(author_name, publications, file_path)
                print(f"Publications for {author_name} saved to {save_path}.")
            else:
                print(f"No publications found for {full_name}.")

        # predict the reviewers' gender 
        reviewers_gender = self.expert_profiler.enrich_data_with_predicted_gender(reviewers_enriched, self.expert_profiler.api_key)

        # directory to save the publications
        save_path = os.path.join("..", "intermediate_data", "reviewer_gender")
        file_path = os.path.join(save_path, 'reviewers_gender.csv')
        reviewers_gender.to_csv(file_path, index=False)

        # define input and output folders
        input_folder = "../intermediate_data/reviewer_publications"
        output_folder = "../intermediate data/reviewer_publications_classified"

        # process all publications
        self.expert_profiler.classify_all_publications(input_folder, output_folder)

        # compute statistics
        research_phase_counts, domain_counts = self.expert_profiler.compute_classification_statistics(output_folder)

        # display statistics
        print("Percentage breakdown of research phases:")
        print(research_phase_counts)

        print("\nPercentage of publications classified in each domain:")
        print(domain_counts)

        # define input and output folders
        input_folder = "../intermediate_data/reviewer_publications_classified"
        output_folder = "../intermediate_data/ranked_mesh_terms"

        # rank mesh terms across all papers
        ranked_mesh_terms = self.expert_profiler.rank_mesh_terms_across_all(input_folder)

        if ranked_mesh_terms is not None:
            # print top 25 mesh terms
            print("Top 25 MeSH Terms:")
            print(ranked_mesh_terms.head(25))

            # save the ranked mesh terms df to a csv file
            ranked_mesh_terms.to_csv(os.path.join(output_folder, "ranked_mesh_terms.csv"), index=True)
        else:
            print("No valid MeSH terms found for TF-IDF vectorization.")

        # define input and output folders
        input_folder = "../intermediate_data/reviewer_publications_classified"
        output_folder = "../intermediate_data/reviewer_seniority"

        # calculate author seniority
        self.expert_profiler.calculate_author_seniority(input_folder, output_folder)

        # load the combined df
        author_seniority_df = pd.read_csv(os.path.join(output_folder, 'author_seniority.csv'))

        # print descriptive statistics for the author seniority df
        print("\nDescriptive Statistics for Author Seniority:")
        print(author_seniority_df.describe())

        # define input and output folders
        input_folder = "../intermediate_data/reviewer_publications_classified"
        output_folder = "../intermediate_data/reviewer_publications_determined"

        # classify publications
        self.expert_profiler.classify_publications(input_folder, output_folder)



        # load and display the combined df
        #output_folder = "../data/07_mental_health_class"
        #classified_files = [pd.read_csv(os.path.join(output_folder, file)) for file in os.listdir(output_folder) if file.endswith('_mental_health.csv')]
        #combined_df_classified = pd.concat(classified_files, ignore_index=True)

        # define the output folder
        #output_combined_folder = "../data"

        # define the path for the combined csv file
        #combined_csv_path = os.path.join(output_combined_folder, "author_characterization.csv")

        # save the combined df to a csv file
        #combined_df_classified.to_csv(combined_csv_path, index=False)
        #print(f"Combined DataFrame saved to {combined_csv_path}\n")

        # compute the proportion of mental health related papers
        #total_papers = len(combined_df_classified)
        #mental_health_papers = combined_df_classified['mental_health_class'].sum() 

        #if total_papers > 0:
            #proportion_mental_health = mental_health_papers / total_papers
        #else:
            #proportion_mental_health = 0

        # print the proportion of mental health realted papers
        #print(f"Total papers: {total_papers}")
        #print(f"Mental health-related papers: {mental_health_papers}")
        #print(f"Proportion of mental health-related papers: {proportion_mental_health:.2f}")



        pass