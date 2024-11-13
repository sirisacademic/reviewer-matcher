# reviewer_matcher.py

from .data_processor import DataProcessor
from .metadata_enricher import MetadataEnricher
from .content_processor import ContentProcessor
from .expert_profiler import ExpertProfiler
from .similarity_calculator import SimilarityCalculator
from .relevance_predictor import RelevancePredictor
from .panel_optimizer import PanelOptimizer

import os

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
        pass