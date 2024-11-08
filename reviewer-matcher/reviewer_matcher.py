# reviewer_matcher.py

from .data_processor import DataProcessor
from .metadata_enricher import MetadataEnricher
from .content_processor import ContentProcessor
from .expert_profiler import ExpertProfiler
from .similarity_calculator import SimilarityCalculator
from .relevance_predictor import RelevancePredictor
from .panel_optimizer import PanelOptimizer

class ReviewerMatcher:
    def __init__(self):
        self.data_processor = DataProcessor()
        self.metadata_enricher = MetadataEnricher()
        self.content_processor = ContentProcessor()
        self.expert_profiler = ExpertProfiler()
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