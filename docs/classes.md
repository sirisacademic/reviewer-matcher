# Class Reference

## Key Classes

1. **DataReader**: Handles data loading and preprocessing.
   - `load_data`: Loads and preprocesses expert and project data.

2. **ContentSummarizer**: Summarizes project abstracts and extracts key content.
   - `summarize_content`: Extracts and summarizes project content.

3. **SimilarityCalculator**: Computes similarity scores between projects and experts.
   - `calculate_similarity`: Calculates similarity based on predefined metrics.

4. **ExpertAssigner**: Optimizes reviewer assignments.
   - `generate_assignments`: Assigns reviewers based on similarity scores.

5. **ExpertProfiler**: Enriches expert profiles with additional metadata.
   - `create_profile`: Generates expert profiles from raw data.

6. **MetadataEnricher**: Adds enriched metadata to projects and experts.
   - `add_mesh_terms`: Associates MeSH terms with project abstracts.
   
## Usage Examples

### Example: Basic Matching Process

```python
from expert_matcher.data_reader import DataReader
from expert_matcher.content_summarizer import ContentSummarizer
from expert_matcher.similarity_calculator import SimilarityCalculator
from expert_matcher.expert_assigner import ExpertAssigner

# Load data
data_reader = DataReader(config_manager)
project_data = data_reader.load_data()

# Summarize project content
content_summarizer = ContentSummarizer(config_manager)
project_summary = content_summarizer.summarize_content(project_data)

# Calculate similarity
similarity_calculator = SimilarityCalculator(config_manager)
similarity_scores = similarity_calculator.calculate_similarity(project_summary)

# Assign reviewers
expert_assigner = ExpertAssigner(config_manager)
assignments = expert_assigner.generate_assignments(similarity_scores)
print(assignments)
```


