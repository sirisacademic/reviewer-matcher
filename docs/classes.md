# Class Reference

## Key Classes Used in the Pipeline

1. **DataReader**: Handles data loading and preprocessing for projects, experts, and publications.
   - `load_data`: Loads and preprocesses data from specified sources.

2. **ContentSummarizer**: Extracts and summarizes the key content of projects and publications.
   - `summarize_content`: Summarizes abstracts, objectives, and research methods.

3. **MeSHLabeler**: Tags projects and publications with MeSH terms.
   - `label_with_mesh`: Adds MeSH terms to the specified input columns of projects or publications.

4. **MeSHSimilarityCalculator**: Computes similarity scores based on MeSH terms.
   - `compute_similarity`: Calculates MeSH-based similarity scores for projects and publications.

5. **LabelSimilarityCalculator**: Compares research topics and approaches.
   - `compute_similarity`: Calculates similarity scores based on research labels.

6. **ContentSimilarityCalculator**: Computes similarity scores between projects and publications based on content.
   - `compute_similarity`: Calculates similarity using content-based metrics.

7. **ResearchTypeSimilarityCalculator**: Computes similarity scores for research types.
   - `compute_similarity`: Calculates similarity between expert and project research types.

8. **FeatureGenerator**: Generates features for ranking expert-project pairs.
   - `generate_features`: Combines similarity scores into a feature set for ranking.

9. **ExpertRanker**: Predicts rankings for expert-project pairs.
   - `generate_predictions`: Produces probabilities and rankings for assignments.

10. **ExpertAssigner**: Finalizes expert-project assignments.
    - `generate_assignments`: Assigns reviewers based on similarity and constraints.

11. **PublicationHandler**: Manages retrieval and processing of publication data.
    - `get_publications_experts`: Retrieves publications associated with experts.

12. **ResearchLabeler**: Assigns research topics and approaches to projects.
    - `label_topics`: Labels projects with research areas and approaches.

13. **DataSaver**: Manages saving of intermediate and final results.
    - `save_data`: Saves data to various formats like CSV, TSV, and Pickle.

---

## Usage Examples

### Example 1: Run the Full Matching Process

```python
from data_processing_pipeline import DataProcessingPipeline
from core.settings_manager import SettingsManager

# Initialize configuration manager
config_manager = SettingsManager(config_path="configs/")

# Initialize pipeline
pipeline = DataProcessingPipeline(config_manager)

# Run all components
pipeline.run_pipeline()
```

### Example 2: Summarize Projects and Publications

```python
from data_processing_pipeline import DataProcessingPipeline
from core.settings_manager import SettingsManager

# Initialize configuration manager
config_manager = SettingsManager(config_path="configs/")

# Initialize pipeline
pipeline = DataProcessingPipeline(config_manager)

# Run project summarization
pipeline.run_pipeline(components=["project_summarization"])

# Run publication summarization
pipeline.run_pipeline(components=["publication_summarization"])
```

### Example 3: Compute Similarity Scores and Assign Reviewers

```python
from data_processing_pipeline import DataProcessingPipeline
from core.settings_manager import SettingsManager

# Initialize configuration manager
config_manager = SettingsManager(config_path="configs/")

# Initialize pipeline
pipeline = DataProcessingPipeline(config_manager)

# Compute similarity scores
pipeline.run_pipeline(components=["similarity_computation"])

# Assign reviewers based on computed scores
pipeline.run_pipeline(components=["expert_assignment"])
```


