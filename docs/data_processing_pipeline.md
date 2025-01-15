# Data Processing Pipeline Documentation

The `DataProcessingPipeline` is the core module for orchestrating various components involved in processing data, calculating similarity scores, and generating assignments for the Reviewer Matcher system.

---

## Overview

This pipeline integrates multiple stages, including:

- **Data Loading**: Loading and preprocessing data for projects, experts, and publications.
- **Metadata Enrichment**: Adding contextual information like MeSH terms and summaries.
- **Similarity Calculation**: Generating similarity scores between projects and experts.
- **Expert Ranking**: Ranking experts for projects based on multiple criteria.
- **Expert Assignment**: Assigning experts to projects based on rankings and constraints.

---

## Initialization

To initialize the pipeline, you need to pass the following parameters:

- `config_manager`: A configuration manager object for handling settings.
- `call` (optional): Call-specific settings to override defaults.
- `all_components` (optional): A list of components to execute in the pipeline.
- `test_mode` (default: `False`): Run the pipeline in test mode with a reduced dataset.
- `test_number` (default: `10`): Number of rows to process in test mode.
- `force_recompute` (default: `False`): Recompute data even if existing outputs are available.

### Example

```python
from data_processing_pipeline import DataProcessingPipeline
from core.settings_manager import SettingsManager

config_manager = SettingsManager(config_path="path/to/configs")
pipeline = DataProcessingPipeline(config_manager, test_mode=True, test_number=5)
```

---

## Running the Pipeline

### Run All Components

You can run the entire pipeline with all components included:

```python
pipeline.run_pipeline()
```

### Run Specific Components

You can specify a list of components to run, or exclude specific ones:

```python
components_to_run = ["project_classification", "similarity_computation"]
pipeline.run_pipeline(components=components_to_run)
```

Exclude components by using the `exclude` parameter:

```python
components_to_exclude = ["publication_data_loading"]
pipeline.run_pipeline(exclude=components_to_exclude)
```

---

## Available Components

The pipeline supports the following components, which can be run individually:

1. **`project_data_loading`**: Loads project data.
2. **`expert_data_loading`**: Loads expert data.
3. **`publication_data_loading`**: Loads publication data.
4. **`project_classification`**: Classifies projects with research areas and approaches.
5. **`project_summarization`**: Summarizes project content.
6. **`project_mesh_tagging`**: Tags projects with MeSH terms.
7. **`publication_summarization`**: Summarizes publication content.
8. **`publication_mesh_tagging`**: Tags publications with MeSH terms.
9. **`similarity_computation`**: Computes similarity scores between experts and projects.
10. **`expert_ranking`**: Ranks experts based on similarity scores.
11. **`expert_assignment`**: Assigns experts to projects.

---

## Examples

### Example 1: Compute Similarity Scores
Run the pipeline to compute similarity scores between experts and projects:

```python
components_to_run = ["similarity_computation"]
pipeline.run_pipeline(components=components_to_run)
```

### Example 2: Assign Experts
Run the pipeline to assign experts to projects:

```python
components_to_run = ["expert_assignment"]
pipeline.run_pipeline(components=components_to_run)
```

---

## Notes

1. The `test_mode` flag reduces the dataset size for faster execution during testing or debugging.

2. Use the `force_recompute` flag to recompute data even if pre-existing outputs are found.

3. All intermediate outputs are saved to directories specified in the configuration.


---

This documentation provides a comprehensive guide to understanding and using the `DataProcessingPipeline` effectively.
