# Example usages:
#
# Run only 'data_loading' and 'project_enrichment' components for a specific call:
# > python main.py --components data_loading project_enrichment --call 2022-Salut_Cardiovascular
#
# Run all components except 'similarity_computation' for a specific call:
# > python main.py --exclude similarity_computation --call 2018-Cancer
#
# Run two specific components ('data_loading', 'expert_ranking') for the default call set in config_general.py
# > python main.py --components data_loading expert_ranking
#
# Run all the components for a specific call
# > python main.py --call 2021-Salut_Mental
#

import argparse
from core.config_handler import ConfigManager
from pipeline.data_processing_pipeline import DataProcessingPipeline

# Define the list of all components
ALL_COMPONENTS = [
    'project_data_loading',
    'expert_data_loading',
    'publication_extraction',
    'pubmed_retrieval',
    'project_enrichment',
    'publication_enrichment',
    'project_classification',
    'project_mesh_tagging',
    'publication_mesh_tagging',
    'similarity_computation',
    'expert_ranking',
    'expert_assignment'
]

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Run data processing pipeline with optional parameters.")
    parser.add_argument(
        '--components', type=str, nargs='+', choices=ALL_COMPONENTS,
        help="Specify which components to run (e.g., 'data_loading project_enrichment')."
    )
    parser.add_argument(
        '--exclude', type=str, nargs='+', choices=ALL_COMPONENTS,
        help="Specify which components to exclude (e.g., 'project_mesh_tagging')."
    )
    parser.add_argument(
        '--call', type=str, help="Specify the name of the call to override the default in config_general."
    )
    parser.add_argument(
        '--test-mode', action='store_true', default=False, help="Run the pipeline in test mode (default: False)."
    )
    parser.add_argument(
        '--test-number', type=int, default=10, help="Specify the number of rows to process in test mode (default: 10)."
    )
    args = parser.parse_args()
      
    # Create a single ConfigManager handling all configuration files.
    config_manager = ConfigManager([
        'configs.config_general',
        'configs.config_llm',
        'configs.config_get_publications'
    ])
    
    # Print all configurations for debugging.
    config_manager.print_all_configs()
        
    # Initialize and run the pipeline.
    pipeline = DataProcessingPipeline(
        config_manager=config_manager,
        call=args.call,
        all_components=ALL_COMPONENTS,
        test_mode=args.test_mode,
        test_number=args.test_number
    )
    pipeline.run_pipeline(components=args.components, exclude=args.exclude)

if __name__ == '__main__':
    main()

