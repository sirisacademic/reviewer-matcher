# File: main.py
#
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

import os
import sys
import datetime

# Whether to redirect standard/error output to files.
REDIRECT_OUTPUT = False

# Define the list of all components
ALL_COMPONENTS = [
    'project_data_loading',
    'expert_data_loading',
    'publication_data_loading',
    'project_classification',
    'project_summarization',
    'project_mesh_tagging',
    'publication_summarization',
    'publication_mesh_tagging',
    'similarity_computation',
    'expert_ranking',
    'expert_assignment'
]

def main():

    import argparse
    from core.config_handler import ConfigManager
    from pipeline.data_processing_pipeline import DataProcessingPipeline

    try:
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
        parser.add_argument(
            '--force-recompute', action='store_true', default=False, help="Force recomputing existing data (default: False)."
        )
        args = parser.parse_args()
        
        # Create a single ConfigManager handling all configuration files.
        config_manager = ConfigManager([
            'configs.config_general',
            'configs.config_llm',
            'configs.config_get_publications',
            'configs.config_similarity_scores',
            'configs.config_expert_profiler',
            'configs.config_expert_ranking',
            'train_eval.config'
        ])
        
        # Print all configurations for debugging.
        config_manager.print_all_configs()
        
        # Initialize and run the pipeline.
        pipeline = DataProcessingPipeline(
            config_manager=config_manager,
            call=args.call,
            all_components=ALL_COMPONENTS,
            test_mode=args.test_mode,
            test_number=args.test_number,
            force_recompute=args.force_recompute
        )
        pipeline.run_pipeline(components=args.components, exclude=args.exclude)

    except Exception as e:
        # Log any exception that occurs
        print(f"An error occurred: {e}")

def redirect_output():
    # Get current date for the log filenames
    current_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    stdout_log_file = f"output_stdout_{current_date}.log"
    stderr_log_file = f"output_stderr_{current_date}.log"
    print(f'Redirecting standard output to {stdout_log_file}', flush=True)
    print(f'Redirecting standard error to {stderr_log_file}', flush=True)
    # Open log files
    stdout_log = open(stdout_log_file, "w", buffering=1)  # Line-buffered
    stderr_log = open(stderr_log_file, "w", buffering=1)  # Line-buffered
    # Redirect low-level system output
    os.dup2(stdout_log.fileno(), 1)  # Redirect stdout
    os.dup2(stderr_log.fileno(), 2)  # Redirect stderr
    print(f"Log files created: {stdout_log_file}, {stderr_log_file}", flush=True)
    return stdout_log, stderr_log

def close_redirections(stdout_log, stderr_log):
    # Close log files and restore stdout/stderr
    stdout_log.close()
    stderr_log.close()
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__  
    
if __name__ == '__main__':

    if REDIRECT_OUTPUT:
        stdout_log, stderr_log = redirect_output()
    # Execute main.
    main()
    if REDIRECT_OUTPUT:
        close_redirections(stdout_log, stderr_log)
        

