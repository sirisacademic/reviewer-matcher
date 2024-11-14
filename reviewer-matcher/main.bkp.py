### main.py
#
# Example usages:
#
# Run only the 'projects' component with a specific call:
# > python main.py --components projects --call 2022-Salut_Cardiovascular
#
# Run all components except 'experts' for a specific call:
# > python main.py --exclude experts --call 2018-Cancer
#
# Run two components ('projects', 'experts') for the default call set in config_general.py
# > python main.py --components projects experts
#
# Run all the components for a specific call
# > python main.py --call 2021-Salut_Mental
#
###

import argparse
from pipeline.data_processing_pipeline import DataProcessingPipeline

def main():

    # Parse arguments
    parser = argparse.ArgumentParser(description="Run data processing pipeline with optional parameters.")
    parser.add_argument(
        '--components', type=str, nargs='+', choices=['projects', 'experts', 'publications'],
        help="Specify which components to run (e.g., 'projects experts')."
    )
    parser.add_argument(
        '--exclude', type=str, nargs='+', choices=['projects', 'experts', 'publications'],
        help="Specify which components to exclude (e.g., 'projects')."
    )
    parser.add_argument(
        '--call', type=str, help="Specify the name of the call to override the default in config_general."
    )
    args = parser.parse_args()

    # Initialize and run the pipeline
    pipeline = DataProcessingPipeline('configs.config_general', call=args.call)
    pipeline.run_pipeline(components=args.components, exclude=args.exclude)

if __name__ == '__main__':
    main()

