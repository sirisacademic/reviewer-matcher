from pipeline.data_processing_pipeline import DataProcessingPipeline

def main():
    # Initialize and run the pipeline
    pipeline = DataProcessingPipeline('configs.config_general')
    pipeline.run_pipeline()

if __name__ == '__main__':
    main()

