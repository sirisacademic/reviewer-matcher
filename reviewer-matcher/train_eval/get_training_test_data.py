# File: get_training_test_data.py

import os
import pandas as pd
from config import (
    CALLS,
    CALL_PATHS,
    TRAIN_EVAL_DATA_PATH,
    EXPERTS_FILE,
    FINAL_ANNOTATIONS_FILE,
    EXPERT_PROJECT_SCORES_FILE,
    MANUAL_ASSIGNMENTS_NOT_TRAINING_FILE,
    MANUAL_ANNOTATIONS_FILE,
    MANUAL_ASSIGNMENTS_FILE,
    TRAIN_EVAL_DATA_FILE,
    ANNOTATION_POSITIVE,
    ANNOTATION_NEGATIVE,
    COLUMN_TASK_GOLD,
    USE_SUBSET_FINAL_ASSIGNMENTS,
    PREFIX
)

def process_annotations():

    # Initialize empty dataframes to accumulate results
    combined_features = pd.DataFrame()
    combined_data_train_eval = pd.DataFrame()
    combined_final_annotations = pd.DataFrame()
    combined_manual_assignments_not_training = pd.DataFrame()

    for call in CALLS:
        year = call.split('-')[0]
        paths = CALL_PATHS[call]

        # Read input data.
        
        ### Experts
        experts = pd.read_pickle(f'{paths["data"]}/{EXPERTS_FILE}')
        #print(experts.dtypes)
        # Clean 'GENDER' data.
        experts['GENDER'] = experts['GENDER'].fillna('missing')
        experts['GENDER'] = experts['GENDER'].map(lambda x: x if x in ['male', 'female'] else 'missing')
        # Encode Gender to 1 (male), 1.5 (missing), 2 (female)
        gender_mapping = {'male': 1, 'missing': 1.5, 'female': 2}
        experts['GENDER'] = experts['GENDER'].map(gender_mapping)

        ### Features
        features = pd.read_csv(f'{paths["scores"]}/{EXPERT_PROJECT_SCORES_FILE}', sep='\t')
        #print(features.dtypes)
        # Add gender column. TODO: Include it when generating features.
        features = features.merge(
            experts[['ID', 'GENDER']],
            left_on='Expert_ID',
            right_on='ID',
            how='inner'
        )
        features.drop(columns=['ID'], inplace=True)
        features.rename(columns={'GENDER': 'Gender'}, inplace=True)
        # Prepend year to IDs.
        features['Project_ID'] = features['Project_ID'].apply(lambda x: f'{year}_{x}')
        features['Expert_ID'] = features['Expert_ID'].apply(lambda x: f'{year}_{x}')

        ### Manual annotations
        manual_annotations = pd.read_csv(f'{paths["data"]}/{MANUAL_ANNOTATIONS_FILE}', sep='\t')
        
        # Add values for columns not present in manual annotations.
        # Annotations were generated excluding the manual assignments, therefore none is an assignment and even less so a final assignment.
        manual_annotations['Final_Assignment'] = 0
        manual_annotations = manual_annotations.merge(
            experts[['ID', 'GENDER']],
            left_on='Expert_ID',
            right_on='ID',
            how='inner'
        )[['Project_ID', 'Expert_ID', 'Annotation', 'Final_Assignment', 'GENDER']]
        
        # Rename the GENDER column to Gender.
        manual_annotations.rename(columns={'GENDER': 'Gender'}, inplace=True)
        
        # Preprend year to identifiers.
        manual_annotations['Project_ID'] = manual_annotations['Project_ID'].apply(lambda x: f'{year}_{x}')
        manual_annotations['Expert_ID'] = manual_annotations['Expert_ID'].apply(lambda x: f'{year}_{x}')
        
        # Filter out annotations for which there was no consensus or that are not valid values.
        filtered_annotations = manual_annotations[manual_annotations[COLUMN_TASK_GOLD].isin([ANNOTATION_POSITIVE, ANNOTATION_NEGATIVE])]
        
        ### Manually assigned pairs.
        manual_assignments = pd.read_csv(f'{paths["data"]}/{MANUAL_ASSIGNMENTS_FILE}', sep='\t')

        # Set negatives as 0.
        manual_assignments['Final_Assignment'] = manual_assignments['Final_Assignment'].fillna(0).astype(int)
        if USE_SUBSET_FINAL_ASSIGNMENTS:
            manual_assignments = manual_assignments[manual_assignments['Final_Assignment']==1]
        # Preprend year to project identifiers.
        manual_assignments['Project_ID'] = manual_assignments['Project_ID'].apply(lambda x: f'{year}_{x}')
        # Adding expert IDs to manual assignments based on expert names (case insensitive).
        manual_assignments_with_ids = manual_assignments.merge(
            experts[['ID', 'GENDER', 'FULL_NAME']],
            left_on=manual_assignments['Expert_Full_Name'].str.lower(),
            right_on=experts['FULL_NAME'].str.lower(),
            how='inner'
        )[['Project_ID', 'ID', 'Final_Assignment', 'GENDER']]
        # Rename columns for clarity.
        manual_assignments_with_ids.columns = ['Project_ID', 'Expert_ID', 'Final_Assignment', 'Gender']
        # Prepend year to expert IDs.
        manual_assignments_with_ids['Expert_ID'] = manual_assignments_with_ids['Expert_ID'].apply(lambda x: f'{year}_{x}')

        ### Add as positive annotations the manual assignments for the projects considered.
        manual_assignments_with_ids[COLUMN_TASK_GOLD] = ANNOTATION_POSITIVE
        subset_manual_assignments_projects = manual_assignments_with_ids[
            manual_assignments_with_ids['Project_ID'].isin(filtered_annotations['Project_ID'])
        ]

        ### Get final annotations combining the manually assigned pairs and the annotated data.
        final_annotations = pd.concat([subset_manual_assignments_projects, filtered_annotations], ignore_index=True)

        ### Add features to final annotations to get the training/eval data.
        # Remove the duplicate 'Gender' column when merging to make sure we keep only one.
        columns_in_both = ['Gender']
        data_train_eval = final_annotations.merge(features.drop(columns=columns_in_both), on=['Expert_ID', 'Project_ID'], how='inner')

        ### Get manual assignments for projects not used for training. We will use these for evaluation.
        manual_assignments_not_training = manual_assignments_with_ids[
            ~manual_assignments_with_ids['Project_ID'].isin(filtered_annotations['Project_ID'])
        ]

        # Save output files for each call.
        os.makedirs(paths["data"], exist_ok=True)
        prefix = f'{PREFIX}_subset_top_reviewers_' if USE_SUBSET_FINAL_ASSIGNMENTS else PREFIX

        data_train_eval_path = f'{paths["data"]}/{prefix}{TRAIN_EVAL_DATA_FILE}'
        final_annotations_path = f'{paths["data"]}/{prefix}{FINAL_ANNOTATIONS_FILE}'
        manual_assignments_not_training_path = f'{paths["data"]}/{prefix}{MANUAL_ASSIGNMENTS_NOT_TRAINING_FILE}'

        print(f'{call}: Training data containing {len(data_train_eval_path)} pairs saved to file {data_train_eval_path}')
        data_train_eval.to_csv(data_train_eval_path, sep='\t', index=False)
        print(f'{call}: Annotations used for training containing {len(final_annotations)} pairs saved to file {final_annotations_path}')
        final_annotations.to_csv(final_annotations_path, sep='\t', index=False)
        print(f'{call}: Manual assignments of projects not used for training containing {len(manual_assignments_not_training)} pairs saved to file {manual_assignments_not_training_path}')
        manual_assignments_not_training.to_csv(manual_assignments_not_training_path, sep='\t', index=False)

        # Accumulate results
        combined_features = pd.concat([combined_features, features], ignore_index=True)
        combined_data_train_eval = pd.concat([combined_data_train_eval, data_train_eval], ignore_index=True)
        combined_final_annotations = pd.concat([combined_final_annotations, final_annotations], ignore_index=True)
        combined_manual_assignments_not_training = pd.concat([combined_manual_assignments_not_training, manual_assignments_not_training], ignore_index=True)

    # Save to output files
    os.makedirs(TRAIN_EVAL_DATA_PATH, exist_ok=True)

    features_path = f'{TRAIN_EVAL_DATA_PATH}/{EXPERT_PROJECT_SCORES_FILE}'
    data_train_eval_path = f'{TRAIN_EVAL_DATA_PATH}/{prefix}{TRAIN_EVAL_DATA_FILE}'
    final_annotations_path = f'{TRAIN_EVAL_DATA_PATH}/{prefix}{FINAL_ANNOTATIONS_FILE}'
    manual_assignments_not_training_path = f'{TRAIN_EVAL_DATA_PATH}/{prefix}{MANUAL_ASSIGNMENTS_NOT_TRAINING_FILE}'

    print(f'All features containing {len(combined_features)} pairs saved to file {features_path}')
    combined_features.to_csv(features_path, sep='\t', index=False)
    print(f'Training data containing {len(combined_data_train_eval)} pairs saved to file {data_train_eval_path}')
    combined_data_train_eval.to_csv(data_train_eval_path, sep='\t', index=False)
    print(f'Combined annotations used for training containing {len(combined_final_annotations)} pairs saved to file {final_annotations_path}')
    combined_final_annotations.to_csv(final_annotations_path, sep='\t', index=False)
    print(f'Combined manual assignments of projects not used for training containing {len(combined_manual_assignments_not_training)} pairs saved to file {manual_assignments_not_training_path}')
    combined_manual_assignments_not_training.to_csv(manual_assignments_not_training_path, sep='\t', index=False)

    missing_rows = final_annotations[~final_annotations.set_index(['Expert_ID', 'Project_ID']).index.isin(features.set_index(['Expert_ID', 'Project_ID']).index)]
    print(f"Number of rows in final_annotations missing from features: {len(missing_rows)}")
    print("Missing rows in final_annotations:")
    print(missing_rows)

if __name__ == '__main__':
    process_annotations()


