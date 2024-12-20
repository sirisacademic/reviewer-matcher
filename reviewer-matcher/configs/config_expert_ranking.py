# Output file.
FILE_EXPERT_PROJECT_FEATURES = 'expert_project_features.tsv'

EXPERT_ID_COLUMN = 'Expert_ID'
PROJECT_ID_COLUMN = 'Project_ID'

FEATURE_GROUPS = {
    'MeSH': [
        'Expert_MeSH_Max_Similarity_Max',
        'Expert_MeSH_Max_Similarity_Avg',
        'Expert_MeSH_Avg_Similarity_Max',
        'Expert_MeSH_Avg_Similarity_Avg',
        'Expert_MeSH_Max_Similarity_Weighted_Max',
        'Expert_MeSH_Max_Similarity_Weighted_Avg',
        'Expert_MeSH_Avg_Similarity_Weighted_Max',
        'Expert_MeSH_Avg_Similarity_Weighted_Avg'
    ],
    'Topic': [
        'Expert_Topic_Similarity_Max',
        'Expert_Topic_Similarity_Avg'
    ],
    'Objectives': [
        'Expert_Objectives_Max_Similarity_Max',
        'Expert_Objectives_Max_Similarity_Avg',
        'Expert_Objectives_Avg_Similarity_Max',
        'Expert_Objectives_Avg_Similarity_Avg'
    ],
    'Methods_Specific': [
       'Expert_Methods_Specific_Max_Similarity_Max',
       'Expert_Methods_Specific_Max_Similarity_Avg',
       'Expert_Methods_Specific_Avg_Similarity_Max',
       'Expert_Methods_Specific_Avg_Similarity_Avg'
    ],
    'Methods': [
        'Expert_Methods_Max_Similarity_Max',
        'Expert_Methods_Max_Similarity_Avg',
        'Expert_Methods_Avg_Similarity_Max',
        'Expert_Methods_Avg_Similarity_Avg',
        'Expert_Methods_Max_Similarity_Weighted_Max',
        'Expert_Methods_Max_Similarity_Weighted_Avg',
        'Expert_Methods_Avg_Similarity_Weighted_Max',
        'Expert_Methods_Avg_Similarity_Weighted_Avg'
    ]
}



