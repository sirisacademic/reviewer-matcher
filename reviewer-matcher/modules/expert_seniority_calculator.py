import pandas as pd
import numpy as np
import re


class ExpertSeniorityCalculator:
    def __init__(self, config_manager):
        """Initialize the ExpertSeniorityCalculator with configuration settings."""
        # Seniority levels
        self.seniority_undetermined = config_manager.get('SENIORITY_UNDETERMINED', 0)
        self.seniority_low = config_manager.get('SENIORITY_LOW', 1)
        self.seniority_middle = config_manager.get('SENIORITY_MIDDLE', 2)
        self.seniority_high = config_manager.get('SENIORITY_HIGH', 3)
        # Percentile thresholds
        self.num_pubs_top_perc = config_manager.get('NUM_PUBS_TOP_PERC_SENIORITY', 30)
        self.num_cits_top_perc = config_manager.get('NUM_CITATIONS_TOP_PERC_SENIORITY', 30)
        # Column names from configuration
        self.col_num_pubs = config_manager.get('COLUMN_NUMBER_PUBLICATIONS', 'NUMBER_PUBLICATIONS')
        self.col_num_cits = config_manager.get('COLUMN_NUMBER_CITATIONS', 'NUMBER_CITATIONS')
        self.col_exp_reviewer = config_manager.get('COLUMN_EXPERIENCE_REVIEWER', 'EXPERIENCE_REVIEWER')
        self.col_exp_panel = config_manager.get('COLUMN_EXPERIENCE_PANEL', 'EXPERIENCE_PANEL')
        self.col_seniority = config_manager.get('COLUMN_SENIORITY', 'SENIORITY')

    def enrich_with_seniority(self, experts):
        """Enrich the dataframe with the seniority level by adding a SENIORITY column."""
        # Preprocess the data
        experts = self.preprocess_data(experts)
        # Calculate thresholds
        pub_threshold, cits_threshold = self.calculate_thresholds(experts)
        # Determine seniority and add the SENIORITY column
        experts = self.determine_seniority(experts, pub_threshold, cits_threshold)
        return experts

    def preprocess_data(self, experts):
        """Preprocess the dataframe by converting relevant columns to numeric values."""
        experts[self.col_num_pubs] = experts[self.col_num_pubs].apply(self._extract_number)
        experts[self.col_num_cits] = experts[self.col_num_cits].apply(self._extract_number)
        return experts

    def calculate_thresholds(self, experts):
        """Calculate thresholds for publications and citations based on percentile values."""
        pub_threshold = np.percentile(experts[self.col_num_pubs], self.num_pubs_top_perc)
        cits_threshold = np.percentile(experts[self.col_num_cits], self.num_cits_top_perc)
        return pub_threshold, cits_threshold

    def determine_seniority(self, experts, pub_threshold, cits_threshold):
        """Compute seniority levels for all experts based on thresholds and experience."""
        experts[self.col_seniority] = experts.apply(
            lambda row: self.seniority_logic(row, pub_threshold, cits_threshold), axis=1
        )
        return experts

    def seniority_logic(self, row, pub_threshold, cits_threshold):
        """Logic to determine the seniority level for a single expert."""
        # If the expert has no publications and no citations.
        if row[self.col_num_pubs] == 0 and row[self.col_num_cits] == 0:
            # Assign seniority as undetermined if reviewer or panel experience exists.
            if row[self.col_exp_reviewer] == 'yes' or row[self.col_exp_panel] == 'yes':
                return self.seniority_undetermined
            # Otherwise, assign seniority as low
            return self.seniority_low
        # High seniority if publications and reviewer experience are both strong even if other factors are moderate.
        elif row[self.col_num_pubs] >= pub_threshold and row[self.col_exp_reviewer] == 'yes':
            return self.seniority_high
        # Middle seniority if publications or reviewer experience is strong, but not both, or other experience factors are met.
        elif row[self.col_num_pubs] >= pub_threshold or row[self.col_exp_reviewer] == 'yes':
            # Assign middle seniority if panel experience or high citation count exists.
            if row[self.col_exp_panel] == 'yes' or row[self.col_num_cits] >= cits_threshold:
                return self.seniority_middle
            # Otherwise, assign seniority as low.
            return self.seniority_low
        # Middle seniority if both reviewer and panel experiences are positive even if publications and citations are not strong.
        elif row[self.col_exp_reviewer] == 'yes' and row[self.col_exp_panel] == 'yes':
            return self.seniority_middle
        # Low seniority if none of the above apply
        return self.seniority_low

    @staticmethod
    def _extract_number(value):
        """Extract the first valid numeric value from a given input."""
        if isinstance(value, int):
            return value
        if pd.isna(value) or not isinstance(value, str) or value.strip() == '':
            return 0
        numbers = re.findall(r'\b\d{1,10}(?:,\d{3})*|\d+\b', value)
        return max([int(num.replace(',', '')) for num in numbers], default=0)



