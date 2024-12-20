import pandas as pd
import numpy as np
import re

class ExpertSeniorityCalculator:
    def __init__(self, config_manager):
        """Initialize the ExpertSeniorityCalculator with configuration settings."""
        # Seniority levels
        self.seniority_undetermined = config_manager.get('SENIORITY_UNDETERMINED', None)
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
        # Output columns.
        self.col_seniority_publications = config_manager.get('COLUMN_SENIORITY_PUBLICATIONS', 'SENIORITY_PUBLICATIONS')
        self.col_seniority_reviewer = config_manager.get('COLUMN_SENIORITY_REVIEWER', 'SENIORITY_REVIEWER')

    def enrich_with_seniority(self, experts):
        """Enrich the dataframe with publication and reviewer expertise scores."""
        # Preprocess the data
        experts = self.preprocess_data(experts)
        # Calculate thresholds
        pub_threshold, cits_threshold = self.calculate_thresholds(experts)
        # Determine seniority and add the SENIORITY columns for publicatoins and reviewer expertise.
        experts[self.col_seniority_publications] = experts.apply(
            lambda row: self._evaluate_publication_expertise(row, pub_threshold, cits_threshold),
            axis=1
        )
        experts[self.col_seniority_reviewer] = experts.apply(
            lambda row: self._evaluate_reviewer_expertise(row),
            axis=1
        )
        return experts

    def preprocess_data(self, experts):
        """Preprocess the dataframe by converting relevant columns to numeric values."""
        if self.col_num_pubs in experts:
            experts[self.col_num_pubs] = experts[self.col_num_pubs].apply(self._extract_number)
        if self.col_num_cits in experts:
            experts[self.col_num_cits] = experts[self.col_num_cits].apply(self._extract_number)
        return experts

    def calculate_thresholds(self, experts):
        """Calculate thresholds for publications and citations based on percentile values."""
        pub_threshold = np.percentile(experts[self.col_num_pubs], self.num_pubs_top_perc) if self.col_num_pubs in experts else None
        cits_threshold = np.percentile(experts[self.col_num_cits], self.num_cits_top_perc) if self.col_num_cits in experts else None
        return pub_threshold, cits_threshold

    def _evaluate_publication_expertise(self, row, pub_threshold, cits_threshold):
        """Evaluate the expertise of the expert based on publications and citations."""
        # Initialize the expertise level to undetermined
        publications_expertise = self.seniority_undetermined
        # Get the values for publications and citations, defaulting to 0 if the column is not present
        pubs_value = row.get(self.col_num_pubs, 0)
        cits_value = row.get(self.col_num_cits, 0)
        # If both values are 0, return undetermined
        if pubs_value == 0 and cits_value == 0:
            return self.seniority_undetermined
        # If both thresholds are available
        if pub_threshold is not None and cits_threshold is not None:
            # If both values are above or equal to their respective thresholds, return high
            if pubs_value >= pub_threshold and cits_value >= cits_threshold:
                publications_expertise = self.seniority_high
            # If one value is above or equal to its threshold and the other is below, return middle
            elif pubs_value >= pub_threshold or cits_value >= cits_threshold:
                publications_expertise = self.seniority_middle
            # If both values are below their respective thresholds, return low
            else:
                publications_expertise = self.seniority_low
        # If only the publication threshold is available
        elif pub_threshold is not None:
            # If the publication value is above or equal to its threshold, return high
            if pubs_value >= pub_threshold:
                publications_expertise = self.seniority_high
            # If the publication value is below its threshold but greater than 0, return low
            elif pubs_value > 0:
                publications_expertise = self.seniority_low
        # If only the citation threshold is available
        elif cits_threshold is not None:
            # If the citation value is above or equal to its threshold, return high
            if cits_value >= cits_threshold:
                publications_expertise = self.seniority_high
            # If the citation value is below its threshold but greater than 0, return low
            elif cits_value > 0:
                publications_expertise = self.seniority_low
        # Return the determined expertise level
        return publications_expertise

    def _evaluate_reviewer_expertise(self, row):
        """Evaluate the expertise of the expert based on experience as a reviewer or panelist."""
        reviewer_expertise = self.seniority_undetermined
        # Check if the expert has reviewer experience
        has_reviewer_experience = row.get(self.col_exp_reviewer, 'no') == 'yes'
        # Check if the expert has panel experience
        has_panel_experience = row.get(self.col_exp_panel, 'no') == 'yes'
        # If both columns are missing, return undetermined
        if self.col_exp_reviewer not in row and self.col_exp_panel not in row:
            return self.seniority_undetermined
        # If the expert has both reviewer and panel experience, return high
        if has_reviewer_experience and has_panel_experience:
            reviewer_expertise = self.seniority_high
        # If the expert has either reviewer or panel experience, return middle
        elif has_reviewer_experience or has_panel_experience:
            reviewer_expertise = self.seniority_middle
        # Otherwise, return low
        else:
            reviewer_expertise = self.seniority_low
        return reviewer_expertise

    @staticmethod
    def _extract_number(value):
        """Extract the first valid numeric value from a given input."""
        if isinstance(value, int):
            return value
        if pd.isna(value) or not isinstance(value, str) or value.strip() == '':
            return 0
        numbers = re.findall(r'\b\d{1,10}(?:,\d{3})*|\d+\b', value)
        return max([int(num.replace(',', '')) for num in numbers], default=0)

