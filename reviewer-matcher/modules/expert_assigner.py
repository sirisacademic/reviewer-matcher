import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Set

@dataclass
class Assignment:
    project_id: int
    proposed_experts: List[int]
    alternative_experts: List[int]

class ExpertAssigner:
    def __init__(self, config_manager):
        """Initialize ExpertAssigner with configuration settings."""
        # Column mappings
        self.project_cols = config_manager.get('OUTPUT_COLUMNS_PROJECTS')
        self.expert_cols = config_manager.get('OUTPUT_COLUMNS_EXPERTS')
        # Input columns
        self.expert_id_input_col = config_manager.get('EXPERT_ID_INPUT_COLUMN', 'ID')
        self.expert_gender_input_col = config_manager.get('EXPERT_GENDER_INPUT_COLUMN', 'GENDER')
        self.expert_max_projects_input_col = config_manager.get('EXPERT_MAX_PROJECTS_INPUT_COLUMN', 'MAX_PROJECTS_REVIEW')
        self.expert_name_input_col = config_manager.get('EXPERT_NAME_INPUT_COLUMN', 'FULL_NAME')
        self.expert_research_types_input_col = config_manager.get('EXPERT_RESEARCH_TYPES_INPUT_COLUMN', 'RESEARCH_TYPES')
        self.project_id_input_col = config_manager.get('PROJECT_ID_INPUT_COLUMN', 'ID')
        self.project_title_input_col = config_manager.get('PROJECT_TITLE_INPUT_COLUMN', 'TITLE')
        self.project_research_types_input_col = config_manager.get('PROJECT_RESEARCH_TYPES_INPUT_COLUMN', 'RESEARCH_TYPE')
        self.predicted_prob_col = config_manager.get('PREDICTED_PROB_COLUMN', 'Predicted_Prob')
        self.predicted_prob_rank_col = config_manager.get('PREDICTED_PROB_RANK_COLUMN', 'Predicted_Prob_Rank')
        # Output columns
        self.expert_id_output_col = self.expert_cols[self.expert_id_input_col]
        self.project_id_output_col = self.project_cols[self.project_id_input_col]
        self.expert_gender_output_col = self.expert_cols[self.expert_gender_input_col]
        self.expert_max_projects_output_col = self.expert_cols[self.expert_max_projects_input_col]
        self.expert_name_output_col = self.expert_cols[self.expert_name_input_col]
        self.expert_research_types_output_col = self.expert_cols[self.expert_research_types_input_col]
        self.project_title_output_col = self.project_cols[self.project_title_input_col]
        self.project_research_types_output_col = self.project_cols[self.project_research_types_input_col]
        # Value used to identify genders
        self.expert_gender_value_women = config_manager.get('EXPERT_GENDER_VALUE_WOMEN', 'female')
        self.expert_gender_value_men = config_manager.get('EXPERT_GENDER_VALUE_MEN', 'male')
        # Assignment configuration
        self.num_proposed_experts = config_manager.get('NUM_PROPOSED_EXPERTS', 3)
        self.num_alternative_experts = config_manager.get('NUM_ALTERNATIVE_EXPERTS', 5)
        self.min_women_proposed = config_manager.get('MIN_WOMEN_PROPOSED', 1)
        self.min_women_alternative = config_manager.get('MIN_WOMEN_ALTERNATIVE', 2)
        self.max_default_projects_per_expert = config_manager.get('MAX_DEFAULT_PROJECTS_PER_EXPERT', 5)
        # Track assignments
        self.expert_assignment_count = {}
        # Min. probability threshold for expert-project pairs.
        self.min_probability_threshold = config_manager.get('MIN_PROBABILITY_THRESHOLD', 0.5)

    def generate_assignments(self, ranked_pairs_df: pd.DataFrame, experts_df: pd.DataFrame, projects_df: pd.DataFrame) -> pd.DataFrame:
        """Generate expert assignments for all projects."""
        try:
            # Data type conversion and validation
            ranked_pairs_df[self.project_id_output_col] = ranked_pairs_df[self.project_id_output_col].astype(int)
            ranked_pairs_df[self.expert_id_output_col] = ranked_pairs_df[self.expert_id_output_col].astype(int)
            ranked_pairs_df[self.predicted_prob_rank_col] = ranked_pairs_df[self.predicted_prob_rank_col].astype(int)
            projects_df[self.project_id_input_col] = projects_df[self.project_id_input_col].astype(int)
            experts_df[self.expert_id_input_col] = experts_df[self.expert_id_input_col].astype(int)
            # Handle max projects per expert
            if self.expert_max_projects_input_col not in experts_df.columns:
                experts_df[self.expert_max_projects_input_col] = self.max_default_projects_per_expert
            else:
                mask = (experts_df[self.expert_max_projects_input_col].isna()) | (experts_df[self.expert_max_projects_input_col] == 0)
                if mask.any():
                    experts_df.loc[mask, self.expert_max_projects_input_col] = self.max_default_projects_per_expert
            # Create expert info lookup without project-specific data
            expert_info_df = experts_df[[
                self.expert_id_input_col,
                self.expert_gender_input_col,
                self.expert_max_projects_input_col,
                self.expert_name_input_col,
                self.expert_research_types_input_col
            ]].rename(columns={
                self.expert_id_input_col: self.expert_id_output_col,
                self.expert_gender_input_col: self.expert_gender_output_col,
                self.expert_max_projects_input_col: self.expert_max_projects_output_col,
                self.expert_name_input_col: self.expert_name_output_col,
                self.expert_research_types_input_col: self.expert_research_types_output_col
            })
            # Generate assignments for each project
            assignments = {}
            projects = sorted(ranked_pairs_df[self.project_id_output_col].unique())
            # First round: Proposed experts
            for project_id in projects:
                project_rankings = ranked_pairs_df[
                    ranked_pairs_df[self.project_id_output_col] == project_id
                ].merge(expert_info_df, on=self.expert_id_output_col, how='left')
                
                proposed_experts = self._assign_proposed_experts(project_id, project_rankings)
                assignments[project_id] = Assignment(
                    project_id=project_id,
                    proposed_experts=proposed_experts,
                    alternative_experts=[]
                )
            # Second round: Alternative experts
            for project_id in projects:
                project_rankings = ranked_pairs_df[
                    ranked_pairs_df[self.project_id_output_col] == project_id
                ].merge(expert_info_df, on=self.expert_id_output_col, how='left')
                alternative_experts = self._assign_alternative_experts(
                    project_id,
                    project_rankings,
                    assignments[project_id].proposed_experts
                )
                assignments[project_id].alternative_experts = alternative_experts
            # Convert assignments to DataFrame
            rows = []
            for project_id, assignment in assignments.items():
                project_info = projects_df[projects_df[self.project_id_input_col] == project_id].iloc[0]
                # Add proposed experts
                for i, expert_id in enumerate(assignment.proposed_experts, 1):
                    expert_info = expert_info_df[expert_info_df[self.expert_id_output_col] == expert_id].iloc[0]
                    expert_ranking = ranked_pairs_df[
                        (ranked_pairs_df[self.project_id_output_col] == project_id) &
                        (ranked_pairs_df[self.expert_id_output_col] == expert_id)
                    ].iloc[0]
                    rows.append({
                        self.project_id_output_col: project_id,
                        self.project_title_output_col: project_info[self.project_title_input_col],
                        self.project_research_types_output_col: project_info[self.project_research_types_input_col],
                        self.expert_id_output_col: expert_id,
                        self.expert_name_output_col: expert_info[self.expert_name_output_col],
                        self.expert_gender_output_col: expert_info[self.expert_gender_output_col],
                        self.expert_research_types_output_col: expert_info[self.expert_research_types_output_col],
                        self.predicted_prob_col: expert_ranking[self.predicted_prob_col],
                        self.predicted_prob_rank_col: expert_ranking[self.predicted_prob_rank_col],
                        'Assignment_Type': 'Proposed',
                        'Position': i
                    })
                # Add alternative experts
                for i, expert_id in enumerate(assignment.alternative_experts, 1):
                    expert_info = expert_info_df[expert_info_df[self.expert_id_output_col] == expert_id].iloc[0]
                    expert_ranking = ranked_pairs_df[
                        (ranked_pairs_df[self.project_id_output_col] == project_id) &
                        (ranked_pairs_df[self.expert_id_output_col] == expert_id)
                    ].iloc[0]
                    rows.append({
                        self.project_id_output_col: project_id,
                        self.project_title_output_col: project_info[self.project_title_input_col],
                        self.project_research_types_output_col: project_info[self.project_research_types_input_col],
                        self.expert_id_output_col: expert_id,
                        self.expert_name_output_col: expert_info[self.expert_name_output_col],
                        self.expert_gender_output_col: expert_info[self.expert_gender_output_col],
                        self.expert_research_types_output_col: expert_info[self.expert_research_types_output_col],
                        self.predicted_prob_col: expert_ranking[self.predicted_prob_col],
                        self.predicted_prob_rank_col: expert_ranking[self.predicted_prob_rank_col],
                        'Assignment_Type': 'Alternative',
                        'Position': i
                    })
            # Create final DataFrame and sort
            assignments_df = pd.DataFrame(rows)
            assignments_df['Assignment_Type'] = pd.Categorical(
                assignments_df['Assignment_Type'],
                categories=['Proposed', 'Alternative'],
                ordered=True
            )
            assignments_df = assignments_df.sort_values(
                [self.project_id_output_col, 'Assignment_Type', 'Position']
            )
            return assignments_df
        except Exception as e:
            raise

    def _assign_proposed_experts(self, project_id: int, ranked_df: pd.DataFrame) -> List[int]:
        """
        Assign proposed experts for a project. Prioritize meeting default constraints, 
        then apply flexibility (50% more projects) if needed.
        """
        proposed_experts = self._try_assign_experts(
            project_id,
            ranked_df,
            num_experts=self.num_proposed_experts,
            min_women=self.min_women_proposed,
            flexible=False  # Strict mode first
        )

        # If we fail to assign enough experts, try again with flexibility
        if len(proposed_experts) < self.num_proposed_experts:
            proposed_experts = self._try_assign_experts(
                project_id,
                ranked_df,
                num_experts=self.num_proposed_experts,
                min_women=self.min_women_proposed,
                flexible=True  # Flexible mode
            )
        return proposed_experts


    def _assign_alternative_experts(self, project_id: int, ranked_df: pd.DataFrame, proposed_experts: List[int]) -> List[int]:
        """
        Assign alternative experts for a project. Ensure at least half the required 
        number is assigned, applying flexibility only if needed.
        """
        alternative_experts = self._try_assign_experts(
            project_id,
            ranked_df,
            num_experts=self.num_alternative_experts,
            min_women=self.min_women_alternative,
            exclude_experts=proposed_experts,
            flexible=False  # Strict mode first
        )

        # If fewer than half the required alternatives are assigned, try again with flexibility
        min_alternative_experts = max(1, -(-self.num_alternative_experts // 2))  # Ceiling division
        if len(alternative_experts) < min_alternative_experts:
            alternative_experts = self._try_assign_experts(
                project_id,
                ranked_df,
                num_experts=min_alternative_experts,
                min_women=self.min_women_alternative,
                exclude_experts=proposed_experts,
                flexible=True  # Flexible mode
            )
        return alternative_experts

    def _try_assign_experts(self, project_id: int, ranked_df: pd.DataFrame, num_experts: int, 
                            min_women: int, exclude_experts: List[int] = None, flexible: bool = False) -> List[int]:
        """
        Attempt to assign a specified number of experts, optionally with flexibility.
        Args:
            project_id (int): The ID of the project.
            ranked_df (pd.DataFrame): Ranked data for the project.
            num_experts (int): The number of experts to assign.
            min_women (int): Minimum number of women experts.
            exclude_experts (List[int], optional): List of experts to exclude from assignment.
            flexible (bool, optional): Whether to apply flexible constraints.
        Returns:
            List[int]: List of assigned expert IDs.
        """
        if exclude_experts is None:
            exclude_experts = []

        # Filter and sort candidates
        candidates = ranked_df[
            (ranked_df[self.project_id_output_col] == project_id) &
            (~ranked_df[self.expert_id_output_col].isin(exclude_experts)) &
            (ranked_df[self.predicted_prob_col] >= self.min_probability_threshold)
        ].sort_values(self.predicted_prob_rank_col)

        assigned_experts = []
        women_count = 0

        for _, expert_row in candidates.iterrows():
            expert_id = expert_row[self.expert_id_output_col]
            current_assignments = self.expert_assignment_count.get(expert_id, 0)
            max_projects = expert_row[self.expert_max_projects_output_col]

            # Adjust capacity if flexibility is enabled
            if flexible:
                max_projects = int(max_projects * 1.5)

            if current_assignments >= max_projects:
                continue

            # Check gender requirements
            is_woman = expert_row[self.expert_gender_output_col].lower() == self.expert_gender_value_women
            slots_remaining = num_experts - len(assigned_experts)

            if slots_remaining == 1 and women_count < min_women:
                # If the last slot must meet the women quota
                remaining_women = sum(
                    1 for _, e in candidates.iterrows()
                    if e[self.expert_id_output_col] not in assigned_experts
                    and e[self.expert_gender_output_col].lower() == self.expert_gender_value_women
                )
                if remaining_women > 0 and not is_woman:
                    continue

            # Assign the expert
            assigned_experts.append(expert_id)
            self.expert_assignment_count[expert_id] = current_assignments + 1
            if is_woman:
                women_count += 1

            # Stop if we have assigned enough experts
            if len(assigned_experts) >= num_experts:
                break

        return assigned_experts


    def get_assignment_stats(self, assignments_df: pd.DataFrame) -> Dict[str, dict]:
        """
        Get statistics about the assignments from a DataFrame.
        Args:
            assignments_df (pd.DataFrame): DataFrame containing assignment details.
        Returns:
            dict: Statistics including expert assignment counts, project coverage, and gender distribution.
        """
        stats = {
            'expert_counts': {},  # Count of assignments per expert
            'gender_distribution': {  # Gender distribution for proposed and alternative experts
                'proposed': {self.expert_gender_value_men: 0, self.expert_gender_value_women: 0},
                'alternative': {self.expert_gender_value_men: 0, self.expert_gender_value_women: 0}
            }
        }
        # Count expert assignments
        for expert_id, count in assignments_df[self.expert_id_output_col].value_counts().items():
            stats['expert_counts'][expert_id] = count
        # Calculate gender distribution
        for assignment_type in ['Proposed', 'Alternative']:
            gender_counts = (
                assignments_df[assignments_df['Assignment_Type'] == assignment_type]
                [self.expert_gender_output_col]
                .value_counts()
            )
            for gender, count in gender_counts.items():
                stats['gender_distribution'][assignment_type.lower()][gender.lower()] += count
        return stats

    def get_problematic_projects(self, assignments_df: pd.DataFrame, projects_df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze problematic projects after assignments, identifying cases where:
        - Not enough proposed or alternative experts were assigned.
        - Gender diversity constraints were not met.
        - Flexibility was applied to meet constraints.
        
        Args:
            assignments_df (pd.DataFrame): DataFrame containing assignment details.
            projects_df (pd.DataFrame): DataFrame containing project details.
            
        Returns:
            pd.DataFrame: Summary of problematic projects with details on unmet constraints.
        """
        results = []

        for project_id in projects_df[self.project_id_input_col].unique():
            # Get project-specific data
            project_assignments = assignments_df[
                assignments_df[self.project_id_output_col] == project_id
            ]
            proposed_assignments = project_assignments[
                project_assignments['Assignment_Type'] == 'Proposed'
            ]
            alternative_assignments = project_assignments[
                project_assignments['Assignment_Type'] == 'Alternative'
            ]
            
            # Count assigned experts
            num_proposed = len(proposed_assignments)
            num_alternative = len(alternative_assignments)
            
            # Count gender diversity
            num_proposed_women = proposed_assignments[
                proposed_assignments[self.expert_gender_output_col].str.lower() == self.expert_gender_value_women
            ].shape[0]
            num_alternative_women = alternative_assignments[
                alternative_assignments[self.expert_gender_output_col].str.lower() == self.expert_gender_value_women
            ].shape[0]

            # Calculate flexibility application
            proposed_flex_applied = any(
                proposed_assignments[self.predicted_prob_col] < self.min_probability_threshold
            )
            alternative_flex_applied = any(
                alternative_assignments[self.predicted_prob_col] < self.min_probability_threshold
            )

            # Determine if this project is problematic
            missing_proposed = max(0, self.num_proposed_experts - num_proposed)
            missing_alternative = max(0, self.num_alternative_experts - num_alternative)
            problematic = (
                missing_proposed > 0 or
                missing_alternative > 0 or
                num_proposed_women < self.min_women_proposed or
                num_alternative_women < self.min_women_alternative
            )
            
            if problematic:
                results.append({
                    'Project_ID': project_id,
                    'Project_Title': projects_df.loc[
                        projects_df[self.project_id_input_col] == project_id,
                        self.project_title_input_col
                    ].iloc[0],
                    'Proposed_Assigned': num_proposed,
                    'Missing_Proposed': missing_proposed,
                    'Proposed_Women_Assigned': num_proposed_women,
                    'Required_Proposed_Women': self.min_women_proposed,
                    'Alternative_Assigned': num_alternative,
                    'Missing_Alternative': missing_alternative,
                    'Alternative_Women_Assigned': num_alternative_women,
                    'Required_Alternative_Women': self.min_women_alternative,
                    'Proposed_Flex_Applied': proposed_flex_applied,
                    'Alternative_Flex_Applied': alternative_flex_applied
                })

        if not results:
            return pd.DataFrame()

        return pd.DataFrame(results).sort_values('Project_ID')


    def get_expert_capacity_issues(self, ranked_pairs_df: pd.DataFrame, experts_df: pd.DataFrame) -> pd.DataFrame:
        """
        Identify experts who might be overloaded based on their maximum project limits
        and the number of high-probability assignments available.
        
        Args:
            ranked_pairs_df (pd.DataFrame): DataFrame with expert-project pairs and predictions
            experts_df (pd.DataFrame): DataFrame with expert information
            
        Returns:
            DataFrame with experts who might exceed their capacity
        """
        # Handle max projects column
        experts_df = experts_df.copy()
        if self.expert_max_projects_input_col not in experts_df.columns:
            experts_df[self.expert_max_projects_input_col] = self.max_default_projects_per_expert
        else:
            mask = (experts_df[self.expert_max_projects_input_col].isna()) | (experts_df[self.expert_max_projects_input_col] == 0)
            if mask.any():
                experts_df.loc[mask, self.expert_max_projects_input_col] = self.max_default_projects_per_expert

        results = []
        
        # Get all qualified assignments (prob >= threshold)
        qualified_assignments = ranked_pairs_df[
            ranked_pairs_df[self.predicted_prob_col] >= self.min_probability_threshold
        ]
        
        # Count potential assignments per expert
        for expert_id in experts_df[self.expert_id_input_col].unique():
            expert_assignments = qualified_assignments[
                qualified_assignments[self.expert_id_output_col] == expert_id
            ]
            
            if len(expert_assignments) > 0:
                max_projects = experts_df.loc[
                    experts_df[self.expert_id_input_col] == expert_id,
                    self.expert_max_projects_input_col
                ].iloc[0]
                
                if len(expert_assignments) > max_projects:
                    results.append({
                        'Expert_ID': expert_id,
                        'Expert_Name': experts_df.loc[
                            experts_df[self.expert_id_input_col] == expert_id,
                            self.expert_name_input_col
                        ].iloc[0],
                        'Expert_Gender': experts_df.loc[
                            experts_df[self.expert_id_input_col] == expert_id,
                            self.expert_gender_input_col
                        ].iloc[0],
                        'Max_Projects': max_projects,
                        'Potential_Assignments': len(expert_assignments),
                        'Excess_Assignments': len(expert_assignments) - max_projects,
                        'Projects': sorted(expert_assignments[self.project_id_output_col].unique()),
                        'Avg_Probability': expert_assignments[self.predicted_prob_col].mean()
                    })
        
        if not results:
            return pd.DataFrame()
            
        return pd.DataFrame(results).sort_values('Potential_Assignments', ascending=False)

    def get_expert_assignment_distribution(self, assignments_df: pd.DataFrame, experts_df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze how the actual assignments are distributed across experts.
        
        Args:
            assignments_df (pd.DataFrame): The final assignments
            experts_df (pd.DataFrame): Expert information including max projects
            
        Returns:
            pd.DataFrame: DataFrame with columns for expert ID, name, gender, number of 
                         assignments, max projects allowed, utilization percentage, and
                         average assignment probability.
        """
        # Handle max projects column
        experts_df = experts_df.copy()
        if self.expert_max_projects_input_col not in experts_df.columns:
            experts_df[self.expert_max_projects_input_col] = self.max_default_projects_per_expert
        else:
            mask = (experts_df[self.expert_max_projects_input_col].isna()) | (experts_df[self.expert_max_projects_input_col] == 0)
            if mask.any():
                experts_df.loc[mask, self.expert_max_projects_input_col] = self.max_default_projects_per_expert
                
        # Count assignments per expert - using groupby instead of value_counts
        assignment_counts = (
            assignments_df.groupby(self.expert_id_output_col)
            .size()
            .reset_index(name='Num_Assignments')
        )
        
        # Merge with expert information
        distribution = assignment_counts.merge(
            experts_df[[
                self.expert_id_input_col,
                self.expert_name_input_col,
                self.expert_gender_input_col,
                self.expert_max_projects_input_col
            ]],
            left_on=self.expert_id_output_col,
            right_on=self.expert_id_input_col,
            how='right'
        )
        
        # Fill NaN values for experts with no assignments
        distribution['Num_Assignments'] = distribution['Num_Assignments'].fillna(0).astype(int)
        
        # Calculate utilization percentage
        distribution['Utilization_Pct'] = (
            distribution['Num_Assignments'] / distribution[self.expert_max_projects_input_col] * 100
        ).round(1)
        
        # Calculate average probability for assigned projects
        avg_probs = assignments_df.groupby(self.expert_id_output_col)[self.predicted_prob_col].mean()
        distribution[self.expert_id_output_col] = distribution[self.expert_id_input_col]  # Ensure matching IDs for mapping
        distribution['Avg_Assignment_Probability'] = distribution[self.expert_id_output_col].map(avg_probs)
        
        # Count proposed and alternative assignments
        proposed_counts = (
            assignments_df[assignments_df['Assignment_Type'] == 'Proposed']
            .groupby(self.expert_id_output_col)
            .size()
        )
        alternative_counts = (
            assignments_df[assignments_df['Assignment_Type'] == 'Alternative']
            .groupby(self.expert_id_output_col)
            .size()
        )
        
        distribution['Num_Proposed'] = distribution[self.expert_id_output_col].map(proposed_counts).fillna(0).astype(int)
        distribution['Num_Alternative'] = distribution[self.expert_id_output_col].map(alternative_counts).fillna(0).astype(int)
        
        # Reorder columns and sort by utilization
        result = distribution[[
            self.expert_id_output_col,
            self.expert_name_input_col,
            self.expert_gender_input_col,
            'Num_Assignments',
            'Num_Proposed',
            'Num_Alternative',
            self.expert_max_projects_input_col,
            'Utilization_Pct',
            'Avg_Assignment_Probability'
        ]].rename(columns={
            self.expert_id_output_col: 'Expert_ID',
            self.expert_name_input_col: 'Expert_Name',
            self.expert_gender_input_col: 'Expert_Gender',
            self.expert_max_projects_input_col: 'Max_Projects'
        })
        
        return result.sort_values('Utilization_Pct', ascending=False)

    def get_project_assignment_status(self, assignments_df: pd.DataFrame, projects_df: pd.DataFrame) -> pd.DataFrame:
        """
        Provides a comprehensive analysis of project assignments, including numbers of
        assigned experts, gender distribution, and any issues or constraints that weren't met.
        
        Args:
            assignments_df (pd.DataFrame): DataFrame containing assignment details
            projects_df (pd.DataFrame): DataFrame containing project details
            
        Returns:
            pd.DataFrame: Comprehensive project status including:
                - Basic counts (assigned/missing experts)
                - Gender distribution
                - Whether constraints were met
                - Total impact metrics
        """
        results = []
        
        for project_id in projects_df[self.project_id_input_col].unique():
            # Get project information
            project_info = projects_df[projects_df[self.project_id_input_col] == project_id].iloc[0]
            
            # Get project assignments
            project_assignments = assignments_df[
                assignments_df[self.project_id_output_col] == project_id
            ]
            
            # Get proposed and alternative assignments
            proposed_assignments = project_assignments[
                project_assignments['Assignment_Type'] == 'Proposed'
            ]
            alternative_assignments = project_assignments[
                project_assignments['Assignment_Type'] == 'Alternative'
            ]
            
            # Count total assignments
            num_proposed = len(proposed_assignments)
            num_alternative = len(alternative_assignments)
            
            # Count gender distribution
            num_proposed_women = sum(
                proposed_assignments[self.expert_gender_output_col].str.lower() == self.expert_gender_value_women
            )
            num_alternative_women = sum(
                alternative_assignments[self.expert_gender_output_col].str.lower() == self.expert_gender_value_women
            )
            
            # Calculate missing experts
            missing_proposed = self.num_proposed_experts - num_proposed
            missing_alternative = self.num_alternative_experts - num_alternative
            
            # Check if gender requirements were met
            proposed_gender_req_met = num_proposed_women >= self.min_women_proposed
            alternative_gender_req_met = num_alternative_women >= self.min_women_alternative
            
            # Check quality of assignments
            avg_proposed_prob = proposed_assignments[self.predicted_prob_col].mean() if not proposed_assignments.empty else 0
            avg_alternative_prob = alternative_assignments[self.predicted_prob_col].mean() if not alternative_assignments.empty else 0
            
            results.append({
                # Basic project info
                'Project_ID': project_id,
                'Project_Title': project_info[self.project_title_input_col],
                
                # Proposed experts status
                'Proposed_Total': num_proposed,
                'Proposed_Women': num_proposed_women,
                'Proposed_Missing': missing_proposed,
                'Proposed_Gender_Req_Met': proposed_gender_req_met,
                'Proposed_Avg_Prob': round(avg_proposed_prob, 3),
                
                # Alternative experts status
                'Alternative_Total': num_alternative,
                'Alternative_Women': num_alternative_women,
                'Alternative_Missing': missing_alternative,
                'Alternative_Gender_Req_Met': alternative_gender_req_met,
                'Alternative_Avg_Prob': round(avg_alternative_prob, 3),
                
                # Overall status
                'Total_Assigned': num_proposed + num_alternative,
                'Total_Required': self.num_proposed_experts + self.num_alternative_experts,
                'Total_Missing': missing_proposed + missing_alternative,
                'All_Requirements_Met': (missing_proposed == 0 and 
                                       missing_alternative == 0 and 
                                       proposed_gender_req_met and 
                                       alternative_gender_req_met)
            })
        
        # Convert to DataFrame and sort
        status_df = pd.DataFrame(results)
        
        if not status_df.empty:
            # Add summary metrics
            status_df['Assignment_Completion'] = (
                (status_df['Total_Assigned'] / status_df['Total_Required'] * 100)
                .round(1)
            )
            
            # Sort by completion percentage and then by project ID
            status_df = status_df.sort_values(
                ['Assignment_Completion', 'Project_ID'],
                ascending=[True, True]
            )
        
        return status_df


    def get_unassigned_projects(self, assignments_df: pd.DataFrame, projects_df: pd.DataFrame) -> pd.DataFrame:
        """
        Identify projects that didn't receive full assignments.
        Returns:
            DataFrame with projects that have fewer than required experts
        """
        results = []
        for project_id in projects_df[self.project_id_input_col].unique():
            project_assignments = assignments_df[
                assignments_df[self.project_id_output_col] == project_id
            ]
            proposed_count = len(project_assignments[
                project_assignments['Assignment_Type'] == 'Proposed'
            ])
            alternative_count = len(project_assignments[
                project_assignments['Assignment_Type'] == 'Alternative'
            ])
            if proposed_count < self.num_proposed_experts or alternative_count < self.num_alternative_experts:
                results.append({
                    'Project_ID': project_id,
                    'Project_Title': projects_df.loc[
                        projects_df[self.project_id_input_col] == project_id,
                        self.project_title_input_col
                    ].iloc[0],
                    'Assigned_Proposed': proposed_count,
                    'Missing_Proposed': self.num_proposed_experts - proposed_count,
                    'Assigned_Alternative': alternative_count,
                    'Missing_Alternative': self.num_alternative_experts - alternative_count,
                    'Total_Assigned': proposed_count + alternative_count,
                    'Total_Missing': (self.num_proposed_experts + self.num_alternative_experts) - 
                                   (proposed_count + alternative_count)
                })
        if not results:
            return pd.DataFrame()
        return pd.DataFrame(results).sort_values('Total_Missing', ascending=False)
    
    def get_multidisciplinary_coverage(self, assignments_df: pd.DataFrame, projects_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute approach coverage scores for multidisciplinary projects and return as a DataFrame.
        Args:
            assignments_df (pd.DataFrame): DataFrame containing assignment details.
            projects_df (pd.DataFrame): DataFrame containing project details.
        Returns:
            pd.DataFrame: DataFrame with columns ['Project_ID', 'Coverage_Score'].
        """
        results = []
        for project_id in assignments_df['Project_ID'].unique():
            # Get project data
            project_data = projects_df[projects_df['Project_ID'] == project_id]
            if project_data.empty or project_data['Project_Type'].iloc[0] != 'multi':
                continue
            # Get combined expert approaches
            proposed_experts = assignments_df[
                (assignments_df['Project_ID'] == project_id) &
                (assignments_df['Assignment_Type'] == 'Proposed')
            ]['Expert_ID']
            expert_approaches = self._get_combined_expert_approaches(proposed_experts, experts_df)
            # Calculate coverage score
            project_approaches = set(project_data['Project_Approaches'].iloc[0].split('|'))
            coverage = len(expert_approaches.intersection(project_approaches)) / len(project_approaches)
            results.append({'Project_ID': project_id, 'Coverage_Score': coverage})
        # Convert results to DataFrame
        coverage_df = pd.DataFrame(results)
        return coverage_df
        

