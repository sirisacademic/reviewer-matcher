# File: expert_assigner.py

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
        self.expert_max_proposed_projects_input_col = config_manager.get('EXPERT_MAX_PROPOSED_PROJECTS_INPUT_COLUMN', 'MAX_PROJECTS_REVIEW')
        self.expert_name_input_col = config_manager.get('EXPERT_NAME_INPUT_COLUMN', 'FULL_NAME')
        self.expert_research_types_input_col = config_manager.get('EXPERT_RESEARCH_TYPES_INPUT_COLUMN', 'RESEARCH_TYPES')
        self.expert_research_approaches_input_col = config_manager.get('EXPERT_RESEARCH_APPROACHES_INPUT_COLUMN', 'RESEARCH_APPROACHES')
        self.project_id_input_col = config_manager.get('PROJECT_ID_INPUT_COLUMN', 'ID')
        self.project_title_input_col = config_manager.get('PROJECT_TITLE_INPUT_COLUMN', 'TITLE')
        self.project_approach_type_input_col = config_manager.get('PROJECT_APPROACH_TYPE_INPUT_COLUMN', 'APPROACH_TYPE')
        self.project_research_types_input_col = config_manager.get('PROJECT_RESEARCH_TYPES_INPUT_COLUMN', 'RESEARCH_TYPE')
        self.project_research_approaches_input_col = config_manager.get('PROJECT_RESEARCH_APPROACHES_INPUT_COLUMN', 'RESEARCH_APPROACHES')
        self.predicted_prob_col = config_manager.get('PREDICTED_PROB_COLUMN', 'Predicted_Prob')
        self.predicted_prob_rank_col = config_manager.get('PREDICTED_PROB_RANK_COLUMN', 'Predicted_Prob_Rank')
        # Output columns
        self.expert_id_output_col = self.expert_cols[self.expert_id_input_col]
        self.project_id_output_col = self.project_cols[self.project_id_input_col]
        self.expert_gender_output_col = self.expert_cols[self.expert_gender_input_col]
        self.expert_max_proposed_projects_output_col = self.expert_cols[self.expert_max_proposed_projects_input_col]
        self.expert_name_output_col = self.expert_cols[self.expert_name_input_col]
        self.expert_research_types_output_col = self.expert_cols[self.expert_research_types_input_col]
        self.expert_research_approaches_output_col = self.expert_cols[self.expert_research_approaches_input_col]
        self.project_title_output_col = self.project_cols[self.project_title_input_col]
        self.project_approach_type_output_col = self.project_cols[self.project_approach_type_input_col]
        self.project_research_types_output_col = self.project_cols[self.project_research_types_input_col]
        self.project_research_approaches_output_col = self.project_cols[self.project_research_approaches_input_col]
        # Value used to identify genders
        self.expert_gender_value_women = config_manager.get('EXPERT_GENDER_VALUE_WOMEN', 'female')
        self.expert_gender_value_men = config_manager.get('EXPERT_GENDER_VALUE_MEN', 'male')
        # Assignment configuration
        self.num_proposed_experts = config_manager.get('NUM_PROPOSED_EXPERTS', 3)
        self.num_alternative_experts = config_manager.get('NUM_ALTERNATIVE_EXPERTS', 20)
        self.min_women_proposed = config_manager.get('MIN_WOMEN_PROPOSED', 1)
        self.min_women_alternative = config_manager.get('MIN_WOMEN_ALTERNATIVE', 2)
        self.min_projects_per_expert = config_manager.get('MIN_PROJECTS_PER_EXPERT', 2)
        self.max_proposed_projects_per_expert = config_manager.get('MAX_DEFAULT_PROPOSED_PROJECTS_PER_EXPERT', 5)
        self.max_total_projects_per_expert = config_manager.get('MAX_DEFAULT_TOTAL_PROJECTS_PER_EXPERT', 20)
        # Track assignments
        self.expert_assignment_count_proposed = {}
        self.expert_assignment_count_total = {}
        # Min. probability threshold for expert-project pairs.
        self.min_probability_threshold = config_manager.get('MIN_PROBABILITY_THRESHOLD', 0.5)

    def generate_assignments(self, ranked_pairs_df: pd.DataFrame, experts_df: pd.DataFrame, projects_df: pd.DataFrame) -> pd.DataFrame:
        #Generate expert assignments for all projects.
        try:
            # Data type conversion and validation
            ranked_pairs_df[self.project_id_output_col] = ranked_pairs_df[self.project_id_output_col].astype(int)
            ranked_pairs_df[self.expert_id_output_col] = ranked_pairs_df[self.expert_id_output_col].astype(int)
            ranked_pairs_df[self.predicted_prob_rank_col] = ranked_pairs_df[self.predicted_prob_rank_col].astype(int)
            projects_df[self.project_id_input_col] = projects_df[self.project_id_input_col].astype(int)
            experts_df[self.expert_id_input_col] = experts_df[self.expert_id_input_col].astype(int)
            # Handle max projects per expert
            if self.expert_max_proposed_projects_input_col not in experts_df.columns:
                experts_df[self.expert_max_proposed_projects_input_col] = self.max_proposed_projects_per_expert
            else:
                mask = (experts_df[self.expert_max_proposed_projects_input_col].isna()) | (experts_df[self.expert_max_proposed_projects_input_col] == 0)
                if mask.any():
                    experts_df.loc[mask, self.expert_max_proposed_projects_input_col] = self.max_proposed_projects_per_expert
 
            # Create expert info lookup without project-specific data
            expert_info_df = experts_df[[
                self.expert_id_input_col,
                self.expert_gender_input_col,
                self.expert_max_proposed_projects_input_col,
                self.expert_name_input_col,
                self.expert_research_types_input_col
            ]].rename(columns={
                self.expert_id_input_col: self.expert_id_output_col,
                self.expert_gender_input_col: self.expert_gender_output_col,
                self.expert_max_proposed_projects_input_col: self.expert_max_proposed_projects_output_col,
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
                        self.project_approach_type_output_col: project_info[self.project_approach_type_input_col],
                        self.project_research_types_output_col: project_info[self.project_research_types_input_col],
                        self.expert_id_output_col: expert_id,
                        self.expert_name_output_col: expert_info[self.expert_name_output_col],
                        self.expert_gender_output_col: expert_info[self.expert_gender_output_col],
                        self.expert_research_types_output_col: expert_info[self.expert_research_types_output_col],
                        self.predicted_prob_col: expert_ranking[self.predicted_prob_col],
                        'Assignment_Type': 'Proposed'
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
                        self.project_approach_type_output_col: project_info[self.project_approach_type_input_col],
                        self.project_research_types_output_col: project_info[self.project_research_types_input_col],
                        self.expert_id_output_col: expert_id,
                        self.expert_name_output_col: expert_info[self.expert_name_output_col],
                        self.expert_gender_output_col: expert_info[self.expert_gender_output_col],
                        self.expert_research_types_output_col: expert_info[self.expert_research_types_output_col],
                        self.predicted_prob_col: expert_ranking[self.predicted_prob_col],
                        'Assignment_Type': 'Alternative'
                    })
            assignments_df = pd.DataFrame(rows)
            assignments_df['Assignment_Type'] = pd.Categorical(
                assignments_df['Assignment_Type'],
                categories=['Proposed', 'Alternative', 'Not assigned'],
                ordered=True
            )
            
            # Store ranked_pairs_df for use in balancing
            self.ranked_pairs_df = ranked_pairs_df.copy()
            
            # Balance assignments to minimize experts with only one "Proposed" assignment.
            # assignments_df = self._balance_assignments(assignments_df)
            def count_single_proposed(df):
                #Returns the number of experts that have exactly one 'Proposed' assignment.
                counts = df[df['Assignment_Type'] == 'Proposed'][self.expert_id_output_col].value_counts()
                return (counts == 1).sum()
            reduction_threshold = 0.02   # Require at least a 2% reduction to continue iterating.
            max_iterations = 50          # Prevent endless looping.
            num_iterations = 0
            prev_single_count = count_single_proposed(assignments_df)
            print(f"Initial experts with one 'Proposed': {prev_single_count}")
            while num_iterations < max_iterations:
                num_iterations += 1
                # Save current state for later comparison.
                prev_assignments_df = assignments_df.copy(deep=True)
                # Run the re-balancing algorithm.
                assignments_df = self._balance_assignments(assignments_df)
                # Count experts with only one Proposed assignment after rebalancing.
                new_single_count = count_single_proposed(assignments_df)
                reduction = 0.0
                if prev_single_count > 0:
                    reduction = (prev_single_count - new_single_count) / prev_single_count
                print(f"Iteration {num_iterations}: Single-assignment experts reduced from {prev_single_count} to {new_single_count} ({reduction*100:.2f}% reduction)")
                # Stop if the reduction is too small.
                if reduction < reduction_threshold:
                    print("Reduction below threshold, stopping iterations.")
                    break
                # Otherwise, update the count and continue.
                prev_single_count = new_single_count
            print(f"Rebalancing finished after {num_iterations} iteration(s).")
   
            # Remove single 'Proposed' assignments and redistribute.
            print(f"Un-assigning projects for remaining single-project experts and re-assigning them...")
            assignments_df = self.remove_single_proposed_assignments(assignments_df, self.ranked_pairs_df, experts_df, projects_df)
            single_count = count_single_proposed(assignments_df)
            if single_count > 0:
                print(f"{single_count} experts remain with one single proposed assignments. Please check them manually.")
   
            # Add a new column: count of 'Proposed' assignments per expert.
            proposed_counts = assignments_df[assignments_df['Assignment_Type'] == 'Proposed'][self.expert_id_output_col].value_counts().to_dict()
            assignments_df['Proposed_Assignment_Count'] = assignments_df[self.expert_id_output_col].apply(lambda x: proposed_counts.get(x, 0))
            
            # Compute 'proposed' experts coverage in terms of research approaches.
            coverage_df = self.get_multidisciplinary_coverage(assignments_df, projects_df, experts_df)
            final_assignments_df = assignments_df.merge(coverage_df, on=self.project_id_output_col, how='left')

            final_assignments_df = final_assignments_df.sort_values(
                by=[self.project_id_output_col, 'Assignment_Type', 'Predicted_Prob'],
                ascending=[True, True, False]
            )

            # Return assignments.
            return final_assignments_df

        except Exception as e:
            raise

    def remove_single_proposed_assignments(self, assignments_df: pd.DataFrame, ranked_pairs_df: pd.DataFrame, experts_df: pd.DataFrame, projects_df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove experts with only one 'Proposed' assignment and reassign their projects,
        ensuring no expert exceeds their individual maximum allowed assignments.
        Promoted 'Alternative' experts are removed from their alternative assignments.
        """
        proposed_counts = assignments_df[assignments_df['Assignment_Type'] == 'Proposed'][self.expert_id_output_col].value_counts()
        experts_with_one_proposed = proposed_counts[proposed_counts == 1].index.tolist()

        iteration = 0
        while experts_with_one_proposed:
            iteration += 1
            print(f"Iteration {iteration}: Processing {len(experts_with_one_proposed)} experts with a single 'Proposed' assignment.")

            for expert_id in experts_with_one_proposed:
                # Get the project assigned to this expert
                project_row = assignments_df[
                    (assignments_df[self.expert_id_output_col] == expert_id) &
                    (assignments_df['Assignment_Type'] == 'Proposed')
                ].iloc[0]
                project_id = project_row[self.project_id_output_col]

                # Retrieve project details
                project_info = projects_df[projects_df[self.project_id_input_col] == project_id].iloc[0]

                # Find experts already assigned as 'Proposed' to this project
                already_assigned_proposed_experts = assignments_df[
                    (assignments_df[self.project_id_output_col] == project_id) &
                    (assignments_df['Assignment_Type'] == 'Proposed')
                ][self.expert_id_output_col].unique()

                # Find candidate experts not already assigned as 'Proposed' to the project
                candidate_experts_df = assignments_df[
                    (assignments_df['Assignment_Type'] == 'Proposed') &
                    (assignments_df[self.expert_id_output_col] != expert_id) &
                    (~assignments_df[self.expert_id_output_col].isin(already_assigned_proposed_experts))
                ][[self.expert_id_output_col]].drop_duplicates()

                # Merge with predicted probabilities and expert details (including max limits)
                candidates_with_details = candidate_experts_df.merge(
                    ranked_pairs_df[
                        ranked_pairs_df[self.project_id_output_col] == project_id
                    ][[self.expert_id_output_col, self.predicted_prob_col]],
                    on=self.expert_id_output_col,
                    how='left'
                ).merge(
                    experts_df[[
                        self.expert_id_input_col,
                        self.expert_name_input_col,
                        self.expert_gender_input_col,
                        self.expert_research_types_input_col,
                        self.expert_max_proposed_projects_input_col  # Include individual max projects
                    ]],
                    left_on=self.expert_id_output_col,
                    right_on=self.expert_id_input_col,
                    how='left'
                )

                # Add workload and gender priority
                candidates_with_details['proposed_count'] = candidates_with_details[self.expert_id_output_col].map(proposed_counts).fillna(0).astype(int)
                candidates_with_details['max_proposed_projects'] = candidates_with_details[self.expert_max_proposed_projects_input_col].fillna(self.max_proposed_projects_per_expert).astype(int)
                candidates_with_details['is_woman'] = candidates_with_details[self.expert_gender_input_col].str.lower() == self.expert_gender_value_women

                # Filter candidates who haven't reached their individual max proposed projects
                eligible_candidates = candidates_with_details[
                    candidates_with_details['proposed_count'] < candidates_with_details['max_proposed_projects']
                ]

                if eligible_candidates.empty:
                    print(f"Could not reassign project {project_id} from expert {expert_id}; keeping the assignment.")
                    continue

                # Prioritize by: probability DESC, workload ASC, gender (women first)
                prioritized_candidates = eligible_candidates.sort_values(
                    by=[self.predicted_prob_col, 'proposed_count', 'is_woman'],
                    ascending=[False, True, False]
                )

                # Select the top candidate
                candidate = prioritized_candidates.iloc[0]
                candidate_id = candidate[self.expert_id_output_col]

                # Final check: Ensure candidate hasn't reached the limit after other reassignments in this iteration
                current_proposed_count = proposed_counts.get(candidate_id, 0)
                max_allowed_projects = candidate['max_proposed_projects']

                if current_proposed_count >= max_allowed_projects:
                    print(f"Skipping reassignment to expert {candidate_id} as they have reached their max proposed projects.")
                    continue  # Skip to next candidate

                # Proceed with reassignment if the expert is within the allowed limit
                new_assignment = {
                    self.project_id_output_col: project_id,
                    self.project_title_output_col: project_info[self.project_title_input_col],
                    self.project_approach_type_output_col: project_info[self.project_approach_type_input_col],
                    self.project_research_types_output_col: project_info[self.project_research_types_input_col],
                    self.expert_id_output_col: candidate_id,
                    self.expert_name_output_col: candidate[self.expert_name_input_col],
                    self.expert_gender_output_col: candidate[self.expert_gender_input_col],
                    self.expert_research_types_output_col: candidate[self.expert_research_types_input_col],
                    self.predicted_prob_col: candidate[self.predicted_prob_col],
                    'Assignment_Type': 'Proposed'
                }

                # Add the new 'Proposed' assignment
                assignments_df = pd.concat([assignments_df, pd.DataFrame([new_assignment])], ignore_index=True)

                # Remove the expert's 'Alternative' assignment for the same project if it exists
                assignments_df = assignments_df[
                    ~(
                        (assignments_df[self.expert_id_output_col] == candidate_id) &
                        (assignments_df[self.project_id_output_col] == project_id) &
                        (assignments_df['Assignment_Type'] == 'Alternative')
                    )
                ]

                # Update proposed count immediately
                proposed_counts[candidate_id] = current_proposed_count + 1

                # Remove the under-utilized expert's assignment
                assignments_df = assignments_df[
                    ~((assignments_df[self.expert_id_output_col] == expert_id) &
                      (assignments_df[self.project_id_output_col] == project_id))
                ]

                print(f"Reassigned project {project_id} from expert {expert_id} to expert {candidate_id}.")

            # Update counts after reassignment
            proposed_counts = assignments_df[assignments_df['Assignment_Type'] == 'Proposed'][self.expert_id_output_col].value_counts()
            experts_with_one_proposed = proposed_counts[proposed_counts == 1].index.tolist()

        print("Final reassignment complete. All possible single 'Proposed' assignments have been addressed.")
        return assignments_df





    def generate_expert_project_alternatives(self, ranked_pairs_df, assignments_df, experts_df, projects_df):
        """
        Generates a file listing up to 20 projects for each expert, including 'Not assigned' pairs.
        
        Args:
            ranked_pairs_df (pd.DataFrame): DataFrame with expert-project ranking information.
            assignments_df (pd.DataFrame): DataFrame with actual expert assignments.
            experts_df (pd.DataFrame): DataFrame with expert details.
            projects_df (pd.DataFrame): DataFrame with project details.
        
        Returns:
            pd.DataFrame: Reformatted expert-project assignments, including high-scoring unassigned pairs.
        """
        # Filter out expert-project pairs below the probability threshold.
        filtered_df = ranked_pairs_df[ranked_pairs_df[self.predicted_prob_col] >= self.min_probability_threshold]

        # Merge assignment information to identify assigned and unassigned pairs.
        merged_df = filtered_df.merge(assignments_df, 
                                      on=[self.expert_id_output_col, self.project_id_output_col], 
                                      how='left', 
                                      suffixes=('', '_assigned'))
        
        # Assign "Not assigned" to pairs without an assignment.
        merged_df['Assignment_Type'] = merged_df['Assignment_Type'].fillna("Not assigned")

        # Merge expert and project details for all rows, including "Not assigned".
        merged_df = merged_df.merge(experts_df[[self.expert_id_input_col, self.expert_name_input_col, self.expert_gender_input_col]],
                                    left_on=self.expert_id_output_col, right_on=self.expert_id_input_col, how='left')

        merged_df = merged_df.merge(projects_df[[self.project_id_input_col, self.project_title_input_col]],
                                    left_on=self.project_id_output_col, right_on=self.project_id_input_col, how='left')
        
        # Select and order relevant columns.
        merged_df = merged_df[[
            self.expert_id_output_col,
            self.expert_name_input_col,
            self.expert_gender_input_col,
            self.project_id_output_col,
            'Assignment_Type',
            self.predicted_prob_col,
            self.project_title_input_col
        ]]

        # Sort by expert and descending probability score.
        merged_df = merged_df.sort_values(
            by=[self.expert_id_output_col, 'Assignment_Type', self.predicted_prob_col], 
            ascending=[True, True, False]
        )

        # Limit to the top 20 projects per expert.
        merged_df = merged_df.groupby(self.expert_id_output_col).head(20)

        return merged_df



    def _assign_proposed_experts(self, project_id: int, ranked_df: pd.DataFrame) -> List[int]:
        """
        Assign proposed experts for a project. Strict limit.
        """
        proposed_experts = self._try_assign_experts(
            project_id,
            ranked_df,
            'Proposed',
            num_experts_to_assign=self.num_proposed_experts,
            min_women=self.min_women_proposed,
            flexible=False  # Strict mode.
        )
        return proposed_experts

    def _assign_alternative_experts(self, project_id: int, ranked_df: pd.DataFrame, proposed_experts: List[int]) -> List[int]:
        """
        Assign alternative experts for a project, applying flexibility only if needed.
        """
        alternative_experts = self._try_assign_experts(
            project_id,
            ranked_df,
            'Alternative',
            num_experts_to_assign=self.num_alternative_experts,
            min_women=self.min_women_alternative,
            exclude_experts=proposed_experts,
            flexible=False  # Strict mode first.
        )

        # If fewer than half the required alternatives are assigned, try again with flexibility
        #min_alternative_experts = max(1, -(-self.num_alternative_experts // 2))  # Ceiling division
        min_alternative_experts = self.num_alternative_experts
        if len(alternative_experts) < min_alternative_experts:
            alternative_experts = self._try_assign_experts(
                project_id,
                ranked_df,
                'Alternative',
                num_experts_to_assign=min_alternative_experts,
                min_women=self.min_women_alternative,
                exclude_experts=proposed_experts,
                flexible=True  # Flexible mode.
            )
        return alternative_experts

    def _try_assign_experts(self, project_id: int, ranked_df: pd.DataFrame, assignment_type: str, num_experts_to_assign: int, 
                            min_women: int, exclude_experts: List[int] = None, flexible: bool = False) -> List[int]:
        """
        Attempt to assign a specified number of experts, optionally with flexibility.
        Args:
            project_id (int): The ID of the project.
            ranked_df (pd.DataFrame): Ranked data for the project.
            assignment_type (str): Whether we are assigning proposed or alternative experts.
            num_experts_to_assign (int): The number of experts to assign.
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
            if assignment_type == 'Proposed':
                max_projects = expert_row[self.expert_max_proposed_projects_output_col]
                assignment_count = self.expert_assignment_count_proposed
            else:
                max_projects = self.max_total_projects_per_expert
                assignment_count = self.expert_assignment_count_total

            # Get current assignments for the assignment type.
            current_assignments = assignment_count.get(expert_id, 0)

            # Adjust capacity if flexibility is enabled
            if flexible:
                max_projects = int(max_projects * 1.5)

            if current_assignments >= max_projects:
                continue

            # Check gender requirements
            is_woman = expert_row[self.expert_gender_output_col].lower() == self.expert_gender_value_women
            slots_remaining = num_experts_to_assign - len(assigned_experts)

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
            assignment_count[expert_id] = current_assignments + 1
            if is_woman:
                women_count += 1

            # Stop if we have assigned enough experts
            if len(assigned_experts) >= num_experts_to_assign:
                break

        return assigned_experts

    def _balance_assignments(self, assignments_df: pd.DataFrame) -> pd.DataFrame:
        """
        Rebalance assignments so that experts who have only one 'Proposed' assignment get an extra
        'Proposed' slot (by swapping from one of their Alternative assignments) if possible, without creating new rows.
        
        For each expert with only 1 Proposed assignment:
          1. For each project in which the expert is assigned as Alternative, check if that project has
             fewer than self.num_proposed_experts Proposed experts. If yes, simply upgrade that row.
          2. Otherwise (the project is full), attempt a swap:
             - Among the experts already assigned as Proposed for that project, select the candidate 
               with the highest total number of Proposed assignments (i.e. the most heavily loaded expert).
             - If that candidate’s overall Proposed count is greater than 1, swap the assignments:
                 * Demote that candidate from Proposed to Alternative.
                 * Upgrade the current expert’s row (from Alternative to Proposed).
             - In either case the total number of rows in the project remains unchanged.
        
        This implementation does not consider gender at this stage.
        """
        # Count the current Proposed assignments per expert.
        proposed_counts = assignments_df[
            assignments_df['Assignment_Type'] == 'Proposed'
        ][self.expert_id_output_col].value_counts()
        experts_with_one_proposed = proposed_counts[proposed_counts == 1].index.tolist()

        for expert_id in experts_with_one_proposed:
            # Find all rows for this expert that are marked as Alternative.
            alt_rows = assignments_df[
                (assignments_df[self.expert_id_output_col] == expert_id) &
                (assignments_df['Assignment_Type'] == 'Alternative')
            ]
            # For each project where this expert is currently Alternative, try to convert that row.
            for idx, row in alt_rows.iterrows():
                project_id = row[self.project_id_output_col]
                # Get the current Proposed assignments in this project.
                current_proposed = assignments_df[
                    (assignments_df[self.project_id_output_col] == project_id) &
                    (assignments_df['Assignment_Type'] == 'Proposed')
                ]
                if len(current_proposed) < self.num_proposed_experts:
                    # There is a free slot: upgrade this Alternative to Proposed.
                    assignments_df.at[idx, 'Assignment_Type'] = 'Proposed'
                    proposed_counts[expert_id] = proposed_counts.get(expert_id, 0) + 1
                    break  # We've fixed this expert; proceed to the next.
                else:
                    # The project already has the maximum number of Proposed experts.
                    # Try to swap by choosing among the current Proposed experts the one with
                    # the highest total number of Proposed assignments.
                    candidate_rows = current_proposed.copy()
                    # Add a temporary column 'total_proposed' based on our overall counts.
                    candidate_rows['total_proposed'] = candidate_rows[self.expert_id_output_col].apply(
                        lambda e: proposed_counts.get(e, 0)
                    )
                    # Sort descending by 'total_proposed' so that the most-loaded expert is first.
                    candidate_rows = candidate_rows.sort_values(by='total_proposed', ascending=False)
                    swapped = False
                    for candidate_idx, candidate_row in candidate_rows.iterrows():
                        candidate_expert_id = candidate_row[self.expert_id_output_col]
                        candidate_count = proposed_counts.get(candidate_expert_id, 0)
                        if candidate_count > 1:
                            # Swap: upgrade our expert's Alternative row and demote the candidate.
                            assignments_df.at[idx, 'Assignment_Type'] = 'Proposed'
                            assignments_df.at[candidate_idx, 'Assignment_Type'] = 'Alternative'
                            # Update our counts.
                            proposed_counts[candidate_expert_id] = candidate_count - 1
                            proposed_counts[expert_id] = proposed_counts.get(expert_id, 0) + 1
                            swapped = True
                            break
                    if swapped:
                        # If after the swap our expert now has at least 2 Proposed assignments, move on.
                        new_count = assignments_df[
                            (assignments_df[self.expert_id_output_col] == expert_id) &
                            (assignments_df['Assignment_Type'] == 'Proposed')
                        ].shape[0]
                        if new_count >= 2:
                            break  # Done for this expert.
        return assignments_df


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


    # Not using this now !!!
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
        if self.expert_max_proposed_projects_input_col not in experts_df.columns:
            experts_df[self.expert_max_proposed_projects_input_col] = self.max_proposed_projects_per_expert
        else:
            mask = (experts_df[self.expert_max_proposed_projects_input_col].isna()) | (experts_df[self.expert_max_proposed_projects_input_col] == 0)
            if mask.any():
                experts_df.loc[mask, self.expert_max_proposed_projects_input_col] = self.max_proposed_projects_per_expert

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
                    self.expert_max_proposed_projects_input_col
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
        # Handle max projects per expert
        if self.expert_max_proposed_projects_input_col not in experts_df.columns:
            experts_df[self.expert_max_proposed_projects_input_col] = self.max_proposed_projects_per_expert
        else:
            mask = (experts_df[self.expert_max_proposed_projects_input_col].isna()) | (experts_df[self.expert_max_proposed_projects_input_col] == 0)
            if mask.any():
                experts_df.loc[mask, self.expert_max_proposed_projects_input_col] = self.max_proposed_projects_per_expert

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
                self.expert_max_proposed_projects_input_col
            ]],
            left_on=self.expert_id_output_col,
            right_on=self.expert_id_input_col,
            how='right'
        )
        
        # Fill NaN values for experts with no assignments
        distribution['Num_Assignments'] = distribution['Num_Assignments'].fillna(0).astype(int)
               
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
        
        # Calculate utilization percentage
        distribution['Utilization_Pct'] = (
            distribution['Num_Proposed'] / distribution[self.expert_max_proposed_projects_input_col] * 100
        ).round(1)
        
        # Reorder columns and sort by utilization
        result = distribution[[
            self.expert_id_output_col,
            self.expert_name_input_col,
            self.expert_gender_input_col,
            'Num_Assignments',
            'Num_Proposed',
            'Num_Alternative',
            self.expert_max_proposed_projects_input_col,
            'Utilization_Pct',
            'Avg_Assignment_Probability'
        ]].rename(columns={
            self.expert_id_output_col: 'Expert_ID',
            self.expert_name_input_col: 'Expert_Name',
            self.expert_gender_input_col: 'Expert_Gender',
            self.expert_max_proposed_projects_input_col: 'Max_Projects'
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
    
    def get_multidisciplinary_coverage(self, final_assignments_df: pd.DataFrame, 
                                         projects_df: pd.DataFrame, 
                                         experts_df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes the coverage score for each project based on the final assignments.
        
        The coverage score is defined as the fraction of the project's research approaches 
        (pipe-separated in projects_df) that are present in the union of research approaches 
        from experts assigned as 'Proposed' (in final_assignments_df).

        Args:
            final_assignments_df (pd.DataFrame): DataFrame containing the final balanced assignments.
            projects_df (pd.DataFrame): DataFrame containing project information.
            experts_df (pd.DataFrame): DataFrame containing expert information.
            
        Returns:
            pd.DataFrame: DataFrame with columns ['Project_ID', 'Coverage_Score'].
        """
        results = []
        # Use the project IDs from projects_df (or you could use final_assignments_df if IDs are consistent)
        for project_id in projects_df[self.project_id_input_col].unique():
            # Retrieve the project row.
            project_row = projects_df[projects_df[self.project_id_input_col] == project_id].iloc[0]
            
            # Get the project's research approaches (pipe-separated string)
            proj_approaches_str = project_row.get(self.project_research_approaches_input_col, "")
            if not isinstance(proj_approaches_str, str) or proj_approaches_str.strip() == "":
                project_approaches = set()
            else:
                project_approaches = set(s.strip() for s in proj_approaches_str.split('|') if s.strip())
            
            # Retrieve all experts assigned as 'Proposed' for this project from the final assignments.
            proposed_assignments = final_assignments_df[
                (final_assignments_df[self.project_id_output_col] == project_id) &
                (final_assignments_df['Assignment_Type'] == 'Proposed')
            ]
            proposed_expert_ids = proposed_assignments[self.expert_id_output_col].unique()
            
            # Build the union of research approaches from all these experts.
            combined_expert_approaches = set()
            for expert_id in proposed_expert_ids:
                expert_rows = experts_df[experts_df[self.expert_id_input_col] == expert_id]
                if expert_rows.empty:
                    continue
                expert_approaches_str = expert_rows.iloc[0].get(self.expert_research_approaches_input_col, "")
                if isinstance(expert_approaches_str, str) and expert_approaches_str.strip() != "":
                    expert_approaches = set(s.strip() for s in expert_approaches_str.split('|') if s.strip())
                    combined_expert_approaches.update(expert_approaches)
            
            # Compute coverage as the fraction of the project approaches that are covered.
            if project_approaches:
                coverage = len(project_approaches.intersection(combined_expert_approaches)) / len(project_approaches)
            else:
                coverage = 0.0
            
            results.append({'Project_ID': project_id, 'Coverage_Score': coverage})
        
        return pd.DataFrame(results)
