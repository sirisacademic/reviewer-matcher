import pandas as pd
from sklearn.metrics import ndcg_score
from config import (
    MODEL_NAME,
    OUTPUT_PREDICTIONS_FINAL_PATH,
    PREDICTIONS_ALL_PAIRS_FILE,
    TRAIN_EVAL_DATA_PATH,
    MANUAL_ASSIGNMENTS_NOT_TRAINING_FILE,
    PREDICTIONS_EVALUATION_FILE,
    USE_SUBSET_FINAL_ASSIGNMENTS,
    PREFIX,
    SET_FEATURES
)

def load_and_prepare_data(input_predictions_file, manual_assignments_file):
    """Load and merge predictions with manual assignments."""
    predictions = pd.read_csv(input_predictions_file, sep="\t")
    manual_assignments = pd.read_csv(manual_assignments_file, sep="\t")
    manual_assignments["Assignment"] = 1
    predictions = predictions[predictions["Project_ID"].isin(manual_assignments["Project_ID"])]
    merged_data = pd.merge(
        predictions,
        manual_assignments[["Project_ID", "Expert_ID", "Assignment"]],
        on=["Project_ID", "Expert_ID"],
        how="left"
    )
    merged_data["Assignment"] = merged_data["Assignment"].fillna(0).astype(int)
    merged_data["Predicted_Prob_Rank"] = merged_data.groupby("Project_ID")["Predicted_Prob"].rank(ascending=False)
    # Extract year from Project_ID
    merged_data["Year"] = merged_data["Project_ID"].str[:4].astype(int)
    return merged_data

def evaluate_project(project_data, top_n_thresholds, threshold_counts):
    """Evaluate metrics for a single project."""
    project_assignment = project_data["Assignment"].tolist()
    project_predicted_probs = project_data["Predicted_Prob"].tolist()
    project_ndcg = ndcg_score([project_assignment], [project_predicted_probs])
    top_n_results = {}
    expert_counts = {f"Top-{n}": {str(count): 0 for count in threshold_counts + ['N+']} for n in top_n_thresholds}
    for n in top_n_thresholds:
        top_n_experts = project_data.nsmallest(n, "Predicted_Prob_Rank")
        num_relevant_in_top_n = top_n_experts["Assignment"].sum()
        for count in threshold_counts:
            if num_relevant_in_top_n >= count:
                expert_counts[f"Top-{n}"][str(count)] += 1
        if num_relevant_in_top_n > max(threshold_counts):
            expert_counts[f"Top-{n}"]['N+'] += 1
        total_relevant = project_data["Assignment"].sum()
        percentage_in_top_n = (num_relevant_in_top_n / total_relevant) * 100 if total_relevant > 0 else 0
        top_n_results[f"Top-{n}"] = percentage_in_top_n
    return project_ndcg, top_n_results, expert_counts

def evaluate_predictions_by_year(merged_data, top_n_thresholds, threshold_counts):
    """Evaluate metrics grouped by year."""
    results_by_year = {}
    for year, year_data in merged_data.groupby("Year"):
        print(f"\n=== Year {year} ===")
        avg_ndcg, median_ndcg, avg_top_n_results, aggregated_expert_counts = evaluate_predictions_for_year(
            year_data, top_n_thresholds, threshold_counts
        )
        results_by_year[year] = {
            "avg_ndcg": avg_ndcg,
            "median_ndcg": median_ndcg,
            "avg_top_n_results": avg_top_n_results,
        }
        # Print Top-N Project Stats for this year
        for n in top_n_thresholds:
            total_projects = len(year_data["Project_ID"].unique())
            print(f"\nTop-{n} Project Stats:")
            for count_label, count in aggregated_expert_counts[f"Top-{n}"].items():
                percentage = (count / total_projects) * 100
                print(f"- Projects with {count_label} manually-assigned experts: {count} ({percentage:.2f}%)")
    return results_by_year

def evaluate_predictions_for_year(year_data, top_n_thresholds, threshold_counts):
    """Evaluate metrics for a single year."""
    all_projects_ndcg = []
    aggregated_top_n_results = {f"Top-{n}": [] for n in top_n_thresholds}
    aggregated_expert_counts = {f"Top-{n}": {str(count): 0 for count in threshold_counts + ['N+']} for n in top_n_thresholds}
    for project_id in year_data["Project_ID"].unique():
        project_data = year_data[year_data["Project_ID"] == project_id]
        project_ndcg, top_n_results, expert_counts = evaluate_project(project_data, top_n_thresholds, threshold_counts)
        all_projects_ndcg.append(project_ndcg)
        for key in aggregated_top_n_results:
            aggregated_top_n_results[key].append(top_n_results[key])
        for n in top_n_thresholds:
            for count_key, count_value in expert_counts[f"Top-{n}"].items():
                aggregated_expert_counts[f"Top-{n}"][count_key] += count_value
    avg_ndcg = sum(all_projects_ndcg) / len(all_projects_ndcg)
    median_ndcg = sorted(all_projects_ndcg)[len(all_projects_ndcg) // 2]
    avg_top_n_results = {key: sum(values) / len(values) for key, values in aggregated_top_n_results.items()}
    print(f"Average nDCG: {avg_ndcg:.3f}")
    print(f"Median nDCG: {median_ndcg:.3f}")
    for key, avg_value in avg_top_n_results.items():
        print(f"Average percentage of manually assigned experts in {key}: {avg_value:.2f}%")
    return avg_ndcg, median_ndcg, avg_top_n_results, aggregated_expert_counts

if __name__ == "__main__":
    NUMBER_ASSIGNED_REVIEWERS_TO_CHECK = 3
    prefix = f'{PREFIX}_subset_top_reviewers_' if USE_SUBSET_FINAL_ASSIGNMENTS else PREFIX
    manual_assignments_not_training_path = f"{TRAIN_EVAL_DATA_PATH}/{prefix}{MANUAL_ASSIGNMENTS_NOT_TRAINING_FILE}"
    input_predictions_file = f"{OUTPUT_PREDICTIONS_FINAL_PATH}/{MODEL_NAME}_{prefix}{SET_FEATURES}_{PREDICTIONS_ALL_PAIRS_FILE}"
    output_predictions_file = f"{OUTPUT_PREDICTIONS_FINAL_PATH}/{MODEL_NAME}_{prefix}{SET_FEATURES}_{PREDICTIONS_EVALUATION_FILE}"
    
    merged_data = load_and_prepare_data(input_predictions_file, manual_assignments_not_training_path)
    merged_data.to_csv(output_predictions_file, sep='\t', index=False)
    print(f'Evaluating predictions for {len(merged_data)} projects in file {output_predictions_file}')
    
    top_n_thresholds = [5, 10, 15, 20]
    threshold_counts = list(range(1, NUMBER_ASSIGNED_REVIEWERS_TO_CHECK + 1))
    evaluate_predictions_by_year(merged_data, top_n_thresholds, threshold_counts)


