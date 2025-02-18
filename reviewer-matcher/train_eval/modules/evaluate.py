# evaluate.py

import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, ndcg_score, precision_score, recall_score

def evaluate_model(model, X_test, y_test):
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    # Compute metrics
    auc_score = roc_auc_score(y_test, y_pred_proba)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    ndcg = ndcg_score([y_test], [y_pred_proba]) if len(y_test) > 0 else 0.0
    # Return metrics and probabilities in a dictionary
    return {
        "AUC": auc_score,
        "Accuracy": accuracy,
        "F1": f1,
        "Precision": precision,
        "Recall": recall,
        "nDCG": ndcg,
        "probs": y_pred_proba
    }

def evaluate_model_with_thresholds(model, X_test, y_test, thresholds=[0.5]):
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    results = {
        "AUC": roc_auc_score(y_test, y_pred_proba),
        "Metrics_by_Threshold": {}
    }
    for threshold in thresholds:
        # Apply threshold to probabilities to get binary predictions
        y_pred = (y_pred_proba >= threshold).astype(int)
        # Compute metrics for the current threshold
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        ndcg = ndcg_score([y_test], [y_pred_proba]) if len(y_test) > 0 else 0.0
        # Store metrics for this threshold
        results["Metrics_by_Threshold"][threshold] = {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "nDCG": ndcg,
        }
    return results

