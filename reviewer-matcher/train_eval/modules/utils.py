# utils.py

import numpy as np
import joblib

def save_scaler(scaler, path):
    joblib.dump(scaler, path)

def load_scaler(path):
    return joblib.load(path)

def rank_experts(data, probabilities, column_name):
    data[column_name] = probabilities
    data[f"{column_name}_Rank"] = data.groupby("Project_ID")[column_name].rank(ascending=False, method="min")
    return data
