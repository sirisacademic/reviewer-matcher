# model_training.py

from sklearn.model_selection import GridSearchCV, StratifiedKFold
import joblib

def train_model(estimator, model_name, X_train, y_train, param_grid, scoring="roc_auc", cv=5, n_jobs=-1):
    """
    Train a classification model using GridSearchCV with cross-validation.
    Parameters:
    - estimator: sklearn-compatible classifier (e.g., LogisticRegression).
    - X_train: Training feature matrix.
    - y_train: Training target vector.
    - param_grid: Dictionary of hyperparameters for GridSearchCV.
    - scoring: Scoring metric for optimization (default: roc_auc).
    - cv: Number of cross-validation folds (default: 5).
    - n_jobs: Number of parallel jobs (default: -1).

    Returns:
    - best_estimator: Best model found by GridSearchCV.
    - best_params: Dictionary of the best parameters found by GridSearchCV.
    """
    grid_search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring=scoring,
        cv=StratifiedKFold(n_splits=cv),
        n_jobs=n_jobs
    )
    grid_search.fit(X_train, y_train)
    print(f"Best Parameters for {model_name}: {grid_search.best_params_}")
    return grid_search.best_estimator_, grid_search.best_params_

def save_model(model, path):
    """Save the trained model to disk."""
    joblib.dump(model, path)

def load_model(path):
    """Load a saved model from disk."""
    return joblib.load(path)

