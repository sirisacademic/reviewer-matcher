import joblib
import argparse
from config import FEATURE_COLUMNS, SET_FEATURES

def main(model_path):
    # Load the saved model.
    pipeline = joblib.load(model_path)

    # Extract LogisticRegression step from the pipeline
    lasso_model = pipeline.named_steps['classifier']

    # Print parameters of the Lasso model.
    print(f"Model Parameters for Lasso with {SET_FEATURES}")
    print(f"Model path: {model_path}")
    print(lasso_model.get_params())

    # Get coefficients from the trained Lasso model.
    lasso_coefs = lasso_model.coef_[0]  # Assuming binary classification

    # Pair feature names with coefficients
    feature_importance = list(zip(FEATURE_COLUMNS[SET_FEATURES], lasso_coefs))

    # Filter and display only non-zero coefficients
    important_features = [(name, coef) for name, coef in feature_importance if coef != 0]

    # Sort by absolute value of the coefficients
    sorted_features = sorted(important_features, key=lambda x: abs(x[1]), reverse=True)

    print("\nRanked Features:")
    for name, coef in sorted_features:
        print(f"{name}: {coef:.3f}")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Inspect a saved Lasso model's parameters.")
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to the saved Lasso model file (e.g., models/final/all_features_final_lasso.pkl)"
    )
    args = parser.parse_args()

    # Run the main function with the provided model path
    main(args.model_path)


