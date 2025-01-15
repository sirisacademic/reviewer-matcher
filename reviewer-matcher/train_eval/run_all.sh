#!/usr/bin/bash

echo "Generating data..."
python3 get_training_test_data.py
echo

echo "Training/evaluating model filtering training instances with cross-validation..."
python3 train_eval_lasso.py
echo

echo "Training final model..."
python3 train_final_lasso.py
echo

latest_model_file=$(ls -t models/final | head -n 1)
echo "Inspecting final model (using ${latest_model_file})..."
python3 inspect_lasso.py "models/final/${latest_model_file}"
echo

echo "Making final predictions..."
python3 make_final_predictions.py
echo

echo "Evaluate ranking (manually assigned pairs)..."
python3 evaluate_predictions_vs_manual_assignments_projects.py
echo
