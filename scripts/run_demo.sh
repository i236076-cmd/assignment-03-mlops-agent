#!/usr/bin/env bash
# Runs the five demo scenarios and saves the agent's answers + tool evidence.
# Needs: minikube + churn-api running, MLflow server (port 5000), Airflow (port 8080), AIRFLOW_PASSWORD exported.
cd "$(dirname "$0")/.." && source .venv/bin/activate
OUT=screenshots/logs; mkdir -p "$OUT"
ask() { echo "### $2" | tee "$OUT/$1.txt"; python -m agent.main "$2" 2>/dev/null | tee -a "$OUT/$1.txt"; echo; }
ask demo1_dataset_provenance  "Which dataset version was used to train the best model, and how do I reproduce it?"
ask demo2_model_selection     "Compare the last five training runs and recommend the best candidate based on F1 score."
ask demo3_pipeline_execution  "Show the status of the recent training pipeline runs and their tasks."
ask demo4_deployment          "Why might my prediction API be unavailable? Check the Kubernetes pods, events and logs."
ask demo5_multi_tool          "Find the best recent model, identify the dataset version used to train it, check the deployment status, and explain whether retraining is necessary."
