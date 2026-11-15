"""Training pipeline: validate -> preprocess -> train -> evaluate -> quality gate."""
import os
from datetime import datetime, timedelta

from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import DAG

PROJECT = os.getenv("MLOPS_PROJECT", os.path.expanduser("~/assignment-03-mlops-agent"))
PYTHON = os.path.join(PROJECT, ".venv", "bin", "python")
STEPS = os.path.join(PROJECT, "training", "pipeline_steps.py")
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")

default_args = {
    "owner": "muneeb",
    "retries": 2,
    "retry_delay": timedelta(seconds=20),
}

# these can be overridden with the conf of a manual trigger
params = {
    "min_rows": 500,
    "test_size": 0.2,
    "model": "rf",
    "n_estimators": 100,
    "max_depth": 10,
    "min_f1": 0.70,
    "min_roc_auc": 0.80,
}

RUN = '--run-id "{{ run_id }}"'
TASK_ENV = {
    "MLFLOW_TRACKING_URI": MLFLOW_URI,
    # fail fast when MLflow is down instead of waiting for minutes
    "MLFLOW_HTTP_REQUEST_MAX_RETRIES": "1",
    "MLFLOW_HTTP_REQUEST_TIMEOUT": "10",
    "PYTHONWARNINGS": "ignore",
}


def stage(task_id, name, extra="", **kwargs):
    return BashOperator(
        task_id=task_id,
        bash_command=PYTHON + " " + STEPS + " " + name + " " + RUN + " " + extra,
        cwd=PROJECT,
        env=TASK_ENV,
        append_env=True,
        **kwargs,
    )


with DAG(
    dag_id="training_pipeline",
    description="Churn model training with a quality gate before candidate registration",
    start_date=datetime(2026, 11, 1),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    default_args=default_args,
    params=params,
    tags=["mlops", "assignment-03"],
) as dag:

    validate_dataset = stage(
        "validate_dataset", "validate", "--min-rows {{ params.min_rows }}", retries=0
    )

    preprocess_data = stage(
        "preprocess_data", "preprocess", "--test-size {{ params.test_size }}"
    )

    train_model = stage(
        "train_model",
        "train",
        "--model {{ params.model }} --n-estimators {{ params.n_estimators }} "
        "--max-depth {{ params.max_depth }} --test-size {{ params.test_size }}",
    )

    evaluate_and_log = stage("evaluate_and_log", "evaluate")

    quality_gate = stage(
        "quality_gate",
        "gate",
        "--min-f1 {{ params.min_f1 }} --min-roc-auc {{ params.min_roc_auc }}",
        retries=0,
    )

    validate_dataset >> preprocess_data >> train_model >> evaluate_and_log >> quality_gate
