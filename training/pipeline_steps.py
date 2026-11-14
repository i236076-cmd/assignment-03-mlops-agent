"""Stages of the training pipeline, one subcommand per Airflow task.

Usage:
    python training/pipeline_steps.py validate --run-id demo1
    python training/pipeline_steps.py preprocess --run-id demo1
    python training/pipeline_steps.py train --run-id demo1
    python training/pipeline_steps.py evaluate --run-id demo1
    python training/pipeline_steps.py gate --run-id demo1

A stage returns exit code 1 when its check fails. That makes the Airflow
task fail, so nothing downstream runs.
"""
import argparse
import hashlib
import json
import logging
import os
import re
import sys

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from training.provenance import DVC_FILE, get_provenance, md5_from_dvc_text  # noqa: E402

DATA_PATH = os.path.join(ROOT, "data", "dataset.csv")
WORK_ROOT = os.path.join(ROOT, "artifacts", "pipeline")
FEATURES = ["tenure", "monthly_charges", "support_calls", "usage_hours", "contract_type"]
TARGET = "churn"
EXPERIMENT = "churn-prediction"
MODEL_NAME = "churn-model"
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("pipeline")


def safe_name(run_id):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", run_id)


def work_dir(run_id):
    path = os.path.join(WORK_ROOT, safe_name(run_id))
    os.makedirs(path, exist_ok=True)
    return path


def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def file_md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def validate(args):
    out = work_dir(args.run_id)
    problems = []

    if not os.path.exists(DATA_PATH):
        log.error("dataset file not found: %s", DATA_PATH)
        return 1
    df = pd.read_csv(DATA_PATH)

    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        problems.append("missing columns: " + ", ".join(missing))
    else:
        if len(df) < args.min_rows:
            problems.append("only %d rows, need at least %d" % (len(df), args.min_rows))
        if df.isnull().sum().sum() > 0:
            problems.append("dataset contains missing values")
        if not set(df[TARGET].unique()) <= {0, 1}:
            problems.append("churn column must contain only 0 and 1")
        else:
            rate = df[TARGET].mean()
            if rate < 0.05 or rate > 0.95:
                problems.append("churn rate %.2f is extremely unbalanced" % rate)

    # the data on disk must match the hash recorded in the dvc pointer file
    with open(os.path.join(ROOT, DVC_FILE)) as f:
        expected_md5 = md5_from_dvc_text(f.read())
    actual_md5 = file_md5(DATA_PATH)
    if actual_md5 != expected_md5:
        problems.append(
            "data changed since it was tracked by DVC (expected %s, found %s)"
            % (expected_md5, actual_md5)
        )

    save_json(
        os.path.join(out, "validation.json"),
        {"rows": len(df), "problems": problems, "provenance": get_provenance()},
    )

    if problems:
        for p in problems:
            log.error("validation problem: %s", p)
        return 1
    log.info("validation passed (%d rows, md5 %s)", len(df), actual_md5)
    return 0


def preprocess(args):
    out = work_dir(args.run_id)
    df = pd.read_csv(DATA_PATH).drop_duplicates()
    train_df, test_df = train_test_split(
        df, test_size=args.test_size, random_state=42, stratify=df[TARGET]
    )
    train_df.to_csv(os.path.join(out, "train.csv"), index=False)
    test_df.to_csv(os.path.join(out, "test.csv"), index=False)
    log.info("preprocess done: %d train rows, %d test rows", len(train_df), len(test_df))
    return 0


def build_model(args):
    if args.model == "logreg":
        return LogisticRegression(max_iter=1000)
    return RandomForestClassifier(
        n_estimators=args.n_estimators, max_depth=args.max_depth, random_state=42
    )


def train(args):
    out = work_dir(args.run_id)
    train_df = pd.read_csv(os.path.join(out, "train.csv"))
    model = build_model(args)
    model.fit(train_df[FEATURES], train_df[TARGET])
    joblib.dump(model, os.path.join(out, "model.joblib"))

    is_rf = args.model == "rf"
    save_json(
        os.path.join(out, "train_info.json"),
        {
            "params": {
                "model_type": args.model,
                "n_estimators": args.n_estimators if is_rf else "na",
                "max_depth": args.max_depth if is_rf else "na",
                "test_size": args.test_size,
                "seed": 42,
                "n_rows": len(train_df),
            }
        },
    )
    log.info("training finished with model %s on %d rows", args.model, len(train_df))
    return 0


def evaluate(args):
    out = work_dir(args.run_id)
    info = load_json(os.path.join(out, "train_info.json"))
    model = joblib.load(os.path.join(out, "model.joblib"))
    test_df = pd.read_csv(os.path.join(out, "test.csv"))

    preds = model.predict(test_df[FEATURES])
    probs = model.predict_proba(test_df[FEATURES])[:, 1]
    y = test_df[TARGET]
    metrics = {
        "accuracy": float(accuracy_score(y, preds)),
        "precision": float(precision_score(y, preds, zero_division=0)),
        "recall": float(recall_score(y, preds, zero_division=0)),
        "f1": float(f1_score(y, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probs)),
    }

    prov = get_provenance()
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name="airflow_" + safe_name(args.run_id)) as run:
        mlflow.set_tags(prov)
        mlflow.set_tags(
            {"source": "airflow", "airflow_run_id": args.run_id, "quality_gate": "pending"}
        )
        mlflow.log_params(info["params"])
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(os.path.join(ROOT, DVC_FILE), artifact_path="provenance")
        # the model is logged but NOT registered here, the quality gate decides that
        model_info = mlflow.sklearn.log_model(
            model, name="model", skops_trusted_types=["sklearn.tree._tree.Tree"]
        )
        mlflow_run_id = run.info.run_id
        model_uri = model_info.model_uri

    save_json(
        os.path.join(out, "evaluation.json"),
        {
            "mlflow_run_id": mlflow_run_id,
            "model_uri": model_uri,
            "metrics": metrics,
            "dataset_version": prov["dataset_version"],
        },
    )
    log.info("logged MLflow run %s (dataset %s) f1=%.4f roc_auc=%.4f",
             mlflow_run_id, prov["dataset_version"], metrics["f1"], metrics["roc_auc"])
    return 0


def gate(args):
    out = work_dir(args.run_id)
    ev = load_json(os.path.join(out, "evaluation.json"))
    metrics = ev["metrics"]

    problems = []
    if metrics["f1"] < args.min_f1:
        problems.append("f1 %.4f is below the minimum %.2f" % (metrics["f1"], args.min_f1))
    if metrics["roc_auc"] < args.min_roc_auc:
        problems.append(
            "roc_auc %.4f is below the minimum %.2f" % (metrics["roc_auc"], args.min_roc_auc)
        )

    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()

    if problems:
        client.set_tag(ev["mlflow_run_id"], "quality_gate", "failed")
        save_json(os.path.join(out, "gate.json"), {"passed": False, "problems": problems})
        for p in problems:
            log.error("quality gate failed: %s", p)
        return 1

    # passing the gate only makes the model a candidate, a human still promotes it
    version = mlflow.register_model(ev["model_uri"], MODEL_NAME)
    client.set_registered_model_alias(MODEL_NAME, "candidate", version.version)
    client.set_tag(ev["mlflow_run_id"], "quality_gate", "passed")
    save_json(
        os.path.join(out, "gate.json"),
        {"passed": True, "registered_version": version.version, "alias": "candidate"},
    )
    log.info("quality gate passed, registered %s version %s as candidate",
             MODEL_NAME, version.version)
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["validate", "preprocess", "train", "evaluate", "gate"])
    parser.add_argument("--run-id", default="manual")
    parser.add_argument("--min-rows", type=int, default=500)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--model", choices=["rf", "logreg"], default="rf")
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=10)
    parser.add_argument("--min-f1", type=float, default=0.70)
    parser.add_argument("--min-roc-auc", type=float, default=0.80)
    args = parser.parse_args()

    os.chdir(ROOT)
    stages = {
        "validate": validate,
        "preprocess": preprocess,
        "train": train,
        "evaluate": evaluate,
        "gate": gate,
    }
    sys.exit(stages[args.stage](args))


if __name__ == "__main__":
    main()
