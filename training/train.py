"""Trains a churn model and logs everything to MLflow.

Example:
    python training/train.py --model rf --n-estimators 100 --max-depth 10
"""
import argparse
import os
import sys

import mlflow
import mlflow.sklearn
import pandas as pd
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

# so that "training.provenance" can be imported when running this file directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.provenance import DVC_FILE, get_provenance  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/dataset.csv")
    p.add_argument("--experiment", default="churn-prediction")
    p.add_argument("--run-name", default=None)
    p.add_argument("--model", choices=["rf", "logreg"], default="rf")
    p.add_argument("--n-estimators", type=int, default=100)
    p.add_argument("--max-depth", type=int, default=8)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--registered-name", default="churn-model")
    p.add_argument("--no-register", action="store_true")
    return p.parse_args()


def build_model(args):
    if args.model == "rf":
        return RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=args.seed,
        )
    return LogisticRegression(max_iter=1000)


def main():
    args = parse_args()
    prov = get_provenance()

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(args.experiment)

    df = pd.read_csv(args.data)
    X = df.drop(columns=["churn"])
    y = df["churn"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    with mlflow.start_run(run_name=args.run_name) as run:
        # provenance goes in as tags so the agent can read it back later
        mlflow.set_tags(prov)

        mlflow.log_params({
            "model_type": args.model,
            "n_estimators": args.n_estimators if args.model == "rf" else "na",
            "max_depth": args.max_depth if args.model == "rf" else "na",
            "test_size": args.test_size,
            "seed": args.seed,
            "n_rows": len(df),
        })

        model = build_model(args)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]
        metrics = {
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds, zero_division=0),
            "recall": recall_score(y_test, preds, zero_division=0),
            "f1": f1_score(y_test, preds, zero_division=0),
            "roc_auc": roc_auc_score(y_test, probs),
        }
        mlflow.log_metrics(metrics)

        # keep a copy of the dvc pointer file with the run
        mlflow.log_artifact(DVC_FILE, artifact_path="provenance")

        mlflow.sklearn.log_model(
            model,
            name="model",
            input_example=X_test.head(3),
            registered_model_name=None if args.no_register else args.registered_name,
        )

        print(f"run_id: {run.info.run_id}")
        print(f"dataset_version: {prov['dataset_version']}  git_commit: {prov['git_commit'][:7]}")
        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()
