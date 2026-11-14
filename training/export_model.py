"""Exports a registered MLflow model to a local folder for the inference image.

Example:
    python training/export_model.py --version 1
"""
import argparse
import json
import os

import joblib
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="churn-model")
    parser.add_argument("--version", default="1")
    parser.add_argument("--out", default="model")
    args = parser.parse_args()

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    client = MlflowClient()

    mv = client.get_model_version(args.name, args.version)
    run = client.get_run(mv.run_id)

    model = mlflow.sklearn.load_model(f"models:/{args.name}/{args.version}")

    os.makedirs(args.out, exist_ok=True)
    joblib.dump(model, os.path.join(args.out, "model.joblib"))

    info = {
        "model_name": args.name,
        "model_version": args.version,
        "run_id": mv.run_id,
        "dataset_version": run.data.tags.get("dataset_version"),
        "git_commit": run.data.tags.get("git_commit"),
        "f1": run.data.metrics.get("f1"),
        "roc_auc": run.data.metrics.get("roc_auc"),
    }
    with open(os.path.join(args.out, "model_info.json"), "w") as f:
        json.dump(info, f, indent=2)

    print("exported to", args.out)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
