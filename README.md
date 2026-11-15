# Assignment 03: MLOps AI Agent

An AI agent that talks to DVC, MLflow, Apache Airflow 3 and Kubernetes through tools and explains the results.
See `architecture.md` for the design and `REPORT.md` for the technical report.

## Requirements
Linux, Python 3.10, Docker, Minikube, kubectl, git. Airflow 3 runs in its own environment (`~/airflow-env`).

## Install
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-ci.txt
```

## Environment variables (never commit them)
| Name | Meaning | Default |
|---|---|---|
| `GROQ_API_KEY` | Groq key. If missing, the rule-based planner is used | none |
| `GROQ_MODEL` | Groq model name | llama-3.3-70b-versatile |
| `MLFLOW_TRACKING_URI` | MLflow server | http://127.0.0.1:5000 |
| `AIRFLOW_URL`, `AIRFLOW_USER`, `AIRFLOW_PASSWORD` | Airflow API login | http://127.0.0.1:8080, admin, none |
| `APPROVED_DAG` | the only DAG that can be triggered | training_pipeline |
| `K8S_NAMESPACE` | the only namespace the agent may use | mlops-demo |

## Start the services (one terminal each)
```bash
# 1. Kubernetes
minikube start --driver=docker
kubectl apply -f kubernetes/            # namespace mlops-demo must exist: kubectl create ns mlops-demo
# 2. MLflow (run from the project folder)
mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
# 3. Airflow 3 (own environment)
source ~/airflow_env.sh && airflow standalone
# 4. Agent API
export AIRFLOW_PASSWORD=...  GROQ_API_KEY=...
uvicorn agent.main:app --port 8001
```

## Talk to the agent
```bash
python -m agent.main "Which dataset version is tracked by DVC?"
python -m agent.main "Find the best recent model, identify the dataset version used to train it, check the deployment status, and explain whether retraining is necessary."
curl -s localhost:8001/chat -H 'Content-Type: application/json' -d '{"query":"Why is my prediction API unavailable?"}'
# write actions only run with confirmation:
python -m agent.main "Trigger the training pipeline" --confirm
```

## Tests and lint
```bash
pytest -q
ruff check .
```
All external services are mocked in tests and in CI (`.github/workflows/ci.yml`).

## Reproduce a dataset version
```bash
git checkout v1 -- data/dataset.csv.dvc && dvc checkout data/dataset.csv.dvc
```
