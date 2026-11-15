"""Settings for the agent. Everything comes from environment variables,
so no passwords or tokens are stored in the repository."""
import os
from pathlib import Path

PROJECT_ROOT = Path(os.getenv("MLOPS_PROJECT", Path(__file__).resolve().parents[1]))

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
DEFAULT_EXPERIMENT = os.getenv("DEFAULT_EXPERIMENT", "churn-prediction")
DEFAULT_MODEL_NAME = os.getenv("DEFAULT_MODEL_NAME", "churn-model")

AIRFLOW_URL = os.getenv("AIRFLOW_URL", "http://127.0.0.1:8080")
AIRFLOW_USER = os.getenv("AIRFLOW_USER", "admin")
AIRFLOW_PASSWORD = os.getenv("AIRFLOW_PASSWORD", "")
APPROVED_DAG = os.getenv("APPROVED_DAG", "training_pipeline")

K8S_NAMESPACE = os.getenv("K8S_NAMESPACE", "mlops-demo")
K8S_ALLOWED_NAMESPACES = {K8S_NAMESPACE}
K8S_DEPLOYMENT = os.getenv("K8S_DEPLOYMENT", "churn-api")

# default time limit for one tool call (seconds)
TOOL_TIMEOUT = float(os.getenv("TOOL_TIMEOUT_SECONDS", "20"))

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = os.getenv("AGENT_LOG_FILE", str(PROJECT_ROOT / "logs" / "agent.log"))

# fail fast when MLflow is down instead of retrying for minutes
os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "10")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "1")
