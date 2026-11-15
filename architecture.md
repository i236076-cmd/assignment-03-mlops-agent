# Architecture

The agent answers questions about an ML project by calling real tools. It does not answer
operational questions from its own knowledge.

```
User / Engineer  ->  FastAPI (agent/main.py: validation, /chat, /tools)
                          |
                    Orchestrator (agent/orchestrator.py)
                    LLM (Groq) picks tools  |  rule-based planner as fallback
                    permissions + tracing + request ids
                          |
      +-----------+-------+-------+-------------+
      |           |               |             |
     DVC        MLflow         Airflow 3     Kubernetes
  dvc_tools   mlflow_tools   airflow_tools  kubernetes_tools
      |           |               |             |
  git + DVC   tracking server  REST API v2   Minikube (ns mlops-demo)
  remote      (sqlite)         (JWT login)   churn-api, 2 replicas
                          |
        Evidence-based answer: known facts, uncertainties, recommended actions
```

## Tool registry (`agent/tools/base.py`)
Every tool is registered with a name, description, JSON schema and a permission (`read` or `write`).
The wrapper gives every tool the same behaviour: argument check, timeout, error mapping
(`ok`, `error`, `unavailable`, `rejected`) and one structured log line with tool, status, duration and request id.

## Tools

| Module | Tool | Permission |
|---|---|---|
| DVC | list_tracked_datasets, get_current_dataset_version, compare_dataset_versions, check_data_changed, explain_dataset_reproduction, get_dataset_for_run | read |
| MLflow | list_experiments, list_runs, get_run_details, find_best_run, compare_runs, compare_experiments, get_model_registry | read |
| Airflow | airflow_list_dags, airflow_get_dag_runs, airflow_get_task_instances, airflow_get_task_logs, airflow_get_failed_tasks | read |
| Airflow | airflow_trigger_dag (approved DAG `training_pipeline` only, allowed parameters only) | write |
| Kubernetes | k8s_list_workloads, k8s_get_pod_status, k8s_get_pod_logs, k8s_list_services, k8s_find_unhealthy_pods, k8s_get_deployed_version, k8s_diagnose_deployment | read |
| Kubernetes | k8s_restart_deployment, k8s_scale_deployment (0 to 5 replicas) | write |

## Safety design
- **Confirmation is outside the LLM.** The orchestrator sets `confirm` for write tools from the `confirm`
  field of the user's request and hides it from the LLM. The LLM cannot approve its own action.
- **Approved scope.** Kubernetes tools only work in the approved namespace (`K8S_NAMESPACE`, default `mlops-demo`).
  Airflow can only trigger the approved DAG, and only with the allowed parameters.
- **No invented results.** A failed tool gives `data = null`. The answer lists it under "Could not verify".
- **Secrets.** Passwords and keys come from environment variables. Logs record argument names only and
  redact tokens and passwords.
- **Retraining is never started automatically.** It needs a separate confirmed request.

## Training DAG (`dags/training_pipeline.py`)
validate_dataset -> preprocess_data -> train_model -> evaluate_and_log -> quality_gate.
Tasks have retries (validation and gate have none), dependencies and logs. A failed validation or gate stops
the downstream steps, so a bad model is never registered as a candidate.

## Data lineage
Each MLflow run stores the DVC file, dataset md5 and dataset version as tags. `get_dataset_for_run` reads
that record, so the dataset answer comes from recorded provenance.
