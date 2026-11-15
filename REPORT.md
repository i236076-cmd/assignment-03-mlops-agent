# Assignment 03: MLOps AI Agent. Technical Report

Name: Sardar Muneeb Ur Rehman  |  ID: 23i-6076  |  Repo: https://github.com/i236076-cmd/assignment-03-mlops-agent

## 1. System architecture
Paste or redraw the diagram from `architecture.md`. In 4 or 5 sentences explain: the FastAPI interface receives
the question, the orchestrator lets the LLM (Groq) choose tools, each tool talks to DVC, MLflow, Airflow 3 or
Kubernetes, and the answer is built only from tool results. If Groq is not available, a rule-based planner takes over.

## 2. Tool definitions and permissions
Copy the table from `architecture.md`. Add: read tools run freely. Write tools (`airflow_trigger_dag`,
`k8s_restart_deployment`, `k8s_scale_deployment`) need `confirm=true` set by the user (not the LLM), and are
limited to the approved DAG and the approved namespace `mlops-demo`.

## 3. Evidence from the four tools
Add screenshots (save them in `screenshots/`):
- DVC: output of `dvc status`, `git tag`, `cat data/dataset.csv.dvc`
- MLflow: experiment page with runs and the model registry page (`localhost:5000`)
- Airflow: DAG graph and run list (`localhost:8080`)
- Kubernetes: `kubectl -n mlops-demo get pods,svc,deploy` and `describe deployment churn-api`

## 4. Five tested agent conversations
The files `screenshots/logs/demo1_*.txt` to `demo5_*.txt` are written by `scripts/run_demo.sh`.
For each one write the question, which tools were called (from the evidence table in the output), and the answer.
| # | Scenario | Question | Tools used | Result |
|---|---|---|---|---|
| 1 | Dataset provenance | | | |
| 2 | Model selection | | | |
| 3 | Pipeline execution | | | |
| 4 | Deployment diagnosis | | | |
| 5 | Multi-tool reasoning | | | |

## 5. Two failure investigations
**Failure 1: quality gate failed.** A pipeline run was started with `min_f1 = 0.95`. Evaluation gave F1 0.739, so
`quality_gate` failed and nothing was promoted. Ask the agent: "Why did the last training run fail?"
It uses `airflow_get_failed_tasks` and shows the log line "f1 0.7390 is below the minimum 0.95".
Root cause: threshold higher than the model quality. Resolution: run again with a realistic threshold (0.70) or improve the model.

**Failure 2: bad image in the deployment.** Run `kubectl -n mlops-demo set image deployment/churn-api churn-api=churn-api:does-not-exist`.
Ask: "Why is my prediction API unavailable?" The agent calls `k8s_diagnose_deployment` and reports ImagePullBackOff/ErrImagePull
with the Warning events. Resolution: `kubectl -n mlops-demo rollout undo deployment/churn-api`. Paste your real output here.

(Optional third) MLflow stopped: the agent says MLflow is unavailable and does not guess metrics.

## 6. Known limitations and improvements
- The rule-based fallback only understands keywords; the Groq model understands free text but has rate limits.
- Retraining advice uses simple evidence rules; a real freshness policy and drift metrics are not implemented.
- Kubernetes access uses the local kubeconfig; a restricted service account (RBAC) would be safer.
- CI mocks all services; a dedicated test cluster could run integration tests.
- Possible next steps: port-forward and call the live `/version` endpoint, stream answers, add model drift checks.

---
## Appendix: actual agent outputs

### demo1_dataset_provenance
```
### Which dataset version was used to train the best model, and how do I reproduce it?
Known facts (from tool results):
- list_tracked_datasets: {"count": 1, "datasets": [{"dvc_file": "data/dataset.csv.dvc", "data_path": "data/dataset.csv", "md5": "ebfad498aa6eec419fff9fc1b4930796", "size_bytes": 31160, "version": "v2"}]}
- get_current_dataset_version: {"dataset": "data/dataset.csv", "version": "v2", "md5": "ebfad498aa6eec419fff9fc1b4930796", "size_bytes": 31160, "commit_that_recorded_it": "417aec0c63d597690e96e0a3880794498d4bedf8", "commit_message": "Add dataset v2 (1500 rows)", "repo_head_commit": "bfd3503a564637994726144181c49859b966432b", "pointer_differs_from_head": false, "available_version
- find_best_run: {"metric": "f1", "runs_considered": 7, "best": {"run_id": "6fc262121a8a42a984eb7fd5b80a5a0c", "run_name": "rf_medium_v1", "status": "FINISHED", "started": "2026-11-14T19:04:46+00:00", "source": "manual", "dataset_version": "v1", "quality_gate": null, "metrics": {"accuracy": 0.77, "precision": 0.7315, "recall": 0.8229, "f1": 0.7745, "roc_auc": 0.850
- get_dataset_for_run: {"run_id": "6fc262121a8a42a984eb7fd5b80a5a0c", "run_name": "rf_medium_v1", "dataset_version": "v1", "dataset_md5": "f4731343419b90cc97b98a25874fbbba", "dataset_commit": "417aec0c63d597690e96e0a3880794498d4bedf8", "code_git_commit": "f7826a29a3b2f25f7ccab9bab6b111fd083a474b", "code_had_uncommitted_changes": "true", "recorded_version_matches_git_tags

Uncertainties:
- none reported by the tools

Recommended actions:
- No pipeline was started and nothing in the cluster was changed.

--- evidence (request a320f451657f, LLM: rule-based) ---
list_tracked_datasets            ok               8 ms  
get_current_dataset_version      ok              13 ms  
find_best_run                    ok              69 ms  
get_dataset_for_run              ok              21 ms  
```

### demo2_model_selection
```
### Compare the last five training runs and recommend the best candidate based on F1 score.
Known facts (from tool results):
- find_best_run: {"metric": "f1", "runs_considered": 7, "best": {"run_id": "6fc262121a8a42a984eb7fd5b80a5a0c", "run_name": "rf_medium_v1", "status": "FINISHED", "started": "2026-11-14T19:04:46+00:00", "source": "manual", "dataset_version": "v1", "quality_gate": null, "metrics": {"accuracy": 0.77, "precision": 0.7315, "recall": 0.8229, "f1": 0.7745, "roc_auc": 0.850
- compare_runs: {"runs": [{"run_id": "7fcf596aa03b4c1a9825a5a7e0db1ed2", "run_name": "airflow_manual__2026-11-15T07_40_08.464802_00_00", "status": "FINISHED", "started": "2026-11-15T07:40:20+00:00", "source": "airflow", "dataset_version": "v2", "quality_gate": "failed", "metrics": {"accuracy": 0.7433, "precision": 0.7124, "recall": 0.7676, "f1": 0.739, "roc_auc": 
- airflow_get_dag_runs: {"dag_id": "training_pipeline", "count": 2, "runs": [{"run_id": "manual__2026-11-15T07:40:08.464802+00:00", "state": "failed", "run_type": "manual", "started": "2026-11-15T07:40:09.823996Z", "ended": "2026-11-15T07:40:28.895651Z", "conf": {"min_f1": 0.95}}, {"run_id": "manual__2026-11-15T07:32:31.885130+00:00", "state": "success", "run_type": "manu

Uncertainties:
- none reported by the tools

Recommended actions:
- The latest training run failed. Look at its failed task and log (airflow_get_failed_tasks) before deciding about retraining.
- No pipeline was started and nothing in the cluster was changed.

--- evidence (request a5095d4482b1, LLM: rule-based) ---
find_best_run                    ok             169 ms  
compare_runs                     ok              35 ms  
airflow_get_dag_runs             ok              25 ms  
```

### demo3_pipeline_execution
```
### Show the status of the recent training pipeline runs and their tasks.
Known facts (from tool results):
- airflow_get_dag_runs: {"dag_id": "training_pipeline", "count": 2, "runs": [{"run_id": "manual__2026-11-15T07:40:08.464802+00:00", "state": "failed", "run_type": "manual", "started": "2026-11-15T07:40:09.823996Z", "ended": "2026-11-15T07:40:28.895651Z", "conf": {"min_f1": 0.95}}, {"run_id": "manual__2026-11-15T07:32:31.885130+00:00", "state": "success", "run_type": "manu

Uncertainties:
- none reported by the tools

Recommended actions:
- The latest training run failed. Look at its failed task and log (airflow_get_failed_tasks) before deciding about retraining.
- No pipeline was started and nothing in the cluster was changed.

--- evidence (request 985027c18587, LLM: rule-based) ---
airflow_get_dag_runs             ok              23 ms  
```

### demo4_deployment
```
### Why might my prediction API be unavailable? Check the Kubernetes pods, events and logs.
Known facts (from tool results):
- k8s_diagnose_deployment: {"truncated_preview": "{\"namespace\": \"mlops-demo\", \"deployment\": {\"name\": \"churn-api\", \"desired_replicas\": 2, \"ready_replicas\": 2, \"updated_replicas\": 2, \"available_replicas\": 2, \"images\": [\"churn-api:1.0.0\"], \"version_label\": \"1.0.0\", \"conditions\": [{\"type\": \"Available\", \"status\": \"True\", \"reason\": \"MinimumRe

Uncertainties:
- none reported by the tools

Recommended actions:
- No pipeline was started and nothing in the cluster was changed.

--- evidence (request c44280418d99, LLM: rule-based) ---
k8s_diagnose_deployment          ok              77 ms  
```

### demo5_multi_tool
```
### Find the best recent model, identify the dataset version used to train it, check the deployment status, and explain whether retraining is necessary.
Known facts (from tool results):
- list_tracked_datasets: {"count": 1, "datasets": [{"dvc_file": "data/dataset.csv.dvc", "data_path": "data/dataset.csv", "md5": "ebfad498aa6eec419fff9fc1b4930796", "size_bytes": 31160, "version": "v2"}]}
- get_current_dataset_version: {"dataset": "data/dataset.csv", "version": "v2", "md5": "ebfad498aa6eec419fff9fc1b4930796", "size_bytes": 31160, "commit_that_recorded_it": "417aec0c63d597690e96e0a3880794498d4bedf8", "commit_message": "Add dataset v2 (1500 rows)", "repo_head_commit": "bfd3503a564637994726144181c49859b966432b", "pointer_differs_from_head": false, "available_version
- find_best_run: {"metric": "f1", "runs_considered": 7, "best": {"run_id": "6fc262121a8a42a984eb7fd5b80a5a0c", "run_name": "rf_medium_v1", "status": "FINISHED", "started": "2026-11-14T19:04:46+00:00", "source": "manual", "dataset_version": "v1", "quality_gate": null, "metrics": {"accuracy": 0.77, "precision": 0.7315, "recall": 0.8229, "f1": 0.7745, "roc_auc": 0.850
- airflow_get_dag_runs: {"dag_id": "training_pipeline", "count": 2, "runs": [{"run_id": "manual__2026-11-15T07:40:08.464802+00:00", "state": "failed", "run_type": "manual", "started": "2026-11-15T07:40:09.823996Z", "ended": "2026-11-15T07:40:28.895651Z", "conf": {"min_f1": 0.95}}, {"run_id": "manual__2026-11-15T07:32:31.885130+00:00", "state": "success", "run_type": "manu
- k8s_get_deployed_version: {"namespace": "mlops-demo", "deployment": "churn-api", "images": ["churn-api:1.0.0"], "version_label": "1.0.0", "configmap_app_version": "1.0.0", "replicas": {"desired": 2, "ready": 2, "updated": 2, "available": 2}, "rollout_complete": true}
- k8s_find_unhealthy_pods: {"namespace": "mlops-demo", "total_pods": 2, "unhealthy_pods": [], "pods_with_restarts": [{"name": "churn-api-7557d78fdb-gt2gb", "phase": "Running", "ready": true, "restarts": 1, "healthy": true, "images": ["churn-api:1.0.0"], "node": "minikube", "started": "2026-11-14T20:32:26+00:00", "problems": ["churn-api last restart was caused by: Unknown (ex
- get_dataset_for_run: {"run_id": "6fc262121a8a42a984eb7fd5b80a5a0c", "run_name": "rf_medium_v1", "dataset_version": "v1", "dataset_md5": "f4731343419b90cc97b98a25874fbbba", "dataset_commit": "417aec0c63d597690e96e0a3880794498d4bedf8", "code_git_commit": "f7826a29a3b2f25f7ccab9bab6b111fd083a474b", "code_had_uncommitted_changes": "true", "recorded_version_matches_git_tags

Uncertainties:
- none reported by the tools

Recommended actions:
- The latest training run failed. Look at its failed task and log (airflow_get_failed_tasks) before deciding about retraining.
- No pipeline was started and nothing in the cluster was changed.

--- evidence (request 232478b8996f, LLM: rule-based) ---
list_tracked_datasets            ok               8 ms  
get_current_dataset_version      ok              14 ms  
find_best_run                    ok              65 ms  
airflow_get_dag_runs             ok              24 ms  
k8s_get_deployed_version         ok              29 ms  
k8s_find_unhealthy_pods          ok              52 ms  
get_dataset_for_run              ok              20 ms  
```

### Failure 1: quality gate failed (root cause: threshold min_f1=0.95 is above the model F1 of about 0.74; fix: use a realistic threshold)
```
Known facts (from tool results):
- airflow_get_dag_runs: {"dag_id": "training_pipeline", "count": 2, "runs": [{"run_id": "manual__2026-11-15T07:40:08.464802+00:00", "state": "failed", "run_type": "manual", "started": "2026-11-15T07:40:09.823996Z", "ended": "2026-11-15T07:40:28.895651Z", "conf": {"min_f1": 0.95}}, {"run_id": "manual__2026-11-15T07:32:31.885130+00:00", "state": "success", "run_type": "manu
- airflow_get_failed_tasks: {"dag_id": "training_pipeline", "run_id": "manual__2026-11-15T07:40:08.464802+00:00", "run_state": "failed", "failed_tasks": [{"task_id": "quality_gate", "state": "failed", "try_number": 1, "max_tries": 0, "started": "2026-11-15T07:40:26.459914Z", "ended": "2026-11-15T07:40:28.515772Z", "duration_s": 4.056}], "blocked_by_failure": [], "diagnostics"

Uncertainties:
- none reported by the tools

Recommended actions:
- The latest training run failed. Look at its failed task and log (airflow_get_failed_tasks) before deciding about retraining.
- No pipeline was started and nothing in the cluster was changed.

--- evidence (request de8159ba38d3, LLM: rule-based) ---
airflow_get_dag_runs             ok              23 ms  
airflow_get_failed_tasks         ok              57 ms  
```

### Failure 2: bad image tag (root cause: churn-api:does-not-exist cannot be pulled; fix: kubectl rollout undo)
```
Known facts (from tool results):
- k8s_diagnose_deployment: {"truncated_preview": "{\"namespace\": \"mlops-demo\", \"deployment\": {\"name\": \"churn-api\", \"desired_replicas\": 2, \"ready_replicas\": 2, \"updated_replicas\": 1, \"available_replicas\": 2, \"images\": [\"churn-api:does-not-exist\"], \"version_label\": \"1.0.0\", \"conditions\": [{\"type\": \"Available\", \"status\": \"True\", \"reason\": \"

Uncertainties:
- none reported by the tools

Recommended actions:
- No pipeline was started and nothing in the cluster was changed.

--- evidence (request 9c5ddc1662cd, LLM: rule-based) ---
k8s_diagnose_deployment          ok              85 ms  
```
