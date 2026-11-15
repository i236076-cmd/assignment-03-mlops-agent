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
- get_current_dataset_version: tracked dataset is v2 (md5 ebfad498aa6eec419fff9fc1b4930796)
- find_best_run: best run by f1 is rf_medium_v1 (6fc26212) with metrics {'accuracy': 0.77, 'precision': 0.7315, 'recall': 0.8229, 'f1': 0.7745, 'roc_auc': 0.8506}, trained on dataset v1
- get_dataset_for_run: run 6fc26212 used dataset v1 (md5 f4731343419b90cc97b98a25874fbbba, data commit 417aec0c)

Uncertainties:
- The best run has no quality-gate record, so a high metric does not prove it is production-ready.
- The best run was trained on dataset v1 but the tracked dataset is now v2. No result here compares a model trained on v2.
- That run was made with uncommitted code changes, so it may not be reproducible from its git commit.

Recommended actions:
- No pipeline was started and nothing in the cluster was changed.

--- evidence (request 0643d2032f1b, LLM: rule-based) ---
list_tracked_datasets            ok               9 ms  
get_current_dataset_version      ok              13 ms  
find_best_run                    ok              66 ms  
get_dataset_for_run              ok              20 ms
```

### demo2_model_selection
```
### Compare the last five training runs and recommend the best candidate based on F1 score.
Known facts (from tool results):
- find_best_run: best run by f1 is rf_medium_v1 (6fc26212) with metrics {'accuracy': 0.77, 'precision': 0.7315, 'recall': 0.8229, 'f1': 0.7745, 'roc_auc': 0.8506}, trained on dataset v1
- compare_runs: {"runs": [{"run_id": "7fcf596aa03b4c1a9825a5a7e0db1ed2", "run_name": "airflow_manual__2026-11-15T07_40_08.464802_00_00", "status": "FINISHED", "started": "2026-11-15T07:40:20+00:00", "source": "airflow", "dataset_version": "v2", "quality_gate": "failed", "metrics": {"accuracy": 0.7433, "precision": 0.7124, "recall": 0.7676, "f1": 0.739, "roc_auc": 
- airflow_get_dag_runs: recent runs of training_pipeline: manual__2026-11-15T07:40:08.464802+00:00 = failed; manual__2026-11-15T07:32:31.885130+00:00 = success

Uncertainties:
- The best run has no quality-gate record, so a high metric does not prove it is production-ready.

Recommended actions:
- The latest training run failed. Look at its failed task and log (airflow_get_failed_tasks) before deciding about retraining.
- No pipeline was started and nothing in the cluster was changed.

--- evidence (request d4a6b8cb6aee, LLM: rule-based) ---
find_best_run                    ok              70 ms  
compare_runs                     ok              44 ms  
airflow_get_dag_runs             ok              28 ms
```

### demo3_pipeline_execution
```
### Show the status of the recent training pipeline runs and their tasks.
Known facts (from tool results):
- airflow_get_dag_runs: recent runs of training_pipeline: manual__2026-11-15T07:40:08.464802+00:00 = failed; manual__2026-11-15T07:32:31.885130+00:00 = success

Uncertainties:
- none reported by the tools

Recommended actions:
- The latest training run failed. Look at its failed task and log (airflow_get_failed_tasks) before deciding about retraining.
- No pipeline was started and nothing in the cluster was changed.

--- evidence (request abdb2fcccb4e, LLM: rule-based) ---
airflow_get_dag_runs             ok              30 ms
```

### demo4_deployment
```
### Why might my prediction API be unavailable? Check the Kubernetes pods, events and logs.
Known facts (from tool results):
- k8s_diagnose_deployment: {"truncated_preview": "{\"findings\": [\"All 2 pod(s) are Running and Ready. No fault found in the cluster data.\"], \"namespace\": \"mlops-demo\", \"deployment\": {\"name\": \"churn-api\", \"desired_replicas\": 2, \"ready_replicas\": 2, \"updated_replicas\": 2, \"available_replicas\": 2, \"images\": [\"churn-api:1.0.0\"], \"version_label\": \"1.0.

Uncertainties:
- none reported by the tools

Recommended actions:
- No pipeline was started and nothing in the cluster was changed.

--- evidence (request 5cf6c3202bbc, LLM: rule-based) ---
k8s_diagnose_deployment          ok             111 ms
```

### demo5_multi_tool
```
### Find the best recent model, identify the dataset version used to train it, check the deployment status, and explain whether retraining is necessary.
Known facts (from tool results):
- list_tracked_datasets: {"count": 1, "datasets": [{"dvc_file": "data/dataset.csv.dvc", "data_path": "data/dataset.csv", "md5": "ebfad498aa6eec419fff9fc1b4930796", "size_bytes": 31160, "version": "v2"}]}
- get_current_dataset_version: tracked dataset is v2 (md5 ebfad498aa6eec419fff9fc1b4930796)
- find_best_run: best run by f1 is rf_medium_v1 (6fc26212) with metrics {'accuracy': 0.77, 'precision': 0.7315, 'recall': 0.8229, 'f1': 0.7745, 'roc_auc': 0.8506}, trained on dataset v1
- airflow_get_dag_runs: recent runs of training_pipeline: manual__2026-11-15T07:40:08.464802+00:00 = failed; manual__2026-11-15T07:32:31.885130+00:00 = success
- k8s_get_deployed_version: image churn-api:1.0.0, version label 1.0.0, ConfigMap APP_VERSION 1.0.0, replicas {'desired': 2, 'ready': 2, 'updated': 2, 'available': 2}
- k8s_find_unhealthy_pods: {"namespace": "mlops-demo", "total_pods": 2, "unhealthy_pods": [], "pods_with_restarts": [{"name": "churn-api-7557d78fdb-gt2gb", "phase": "Running", "ready": true, "restarts": 1, "healthy": true, "images": ["churn-api:1.0.0"], "node": "minikube", "started": "2026-11-14T20:32:26+00:00", "problems": ["churn-api last restart was caused by: Unknown (ex
- get_dataset_for_run: run 6fc26212 used dataset v1 (md5 f4731343419b90cc97b98a25874fbbba, data commit 417aec0c)

Uncertainties:
- The best run has no quality-gate record, so a high metric does not prove it is production-ready.
- The best run was trained on dataset v1 but the tracked dataset is now v2. No result here compares a model trained on v2.
- That run was made with uncommitted code changes, so it may not be reproducible from its git commit.
- Nothing in these results links the deployed image to this MLflow run, so I cannot confirm the deployed model is the best run.

Recommended actions:
- The latest training run failed. Look at its failed task and log (airflow_get_failed_tasks) before deciding about retraining.
- No pipeline was started and nothing in the cluster was changed.

--- evidence (request 5d84a4d10690, LLM: rule-based) ---
list_tracked_datasets            ok              11 ms  
get_current_dataset_version      ok              20 ms  
find_best_run                    ok              80 ms  
airflow_get_dag_runs             ok              27 ms  
k8s_get_deployed_version         ok              31 ms  
k8s_find_unhealthy_pods          ok              57 ms  
get_dataset_for_run              ok              18 ms
```

### Failure 1: quality gate failed (root cause: min_f1=0.95 is above the model F1 of about 0.74; fix: use a realistic threshold such as 0.70)
```
Known facts (from tool results):
- airflow_get_dag_runs: recent runs of training_pipeline: manual__2026-11-15T07:40:08.464802+00:00 = failed; manual__2026-11-15T07:32:31.885130+00:00 = success
- airflow_get_failed_tasks: {"dag_id": "training_pipeline", "run_id": "manual__2026-11-15T07:40:08.464802+00:00", "run_state": "failed", "failed_tasks": [{"task_id": "quality_gate", "state": "failed", "try_number": 1, "max_tries": 0, "started": "2026-11-15T07:40:26.459914Z", "ended": "2026-11-15T07:40:28.515772Z", "duration_s": 4.056}], "blocked_by_failure": [], "diagnostics"

Uncertainties:
- none reported by the tools

Recommended actions:
- The latest training run failed. Look at its failed task and log (airflow_get_failed_tasks) before deciding about retraining.
- No pipeline was started and nothing in the cluster was changed.

--- evidence (request f39fb0558c18, LLM: rule-based) ---
airflow_get_dag_runs             ok              22 ms  
airflow_get_failed_tasks         ok              55 ms
```

### Failure 2: bad image tag (root cause: churn-api:does-not-exist cannot be pulled, so the new pod stays in ImagePullBackOff while the old pods keep serving; fix: kubectl rollout undo)
```
Known facts (from tool results):
- k8s_diagnose_deployment: {"truncated_preview": "{\"findings\": [\"churn-api-7875675f9d-m6bdq cannot start because its image cannot be pulled (churn-api:does-not-exist). Check the image name and tag.\", \"churn-api-7875675f9d-m6bdq is Pending, it has not been scheduled or started yet.\"], \"namespace\": \"mlops-demo\", \"deployment\": {\"name\": \"churn-api\", \"desired_rep

Uncertainties:
- none reported by the tools

Recommended actions:
- No pipeline was started and nothing in the cluster was changed.

--- evidence (request e4d4db49e6f3, LLM: rule-based) ---
k8s_diagnose_deployment          ok             110 ms
```

### Failure 2: pods during the failure
```
NAME                         READY   STATUS         RESTARTS      AGE
churn-api-7557d78fdb-gt2gb   1/1     Running        1 (69m ago)   18h
churn-api-7557d78fdb-zrv6r   1/1     Running        1 (69m ago)   18h
churn-api-7875675f9d-m6bdq   0/1     ErrImagePull   0             27s
```

### Kubernetes state after recovery
```
NAME                             READY   STATUS        RESTARTS      AGE
pod/churn-api-7557d78fdb-gt2gb   1/1     Running       1 (69m ago)   18h
pod/churn-api-7557d78fdb-zrv6r   1/1     Running       1 (69m ago)   18h
pod/churn-api-7875675f9d-m6bdq   0/1     Terminating   0             27s

NAME                TYPE        CLUSTER-IP      EXTERNAL-IP   PORT(S)   AGE
service/churn-api   ClusterIP   10.109.39.215   <none>        80/TCP    18h

NAME                        READY   UP-TO-DATE   AVAILABLE   AGE
deployment.apps/churn-api   2/2     2            2           18h
```

### DVC status and dataset tags
```
Data and pipelines are up to date.
v1
v2
```
