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
