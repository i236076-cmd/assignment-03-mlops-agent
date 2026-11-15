"""MLflow tools: experiments, runs, comparison and the model registry. Read only.

Uses MlflowClient (not the web page). Before any call the server is checked,
so a stopped server gives a clear 'unavailable' result instead of a long wait.
"""
import re
from datetime import datetime, timezone

import requests
from mlflow import MlflowClient

from agent.config import DEFAULT_EXPERIMENT, DEFAULT_MODEL_NAME, MLFLOW_URI
from agent.schemas import ToolError, ToolUnavailable
from agent.tools.base import register

SAFE_TEXT = re.compile(r"^[A-Za-z0-9_.\-:+ ]{1,200}$")
HEX_ID = re.compile(r"^[0-9a-f]{32}$")
MAX_RUNS = 50


def get_client():
    try:
        reply = requests.get(MLFLOW_URI.rstrip("/") + "/health", timeout=5)
    except requests.exceptions.RequestException:
        raise ToolUnavailable("MLflow server at %s is not reachable" % MLFLOW_URI)
    if reply.status_code != 200:
        raise ToolUnavailable("MLflow health check returned HTTP %s" % reply.status_code)
    return MlflowClient(tracking_uri=MLFLOW_URI, registry_uri=MLFLOW_URI)


def clean(value, label):
    if value is None or not SAFE_TEXT.match(str(value)):
        raise ToolError("invalid %s: %r" % (label, value))
    return str(value)


def get_experiment(client, name):
    exp = client.get_experiment_by_name(clean(name, "experiment name"))
    if exp is None:
        names = ", ".join(e.name for e in client.search_experiments())
        raise ToolError("experiment %r not found, available: %s" % (name, names))
    return exp


def resolve_run(client, ref):
    """Find a run by run id, run name or Airflow run id."""
    ref = clean(ref, "run")
    if HEX_ID.match(ref):
        return client.get_run(ref)

    ids = [e.experiment_id for e in client.search_experiments()]
    for tag in ("mlflow.runName", "airflow_run_id"):
        runs = client.search_runs(
            ids, filter_string="tags.`%s` = '%s'" % (tag, ref),
            order_by=["attributes.start_time DESC"], max_results=1,
        )
        if runs:
            return runs[0]
    raise ToolError("no run found with id, name or Airflow run id %r" % ref)


def iso(ms):
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat(timespec="seconds") if ms else None


def summarize(run):
    d = run.data
    return {
        "run_id": run.info.run_id,
        "run_name": d.tags.get("mlflow.runName"),
        "status": run.info.status,
        "started": iso(run.info.start_time),
        "source": d.tags.get("source", "manual"),
        "dataset_version": d.tags.get("dataset_version"),
        "quality_gate": d.tags.get("quality_gate"),
        "metrics": {k: round(v, 4) for k, v in d.metrics.items()},
        "params": dict(d.params),
    }


def recent_runs(client, experiment, limit, only_finished=False):
    exp = get_experiment(client, experiment)
    limit = max(1, min(int(limit), MAX_RUNS))
    query = "attributes.status = 'FINISHED'" if only_finished else ""
    return client.search_runs(
        [exp.experiment_id], filter_string=query,
        order_by=["attributes.start_time DESC"], max_results=limit,
    )


EXPERIMENT_PARAM = {"type": "string", "description": "Experiment name, default churn-prediction"}
METRIC_PARAM = {"type": "string", "description": "Metric name such as f1, roc_auc, accuracy, precision or recall"}


@register("list_experiments", "List MLflow experiments with their number of runs.")
def list_experiments():
    client = get_client()
    items = []
    for exp in client.search_experiments():
        runs = client.search_runs([exp.experiment_id], max_results=1000)
        items.append({"name": exp.name, "experiment_id": exp.experiment_id, "runs": len(runs)})
    return {"experiments": items}


@register(
    "list_runs",
    "List the most recent training runs of an experiment, newest first, with metrics, parameters and dataset version.",
    {
        "type": "object",
        "properties": {
            "experiment": EXPERIMENT_PARAM,
            "limit": {"type": "integer", "description": "How many runs, default 10, maximum 50"},
        },
    },
)
def list_runs(experiment=DEFAULT_EXPERIMENT, limit=10):
    client = get_client()
    runs = recent_runs(client, experiment, limit)
    return {"experiment": experiment, "count": len(runs), "runs": [summarize(r) for r in runs]}


@register(
    "get_run_details",
    "Get the full record of one run: metrics, parameters, provenance tags, artifacts and registered model versions. "
    "Accepts a run id, run name or Airflow run id.",
    {
        "type": "object",
        "properties": {"run": {"type": "string", "description": "run id, run name or Airflow run id"}},
        "required": ["run"],
    },
)
def get_run_details(run):
    client = get_client()
    found = resolve_run(client, run)
    tags = found.data.tags

    versions = client.search_model_versions("run_id = '%s'" % found.info.run_id)
    detail = summarize(found)
    detail["provenance"] = {
        k: tags.get(k)
        for k in ("dataset_version", "dataset_md5", "dataset_commit", "git_commit", "git_dirty", "airflow_run_id")
        if tags.get(k) is not None
    }
    detail["artifacts"] = [a.path for a in client.list_artifacts(found.info.run_id)]
    detail["registered_versions"] = [
        {"model": v.name, "version": v.version} for v in versions
    ]
    return detail


@register(
    "find_best_run",
    "Find the best finished run by a metric, with a list of caveats. "
    "A high metric alone does not make a model ready for production.",
    {
        "type": "object",
        "properties": {
            "metric": METRIC_PARAM,
            "experiment": EXPERIMENT_PARAM,
            "limit": {"type": "integer", "description": "How many recent runs to consider, default 20"},
            "dataset_version": {"type": "string", "description": "Only consider runs trained on this dataset version"},
        },
    },
)
def find_best_run(metric="f1", experiment=DEFAULT_EXPERIMENT, limit=20, dataset_version=None):
    client = get_client()
    metric = clean(metric, "metric")
    runs = [r for r in recent_runs(client, experiment, limit, only_finished=True) if metric in r.data.metrics]
    if dataset_version:
        wanted = clean(dataset_version, "dataset version")
        runs = [r for r in runs if r.data.tags.get("dataset_version") == wanted]
    if not runs:
        raise ToolError("no finished runs with metric %r found" % metric)

    ranked = sorted(runs, key=lambda r: r.data.metrics[metric], reverse=True)
    best, tags = ranked[0], ranked[0].data.tags

    caveats = []
    versions = sorted({r.data.tags.get("dataset_version", "unknown") for r in ranked})
    if len(versions) > 1:
        caveats.append(
            "the compared runs used different dataset versions (%s), so their test sets differ "
            "and the scores are not directly comparable" % ", ".join(versions)
        )
    if len(ranked) > 1:
        gap = best.data.metrics[metric] - ranked[1].data.metrics[metric]
        if gap < 0.01:
            caveats.append("the lead over the runner-up is only %.4f, which is within normal noise" % gap)
    if tags.get("source") == "airflow" and tags.get("quality_gate") != "passed":
        caveats.append("this run did not pass the quality gate")
    if tags.get("git_dirty") == "true":
        caveats.append("this run was recorded with uncommitted code changes")
    if not client.search_model_versions("run_id = '%s'" % best.info.run_id):
        caveats.append("this run has no registered model version")
    caveats.append(
        "the score comes from one hold-out split; production readiness also needs a healthy "
        "deployment, latency checks, drift monitoring and an approval step"
    )

    return {
        "metric": metric,
        "runs_considered": len(ranked),
        "best": summarize(best),
        "runner_up": summarize(ranked[1]) if len(ranked) > 1 else None,
        "caveats": caveats,
    }


@register(
    "compare_runs",
    "Compare several runs side by side. Give run ids or names, or leave empty to compare the last N runs.",
    {
        "type": "object",
        "properties": {
            "runs": {"type": "array", "items": {"type": "string"}, "description": "run ids or names to compare"},
            "last_n": {"type": "integer", "description": "compare the last N runs when no runs are given, default 5"},
            "experiment": EXPERIMENT_PARAM,
        },
    },
)
def compare_runs(runs=None, last_n=5, experiment=DEFAULT_EXPERIMENT):
    client = get_client()
    if runs:
        if len(runs) > 10:
            raise ToolError("compare at most 10 runs at a time")
        chosen = [resolve_run(client, r) for r in runs]
    else:
        chosen = recent_runs(client, experiment, last_n)

    rows = [summarize(r) for r in chosen]
    best = {}
    for name in sorted({m for row in rows for m in row["metrics"]}):
        scored = [row for row in rows if name in row["metrics"]]
        best[name] = max(scored, key=lambda row: row["metrics"][name])["run_name"]

    versions = sorted({row["dataset_version"] or "unknown" for row in rows})
    return {
        "runs": rows,
        "best_run_per_metric": best,
        "dataset_versions_involved": versions,
        "comparable_datasets": len(versions) == 1,
    }


@register(
    "compare_experiments",
    "Compare two or more experiments by their run counts and best and average scores for a metric.",
    {
        "type": "object",
        "properties": {
            "experiments": {"type": "array", "items": {"type": "string"}, "description": "experiment names"},
            "metric": METRIC_PARAM,
        },
        "required": ["experiments"],
    },
)
def compare_experiments(experiments, metric="f1"):
    client = get_client()
    metric = clean(metric, "metric")
    if len(experiments) < 1 or len(experiments) > 5:
        raise ToolError("give between 1 and 5 experiment names")

    summary = []
    for name in experiments:
        exp = get_experiment(client, name)
        runs = client.search_runs([exp.experiment_id], max_results=1000)
        scores = [r.data.metrics[metric] for r in runs if metric in r.data.metrics]
        summary.append({
            "experiment": name,
            "runs": len(runs),
            "runs_with_metric": len(scores),
            "best": round(max(scores), 4) if scores else None,
            "average": round(sum(scores) / len(scores), 4) if scores else None,
        })
    return {"metric": metric, "experiments": summary}


@register(
    "get_model_registry",
    "Get the registered model versions with their source run, dataset version, scores and aliases such as candidate.",
    {
        "type": "object",
        "properties": {"model_name": {"type": "string", "description": "Registered model name, default churn-model"}},
    },
)
def get_model_registry(model_name=DEFAULT_MODEL_NAME):
    client = get_client()
    name = clean(model_name, "model name")
    try:
        model = client.get_registered_model(name)
    except Exception:
        raise ToolError("registered model %r not found" % name)

    by_version = {}
    for alias, version in (model.aliases or {}).items():
        by_version.setdefault(str(version), []).append(alias)

    versions = []
    for mv in sorted(client.search_model_versions("name = '%s'" % name), key=lambda v: int(v.version)):
        run = client.get_run(mv.run_id)
        versions.append({
            "version": mv.version,
            "run_name": run.data.tags.get("mlflow.runName"),
            "run_id": mv.run_id,
            "dataset_version": run.data.tags.get("dataset_version"),
            "f1": round(run.data.metrics.get("f1", 0), 4) if "f1" in run.data.metrics else None,
            "roc_auc": round(run.data.metrics.get("roc_auc", 0), 4) if "roc_auc" in run.data.metrics else None,
            "aliases": by_version.get(str(mv.version), []),
        })
    return {"model": name, "versions": versions}
