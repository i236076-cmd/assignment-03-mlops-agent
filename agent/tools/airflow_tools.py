"""Airflow 3 tools. They call the Airflow REST API (v2) with a JWT token.

Read tools only look at DAGs, runs, tasks and logs. The one write tool can only
trigger the approved training DAG, and only after explicit user confirmation.
"""
import re
import time
from urllib.parse import quote

import requests

from agent import config
from agent.logging_utils import redact
from agent.schemas import ToolError, ToolRejected, ToolUnavailable
from agent.tools.base import register

HTTP_TIMEOUT = 8  # seconds, smaller than the tool timeout
# only these DAG parameters may be changed from the agent
ALLOWED_CONF = {
    "min_rows": (int, float),
    "test_size": (int, float),
    "model": (str,),
    "n_estimators": (int,),
    "max_depth": (int,),
    "min_f1": (int, float),
    "min_roc_auc": (int, float),
}
_token = {"value": None, "expires": 0.0}


def _request(method, path, headers=None, **kwargs):
    url = config.AIRFLOW_URL.rstrip("/") + path
    try:
        return requests.request(method, url, headers=headers or {}, timeout=HTTP_TIMEOUT, **kwargs)
    except (requests.ConnectionError, requests.Timeout) as exc:
        raise ToolUnavailable(
            "Airflow API is not reachable at %s (%s)" % (config.AIRFLOW_URL, type(exc).__name__)
        )


def _get_token():
    if _token["value"] and time.time() < _token["expires"]:
        return _token["value"]
    if not config.AIRFLOW_PASSWORD:
        raise ToolError("AIRFLOW_PASSWORD is not set, so the agent cannot log in to Airflow")
    resp = _request(
        "POST", "/auth/token",
        json={"username": config.AIRFLOW_USER, "password": config.AIRFLOW_PASSWORD},
    )
    if resp.status_code in (401, 403):
        raise ToolError("Airflow rejected the configured username/password")
    if resp.status_code >= 500:
        raise ToolUnavailable("Airflow auth endpoint failed (HTTP %s)" % resp.status_code)
    try:
        token = resp.json()["access_token"]
    except (ValueError, KeyError):
        raise ToolError("Airflow did not return an access token (HTTP %s)" % resp.status_code)
    _token["value"] = token
    _token["expires"] = time.time() + 15 * 60
    return token


def _http(method, path, **kwargs):
    """Authenticated call. Returns the decoded JSON body."""
    for attempt in (1, 2):
        resp = _request(method, path, headers={
            "Authorization": "Bearer " + _get_token(),
            "Accept": "application/json",
        }, **kwargs)
        if resp.status_code == 401 and attempt == 1:
            _token["value"] = None  # token expired, log in again once
            continue
        break
    if resp.status_code in (401, 403):
        raise ToolError("Airflow denied access (HTTP %s)" % resp.status_code)
    if resp.status_code == 404:
        raise ToolError("not found in Airflow: %s" % path)
    if resp.status_code in (400, 409, 422):
        raise ToolError("Airflow rejected the request (HTTP %s): %s" % (resp.status_code, redact(resp.text[:200])))
    if resp.status_code >= 500:
        raise ToolUnavailable("Airflow API error (HTTP %s)" % resp.status_code)
    try:
        return resp.json()
    except ValueError:
        return {"content": resp.text}


def _dag_id(dag_id):
    dag_id = dag_id or config.APPROVED_DAG
    if not isinstance(dag_id, str) or not re.fullmatch(r"[A-Za-z0-9_.\-]+", dag_id):
        raise ToolError("invalid dag_id")
    return dag_id


def _path(*parts):
    return "/api/v2/" + "/".join(quote(str(p), safe="") for p in parts)


def _run_summary(run):
    return {
        "run_id": run.get("dag_run_id"),
        "state": run.get("state"),
        "run_type": run.get("run_type"),
        "started": run.get("start_date"),
        "ended": run.get("end_date"),
        "conf": run.get("conf"),
    }


def _task_summary(task):
    return {
        "task_id": task.get("task_id"),
        "state": task.get("state"),
        "try_number": task.get("try_number"),
        "max_tries": task.get("max_tries"),
        "started": task.get("start_date"),
        "ended": task.get("end_date"),
        "duration_s": task.get("duration"),
    }


def _latest_run_id(dag_id):
    body = _http("GET", _path("dags", dag_id, "dagRuns"), params={"limit": 1, "order_by": "-run_after"})
    runs = body.get("dag_runs") or []
    if not runs:
        raise ToolError("DAG %s has no runs yet" % dag_id)
    return runs[0]["dag_run_id"]


def _log_text(body):
    content = body.get("content") if isinstance(body, dict) else body
    if isinstance(content, list):
        lines = []
        for item in content:
            lines.append(str(item.get("event", item)) if isinstance(item, dict) else str(item))
        text = "\n".join(lines)
    else:
        text = str(content)
    return redact(text)[-4000:]  # keep the end, that is where errors are


@register("airflow_list_dags", "List the DAGs known to Airflow and whether they are paused.",
          {"type": "object", "properties": {"limit": {"type": "integer", "description": "max DAGs, default 50"}}})
def airflow_list_dags(limit=50):
    body = _http("GET", _path("dags"), params={"limit": int(limit)})
    dags = []
    for d in body.get("dags", []):
        tags = [t.get("name") if isinstance(t, dict) else t for t in (d.get("tags") or [])]
        dags.append({
            "dag_id": d.get("dag_id"),
            "is_paused": d.get("is_paused"),
            "description": d.get("description"),
            "tags": tags,
            "approved_for_triggering": d.get("dag_id") == config.APPROVED_DAG,
        })
    return {"count": len(dags), "dags": dags}


@register("airflow_get_dag_runs", "Get the most recent runs of a DAG with their state (success, failed, running).",
          {"type": "object", "properties": {
              "dag_id": {"type": "string", "description": "default is the training DAG"},
              "limit": {"type": "integer", "description": "number of runs, default 5"}}})
def airflow_get_dag_runs(dag_id=None, limit=5):
    dag_id = _dag_id(dag_id)
    body = _http("GET", _path("dags", dag_id, "dagRuns"), params={"limit": int(limit), "order_by": "-run_after"})
    runs = [_run_summary(r) for r in body.get("dag_runs", [])]
    return {"dag_id": dag_id, "count": len(runs), "runs": runs}


@register("airflow_get_task_instances", "Get the state, tries and timing of every task in a DAG run (latest run by default).",
          {"type": "object", "properties": {"dag_id": {"type": "string"}, "run_id": {"type": "string"}}})
def airflow_get_task_instances(dag_id=None, run_id=None):
    dag_id = _dag_id(dag_id)
    run_id = run_id or _latest_run_id(dag_id)
    body = _http("GET", _path("dags", dag_id, "dagRuns", run_id, "taskInstances"))
    tasks = [_task_summary(t) for t in body.get("task_instances", [])]
    return {"dag_id": dag_id, "run_id": run_id, "tasks": tasks}


@register("airflow_get_task_logs", "Read the log of one task (end of the log). Use it to find why a task failed.",
          {"type": "object", "properties": {
              "dag_id": {"type": "string"}, "run_id": {"type": "string", "description": "default is the latest run"},
              "task_id": {"type": "string"}, "try_number": {"type": "integer"}},
           "required": ["task_id"]})
def airflow_get_task_logs(task_id, dag_id=None, run_id=None, try_number=None):
    dag_id = _dag_id(dag_id)
    run_id = run_id or _latest_run_id(dag_id)
    if try_number is None:
        ti = _http("GET", _path("dags", dag_id, "dagRuns", run_id, "taskInstances", task_id))
        try_number = ti.get("try_number") or 1
    body = _http("GET", _path("dags", dag_id, "dagRuns", run_id, "taskInstances", task_id, "logs", int(try_number)),
                 params={"map_index": -1})
    return {"dag_id": dag_id, "run_id": run_id, "task_id": task_id,
            "try_number": int(try_number), "log_tail": _log_text(body)}


@register("airflow_get_failed_tasks", "Find failed tasks in a DAG run (latest by default) and return the log of the first failure.",
          {"type": "object", "properties": {"dag_id": {"type": "string"}, "run_id": {"type": "string"}}})
def airflow_get_failed_tasks(dag_id=None, run_id=None):
    dag_id = _dag_id(dag_id)
    run_id = run_id or _latest_run_id(dag_id)
    run = _http("GET", _path("dags", dag_id, "dagRuns", run_id))
    tasks = _http("GET", _path("dags", dag_id, "dagRuns", run_id, "taskInstances")).get("task_instances", [])
    failed = [_task_summary(t) for t in tasks if t.get("state") == "failed"]
    skipped = [t.get("task_id") for t in tasks if t.get("state") == "upstream_failed"]
    result = {"dag_id": dag_id, "run_id": run_id, "run_state": run.get("state"),
              "failed_tasks": failed, "blocked_by_failure": skipped, "diagnostics": None}
    if failed:
        first = failed[0]
        try:
            result["diagnostics"] = airflow_get_task_logs.__wrapped__(
                task_id=first["task_id"], dag_id=dag_id, run_id=run_id, try_number=first.get("try_number"))
        except (ToolError, ToolUnavailable) as exc:
            result["diagnostics"] = {"log_error": str(exc)}
    return result


@register("airflow_trigger_dag",
          "Start a run of the approved training DAG. WRITE action: needs explicit user confirmation. "
          "Optional conf keys: min_rows, test_size, model, n_estimators, max_depth, min_f1, min_roc_auc.",
          {"type": "object", "properties": {
              "dag_id": {"type": "string"}, "conf": {"type": "object", "description": "optional DAG parameters"}}},
          permission="write")
def airflow_trigger_dag(dag_id=None, conf=None, confirm=False):
    dag_id = _dag_id(dag_id)
    if dag_id != config.APPROVED_DAG:
        raise ToolRejected("only the approved DAG '%s' can be triggered, not '%s'" % (config.APPROVED_DAG, dag_id))
    conf = conf or {}
    if not isinstance(conf, dict):
        raise ToolError("conf must be an object")
    for key, value in conf.items():
        if key not in ALLOWED_CONF:
            raise ToolRejected("parameter '%s' may not be changed from the agent" % key)
        if isinstance(value, bool) or not isinstance(value, ALLOWED_CONF[key]):
            raise ToolError("parameter '%s' has the wrong type" % key)
    if confirm is not True:
        raise ToolRejected("triggering a DAG needs explicit confirmation from the user (confirm=true)")
    body = _http("POST", _path("dags", dag_id, "dagRuns"), json={"logical_date": None, "conf": conf})
    return {"dag_id": dag_id, "triggered": True, "run": _run_summary(body),
            "note": "run started; check it with airflow_get_dag_runs"}
