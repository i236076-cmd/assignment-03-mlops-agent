import requests

from agent.tools import airflow_tools
from agent.tools.base import call_tool


def fake_http(responses):
    def _http(method, path, **kwargs):
        for fragment, body in responses.items():
            if path.endswith(fragment):
                return body
        raise AssertionError("unexpected call %s %s" % (method, path))
    return _http


def test_get_dag_runs_returns_states(monkeypatch):
    body = {"dag_runs": [{"dag_run_id": "manual__1", "state": "success", "run_type": "manual",
                          "start_date": "t1", "end_date": "t2", "conf": {}}]}
    monkeypatch.setattr(airflow_tools, "_http", fake_http({"/dagRuns": body}))
    result = call_tool("airflow_get_dag_runs")
    assert result["status"] == "ok"
    assert result["data"]["runs"][0]["state"] == "success"


def test_failed_task_is_found_with_log(monkeypatch):
    responses = {
        "/dagRuns/run1/taskInstances/quality_gate/logs/1": {"content": ["f1 0.74 is below the minimum 0.95"]},
        "/dagRuns/run1/taskInstances": {"task_instances": [
            {"task_id": "train_model", "state": "success", "try_number": 1},
            {"task_id": "quality_gate", "state": "failed", "try_number": 1}]},
        "/dagRuns/run1": {"dag_run_id": "run1", "state": "failed"},
    }
    monkeypatch.setattr(airflow_tools, "_http", fake_http(responses))
    result = call_tool("airflow_get_failed_tasks", run_id="run1")
    assert result["status"] == "ok"
    assert result["data"]["failed_tasks"][0]["task_id"] == "quality_gate"
    assert "below the minimum" in result["data"]["diagnostics"]["log_tail"]


def test_airflow_down_is_reported_as_unavailable(monkeypatch):
    def boom(*args, **kwargs):
        raise requests.ConnectionError("refused")
    monkeypatch.setattr(airflow_tools.requests, "request", boom)
    airflow_tools._token["value"] = None
    result = call_tool("airflow_get_dag_runs")
    assert result["status"] == "unavailable"
    assert "not reachable" in result["error"]


def test_trigger_needs_confirmation_and_approved_dag(monkeypatch):
    sent = []
    monkeypatch.setattr(airflow_tools, "_http",
                        lambda m, p, **k: sent.append(p) or {"dag_run_id": "new", "state": "queued"})
    assert call_tool("airflow_trigger_dag")["status"] == "rejected"                       # no confirmation
    assert call_tool("airflow_trigger_dag", dag_id="other_dag", confirm=True)["status"] == "rejected"
    assert call_tool("airflow_trigger_dag", conf={"secret_key": 1}, confirm=True)["status"] == "rejected"
    assert sent == []
    ok = call_tool("airflow_trigger_dag", conf={"min_f1": 0.8}, confirm=True)
    assert ok["status"] == "ok" and ok["data"]["run"]["run_id"] == "new"


class FakeResponse:
    def __init__(self, status, body):
        self.status_code, self._body, self.text = status, body, str(body)

    def json(self):
        return self._body


def test_login_gets_token_and_retries_once_after_401(monkeypatch):
    calls = []

    def fake_request(method, url, headers=None, **kwargs):
        calls.append((method, url.replace("http://airflow.test", "")))
        if url.endswith("/auth/token"):
            return FakeResponse(201, {"access_token": "tok-%d" % len(calls)})
        if len([c for c in calls if c[1] != "/auth/token"]) == 1:
            return FakeResponse(401, {})                      # first call: token expired
        assert headers["Authorization"].startswith("Bearer tok-")
        return FakeResponse(200, {"dags": [{"dag_id": "training_pipeline", "is_paused": False, "tags": [{"name": "mlops"}]}]})

    monkeypatch.setattr(airflow_tools.requests, "request", fake_request)
    airflow_tools._token["value"] = None
    result = call_tool("airflow_list_dags")
    assert result["status"] == "ok"
    assert result["data"]["dags"][0]["approved_for_triggering"] is True
    assert [c[1] for c in calls].count("/auth/token") == 2    # logged in again after the 401
