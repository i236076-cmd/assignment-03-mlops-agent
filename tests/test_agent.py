import json

from agent.orchestrator import Agent, FallbackLLM
from agent.tools import airflow_tools
from agent.tools.base import call_tool
from tests.conftest import ScriptedLLM, tool_call


def test_agent_selects_and_runs_tool(dummy_tools):
    llm = ScriptedLLM([
        {"content": None, "tool_calls": [tool_call("dummy_read", x=7)]},
        {"content": "Known facts: value is 42.", "tool_calls": []},
    ])
    result = Agent(llm=llm).run("read the dummy value")
    assert [t["tool"] for t in result["tool_calls"]] == ["dummy_read"]
    assert result["tool_calls"][0]["status"] == "ok"
    assert result["tool_calls"][0]["data"] == {"x": 7, "value": 42}
    assert result["unverified"] == []


def test_agent_handles_request_needing_two_tools(dummy_tools):
    llm = ScriptedLLM([
        {"content": None, "tool_calls": [tool_call("dummy_read"), tool_call("dummy_write")]},
        {"content": "done", "tool_calls": []},
    ])
    result = Agent(llm=llm).run("use both tools")
    assert [t["tool"] for t in result["tool_calls"]] == ["dummy_read", "dummy_write"]


def test_invalid_arguments_and_unknown_tool_are_errors(dummy_tools):
    assert call_tool("dummy_read", bogus=1)["status"] == "error"
    assert "invalid arguments" in call_tool("dummy_read", bogus=1)["error"]
    assert call_tool("no_such_tool")["status"] == "error"


def test_tool_timeout_is_reported_as_unavailable(dummy_tools):
    result = call_tool("dummy_slow")
    assert result["status"] == "unavailable"
    assert "timed out" in result["error"]


def test_agent_does_not_invent_results_when_tool_is_down(dummy_tools):
    llm = ScriptedLLM([
        {"content": None, "tool_calls": [tool_call("dummy_down")]},
        {"content": "Everything looks fine.", "tool_calls": []},
    ])
    result = Agent(llm=llm).run("check the service")
    assert result["tool_calls"][0]["data"] is None          # no data was made up
    assert result["unverified"][0]["tool"] == "dummy_down"
    assert "Could not verify" in result["answer"]


def test_fallback_planner_reports_failure_instead_of_guessing(dummy_tools, monkeypatch):
    # every tool the planner picks is down -> answer must list it as unknown
    from agent import orchestrator
    monkeypatch.setattr(orchestrator, "call_tool",
                        lambda name, **kw: {"tool": name, "status": "unavailable", "request_id": "r",
                                            "duration_ms": 1, "data": None, "error": "service is down"})
    result = Agent(llm=FallbackLLM()).run("which pods are unhealthy in kubernetes?")
    assert result["unverified"]
    assert "Uncertainties" in result["answer"] and "service is down" in result["answer"]


def test_llm_cannot_confirm_write_actions_itself(dummy_tools, monkeypatch):
    # the LLM asks to trigger the DAG with confirm=true; the user did not confirm
    sent = []
    monkeypatch.setattr(airflow_tools, "_http", lambda *a, **k: sent.append(a) or {})
    llm = ScriptedLLM([
        {"content": None, "tool_calls": [tool_call("airflow_trigger_dag", confirm=True)]},
        {"content": "ok", "tool_calls": []},
    ])
    result = Agent(llm=llm).run("find the best model and tell me if I should retrain")
    assert result["tool_calls"][0]["status"] == "rejected"
    assert sent == []                                        # Airflow was never called


def test_fallback_planner_chooses_expected_tools():
    plan = FallbackLLM().plan("Why is my prediction API unavailable on kubernetes?")
    assert [name for name, _ in plan] == ["k8s_diagnose_deployment"]
    plan = FallbackLLM().plan("Find the best model, the dataset version, deployment status and say if retraining is needed")
    names = [name for name, _ in plan]
    assert "find_best_run" in names and "k8s_get_deployed_version" in names and "airflow_get_dag_runs" in names
    assert "airflow_trigger_dag" not in names


def test_fallback_runs_dependent_tool_after_finding_best_run():
    run_id = "a" * 32
    messages = [
        {"role": "user", "content": "best model and which dataset version it used"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "1", "type": "function", "function": {"name": "find_best_run", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "1", "content": json.dumps(
            {"status": "ok", "data": {"best": {"run_id": run_id}}})},
    ]
    reply = FallbackLLM().chat(messages)
    assert reply["tool_calls"][0]["name"] == "get_dataset_for_run"
    assert reply["tool_calls"][0]["arguments"] == {"run": run_id}


def test_structured_log_has_fields_and_no_secrets(dummy_tools):
    from agent import config
    from agent.logging_utils import new_request_id, redact
    rid = new_request_id()
    call_tool("dummy_read", x=1)
    last = json.loads(open(config.LOG_FILE).read().strip().splitlines()[-1])
    assert last["tool"] == "dummy_read" and last["status"] == "ok"
    assert last["request_id"] == rid and "duration_ms" in last
    hidden = redact("Authorization: Bearer abc.def.ghi password=hunter2 api_key=xyz")
    assert "abc.def.ghi" not in hidden and "hunter2" not in hidden and "xyz" not in hidden


def test_write_requests_are_planned_but_blocked_without_confirmation(dummy_tools, monkeypatch):
    plan = [name for name, _ in FallbackLLM().plan("Trigger the training pipeline")]
    assert "airflow_trigger_dag" in plan
    sent = []
    monkeypatch.setattr(airflow_tools, "_http", lambda *a, **k: sent.append(a) or {})
    result = Agent(llm=FallbackLLM()).run("Trigger the training pipeline")
    blocked = [t for t in result["tool_calls"] if t["tool"] == "airflow_trigger_dag"][0]
    assert blocked["status"] == "rejected" and "confirmation" in blocked["error"]
    assert not any("dagRuns" in str(a) and "POST" in str(a) for a in sent)
