from fastapi.testclient import TestClient

from agent import main
from agent.orchestrator import Agent
from tests.conftest import ScriptedLLM


def test_api_validates_requests_and_answers(dummy_tools):
    main._agent["instance"] = Agent(llm=ScriptedLLM([{"content": "hello", "tool_calls": []}]))
    client = TestClient(main.app)
    assert client.post("/chat", json={"query": "x"}).status_code == 422          # too short
    assert client.post("/chat", json={"nope": 1}).status_code == 422
    reply = client.post("/chat", json={"query": "say hello please"}).json()
    assert reply["answer"].startswith("hello") and reply["request_id"]
    assert any(t["name"] == "airflow_trigger_dag" and t["permission"] == "write" for t in client.get("/tools").json())
