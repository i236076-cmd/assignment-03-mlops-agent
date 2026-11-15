"""Test setup. Environment values are set before the agent package is imported,
so tests never touch real services, real logs or real secrets."""
import os
import tempfile

_tmp = tempfile.mkdtemp()
os.environ["AGENT_LOG_FILE"] = os.path.join(_tmp, "agent.log")
os.environ["AIRFLOW_PASSWORD"] = "test-password-123"
os.environ["AIRFLOW_URL"] = "http://airflow.test"
os.environ.pop("GROQ_API_KEY", None)

import pytest  # noqa: E402

from agent.tools import base  # noqa: E402


@pytest.fixture
def dummy_tools():
    """Small fake tools so agent tests do not need any real service."""
    calls = {"write": []}

    @base.register("dummy_read", "fake read tool", {"type": "object", "properties": {"x": {"type": "integer"}}})
    def dummy_read(x=1):
        return {"x": x, "value": 42}

    @base.register("dummy_down", "fake tool that is down")
    def dummy_down():
        from agent.schemas import ToolUnavailable
        raise ToolUnavailable("service is down")

    @base.register("dummy_slow", "fake slow tool", timeout=0.2)
    def dummy_slow():
        import time
        time.sleep(1)
        return {"never": "returned"}

    @base.register("dummy_write", "fake write tool", {"type": "object", "properties": {}}, permission="write")
    def dummy_write(confirm=False):
        from agent.schemas import ToolRejected
        if confirm is not True:
            raise ToolRejected("needs confirmation")
        calls["write"].append(1)
        return {"done": True}

    yield calls
    for name in ("dummy_read", "dummy_down", "dummy_slow", "dummy_write"):
        base.REGISTRY.pop(name, None)


class ScriptedLLM:
    """LLM stand-in that returns prepared replies, one per call."""
    name = "scripted"

    def __init__(self, replies):
        self.replies = list(replies)

    def chat(self, messages, tools=None):
        return self.replies.pop(0)


def tool_call(name, **arguments):
    return {"id": "c-" + name, "name": name, "arguments": arguments}
