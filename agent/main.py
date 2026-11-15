"""FastAPI interface for the agent, plus a small command line.

Run the API:   uvicorn agent.main:app --port 8001
Ask from CLI:  python -m agent.main "Which pods are unhealthy?"
"""
import argparse
import json

from fastapi import FastAPI
from pydantic import BaseModel, Field

from agent.orchestrator import build_agent
from agent.tools.base import REGISTRY

app = FastAPI(title="MLOps AI Agent")
_agent = {"instance": None}


def get_agent():
    if _agent["instance"] is None:
        _agent["instance"] = build_agent()
    return _agent["instance"]


class ChatRequest(BaseModel):
    query: str = Field(min_length=3, max_length=1000)
    confirm: bool = Field(
        default=False,
        description="Set to true only when you explicitly approve changes (trigger DAG, restart, scale).",
    )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/tools")
def tools():
    return [{"name": t["name"], "permission": t["permission"], "description": t["description"]}
            for t in REGISTRY.values()]


@app.post("/chat")
def chat(request: ChatRequest):
    return get_agent().run(request.query, confirmed=request.confirm)


def cli():
    parser = argparse.ArgumentParser(description="Ask the MLOps agent a question")
    parser.add_argument("query")
    parser.add_argument("--confirm", action="store_true", help="approve write actions for this request")
    parser.add_argument("--json", action="store_true", help="print the full result as JSON")
    args = parser.parse_args()
    result = get_agent().run(args.query, confirmed=args.confirm)
    if args.json:
        print(json.dumps(result, indent=2, default=str))
        return
    print(result["answer"])
    print("\n--- evidence (request %s, LLM: %s) ---" % (result["request_id"], result["llm"]))
    for t in result["tool_calls"]:
        print("%-32s %-12s %5d ms  %s" % (t["tool"], t["status"], t["duration_ms"], t["error"] or ""))


if __name__ == "__main__":
    cli()
