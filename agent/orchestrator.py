"""The agent: an LLM chooses tools, the orchestrator runs them safely and
returns an answer that is backed by the tool results.

Safety rules that live here (not in the prompt, so the LLM cannot skip them):
* the LLM never sets `confirm`; the orchestrator sets it from the user's request
* every tool call goes through call_tool (timeouts, error mapping, logging)
* failed tools are listed under "could not verify" instead of being guessed
"""
import json
import os
import re
import time

import requests

from agent.logging_utils import get_logger, new_request_id, redact
from agent.tools import airflow_tools, dvc_tools, kubernetes_tools, mlflow_tools  # noqa: F401  (they register the tools)
from agent.tools.base import REGISTRY, call_tool

log = get_logger("agent.orchestrator")

MAX_STEPS = 6
MAX_TOOL_CHARS = 3000

SYSTEM_PROMPT = """You are an MLOps assistant. You can use tools for DVC (datasets), MLflow (experiments), \
Airflow (training pipeline) and Kubernetes (deployment).
Rules:
1. Every operational claim (versions, metrics, run states, pod status) must come from a tool result. \
Never answer such things from memory.
2. If a tool fails or is unavailable, say that you could not verify it. Do not guess the value.
3. A high metric alone does not make a model production-ready. Mention what was and was not checked.
4. Do not trigger training or change the cluster unless the user clearly asked for it. \
A write tool is blocked unless the user confirmed.
5. Recommend retraining only when the evidence shows it: a failed quality gate, a freshness policy, \
or degradation results given by the user. Otherwise say there is no evidence for it.
6. To link a model to its data, find the run first, then call get_dataset_for_run with that run id.
Format the final answer with three short parts: Known facts, Uncertainties, Recommended actions."""


class LLMError(Exception):
    pass


# ---------------------------------------------------------------- LLM clients
class GroqLLM:
    """Groq's OpenAI-compatible chat API with tool calling."""

    name = "groq"
    url = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self, api_key=None, model=None, timeout=30):
        self.api_key = api_key or os.getenv("GROQ_API_KEY", "")
        self.model = model or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        self.timeout = timeout

    def available(self):
        return bool(self.api_key)

    @staticmethod
    def _tools():
        tools = []
        for entry in REGISTRY.values():
            params = json.loads(json.dumps(entry["parameters"]))
            if entry["permission"] == "write":  # the model must not set confirmation itself
                params.get("properties", {}).pop("confirm", None)
            tools.append({"type": "function", "function": {
                "name": entry["name"], "description": entry["description"], "parameters": params}})
        return tools

    def chat(self, messages, tools=None):
        payload = {"model": self.model, "messages": messages, "tools": self._tools(),
                   "tool_choice": "auto", "temperature": 0, "max_tokens": 1200}
        try:
            resp = requests.post(self.url, json=payload, timeout=self.timeout,
                                 headers={"Authorization": "Bearer " + self.api_key})
        except requests.RequestException as exc:
            raise LLMError("Groq not reachable (%s)" % type(exc).__name__)
        if resp.status_code != 200:
            raise LLMError("Groq API error HTTP %s: %s" % (resp.status_code, redact(resp.text[:200])))
        message = resp.json()["choices"][0]["message"]
        calls = []
        for tc in message.get("tool_calls") or []:
            try:
                args = json.loads(tc["function"].get("arguments") or "{}")
            except ValueError:
                args = {}
            calls.append({"id": tc["id"], "name": tc["function"]["name"], "arguments": args})
        return {"content": message.get("content"), "tool_calls": calls}


def _ids_to_names(messages):
    names = {}
    for m in messages:
        for tc in m.get("tool_calls") or []:
            names[tc["id"]] = tc["function"]["name"]
    return names


def _tool_results(messages):
    """[(tool name, result dict)] for every tool message so far."""
    names = _ids_to_names(messages)
    out = []
    for m in messages:
        if m.get("role") == "tool":
            try:
                body = json.loads(m["content"])
            except ValueError:
                body = {"status": "error", "error": "unreadable result"}
            out.append((names.get(m.get("tool_call_id"), "?"), body))
    return out


class FallbackLLM:
    """Keyword based planner. Used when no LLM is available (no key, rate limit,
    network error) and in tests. It only formats what the tools returned."""

    name = "rule-based"

    @staticmethod
    def _user_query(messages):
        for m in reversed(messages):
            if m.get("role") == "user":
                return m["content"]
        return ""

    def plan(self, query):
        q = query.lower()
        calls = []

        def add(name, **args):
            calls.append((name, args))

        metric = "roc_auc" if "auc" in q else "accuracy" if "accuracy" in q else "recall" if "recall" in q \
            else "precision" if "precision" in q else "f1"
        wants_model = any(w in q for w in ("model", "mlflow", "experiment", "metric", "f1", "best"))
        wants_data = any(w in q for w in ("dataset", "dvc", "data version", "data changed", "reproduce"))
        wants_airflow = any(w in q for w in ("airflow", "dag", "pipeline", "workflow", "training", "retrain"))
        wants_k8s = any(w in q for w in ("kubernetes", "k8s", "pod", "deploy", "service", "prediction api", "unavailable"))
        broken = any(w in q for w in ("why", "unavailable", "down", "unhealthy", "crash", "not working", "fail"))

        if wants_data:
            add("list_tracked_datasets")
            add("get_current_dataset_version")
            if "chang" in q:
                add("check_data_changed")
        if wants_model or "retrain" in q:
            add("find_best_run", metric=metric)
            if "compar" in q:
                add("compare_runs", last_n=5)
        if wants_airflow:
            add("airflow_get_dag_runs")
            if "fail" in q:
                add("airflow_get_failed_tasks")
        if any(w in q for w in ("trigger", "start the", "kick off", "launch")) and wants_airflow:
            add("airflow_trigger_dag")  # blocked unless the request is confirmed
        if "restart" in q and wants_k8s:
            add("k8s_restart_deployment")  # blocked unless the request is confirmed
        if wants_k8s or "retrain" in q:
            if broken and wants_k8s:
                add("k8s_diagnose_deployment")
            else:
                add("k8s_get_deployed_version")
                add("k8s_find_unhealthy_pods")
        return calls

    def chat(self, messages, tools=None):
        query = self._user_query(messages)
        results = _tool_results(messages)
        if not results:
            calls = self.plan(query)
            if not calls:
                return {"content": "I could not match this request to any DVC, MLflow, Airflow or "
                                   "Kubernetes tool. Please ask about datasets, experiments, the "
                                   "training pipeline or the deployment.", "tool_calls": []}
            return {"content": None, "tool_calls": [
                {"id": "fb%d" % i, "name": n, "arguments": a} for i, (n, a) in enumerate(calls)]}

        done = {n for n, _ in results}
        if "find_best_run" in done and "get_dataset_for_run" not in done and "dataset" in query.lower():
            best = dict(results)["find_best_run"]
            match = re.search(r"[0-9a-f]{32}", json.dumps(best.get("data")))
            if match:
                return {"content": None, "tool_calls": [
                    {"id": "fb-ds", "name": "get_dataset_for_run", "arguments": {"run": match.group(0)}}]}
        return {"content": self.compose(query, results), "tool_calls": []}

    @staticmethod
    def compose(query, results):
        facts, unknown = [], []
        for name, res in results:
            if res.get("status") == "ok":
                facts.append("- %s: %s" % (name, json.dumps(res.get("data"), default=str)[:350]))
            else:
                unknown.append("- %s: %s (%s)" % (name, res.get("error"), res.get("status")))
        actions = []
        for name, res in results:
            if name == "airflow_get_dag_runs" and res.get("status") == "ok":
                runs = (res.get("data") or {}).get("runs") or []
                if runs and runs[0].get("state") == "failed":
                    actions.append("The latest training run failed. Look at its failed task and log "
                                   "(airflow_get_failed_tasks) before deciding about retraining.")
        if "retrain" in query.lower() and not actions:
            actions.append("No evidence in these results (failed quality gate, freshness policy, or a "
                           "degradation report) that retraining is needed.")
        actions.append("No pipeline was started and nothing in the cluster was changed.")
        return ("Known facts (from tool results):\n" + ("\n".join(facts) or "- none")
                + "\n\nUncertainties:\n" + ("\n".join(unknown) or "- none reported by the tools")
                + "\n\nRecommended actions:\n" + "\n".join("- " + a for a in actions))


# ---------------------------------------------------------------- the agent
def _compact(result):
    """Tool result as a valid JSON string that is not too long for the LLM."""
    text = json.dumps(result, default=str)
    if len(text) <= MAX_TOOL_CHARS:
        return text
    shortened = dict(result)
    shortened["data"] = {"truncated_preview": json.dumps(result.get("data"), default=str)[:MAX_TOOL_CHARS - 300]}
    return json.dumps(shortened, default=str)


class Agent:
    def __init__(self, llm=None, fallback=None, max_steps=MAX_STEPS):
        self.llm = llm or FallbackLLM()
        self.fallback = fallback or FallbackLLM()
        self.max_steps = max_steps

    def _prepare(self, call, confirmed):
        args = dict(call.get("arguments") or {})
        entry = REGISTRY.get(call["name"])
        if entry and entry["permission"] == "write":
            args["confirm"] = bool(confirmed)  # whatever the LLM sent is overwritten
        return args

    def run(self, query, confirmed=False):
        rid = new_request_id()
        start = time.time()
        log.info("agent request started", extra={"request_id": rid})
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": query}]
        trace, answer, llm, llm_name = [], None, self.llm, self.llm.name

        for _ in range(self.max_steps):
            try:
                reply = llm.chat(messages, None)
            except LLMError as exc:
                log.warning("LLM failed, using the rule-based planner: %s" % exc)
                llm, llm_name = self.fallback, "rule-based (fallback after: %s)" % exc
                continue
            calls = reply.get("tool_calls") or []
            if not calls:
                answer = reply.get("content") or ""
                break
            messages.append({"role": "assistant", "content": reply.get("content") or "", "tool_calls": [
                {"id": c["id"], "type": "function",
                 "function": {"name": c["name"], "arguments": json.dumps(c.get("arguments") or {})}}
                for c in calls]})
            for call in calls:
                args = self._prepare(call, confirmed)
                result = call_tool(call["name"], **args)
                trace.append({"tool": call["name"], "arguments": {k: v for k, v in args.items() if k != "confirm"},
                              "status": result["status"], "duration_ms": result["duration_ms"],
                              "error": result["error"], "data": result["data"]})
                messages.append({"role": "tool", "tool_call_id": call["id"], "content": _compact(result)})
        if answer is None:
            answer = "I stopped after %d steps without a final answer. See the tool results." % self.max_steps

        unverified = [{"tool": t["tool"], "status": t["status"], "reason": t["error"]}
                      for t in trace if t["status"] != "ok"]
        if unverified:
            answer += "\n\nCould not verify (tool problems, nothing was guessed):\n" + "\n".join(
                "- %s: %s (%s)" % (u["tool"], u["reason"], u["status"]) for u in unverified)
        if not trace:
            answer += "\n\nNote: no tool was called, so this answer is not backed by live evidence."

        duration = int((time.time() - start) * 1000)
        log.info("agent request finished", extra={"request_id": rid, "duration_ms": duration})
        return {"request_id": rid, "answer": answer, "llm": llm_name, "tool_calls": trace,
                "unverified": unverified, "duration_ms": duration}


def build_agent():
    """Groq when a key is set, otherwise the rule-based planner."""
    groq = GroqLLM()
    return Agent(llm=groq if groq.available() else FallbackLLM())
