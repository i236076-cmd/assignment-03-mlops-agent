"""Tool registry. Every tool is registered with a name, a description and a
JSON schema for its arguments, so the LLM can pick tools and the rest of the
code can call them in one safe way (timeouts, error mapping, logging)."""
import contextvars
import functools
import inspect
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout

from agent.config import TOOL_TIMEOUT
from agent.logging_utils import get_logger, request_id_var
from agent.schemas import ToolError, ToolRejected, ToolResult, ToolUnavailable

log = get_logger("agent.tools")

REGISTRY = {}
_executor = ThreadPoolExecutor(max_workers=8)


def register(name, description, parameters=None, permission="read", timeout=None):
    schema = parameters or {"type": "object", "properties": {}}
    limit = timeout or TOOL_TIMEOUT

    def decorator(func):
        @functools.wraps(func)
        def wrapper(**kwargs):
            rid = request_id_var.get()
            start = time.time()
            data, error, status = None, None, "ok"

            try:
                inspect.signature(func).bind(**kwargs)
            except TypeError as exc:
                status, error = "error", "invalid arguments: %s" % exc
            else:
                try:
                    ctx = contextvars.copy_context()
                    future = _executor.submit(ctx.run, func, **kwargs)
                    data = future.result(timeout=limit)
                except FutureTimeout:
                    status, error = "unavailable", "tool timed out after %s seconds" % limit
                except ToolRejected as exc:
                    status, error = "rejected", str(exc)
                except ToolUnavailable as exc:
                    status, error = "unavailable", str(exc)
                except ToolError as exc:
                    status, error = "error", str(exc)
                except Exception as exc:  # anything unexpected
                    status, error = "error", "%s: %s" % (type(exc).__name__, exc)

            duration = int((time.time() - start) * 1000)
            log.info(
                "tool call finished",
                extra={
                    "request_id": rid,
                    "tool": name,
                    "status": status,
                    "duration_ms": duration,
                    "arg_names": sorted(kwargs.keys()),
                },
            )
            return ToolResult(
                tool=name, status=status, request_id=rid,
                duration_ms=duration, data=data, error=error,
            ).model_dump()

        REGISTRY[name] = {
            "name": name,
            "description": description,
            "parameters": schema,
            "permission": permission,
            "function": wrapper,
        }
        return wrapper

    return decorator


def call_tool(name, **kwargs):
    """Call a registered tool by name. Unknown names return an error result."""
    entry = REGISTRY.get(name)
    if entry is None:
        return ToolResult(
            tool=name, status="error", request_id=request_id_var.get(),
            duration_ms=0, error="unknown tool: %s" % name,
        ).model_dump()
    return entry["function"](**kwargs)


def tool_specs():
    """Tool descriptions in the shape LLM APIs expect."""
    return [
        {"name": t["name"], "description": t["description"], "input_schema": t["parameters"]}
        for t in REGISTRY.values()
    ]
