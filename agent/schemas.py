"""Shared shapes for tool results and the errors a tool can raise."""
from typing import Any, Optional

from pydantic import BaseModel


class ToolError(Exception):
    """The request was invalid, for example an unknown item or a bad argument."""


class ToolUnavailable(Exception):
    """The external service could not be reached or the call timed out."""


class ToolRejected(Exception):
    """The operation is blocked by the safety rules."""


class ToolResult(BaseModel):
    tool: str
    status: str  # ok, error, unavailable or rejected
    request_id: str
    duration_ms: int
    data: Optional[Any] = None
    error: Optional[str] = None
