"""Structured (JSON) logging with request ids and secret redaction."""
import contextvars
import json
import logging
import os
import re
import sys
import uuid
from datetime import datetime, timezone

from agent.config import LOG_FILE, LOG_LEVEL

request_id_var = contextvars.ContextVar("request_id", default="-")

# key=value style secrets and bearer tokens
KEY_VALUE = re.compile(
    r"(?i)\b(password|passwd|token|secret|api[_-]?key|authorization)\b([\"']?\s*[:=]\s*[\"']?)([^\s\"',}]+)"
)
BEARER = re.compile(r"(?i)(bearer\s+)([A-Za-z0-9._\-]+)")


def redact(text):
    # bearer tokens first, otherwise "Authorization: Bearer xyz" would only hide the word Bearer
    text = BEARER.sub(lambda m: m.group(1) + "***", str(text))
    return KEY_VALUE.sub(lambda m: m.group(1) + m.group(2) + "***", text)


def new_request_id():
    rid = uuid.uuid4().hex[:12]
    request_id_var.set(rid)
    return rid


class JsonFormatter(logging.Formatter):
    EXTRA_KEYS = ("request_id", "tool", "status", "duration_ms", "arg_names")

    def format(self, record):
        entry = {
            "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": redact(record.getMessage()),
        }
        for key in self.EXTRA_KEYS:
            if hasattr(record, key):
                entry[key] = getattr(record, key)
        return json.dumps(entry)


def get_logger(name="agent"):
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(LOG_LEVEL)
    logger.propagate = False
    formatter = JsonFormatter()

    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    try:
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
        file_handler = logging.FileHandler(LOG_FILE)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except OSError:
        pass  # logging to the file is optional
    return logger
