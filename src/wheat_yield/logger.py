"""Structured JSON logging utility for Wheat Yield Prediction Service.

Emits structured JSON log lines to standard output in compliance with
production MLOps and cloud-native logging guidelines.
"""

from datetime import datetime, timezone
import json
import logging
import sys
from typing import Any, Dict


class JSONFormatter(logging.Formatter):
    """Custom logging formatter that outputs log records as single-line JSON."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Include structured extra fields if present
        if hasattr(record, "event_type"):
            log_entry["event_type"] = record.event_type
        if hasattr(record, "details") and isinstance(record.details, dict):
            log_entry.update(record.details)
        if hasattr(record, "latency_ms"):
            log_entry["latency_ms"] = record.latency_ms
        if hasattr(record, "status_code"):
            log_entry["status_code"] = record.status_code

        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry)


def get_logger(name: str = "wheat_yield") -> logging.Logger:
    """Get or configure a structured JSON logger for the application."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers if logger already initialized
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        logger.propagate = False

    return logger
