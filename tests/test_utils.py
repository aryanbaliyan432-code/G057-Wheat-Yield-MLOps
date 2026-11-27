"""Unit tests for utility functions, configuration loading, and logging."""

import json
import logging
import os
import pytest

from src.wheat_yield.logger import JSONFormatter, get_logger
from src.wheat_yield.model_training import compute_data_sha256, load_config


def test_json_formatter_output():
    """Verify JSONFormatter generates parseable JSON string with expected keys."""
    formatter = JSONFormatter()
    record = logging.LogRecord(
        name="test_logger",
        level=logging.INFO,
        pathname="test.py",
        lineno=1,
        msg="Test message",
        args=(),
        exc_info=None,
    )
    record.event_type = "test_event"

    formatted = formatter.format(record)
    parsed = json.loads(formatted)

    assert parsed["level"] == "INFO"
    assert parsed["logger"] == "test_logger"
    assert parsed["message"] == "Test message"
    assert parsed["event_type"] == "test_event"
    assert "timestamp" in parsed


def test_get_logger_singleton_behavior():
    """Verify get_logger returns a properly configured logger instance."""
    logger = get_logger("unit_test_logger")
    assert logger.name == "unit_test_logger"
    assert len(logger.handlers) >= 1
    assert isinstance(logger.handlers[0].formatter, JSONFormatter)


def test_load_config_valid():
    """Verify load_config reads YAML configuration cleanly."""
    config = load_config("params.yaml")
    assert "data" in config
    assert "target" in config
    assert "model" in config
    assert config["target"] == "yield_qtl_per_acre"


def test_compute_data_sha256():
    """Verify SHA-256 calculation produces 64-character hexadecimal string."""
    sha = compute_data_sha256("params.yaml")
    assert len(sha) == 64
    assert all(c in "0123456789abcdef" for c in sha)
