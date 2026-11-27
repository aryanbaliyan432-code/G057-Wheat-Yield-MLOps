"""Unit tests for Population Stability Index (PSI) and data drift calculation."""

import numpy as np
import pytest

from src.wheat_yield.drift_detection import calculate_psi


def test_psi_identical_distributions():
    """Verify that identical distributions yield approximately 0.0 PSI."""
    np.random.seed(42)
    data = np.random.normal(50, 10, size=500)
    psi = calculate_psi(data, data)
    assert psi == pytest.approx(0.0, abs=1e-3)


def test_psi_shifted_distribution():
    """Verify that a significantly shifted distribution produces high PSI (> 0.2)."""
    np.random.seed(42)
    expected = np.random.normal(50, 10, size=500)
    actual = np.random.normal(80, 15, size=500)  # Substantial distribution shift
    psi = calculate_psi(expected, actual)
    assert psi > 0.2


def test_psi_handles_empty_or_nan_gracefully():
    """Verify calculate_psi handles edge cases like NaNs without raising exceptions."""
    expected = np.array([np.nan, 10.0, 20.0, 30.0, np.nan])
    actual = np.array([12.0, 22.0, 32.0, np.nan, 25.0])
    psi = calculate_psi(expected, actual)
    assert isinstance(psi, float)
    assert psi >= 0.0
