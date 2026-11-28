"""FastAPI integration and endpoint validation tests using TestClient."""

from fastapi.testclient import TestClient
import pytest

from app.main import app

client = TestClient(app)


def test_get_root():
    """Verify GET / returns 200 and service discovery info."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "service" in data
    assert "version" in data
    assert data["status"] == "running"

def test_get_health():
    """Verify GET /health returns complete service health information."""
    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "ok"
    assert isinstance(data["model_loaded"], bool)
    assert data["service"] == "wheat-yield-api"

def test_post_predict_valid_payload():
    """Verify POST /predict returns 200 and valid predicted yield for normal input."""
    payload = {
        "rainfall_mm": 650.0,
        "fertilizer_kg": 110.0,
        "sowing_delay_days": 10.0,
        "irrigation_count": 4.0,
        "soil_ph": 6.8,
        "farm_acres": 5.5,
        "seed_variety": "HD-2967",
        "tehsil": "Tehsil South",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_yield_qtl_per_acre" in data
    assert data["status"] == "success"
    assert data["unit"] == "quintals_per_acre"
    assert data["predicted_yield_qtl_per_acre"] > 0


def test_post_predict_client_rule_1_fertilizer_above_300():
    """Client Rule 1: fertilizer_kg > 300 must be rejected with HTTP 422."""
    bad_payload = {
        "rainfall_mm": 650.0,
        "fertilizer_kg": 350.0,  # Invalid: > 300
        "sowing_delay_days": 10.0,
        "irrigation_count": 4.0,
        "soil_ph": 6.8,
        "farm_acres": 5.5,
        "seed_variety": "HD-2967",
        "tehsil": "Tehsil South",
    }
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422


def test_post_predict_client_rule_2_farm_acres_negative():
    """Client Rule 2: farm_acres < 0 must be rejected with HTTP 422."""
    bad_payload = {
        "rainfall_mm": 650.0,
        "fertilizer_kg": 100.0,
        "sowing_delay_days": 10.0,
        "irrigation_count": 4.0,
        "soil_ph": 6.8,
        "farm_acres": -2.5,  # Invalid: < 0
        "seed_variety": "HD-2967",
        "tehsil": "Tehsil South",
    }
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422


def test_post_predict_missing_required_fields():
    """Verify that omitting a required field fails with HTTP 422."""
    incomplete_payload = {
        "rainfall_mm": 650.0,
        "fertilizer_kg": 100.0,
        # missing sowing_delay_days, irrigation_count, etc.
    }
    response = client.post("/predict", json=incomplete_payload)
    assert response.status_code == 422


def test_post_predict_extra_field_rejected():
    """Verify that unexpected extra fields are rejected under strict schema configuration."""
    extra_payload = {
        "rainfall_mm": 650.0,
        "fertilizer_kg": 100.0,
        "sowing_delay_days": 10.0,
        "irrigation_count": 4.0,
        "soil_ph": 6.8,
        "farm_acres": 5.0,
        "seed_variety": "HD-2967",
        "tehsil": "Tehsil South",
        "unexpected_extra_field": 12345,
    }
    response = client.post("/predict", json=extra_payload)
    assert response.status_code == 422
