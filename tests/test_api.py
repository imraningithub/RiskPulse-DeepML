from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert "RiskPulse DeepML" in response.json()["message"]

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data

def test_predict_endpoint():
    payload = {
        "person_age": 28,
        "person_income": 65000,
        "person_home_ownership": "RENT",
        "person_emp_length": 4.0,
        "loan_intent": "EDUCATION",
        "loan_grade": "B",
        "loan_amnt": 10000,
        "loan_int_rate": 11.14,
        "loan_percent_income": 0.15,
        "cb_person_default_on_file": "N",
        "cb_person_cred_hist_length": 4
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "default_probability" in data
    assert "is_default" in data
    assert "risk_category" in data
