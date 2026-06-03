import sys
import os
from fastapi.testclient import TestClient

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.api.main import app

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["Status"] == "ONLINE"

def test_predict_endpoint_valid():
    payload = {
        "Recency": 3.0,
        "Frequency": 10.0,
        "Total_Spending": 50000.0,
        "Average_Transaction_Value": 5000.0,
        "Max_Transaction_Value": 20000.0,
        "Transaction_Value_Std": 500.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "Credit_Risk_Assessment" in data
    assert data["Credit_Risk_Assessment"] in ["APPROVED", "DENIED"]