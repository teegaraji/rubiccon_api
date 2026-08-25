def test_health_check(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["success"] is True
    assert json_data["data"]["status"] == "healthy"
    assert json_data["data"]["engine"] == "Kociemba Two-Phase Algorithm"
    assert "uptimeSeconds" in json_data["data"]
    assert "meta" in json_data
    assert "executionTimeMs" in json_data["meta"]
    assert "timestamp" in json_data["meta"]


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "service" in data
    assert "version" in data


