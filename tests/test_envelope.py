def test_api_meta_structure(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    body = response.json()
    assert "success" in body
    assert body["success"] is True
    assert "data" in body
    assert "meta" in body
    meta = body["meta"]
    assert "timestamp" in meta
    assert "executionTimeMs" in meta
    assert "version" in meta
    assert isinstance(meta["executionTimeMs"], (int, float))


def test_404_error_envelope(client):
    response = client.get("/api/v1/non-existent-path")
    assert response.status_code == 404
    body = response.json()
    assert body["success"] is False
    assert "error" in body
    assert body["error"]["code"] == "HTTP_404"
    assert "meta" in body
