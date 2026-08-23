from tests.conftest import SCRAMBLED_STATE, SOLVED_STATE


def test_validate_solved_state(client):
    response = client.post("/api/v1/validate", json={"state": SOLVED_STATE})
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["data"]["isValid"] is True
    assert data["data"]["hasUniqueCenters"] is True
    assert data["data"]["cornerParityValid"] is True
    assert data["data"]["edgeParityValid"] is True
    assert data["data"]["permutationParityValid"] is True
    assert len(data["data"]["issues"]) == 0
    assert all(count == 9 for count in data["data"]["colorCounts"].values())


def test_validate_scrambled_valid_state(client):
    response = client.post("/api/v1/validate", json={"state": SCRAMBLED_STATE})
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["data"]["isValid"] is True
    assert len(data["data"]["issues"]) == 0


def test_validate_invalid_length(client):
    response = client.post("/api/v1/validate", json={"state": "UUUU"})
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["data"]["isValid"] is False
    assert any(i["code"] == "INVALID_STATE_LENGTH" for i in data["data"]["issues"])


def test_validate_invalid_characters(client):
    invalid_state = "X" + SOLVED_STATE[1:]
    response = client.post("/api/v1/validate", json={"state": invalid_state})
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is False or data["data"]["isValid"] is False
    assert any(i["code"] == "INVALID_FACELET_CHARS" for i in data["data"]["issues"])


def test_validate_color_count_mismatch(client):
    # 10 U and 8 R
    bad_counts = list(SOLVED_STATE)
    bad_counts[9] = "U"
    bad_state = "".join(bad_counts)
    response = client.post("/api/v1/validate", json={"state": bad_state})
    assert response.status_code == 200
    data = response.json()
    assert data["data"]["isValid"] is False
    assert any(i["code"] == "COLOR_COUNT_MISMATCH" for i in data["data"]["issues"])


def test_validate_edge_flip_parity_error(client):
    # Flipped UF edge
    bad_state = list(SOLVED_STATE)
    bad_state[7] = "F"
    bad_state[19] = "U"
    response = client.post("/api/v1/validate", json={"state": "".join(bad_state)})
    assert response.status_code == 200
    data = response.json()
    assert data["data"]["isValid"] is False
    assert data["data"]["edgeParityValid"] is False
    assert any(i["code"] == "EDGE_FLIP_ERROR" for i in data["data"]["issues"])


def test_validate_corner_twist_parity_error(client):
    # Twisted URF corner
    bad_state = list(SOLVED_STATE)
    bad_state[8] = "R"
    bad_state[9] = "F"
    bad_state[20] = "U"
    response = client.post("/api/v1/validate", json={"state": "".join(bad_state)})
    assert response.status_code == 200
    data = response.json()
    assert data["data"]["isValid"] is False
    assert data["data"]["cornerParityValid"] is False
    assert any(i["code"] == "CORNER_TWIST_ERROR" for i in data["data"]["issues"])


def test_validate_permutation_parity_error(client):
    # Swapped UB and UL edges
    bad_state = list(SOLVED_STATE)
    bad_state[46] = "L"
    bad_state[37] = "B"
    response = client.post("/api/v1/validate", json={"state": "".join(bad_state)})
    assert response.status_code == 200
    data = response.json()
    assert data["data"]["isValid"] is False
    assert data["data"]["permutationParityValid"] is False
    assert any(i["code"] == "PERMUTATION_PARITY_ERROR" for i in data["data"]["issues"])
