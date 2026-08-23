from tests.conftest import SCRAMBLED_STATE, SOLVED_STATE


def test_solve_already_solved_cube(client):
    response = client.post("/api/v1/solve", json={"state": SOLVED_STATE})
    assert response.status_code == 200
    res = response.json()
    assert res["success"] is True
    assert res["data"]["isSolved"] is True
    assert res["data"]["totalMoves"] == 0
    assert res["data"]["solutionString"] == ""
    assert len(res["data"]["moves"]) == 0
    assert "meta" in res
    assert "executionTimeMs" in res["meta"]


def test_solve_scrambled_cube(client):
    response = client.post("/api/v1/solve", json={"state": SCRAMBLED_STATE})
    assert response.status_code == 200
    res = response.json()
    assert res["success"] is True
    assert res["data"]["isSolved"] is False
    assert res["data"]["totalMoves"] > 0
    assert len(res["data"]["moves"]) == res["data"]["totalMoves"]

    first_move = res["data"]["moves"][0]
    assert "index" in first_move
    assert "notation" in first_move
    assert "face" in first_move
    assert "direction" in first_move
    assert "angle" in first_move
    assert "instruction" in first_move
    assert "humanGuidance" in first_move
    assert first_move["face"] in ["U", "R", "F", "D", "L", "B"]


def test_solve_invalid_cube_returns_422(client):
    # Flipped edge invalid cube
    bad_state = list(SOLVED_STATE)
    bad_state[7] = "F"
    bad_state[19] = "U"
    response = client.post("/api/v1/solve", json={"state": "".join(bad_state)})
    assert response.status_code == 422
    res = response.json()
    assert res["success"] is False
    assert res["error"]["code"] == "CUBE_STATE_INVALID"
    assert "details" in res["error"]
    assert len(res["error"]["details"]) > 0


def test_solve_malformed_payload(client):
    response = client.post("/api/v1/solve", json={})
    assert response.status_code == 422
    res = response.json()
    assert res["success"] is False
    assert res["error"]["code"] == "INVALID_PAYLOAD_FORMAT"
