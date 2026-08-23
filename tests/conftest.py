import pytest
from fastapi.testclient import TestClient

from app.main import app

SOLVED_STATE = "UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB"
SCRAMBLED_STATE = "BBURUDBFUFFFRRFUUFLULUFUDLRRDBBDBDBLUDDFLLRRBRLLLBRDDF"


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client
