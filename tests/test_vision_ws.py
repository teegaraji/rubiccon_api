import base64
import io
import json

from fastapi.testclient import TestClient
from PIL import Image


def create_sample_b64_image(color=(255, 255, 255), width=90, height=90) -> str:
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


def test_vision_ws_lifecycle(client: TestClient):
    with client.websocket_connect("/api/v1/ws/vision") as ws:
        # 1. Initialize session
        init_payload = {
            "action": "init_session",
            "sessionId": "test_sess_01",
            "cubeType": "3x3",
            "initialFace": "Front",
            "colorScheme": "western",
        }
        ws.send_text(json.dumps(init_payload))

        # 2. Process frame
        img_b64 = create_sample_b64_image(color=(239, 68, 68))  # Red
        frame_payload = {
            "action": "process_frame",
            "frameId": 1,
            "activeFace": "Front",
            "timestamp": 1724426400000,
            "image": img_b64,
            "roiGuide": {
                "normalizedX": 0.1,
                "normalizedY": 0.1,
                "normalizedWidth": 0.8,
                "normalizedHeight": 0.8,
            },
        }
        ws.send_text(json.dumps(frame_payload))

        res_text = ws.receive_text()
        res = json.loads(res_text)
        assert res["event"] == "detection_result"
        assert res["frameId"] == 1
        assert res["activeFace"] == "Front"
        assert len(res["tiles"]) == 9
        assert res["tiles"][0]["faceletCode"] == "F"
        assert "confidence" in res["tiles"][0]
        assert "hsv" in res["tiles"][0]
        assert "rgb" in res["tiles"][0]

        # 3. Change active face
        set_face_payload = {
            "action": "set_active_face",
            "face": "Right",
        }
        ws.send_text(json.dumps(set_face_payload))

        # 4. Lock face with override tiles
        lock_payload = {
            "action": "lock_face",
            "face": "Front",
            "overrideTiles": ["F"] * 9,
        }
        ws.send_text(json.dumps(lock_payload))

        lock_res = json.loads(ws.receive_text())
        assert lock_res["event"] == "face_locked"
        assert lock_res["face"] == "Front"
        assert lock_res["completedCount"] == 1
        assert lock_res["isFullyMapped"] is False
        assert "Front" in lock_res["completedFaces"]
        assert lock_res["nextSuggestedFace"] is not None

        # 5. Lock remaining 5 faces to trigger session_completed
        faces_to_lock = [
            ("Up", "U"),
            ("Right", "R"),
            ("Down", "D"),
            ("Left", "L"),
            ("Back", "B"),
        ]
        for face_name, code in faces_to_lock:
            ws.send_text(
                json.dumps(
                    {
                        "action": "lock_face",
                        "face": face_name,
                        "overrideTiles": [code] * 9,
                    }
                )
            )
            resp1 = json.loads(ws.receive_text())
            assert resp1["event"] == "face_locked"

        # The last face lock should also send session_completed
        resp_completed = json.loads(ws.receive_text())
        assert resp_completed["event"] == "session_completed"
        assert resp_completed["isFullyMapped"] is True
        assert (
            resp_completed["fullStateString"]
            == "UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB"
        )
        assert resp_completed["isValid"] is True

        # 6. Reset session
        ws.send_text(json.dumps({"action": "reset_session"}))


def test_vision_ws_invalid_messages(client: TestClient):
    with client.websocket_connect("/api/v1/ws/vision") as ws:
        # Send malformed JSON
        ws.send_text("not-a-json")
        res = json.loads(ws.receive_text())
        assert res["event"] == "vision_error"
        assert res["code"] == "INVALID_JSON"

        # Send unknown action
        ws.send_text(json.dumps({"action": "unknown_action_foo"}))
        res = json.loads(ws.receive_text())
        assert res["event"] == "vision_error"
        assert res["code"] == "UNKNOWN_ACTION"

        # Send invalid payload for process_frame
        ws.send_text(json.dumps({"action": "process_frame"}))
        res = json.loads(ws.receive_text())
        assert res["event"] == "vision_error"
        assert res["code"] == "INVALID_PAYLOAD"
