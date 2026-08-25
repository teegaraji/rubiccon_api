import json
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from app.schemas.vision import (
    VisionCalibrateColorsMessage,
    VisionErrorEvent,
    VisionInitSessionMessage,
    VisionLockFaceMessage,
    VisionProcessFrameMessage,
    VisionResetSessionMessage,
    VisionSetActiveFaceMessage,
)
from app.services.vision import VisionSession

logger = logging.getLogger(__name__)
router = APIRouter()


@router.websocket("/ws/vision")
async def vision_websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session = VisionSession()

    try:
        while True:
            raw_data = await websocket.receive_text()
            try:
                payload = json.loads(raw_data)
            except json.JSONDecodeError:
                err_event = VisionErrorEvent(
                    code="INVALID_JSON",
                    message="Received malformed JSON string.",
                )
                await websocket.send_text(err_event.model_dump_json())
                continue

            action = payload.get("action")

            if action == "init_session":
                try:
                    init_msg = VisionInitSessionMessage.model_validate(payload)
                    session = VisionSession(
                        session_id=init_msg.sessionId,
                        initial_face=init_msg.initialFace,
                        calibration_palette=init_msg.calibrationPalette,
                    )
                except ValidationError as e:
                    err_event = VisionErrorEvent(
                        code="INVALID_PAYLOAD",
                        message=f"Invalid init_session payload: {str(e)}",
                    )
                    await websocket.send_text(err_event.model_dump_json())

            elif action == "calibrate_colors":
                try:
                    calib_msg = VisionCalibrateColorsMessage.model_validate(payload)
                    calib_evt = session.calibrate_colors(calib_msg.palette)
                    await websocket.send_text(calib_evt.model_dump_json())
                except ValidationError as e:
                    err_event = VisionErrorEvent(
                        code="INVALID_PAYLOAD",
                        message=f"Invalid calibrate_colors payload: {str(e)}",
                    )
                    await websocket.send_text(err_event.model_dump_json())

            elif action == "process_frame":
                try:
                    frame_msg = VisionProcessFrameMessage.model_validate(payload)
                    detection_result = session.process_frame(
                        image_b64=frame_msg.image,
                        frame_id=frame_msg.frameId,
                        active_face=frame_msg.activeFace,
                        timestamp=frame_msg.timestamp,
                        roi_guide=frame_msg.roiGuide,
                    )
                    await websocket.send_text(detection_result.model_dump_json())
                except ValidationError as e:
                    err_event = VisionErrorEvent(
                        code="INVALID_PAYLOAD",
                        message=f"Invalid process_frame payload: {str(e)}",
                    )
                    await websocket.send_text(err_event.model_dump_json())
                except Exception as e:
                    logger.exception("Error processing frame")
                    err_event = VisionErrorEvent(
                        code="FRAME_PROCESSING_ERROR",
                        message=f"Error processing frame: {str(e)}",
                    )
                    await websocket.send_text(err_event.model_dump_json())

            elif action == "set_active_face":
                try:
                    set_face_msg = VisionSetActiveFaceMessage.model_validate(payload)
                    session.set_active_face(set_face_msg.face)
                except ValidationError as e:
                    err_event = VisionErrorEvent(
                        code="INVALID_PAYLOAD",
                        message=f"Invalid set_active_face payload: {str(e)}",
                    )
                    await websocket.send_text(err_event.model_dump_json())

            elif action == "lock_face":
                try:
                    lock_msg = VisionLockFaceMessage.model_validate(payload)
                    locked_evt, completed_evt = session.lock_face(
                        face=lock_msg.face,
                        override_tiles=lock_msg.overrideTiles,
                    )
                    await websocket.send_text(locked_evt.model_dump_json())
                    if completed_evt:
                        await websocket.send_text(completed_evt.model_dump_json())
                except ValidationError as e:
                    err_event = VisionErrorEvent(
                        code="INVALID_PAYLOAD",
                        message=f"Invalid lock_face payload: {str(e)}",
                    )
                    await websocket.send_text(err_event.model_dump_json())

            elif action == "reset_session":
                try:
                    _ = VisionResetSessionMessage.model_validate(payload)
                    session.reset_session()
                except ValidationError as e:
                    err_event = VisionErrorEvent(
                        code="INVALID_PAYLOAD",
                        message=f"Invalid reset_session payload: {str(e)}",
                    )
                    await websocket.send_text(err_event.model_dump_json())

            else:
                err_event = VisionErrorEvent(
                    code="UNKNOWN_ACTION",
                    message=f"Unsupported WebSocket action '{action}'.",
                )
                await websocket.send_text(err_event.model_dump_json())

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected.")
    except Exception:
        logger.exception("Unexpected error in vision WebSocket loop")
