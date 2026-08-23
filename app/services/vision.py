import base64
import colorsys
import io
import time
from collections import Counter, deque

import numpy as np
from PIL import Image

from app.schemas.vision import (
    FACE_NAME_TO_CODE,
    STANDARD_FACE_ORDER,
    CubeFace,
    CubeFaceName,
    FaceletColor,
    VisionDetectionMeta,
    VisionDetectionResultEvent,
    VisionFaceLockedEvent,
    VisionRoiGuide,
    VisionSessionCompletedEvent,
    VisionTileDetection,
)
from app.services.validator import validate_cube_state


def classify_tile_color(
    r: float, g: float, b: float
) -> tuple[FaceletColor, CubeFace, str, float, list[int], list[int]]:
    rf, gf, bf = r / 255.0, g / 255.0, b / 255.0
    h, s, v = colorsys.rgb_to_hsv(rf, gf, bf)
    hue_deg = round(h * 360.0)
    sat_pct = round(s * 100.0)
    val_pct = round(v * 100.0)
    rgb_list = [int(r), int(g), int(b)]
    hsv_list = [hue_deg, sat_pct, val_pct]

    # Neutral / White detection (low saturation, moderate-high value)
    if sat_pct < 22 and val_pct > 35:
        return "white", "U", "#f8fafc", 0.95, hsv_list, rgb_list

    # Chromatic classification
    if hue_deg >= 345 or hue_deg < 15:
        return "red", "F", "#ef4444", 0.96, hsv_list, rgb_list
    elif 15 <= hue_deg < 42:
        return "orange", "B", "#f97316", 0.94, hsv_list, rgb_list
    elif 42 <= hue_deg < 75:
        return "yellow", "D", "#fde047", 0.95, hsv_list, rgb_list
    elif 75 <= hue_deg < 165:
        return "green", "L", "#22c55e", 0.96, hsv_list, rgb_list
    elif 165 <= hue_deg < 265:
        return "blue", "R", "#3b82f6", 0.95, hsv_list, rgb_list
    else:
        return "red", "F", "#ef4444", 0.90, hsv_list, rgb_list


class VisionSession:
    def __init__(self, session_id: str | None = None, initial_face: CubeFaceName = "Front"):
        self.session_id = session_id or f"sess_{int(time.time()*1000)}"
        self.active_face: CubeFaceName = initial_face
        self.locked_faces: dict[CubeFaceName, list[str]] = {}
        self.recent_face_detections: dict[CubeFaceName, deque[list[str]]] = {
            face: deque(maxlen=10) for face in STANDARD_FACE_ORDER
        }
        self.last_detection_tiles: list[VisionTileDetection] = []

    def set_active_face(self, face: CubeFaceName) -> None:
        self.active_face = face

    def reset_session(self) -> None:
        self.locked_faces.clear()
        for d in self.recent_face_detections.values():
            d.clear()
        self.last_detection_tiles.clear()
        self.active_face = "Front"

    def process_frame(
        self,
        image_b64: str,
        frame_id: int,
        active_face: CubeFaceName,
        timestamp: float | int,
        roi_guide: VisionRoiGuide | None = None,
    ) -> VisionDetectionResultEvent:
        start_time = time.perf_counter()
        self.active_face = active_face

        # 1. Clean and decode Base64 Image
        clean_b64 = image_b64.split(",", 1)[1] if "," in image_b64 else image_b64
        image_bytes = base64.b64decode(clean_b64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_w, img_h = image.size

        # 2. Compute ROI Bounding Box
        roi = roi_guide or VisionRoiGuide()
        x1 = max(0, min(img_w - 1, int(roi.normalizedX * img_w)))
        y1 = max(0, min(img_h - 1, int(roi.normalizedY * img_h)))
        x2 = max(x1 + 10, min(img_w, int((roi.normalizedX + roi.normalizedWidth) * img_w)))
        y2 = max(y1 + 10, min(img_h, int((roi.normalizedY + roi.normalizedHeight) * img_h)))

        np_img = np.array(image)
        roi_img = np_img[y1:y2, x1:x2]
        roi_h, roi_w, _ = roi_img.shape

        cell_w = roi_w / 3.0
        cell_h = roi_h / 3.0

        tiles: list[VisionTileDetection] = []
        current_facelet_codes: list[str] = []

        # 3. Sample 3x3 grid cells
        tile_idx = 0
        for r in range(3):
            for c in range(3):
                # Sample inner 50% of the cell to avoid borders/stickers gap
                cx1 = int((c + 0.25) * cell_w)
                cx2 = max(cx1 + 1, int((c + 0.75) * cell_w))
                cy1 = int((r + 0.25) * cell_h)
                cy2 = max(cy1 + 1, int((r + 0.75) * cell_h))

                cell_sample = roi_img[cy1:cy2, cx1:cx2]
                mean_r = float(np.mean(cell_sample[:, :, 0]))
                mean_g = float(np.mean(cell_sample[:, :, 1]))
                mean_b = float(np.mean(cell_sample[:, :, 2]))

                color_name, facelet_code, hex_color, conf, hsv, rgb = classify_tile_color(
                    mean_r, mean_g, mean_b
                )

                tiles.append(
                    VisionTileDetection(
                        index=tile_idx,
                        color=color_name,
                        faceletCode=facelet_code,
                        hex=hex_color,
                        confidence=conf,
                        hsv=hsv,
                        rgb=rgb,
                    )
                )
                current_facelet_codes.append(facelet_code)
                tile_idx += 1

        self.last_detection_tiles = tiles

        # 4. Stability Score Calculation
        history = self.recent_face_detections[active_face]
        history.append(current_facelet_codes)

        agreement_sum = 0.0
        for i in range(9):
            col_history = [frame[i] for frame in history]
            most_common_count = Counter(col_history).most_common(1)[0][1]
            agreement_sum += most_common_count / len(col_history)

        stability_score = round(agreement_sum / 9.0, 2)
        is_ready_to_lock = bool(stability_score >= 0.85 and len(history) >= 3)

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return VisionDetectionResultEvent(
            frameId=frame_id,
            activeFace=active_face,
            isGridAligned=True,
            stabilityScore=stability_score,
            isReadyToLock=is_ready_to_lock,
            tiles=tiles,
            meta=VisionDetectionMeta(
                serverProcessingTimeMs=elapsed_ms,
                timestamp=timestamp,
            ),
        )

    def lock_face(
        self,
        face: CubeFaceName,
        override_tiles: list[str] | None = None,
    ) -> tuple[VisionFaceLockedEvent, VisionSessionCompletedEvent | None]:
        if override_tiles and len(override_tiles) == 9:
            locked_tiles = override_tiles
        elif self.last_detection_tiles and len(self.last_detection_tiles) == 9:
            locked_tiles = [t.faceletCode for t in self.last_detection_tiles]
        else:
            default_code = FACE_NAME_TO_CODE.get(face, "U")
            locked_tiles = [default_code] * 9

        self.locked_faces[face] = locked_tiles
        completed_faces = [f for f in STANDARD_FACE_ORDER if f in self.locked_faces]
        completed_count = len(completed_faces)
        is_fully_mapped = completed_count == 6

        full_state_string = None
        session_completed_event = None

        if is_fully_mapped:
            # Build standard 54-char string:
            # U(0..8) + R(0..8) + F(0..8) + D(0..8) + L(0..8) + B(0..8)
            u_tiles = "".join(self.locked_faces["Up"])
            r_tiles = "".join(self.locked_faces["Right"])
            f_tiles = "".join(self.locked_faces["Front"])
            d_tiles = "".join(self.locked_faces["Down"])
            l_tiles = "".join(self.locked_faces["Left"])
            b_tiles = "".join(self.locked_faces["Back"])
            full_state_string = u_tiles + r_tiles + f_tiles + d_tiles + l_tiles + b_tiles

            val_res = validate_cube_state(full_state_string)
            session_completed_event = VisionSessionCompletedEvent(
                isFullyMapped=True,
                fullStateString=full_state_string,
                isValid=val_res.isValid,
            )
            next_suggested = None
        else:
            next_suggested = next(
                (f for f in STANDARD_FACE_ORDER if f not in self.locked_faces),
                None,
            )
            if next_suggested:
                self.active_face = next_suggested

        locked_event = VisionFaceLockedEvent(
            face=face,
            tiles=locked_tiles,
            completedFaces=completed_faces,
            completedCount=completed_count,
            isFullyMapped=is_fully_mapped,
            nextSuggestedFace=next_suggested,
            fullStateString=full_state_string,
        )

        return locked_event, session_completed_event
