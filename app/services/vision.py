import base64
import io
import time
from collections import Counter, deque

import cv2
import numpy as np
from PIL import Image

from app.schemas.vision import (
    FACE_NAME_TO_CODE,
    STANDARD_FACE_ORDER,
    CubeFace,
    CubeFaceName,
    FaceletColor,
    VisionColorCalibratedEvent,
    VisionDetectionMeta,
    VisionDetectionResultEvent,
    VisionFaceLockedEvent,
    VisionRoiGuide,
    VisionSessionCompletedEvent,
    VisionTileDetection,
)
from app.services.validator import validate_cube_state

FACE_CODE_TO_COLOR: dict[CubeFace, FaceletColor] = {
    "U": "white",
    "R": "blue",
    "F": "red",
    "D": "yellow",
    "L": "green",
    "B": "orange",
}

FACE_CODE_TO_HEX: dict[CubeFace, str] = {
    "U": "#f8fafc",
    "R": "#2563eb",
    "F": "#dc2626",
    "D": "#eab308",
    "L": "#16a34a",
    "B": "#ea580c",
}

DEFAULT_PALETTE_RGB: dict[CubeFace, tuple[int, int, int]] = {
    "U": (245, 245, 245),  # White
    "R": (37, 99, 235),    # Blue
    "F": (220, 38, 38),    # Red
    "D": (234, 179, 8),    # Yellow
    "L": (22, 163, 74),    # Green
    "B": (234, 88, 12),    # Orange
}


def rgb_to_cielab(r: float, g: float, b: float) -> tuple[float, float, float]:
    """Convert sRGB (0..255) to CIELAB (L*, a*, b*) under D65 standard illuminant."""
    def pivot_rgb(c: float) -> float:
        c_norm = max(0.0, min(255.0, c)) / 255.0
        return c_norm / 12.92 if c_norm <= 0.04045 else ((c_norm + 0.055) / 1.055) ** 2.4

    r_lin = pivot_rgb(r)
    g_lin = pivot_rgb(g)
    b_lin = pivot_rgb(b)

    x = r_lin * 0.4124564 + g_lin * 0.3575761 + b_lin * 0.1804375
    y = r_lin * 0.2126729 + g_lin * 0.7151522 + b_lin * 0.0721750
    z = r_lin * 0.0193339 + g_lin * 0.1191920 + b_lin * 0.9503041

    xr = x / 0.95047
    yr = y / 1.00000
    zr = z / 1.08883

    def pivot_xyz(t: float) -> float:
        return t ** (1.0 / 3.0) if t > 0.008856 else (7.787 * t) + (16.0 / 116.0)

    fx = pivot_xyz(xr)
    fy = pivot_xyz(yr)
    fz = pivot_xyz(zr)

    l_val = max(0.0, min(100.0, (116.0 * fy) - 16.0))
    a_val = 500.0 * (fx - fy)
    b_val = 200.0 * (fy - fz)

    return (l_val, a_val, b_val)


def classify_hsv_tile(
    h_ocv: float,
    s_ocv: float,
    v_ocv: float,
    r: float,
    g: float,
    b: float,
    palette_rgb: dict[CubeFace, tuple[float, float, float]] | None = None,
    palette_hex: dict[CubeFace, str] | None = None,
) -> tuple[FaceletColor, CubeFace, str, float, list[int], list[int]]:
    """Classify a single tile sample using OpenCV HSV Color Recognition:

    OpenCV Ranges:
      Hue (H):        0 .. 179  (integer-based circular color angle)
      Saturation (S): 0 .. 255  (color purity / intensity)
      Value (V):      0 .. 255  (brightness / lightness)

    Stages:
      Stage 1: White / Neutral Check (Low Saturation gate, invariant to warm lighting).
      Stage 2: Explicit Custom Prototype Matching (if calibrated).
      Stage 3: Angular Hue Sector Mapping in OpenCV scale + Red/Orange G/R discriminator.
    """
    hue_deg = int(round(h_ocv * 2.0))
    sat_pct = int(round((s_ocv / 255.0) * 100.0))
    val_pct = int(round((v_ocv / 255.0) * 100.0))
    hsv_list = [hue_deg, sat_pct, val_pct]
    rgb_list = [int(round(r)), int(round(g)), int(round(b))]

    hex_map = palette_hex or FACE_CODE_TO_HEX
    gr_ratio = (g + 1e-5) / (r + 1e-5)

    # ----------------------------------------------------
    # STAGE 1: White / Neutral Check
    # ----------------------------------------------------
    # White has low saturation (S < 60 / 255) and adequate brightness (V >= 65 / 255).
    # Yellow always has S >= 140 under similar conditions.
    is_neutral = (
        (s_ocv < 60 and v_ocv >= 65)
        or (s_ocv < 72 and v_ocv >= 120 and gr_ratio > 0.85)
    )

    if is_neutral:
        best_face: CubeFace = "U"
        confidence = round(max(0.85, min(0.99, 1.0 - (s_ocv / 120.0))), 2)
        return "white", best_face, hex_map.get(best_face, "#f8fafc"), confidence, hsv_list, rgb_list

    # ----------------------------------------------------
    # STAGE 2: Explicit Custom Prototype Matching (if calibrated)
    # ----------------------------------------------------
    if palette_rgb:
        scores: list[tuple[CubeFace, float]] = []
        for f_code, p_rgb in palette_rgb.items():
            pr, pg, pb = p_rgb
            p_arr = np.array([[[int(round(pr)), int(round(pg)), int(round(pb))]]], dtype=np.uint8)
            p_hsv = cv2.cvtColor(p_arr, cv2.COLOR_RGB2HSV)
            ph = float(p_hsv[0, 0, 0])
            ps = float(p_hsv[0, 0, 1])
            pv = float(p_hsv[0, 0, 2])

            if f_code == "U":
                dist = s_ocv * 1.5 + (255.0 - v_ocv) * 0.2
            else:
                h_diff = abs(h_ocv - ph)
                if h_diff > 90.0:
                    h_diff = 180.0 - h_diff

                dist = h_diff * 2.5 + abs(s_ocv - ps) * 0.25 + abs(v_ocv - pv) * 0.05

                # Physical penalty for Red vs Orange cross-assignment
                if f_code == "F" and gr_ratio > 0.40:
                    dist += 50.0
                elif f_code == "B" and gr_ratio < 0.28:
                    dist += 50.0

            scores.append((f_code, dist))

        scores.sort(key=lambda item: item[1])
        best_face = scores[0][0]
        second_dist = scores[1][1] if len(scores) > 1 else scores[0][1] + 25.0
        diff = second_dist - scores[0][1]
        confidence = round(max(0.75, min(0.99, 0.75 + (diff / 60.0) * 0.24)), 2)

        color_name = FACE_CODE_TO_COLOR.get(best_face, "white")
        hex_val = hex_map.get(best_face, "#ffffff")
        return color_name, best_face, hex_val, confidence, hsv_list, rgb_list

    # ----------------------------------------------------
    # STAGE 3: Baseline Angular Hue Sector Tree (OpenCV scale 0..179)
    # ----------------------------------------------------
    if 38 <= h_ocv < 82:
        best_face = "L"  # Green
        confidence = 0.95 if 45 <= h_ocv <= 75 else 0.84
    elif 82 <= h_ocv < 132:
        best_face = "R"  # Blue
        confidence = 0.96 if 92 <= h_ocv <= 122 else 0.85
    elif 18 <= h_ocv < 38:
        best_face = "D"  # Yellow
        confidence = 0.95 if 21 <= h_ocv <= 32 else 0.83
    else:
        # Red / Orange Zone (H < 18 or H >= 168)
        if h_ocv < 7 or h_ocv >= 168 or gr_ratio < 0.30:
            best_face = "F"  # Red
            confidence = 0.94 if (h_ocv < 5 or h_ocv >= 172 or gr_ratio < 0.25) else 0.82
        else:
            best_face = "B"  # Orange
            confidence = 0.93 if (8 <= h_ocv <= 16 and gr_ratio >= 0.35) else 0.80

    color_name = FACE_CODE_TO_COLOR.get(best_face, "white")
    hex_val = hex_map.get(best_face, "#ffffff")
    return color_name, best_face, hex_val, confidence, hsv_list, rgb_list


def classify_tile_color_calibrated(
    r: float,
    g: float,
    b: float,
    palette_rgb: dict[CubeFace, tuple[float, float, float]] | None = None,
    palette_hex: dict[CubeFace, str] | None = None,
) -> tuple[FaceletColor, CubeFace, str, float, list[int], list[int]]:
    """Convert scalar RGB to OpenCV HSV and classify."""
    rgb_arr = np.array([[[int(round(r)), int(round(g)), int(round(b))]]], dtype=np.uint8)
    hsv_arr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2HSV)
    h_ocv = float(hsv_arr[0, 0, 0])
    s_ocv = float(hsv_arr[0, 0, 1])
    v_ocv = float(hsv_arr[0, 0, 2])
    return classify_hsv_tile(
        h_ocv, s_ocv, v_ocv, r, g, b, palette_rgb, palette_hex
    )


def classify_tile_color(
    r: float, g: float, b: float
) -> tuple[FaceletColor, CubeFace, str, float, list[int], list[int]]:
    """Legacy helper: classifies tile color using the baseline OpenCV HSV pipeline."""
    return classify_tile_color_calibrated(r, g, b, None)


class VisionSession:
    def __init__(
        self,
        session_id: str | None = None,
        initial_face: CubeFaceName = "Front",
        calibration_palette: dict[CubeFace, list[int]] | None = None,
    ):
        self.session_id = session_id or f"sess_{int(time.time()*1000)}"
        self.active_face: CubeFaceName = initial_face
        self.locked_faces: dict[CubeFaceName, list[str]] = {}
        self.recent_face_detections: dict[CubeFaceName, deque[list[str]]] = {
            face: deque(maxlen=10) for face in STANDARD_FACE_ORDER
        }
        self.last_detection_tiles: list[VisionTileDetection] = []

        # Layer 1 & Layer 2 Calibrated Palette storage
        self.calibrated_rgb: dict[CubeFace, tuple[float, float, float]] = {
            face: (float(rgb[0]), float(rgb[1]), float(rgb[2]))
            for face, rgb in DEFAULT_PALETTE_RGB.items()
        }
        self.calibrated_lab: dict[CubeFace, tuple[float, float, float]] = {
            face: rgb_to_cielab(*rgb) for face, rgb in self.calibrated_rgb.items()
        }
        self.calibrated_hex: dict[CubeFace, str] = dict(FACE_CODE_TO_HEX)

        if calibration_palette:
            self.apply_calibration(calibration_palette)

    def apply_calibration(self, palette: dict[CubeFace, list[int]]) -> None:
        """Update reference color profile from explicit calibration (Layer 1)."""
        for face_code, rgb_vals in palette.items():
            if len(rgb_vals) == 3:
                r, g, b = float(rgb_vals[0]), float(rgb_vals[1]), float(rgb_vals[2])
                self.calibrated_rgb[face_code] = (r, g, b)
                self.calibrated_lab[face_code] = rgb_to_cielab(r, g, b)
                self.calibrated_hex[face_code] = f"#{int(r):02x}{int(g):02x}{int(b):02x}"

    def calibrate_colors(self, palette: dict[CubeFace, list[int]]) -> VisionColorCalibratedEvent:
        """Handle explicit calibration message from client."""
        self.apply_calibration(palette)
        return VisionColorCalibratedEvent(
            calibratedPalette={
                face: [int(round(c)) for c in rgb]
                for face, rgb in self.calibrated_rgb.items()
            },
            source="manual",
        )

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

        np_img = np.array(image, dtype=np.uint8)
        roi_rgb = np_img[y1:y2, x1:x2]
        roi_hsv = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2HSV)
        roi_h, roi_w, _ = roi_rgb.shape

        cell_w = roi_w / 3.0
        cell_h = roi_h / 3.0

        tiles: list[VisionTileDetection] = []
        current_facelet_codes: list[str] = []

        # 3. Sample 3x3 grid cells using tight circular center spot (matching visual dot)
        tile_idx = 0

        for r in range(3):
            for c in range(3):
                # Calculate cell center coordinates
                center_x = (c + 0.5) * cell_w
                center_y = (r + 0.5) * cell_h

                # Sample tight circular radius around center (16% of cell dimension)
                # to strictly avoid white speedcube borders, bevels, and inter-cubie gaps
                rad = max(2, int(min(cell_w, cell_h) * 0.16))

                cy1 = max(0, int(center_y - rad))
                cy2 = min(roi_h, int(center_y + rad + 1))
                cx1 = max(0, int(center_x - rad))
                cx2 = min(roi_w, int(center_x + rad + 1))

                cell_rgb = roi_rgb[cy1:cy2, cx1:cx2]
                cell_hsv = roi_hsv[cy1:cy2, cx1:cx2]

                # Create circular mask within the sampled square patch
                ph, pw, _ = cell_rgb.shape
                py, px = np.ogrid[:ph, :pw]
                dist_from_center = np.sqrt((px - (pw - 1) / 2.0) ** 2 + (py - (ph - 1) / 2.0) ** 2)
                circle_mask = dist_from_center <= rad

                if np.any(circle_mask):
                    sampled_hsv = cell_hsv[circle_mask]
                    sampled_rgb = cell_rgb[circle_mask]
                else:
                    sampled_hsv = cell_hsv.reshape(-1, 3)
                    sampled_rgb = cell_rgb.reshape(-1, 3)

                median_h = float(np.median(sampled_hsv[:, 0]))
                mean_s = float(np.mean(sampled_hsv[:, 1]))
                mean_v = float(np.mean(sampled_hsv[:, 2]))

                mean_r = float(np.mean(sampled_rgb[:, 0]))
                mean_g = float(np.mean(sampled_rgb[:, 1]))
                mean_b = float(np.mean(sampled_rgb[:, 2]))

                (
                    color_name,
                    facelet_code,
                    hex_color,
                    conf,
                    hsv,
                    rgb,
                ) = classify_hsv_tile(
                    median_h,
                    mean_s,
                    mean_v,
                    mean_r,
                    mean_g,
                    mean_b,
                    self.calibrated_rgb,
                    self.calibrated_hex,
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
        face_code = FACE_NAME_TO_CODE.get(face, "U")

        if override_tiles and len(override_tiles) == 9:
            locked_tiles = override_tiles
        elif self.last_detection_tiles and len(self.last_detection_tiles) == 9:
            locked_tiles = [t.faceletCode for t in self.last_detection_tiles]
        else:
            locked_tiles = [face_code] * 9

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
