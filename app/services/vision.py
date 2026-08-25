import base64
import colorsys
import io
import math
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
    # 1. Linearize sRGB
    def pivot_rgb(c: float) -> float:
        c_norm = max(0.0, min(255.0, c)) / 255.0
        return c_norm / 12.92 if c_norm <= 0.04045 else ((c_norm + 0.055) / 1.055) ** 2.4

    r_lin = pivot_rgb(r)
    g_lin = pivot_rgb(g)
    b_lin = pivot_rgb(b)

    # 2. Linear RGB to CIE XYZ (D65 illuminant matrix)
    x = r_lin * 0.4124564 + g_lin * 0.3575761 + b_lin * 0.1804375
    y = r_lin * 0.2126729 + g_lin * 0.7151522 + b_lin * 0.0721750
    z = r_lin * 0.0193339 + g_lin * 0.1191920 + b_lin * 0.9503041

    # 3. Normalize for D65 standard reference white (Xn=0.95047, Yn=1.00000, Zn=1.08883)
    xr = x / 0.95047
    yr = y / 1.00000
    zr = z / 1.08883

    # 4. Convert XYZ to CIELAB
    def pivot_xyz(t: float) -> float:
        return t ** (1.0 / 3.0) if t > 0.008856 else (7.787 * t) + (16.0 / 116.0)

    fx = pivot_xyz(xr)
    fy = pivot_xyz(yr)
    fz = pivot_xyz(zr)

    l_val = max(0.0, min(100.0, (116.0 * fy) - 16.0))
    a_val = 500.0 * (fx - fy)
    b_val = 200.0 * (fy - fz)

    return (l_val, a_val, b_val)


def calculate_color_distance(
    sample_lab: tuple[float, float, float],
    prototype_lab: tuple[float, float, float],
    is_white_prototype: bool = False,
) -> float:
    """Compute weighted perceptual distance in CIELAB color space.

    Luminance (L*) weight is slightly reduced (0.6) to provide robustness against
    surface lighting gradients and directional shadows.
    """
    sl, sa, sb = sample_lab
    pl, pa, pb = prototype_lab

    # If prototype is White, penalize excessive saturation/chroma
    if is_white_prototype:
        sample_chroma = math.sqrt(sa * sa + sb * sb)
        # Moderate penalty if sample chroma exceeds neutral threshold
        chroma_penalty = max(0.0, (sample_chroma - 20.0) * 1.5)
        return math.sqrt(0.7 * (sl - pl) ** 2 + (sa - pa) ** 2 + (sb - pb) ** 2) + chroma_penalty

    return math.sqrt(0.6 * (sl - pl) ** 2 + (sa - pa) ** 2 + (sb - pb) ** 2)


def classify_tile_color_calibrated(
    r: float,
    g: float,
    b: float,
    palette_lab: dict[CubeFace, tuple[float, float, float]],
    palette_hex: dict[CubeFace, str] | None = None,
) -> tuple[FaceletColor, CubeFace, str, float, list[int], list[int]]:
    """Classify a single tile sample by matching against calibrated CIELAB prototypes."""
    rf, gf, bf = r / 255.0, g / 255.0, b / 255.0
    h, s, v = colorsys.rgb_to_hsv(rf, gf, bf)
    hue_deg = round(h * 360.0)
    sat_pct = round(s * 100.0)
    val_pct = round(v * 100.0)
    rgb_list = [int(round(r)), int(round(g)), int(round(b))]
    hsv_list = [hue_deg, sat_pct, val_pct]

    sample_lab = rgb_to_cielab(r, g, b)
    hex_map = palette_hex or FACE_CODE_TO_HEX

    distances: list[tuple[CubeFace, float]] = []
    for face_code, proto_lab in palette_lab.items():
        is_white = (face_code == "U")
        dist = calculate_color_distance(sample_lab, proto_lab, is_white_prototype=is_white)
        distances.append((face_code, dist))

    distances.sort(key=lambda item: item[1])
    best_face, best_dist = distances[0]
    second_best_dist = distances[1][1] if len(distances) > 1 else best_dist + 10.0

    # Calculate confidence score based on margin between best and second-best candidate
    margin = (second_best_dist - best_dist) / (second_best_dist + 1e-5)
    confidence = round(max(0.65, min(0.99, 0.65 + 0.34 * margin)), 2)

    color_name = FACE_CODE_TO_COLOR.get(best_face, "white")
    hex_val = hex_map.get(best_face, "#ffffff")

    return color_name, best_face, hex_val, confidence, hsv_list, rgb_list


def classify_tile_color(
    r: float, g: float, b: float
) -> tuple[FaceletColor, CubeFace, str, float, list[int], list[int]]:
    """Legacy helper: classifies tile color using the standard default baseline palette."""
    default_lab = {
        face: rgb_to_cielab(*rgb) for face, rgb in DEFAULT_PALETTE_RGB.items()
    }
    return classify_tile_color_calibrated(r, g, b, default_lab)


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

        np_img = np.array(image)
        roi_img = np_img[y1:y2, x1:x2]
        roi_h, roi_w, _ = roi_img.shape

        cell_w = roi_w / 3.0
        cell_h = roi_h / 3.0

        tiles: list[VisionTileDetection] = []
        current_facelet_codes: list[str] = []

        # 3. Sample 3x3 grid cells
        tile_idx = 0
        center_rgb: tuple[float, float, float] | None = None

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

                if tile_idx == 4:
                    center_rgb = (mean_r, mean_g, mean_b)

                (
                    color_name,
                    facelet_code,
                    hex_color,
                    conf,
                    hsv,
                    rgb,
                ) = classify_tile_color_calibrated(
                    mean_r, mean_g, mean_b, self.calibrated_lab, self.calibrated_hex
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

        # Layer 2: Auto-adaptive center tile refinement when detection is highly stable
        if is_ready_to_lock and center_rgb:
            center_code = FACE_NAME_TO_CODE.get(active_face)
            if center_code:
                # Smoothly update prototype with small learning rate (alpha=0.08)
                cr, cg, cb = center_rgb
                old_r, old_g, old_b = self.calibrated_rgb[center_code]
                new_r = 0.92 * old_r + 0.08 * cr
                new_g = 0.92 * old_g + 0.08 * cg
                new_b = 0.92 * old_b + 0.08 * cb
                self.calibrated_rgb[center_code] = (new_r, new_g, new_b)
                self.calibrated_lab[center_code] = rgb_to_cielab(new_r, new_g, new_b)

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
            # Layer 2: Lock in center tile color sample to reference profile
            center_tile = self.last_detection_tiles[4]
            if center_tile.rgb and len(center_tile.rgb) == 3:
                r = float(center_tile.rgb[0])
                g = float(center_tile.rgb[1])
                b = float(center_tile.rgb[2])
                self.calibrated_rgb[face_code] = (r, g, b)
                self.calibrated_lab[face_code] = rgb_to_cielab(r, g, b)
                self.calibrated_hex[face_code] = f"#{int(r):02x}{int(g):02x}{int(b):02x}"
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
