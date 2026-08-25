from app.services.vision import (
    VisionSession,
    classify_tile_color,
    classify_tile_color_calibrated,
    rgb_to_cielab,
)


def test_rgb_to_cielab_reference_points():
    # Pure white (255, 255, 255) -> L* approx 100, a* approx 0, b* approx 0
    l_w, a_w, b_w = rgb_to_cielab(255, 255, 255)
    assert 99.0 <= l_w <= 100.0
    assert abs(a_w) < 1.0
    assert abs(b_w) < 1.0

    # Pure black (0, 0, 0) -> L* = 0, a* = 0, b* = 0
    l_k, a_k, b_k = rgb_to_cielab(0, 0, 0)
    assert l_k == 0.0
    assert a_k == 0.0
    assert b_k == 0.0

    # Vibrant Red (220, 38, 38) -> high a* (positive red axis)
    _, a_r, _ = rgb_to_cielab(220, 38, 38)
    assert a_r > 50.0

    # Vibrant Green (22, 163, 74) -> negative a* (green axis)
    _, a_g, _ = rgb_to_cielab(22, 163, 74)
    assert a_g < -30.0

    # Vibrant Blue (37, 99, 235) -> negative b* (blue axis)
    _, _, b_b = rgb_to_cielab(37, 99, 235)
    assert b_b < -40.0

    # Vibrant Yellow (234, 179, 8) -> high positive b* (yellow axis)
    _, _, b_y = rgb_to_cielab(234, 179, 8)
    assert b_y > 50.0


def test_default_classify_tile_color_standard_colors():
    # White
    name, code, _, conf, _, _ = classify_tile_color(248, 250, 252)
    assert code == "U"
    assert name == "white"
    assert conf >= 0.80

    # Red
    name, code, _, conf, _, _ = classify_tile_color(220, 38, 38)
    assert code == "F"
    assert name == "red"
    assert conf >= 0.80

    # Orange
    name, code, _, conf, _, _ = classify_tile_color(234, 88, 12)
    assert code == "B"
    assert name == "orange"
    assert conf >= 0.80

    # Yellow
    name, code, _, conf, _, _ = classify_tile_color(234, 179, 8)
    assert code == "D"
    assert name == "yellow"
    assert conf >= 0.80

    # Green
    name, code, _, conf, _, _ = classify_tile_color(22, 163, 74)
    assert code == "L"
    assert name == "green"
    assert conf >= 0.80

    # Blue
    name, code, _, conf, _, _ = classify_tile_color(37, 99, 235)
    assert code == "R"
    assert name == "blue"
    assert conf >= 0.80


def test_custom_calibrated_palette_variations():
    # Non-standard pastel / fluorescent speedcube palette
    custom_palette = {
        "U": (245, 238, 225),  # Warm white
        "R": (6, 182, 212),    # Cyan / light turquoise blue
        "F": (244, 63, 94),    # Pinkish / light magenta red
        "D": (250, 204, 21),   # Light lemon yellow
        "L": (74, 222, 128),   # Light mint green
        "B": (251, 146, 60),   # Light peach orange
    }

    # Test Pinkish red classification against calibrated palette
    name, code, _, conf, _, _ = classify_tile_color_calibrated(240, 60, 95, custom_palette)
    assert code == "F"
    assert name == "red"
    assert conf >= 0.70

    # Test Warm white under indoor incandescent lighting
    name, code, _, conf, _, _ = classify_tile_color_calibrated(242, 235, 222, custom_palette)
    assert code == "U"
    assert name == "white"

    # Test Cyan Blue
    name, code, _, conf, _, _ = classify_tile_color_calibrated(10, 180, 210, custom_palette)
    assert code == "R"
    assert name == "blue"

    # Test Light peach orange vs Pinkish red
    name, code, _, _, _, _ = classify_tile_color_calibrated(248, 140, 55, custom_palette)
    assert code == "B"
    assert name == "orange"


def test_vision_session_zero_drift_calibration():
    session = VisionSession(initial_face="Front")
    assert "F" in session.calibrated_rgb

    # Explicit calibration update
    session.calibrate_colors({"F": [245, 50, 80]})
    assert session.calibrated_rgb["F"] == (245.0, 50.0, 80.0)

    # Locking face does not corrupt or drift calibrated reference profile
    session.last_detection_tiles = [
        type(
            "TileMock",
            (),
            {
                "faceletCode": "F",
                "rgb": [200, 200, 200],  # Mock background color
            },
        )()
        for _ in range(9)
    ]
    locked_evt, _ = session.lock_face("Front")
    assert locked_evt.face == "Front"
    assert session.calibrated_rgb["F"] == (245.0, 50.0, 80.0)


def test_opencv_hsv_difficult_lighting_and_plastic_variations():
    # Warm lamp white (incandescent lighting)
    name, code, _, _, _, _ = classify_tile_color(240, 235, 210)
    assert code == "U"
    assert name == "white"

    # Shadowed white
    name, code, _, _, _, _ = classify_tile_color(180, 180, 175)
    assert code == "U"
    assert name == "white"

    # Dark / Deep red speedcube plastic
    name, code, _, _, _, _ = classify_tile_color(160, 20, 20)
    assert code == "F"
    assert name == "red"

    # Lime green plastic
    name, code, _, _, _, _ = classify_tile_color(100, 220, 30)
    assert code == "L"
    assert name == "green"

    # Cyan / light turquoise blue
    name, code, _, _, _, _ = classify_tile_color(0, 180, 240)
    assert code == "R"
    assert name == "blue"

    # Pastel yellow vs warm white discrimination
    name, code, _, _, _, _ = classify_tile_color(250, 240, 100)
    assert code == "D"
    assert name == "yellow"


