from kociemba.pykociemba import facecube

from app.schemas.validate import ValidateResponseData, ValidationIssue

VALID_FACES = ("U", "R", "F", "D", "L", "B")
FACE_NAMES = {
    "U": "Up (Top)",
    "R": "Right",
    "F": "Front",
    "D": "Down (Bottom)",
    "L": "Left",
    "B": "Back",
}
CENTER_INDICES = {
    4: "U",
    13: "R",
    22: "F",
    31: "D",
    40: "L",
    49: "B",
}


def validate_cube_state(state: str) -> ValidateResponseData:
    issues: list[ValidationIssue] = []
    color_counts: dict[str, int] = {face: 0 for face in VALID_FACES}
    has_unique_centers = True
    corner_parity_valid = True
    edge_parity_valid = True
    permutation_parity_valid = True

    # 1. Length check
    if len(state) != 54:
        issues.append(
            ValidationIssue(
                code="INVALID_STATE_LENGTH",
                message=(
                    f"State string must be exactly 54 characters (received {len(state)})."
                ),
            )
        )
        return ValidateResponseData(
            isValid=False,
            colorCounts=color_counts,
            hasUniqueCenters=False,
            cornerParityValid=False,
            edgeParityValid=False,
            permutationParityValid=False,
            issues=issues,
        )

    # 2. Character validity and color counts
    invalid_chars = set()
    for idx, char in enumerate(state):
        if char in color_counts:
            color_counts[char] += 1
        else:
            invalid_chars.add(char)
            issues.append(
                ValidationIssue(
                    code="INVALID_FACELET_CHARS",
                    tileIndex=idx,
                    message=f"Invalid facelet character '{char}' at index {idx}.",
                )
            )

    # 3. Check color counts (must be exactly 9 of each color)
    for face in VALID_FACES:
        count = color_counts[face]
        if count != 9:
            issues.append(
                ValidationIssue(
                    code="COLOR_COUNT_MISMATCH",
                    face=face,
                    message=(
                        f"{FACE_NAMES.get(face, face)} face ({face}) has {count} "
                        "tiles (expected 9)."
                    ),
                )
            )

    # 4. Center tiles uniqueness check
    center_values = {}
    for idx, expected_face in CENTER_INDICES.items():
        actual_face = state[idx]
        center_values[idx] = actual_face
        if actual_face != expected_face:
            has_unique_centers = False
            issues.append(
                ValidationIssue(
                    code="DUPLICATE_OR_INVALID_CENTERS",
                    face=expected_face,
                    tileIndex=idx,
                    message=(
                        f"Center tile at index {idx} is '{actual_face}', "
                        f"expected standard center '{expected_face}'."
                    ),
                )
            )

    if len(set(center_values.values())) != 6:
        has_unique_centers = False

    # 5. Mathematical validation (only if basic checks pass)
    if not invalid_chars and all(c == 9 for c in color_counts.values()) and has_unique_centers:
        try:
            fc = facecube.FaceCube(state)
            cc = fc.toCubieCube()
            verify_code = cc.verify()

            if verify_code == 0:
                pass  # Cube is solvable
            elif verify_code == -2:
                edge_parity_valid = False
                issues.append(
                    ValidationIssue(
                        code="MISSING_OR_DUPLICATE_EDGES",
                        message="Not all 12 edges exist exactly once.",
                    )
                )
            elif verify_code == -3:
                edge_parity_valid = False
                issues.append(
                    ValidationIssue(
                        code="EDGE_FLIP_ERROR",
                        message="Edge piece parity violation: One edge has to be flipped.",
                    )
                )
            elif verify_code == -4:
                corner_parity_valid = False
                issues.append(
                    ValidationIssue(
                        code="MISSING_OR_DUPLICATE_CORNERS",
                        message="Not all 8 corners exist exactly once.",
                    )
                )
            elif verify_code == -5:
                corner_parity_valid = False
                issues.append(
                    ValidationIssue(
                        code="CORNER_TWIST_ERROR",
                        message="Corner piece parity violation: One corner has to be twisted.",
                    )
                )
            elif verify_code == -6:
                permutation_parity_valid = False
                issues.append(
                    ValidationIssue(
                        code="PERMUTATION_PARITY_ERROR",
                        message=(
                            "Permutation parity violation: Two corners or "
                            "two edges must be exchanged."
                        ),
                    )
                )
            else:
                issues.append(
                    ValidationIssue(
                        code="CUBE_STATE_INVALID",
                        message=f"Verification failed with code {verify_code}.",
                    )
                )
        except Exception as e:
            issues.append(
                ValidationIssue(
                    code="CUBE_STATE_INVALID",
                    message=f"Mathematical cube verification failed: {str(e)}",
                )
            )

    is_valid = (
        len(issues) == 0
        and has_unique_centers
        and corner_parity_valid
        and edge_parity_valid
        and permutation_parity_valid
    )

    return ValidateResponseData(
        isValid=is_valid,
        colorCounts=color_counts,
        hasUniqueCenters=has_unique_centers,
        cornerParityValid=corner_parity_valid,
        edgeParityValid=edge_parity_valid,
        permutationParityValid=permutation_parity_valid,
        issues=issues,
    )
