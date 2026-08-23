import concurrent.futures

import kociemba

from app.core.exceptions import CubeValidationError, SolverExecutionError, SolverTimeoutError
from app.schemas.solve import CubeMoveItem, SolveResponseData
from app.services.validator import validate_cube_state

SOLVED_STATE = "UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB"

ENGLISH_FACES = {
    "U": "Top",
    "R": "Right",
    "F": "Front",
    "D": "Bottom",
    "L": "Left",
    "B": "Back",
}

INDONESIAN_FACES = {
    "U": "ATAS",
    "R": "KANAN",
    "F": "DEPAN",
    "D": "BAWAH",
    "L": "KIRI",
    "B": "BELAKANG",
}

PHASE2_MOVES = {
    "U", "U'", "U2",
    "D", "D'", "D2",
    "R2", "L2", "F2", "B2",
}


def parse_move(index: int, notation: str) -> CubeMoveItem:
    face = notation[0]
    face_en = ENGLISH_FACES.get(face, face)
    face_id = INDONESIAN_FACES.get(face, face)

    if notation.endswith("'"):
        direction = "CCW"
        angle = -90
        instruction = f"Rotate {face_en} face counter-clockwise 90°"
        human_guidance = f"Putar sisi {face_id} berlawanan jarum jam 90°"
    elif notation.endswith("2"):
        direction = "DOUBLE"
        angle = 180
        instruction = f"Rotate {face_en} face 180°"
        human_guidance = f"Putar sisi {face_id} 180°"
    else:
        direction = "CW"
        angle = 90
        instruction = f"Rotate {face_en} face clockwise 90°"
        human_guidance = f"Putar sisi {face_id} searah jarum jam 90°"

    return CubeMoveItem(
        index=index,
        notation=notation,
        face=face,
        direction=direction,
        angle=angle,
        instruction=instruction,
        humanGuidance=human_guidance,
    )


def solve_cube(
    state: str,
    max_depth: int = 24,
    timeout_seconds: float = 2.0,
) -> SolveResponseData:
    # 1. Pre-flight state validation
    validation = validate_cube_state(state)
    if not validation.isValid:
        details = []
        for issue in validation.issues:
            detail = {"reason": issue.message}
            if issue.face:
                detail["face"] = issue.face
            if issue.tileIndex is not None:
                detail["tileIndex"] = issue.tileIndex
            details.append(detail)

        raise CubeValidationError(
            code="CUBE_STATE_INVALID",
            message="The cube configuration is mathematically impossible to solve.",
            details=details,
        )

    # 2. Check if already solved
    if state == SOLVED_STATE:
        return SolveResponseData(
            isSolved=True,
            totalMoves=0,
            solutionString="",
            moves=[],
            phase1Depth=0,
            phase2Depth=0,
        )

    # 3. Execute Kociemba solver with timeout
    def _execute():
        return kociemba.solve(state, max_depth=max_depth)

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_execute)
            solution_str = future.result(timeout=timeout_seconds).strip()
    except concurrent.futures.TimeoutError as err:
        raise SolverTimeoutError() from err
    except Exception as e:
        raise SolverExecutionError(
            code="SOLVER_EXECUTION_ERROR",
            message=f"Kociemba solver failed: {str(e)}",
        ) from e

    raw_moves = solution_str.split() if solution_str else []
    moves = [parse_move(i, m) for i, m in enumerate(raw_moves, start=1)]

    # Estimate phase depths if possible
    # Phase 2 suffix consists only of Phase 2 allowable moves
    p2_count = 0
    for m in reversed(raw_moves):
        if m in PHASE2_MOVES:
            p2_count += 1
        else:
            break
    p1_count = len(raw_moves) - p2_count

    return SolveResponseData(
        isSolved=False,
        totalMoves=len(moves),
        solutionString=solution_str,
        moves=moves,
        phase1Depth=p1_count if p1_count > 0 else None,
        phase2Depth=p2_count if p2_count > 0 else None,
    )
