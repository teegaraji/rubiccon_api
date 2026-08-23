from typing import Any


class CubeValidationError(Exception):
    def __init__(
        self,
        code: str = "CUBE_STATE_INVALID",
        message: str = "The cube configuration is mathematically impossible to solve.",
        details: list[dict[str, Any]] | None = None,
    ):
        self.code = code
        self.message = message
        self.details = details or []
        super().__init__(message)


class SolverTimeoutError(Exception):
    def __init__(
        self,
        message: str = "Algorithm exceeded maximum allotted calculation time.",
    ):
        self.code = "SOLVER_TIMEOUT"
        self.message = message
        super().__init__(message)


class SolverExecutionError(Exception):
    def __init__(
        self,
        code: str = "SOLVER_ERROR",
        message: str = "An error occurred during solver execution.",
        details: list[dict[str, Any]] | None = None,
    ):
        self.code = code
        self.message = message
        self.details = details or []
        super().__init__(message)
