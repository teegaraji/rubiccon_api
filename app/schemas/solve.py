from pydantic import BaseModel, Field


class SolveRequest(BaseModel):
    state: str = Field(
        ...,
        description="54-character facelet representation of the cube",
        examples=["BBURUDBFUFFFRRFUUFLULUFUDLRRDBBDBDBLUDDFLLRRBRLLLBRDDF"],
    )
    maxDepth: int = Field(
        default=24,
        ge=1,
        le=30,
        description="Maximum search depth in moves",
    )
    timeoutSeconds: float = Field(
        default=2.0,
        gt=0.0,
        le=10.0,
        description="Execution timeout limit in seconds",
    )


class CubeMoveItem(BaseModel):
    index: int
    notation: str
    face: str
    direction: str
    angle: int
    instruction: str
    humanGuidance: str


class SolveResponseData(BaseModel):
    isSolved: bool
    totalMoves: int
    solutionString: str
    moves: list[CubeMoveItem]
    phase1Depth: int | None = None
    phase2Depth: int | None = None
