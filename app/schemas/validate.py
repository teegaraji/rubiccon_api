from pydantic import BaseModel, Field


class ValidateRequest(BaseModel):
    state: str = Field(
        ...,
        description="54-character facelet representation of the cube",
        examples=["UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB"],
    )


class ValidationIssue(BaseModel):
    code: str
    face: str | None = None
    tileIndex: int | None = None
    message: str


class ValidateResponseData(BaseModel):
    isValid: bool
    colorCounts: dict[str, int]
    hasUniqueCenters: bool
    cornerParityValid: bool
    edgeParityValid: bool
    permutationParityValid: bool
    issues: list[ValidationIssue]
