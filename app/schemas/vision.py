from typing import Annotated, Literal

from pydantic import BaseModel, Field

CubeFace = Literal["U", "R", "F", "D", "L", "B"]
CubeFaceName = Literal["Up", "Right", "Front", "Down", "Left", "Back"]
FaceletColor = Literal["white", "red", "green", "yellow", "orange", "blue"]

FACE_NAME_TO_CODE: dict[CubeFaceName, CubeFace] = {
    "Up": "U",
    "Right": "R",
    "Front": "F",
    "Down": "D",
    "Left": "L",
    "Back": "B",
}

FACE_CODE_TO_NAME: dict[CubeFace, CubeFaceName] = {
    "U": "Up",
    "R": "Right",
    "F": "Front",
    "D": "Down",
    "L": "Left",
    "B": "Back",
}

STANDARD_FACE_ORDER: list[CubeFaceName] = [
    "Up",
    "Right",
    "Front",
    "Down",
    "Left",
    "Back",
]


class VisionRoiGuide(BaseModel):
    normalizedX: float = 0.2
    normalizedY: float = 0.2
    normalizedWidth: float = 0.6
    normalizedHeight: float = 0.6


# Client Messages (Upstream)
class VisionInitSessionMessage(BaseModel):
    action: Literal["init_session"] = "init_session"
    sessionId: str | None = None
    cubeType: Literal["3x3"] = "3x3"
    initialFace: CubeFaceName = "Front"
    colorScheme: Literal["western"] = "western"
    calibrationPalette: dict[CubeFace, list[int]] | None = None


class VisionCalibrateColorsMessage(BaseModel):
    action: Literal["calibrate_colors"] = "calibrate_colors"
    palette: dict[CubeFace, list[int]]


class VisionProcessFrameMessage(BaseModel):
    action: Literal["process_frame"] = "process_frame"
    frameId: int
    activeFace: CubeFaceName = "Front"
    timestamp: int | float
    image: str
    roiGuide: VisionRoiGuide | None = None


class VisionSetActiveFaceMessage(BaseModel):
    action: Literal["set_active_face"] = "set_active_face"
    face: CubeFaceName


class VisionLockFaceMessage(BaseModel):
    action: Literal["lock_face"] = "lock_face"
    face: CubeFaceName
    overrideTiles: list[str] | None = None


class VisionResetSessionMessage(BaseModel):
    action: Literal["reset_session"] = "reset_session"


VisionClientMessage = Annotated[
    VisionInitSessionMessage
    | VisionCalibrateColorsMessage
    | VisionProcessFrameMessage
    | VisionSetActiveFaceMessage
    | VisionLockFaceMessage
    | VisionResetSessionMessage,
    Field(discriminator="action"),
]


# Server Messages (Downstream)
class VisionTileDetection(BaseModel):
    index: int
    color: FaceletColor | str
    faceletCode: CubeFace | str
    hex: str
    confidence: float
    hsv: list[int]
    rgb: list[int]


class VisionDetectionMeta(BaseModel):
    serverProcessingTimeMs: float
    timestamp: int | float


class VisionDetectionResultEvent(BaseModel):
    event: Literal["detection_result"] = "detection_result"
    frameId: int
    activeFace: CubeFaceName
    isGridAligned: bool = True
    stabilityScore: float
    isReadyToLock: bool
    tiles: list[VisionTileDetection]
    meta: VisionDetectionMeta | None = None


class VisionFaceLockedEvent(BaseModel):
    event: Literal["face_locked"] = "face_locked"
    face: CubeFaceName
    tiles: list[str]
    completedFaces: list[CubeFaceName]
    completedCount: int
    isFullyMapped: bool
    nextSuggestedFace: CubeFaceName | None = None
    fullStateString: str | None = None


class VisionSessionCompletedEvent(BaseModel):
    event: Literal["session_completed"] = "session_completed"
    isFullyMapped: Literal[True] = True
    fullStateString: str
    isValid: bool


class VisionColorCalibratedEvent(BaseModel):
    event: Literal["color_calibrated"] = "color_calibrated"
    calibratedPalette: dict[CubeFace, list[int]]
    source: Literal["manual", "auto_center"] = "manual"


class VisionErrorEvent(BaseModel):
    event: Literal["vision_error"] = "vision_error"
    code: str
    message: str
