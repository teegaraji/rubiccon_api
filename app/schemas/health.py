from pydantic import BaseModel


class HealthResponseData(BaseModel):
    status: str = "healthy"
    engine: str = "Kociemba Two-Phase Algorithm"
    version: str = "1.0.0"
    pruningTablesLoaded: bool = True
    uptimeSeconds: float = 0.0
