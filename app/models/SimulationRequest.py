from pydantic import BaseModel


# -----------------------------
# Input Model
# -----------------------------
class SimulationRequest(BaseModel):
    user_id: str
    avatar_id: str
