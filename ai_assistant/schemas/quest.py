from pydantic import BaseModel, Field
import uuid
from enum import Enum

class Sex(Enum):
    male = "male"
    female = "female"

class RecommendatinId(BaseModel):
    user_id: uuid.UUID = Field(...)
    sex: Sex = Field(...)
    age: int = Field(...)




