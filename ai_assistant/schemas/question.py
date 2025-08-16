from pydantic import BaseModel, Field
import uuid

class QuestionId(BaseModel):
    user_id: int = Field(...)
    text: str = Field(...)


class Question(BaseModel):
    user_id: uuid.UUID = Field(...)
    text: str = Field(...)




