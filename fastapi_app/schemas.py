# app/schemas.py

from pydantic import BaseModel

class Detection(BaseModel):

    class_name: str
    confidence: float
    bbox: list
