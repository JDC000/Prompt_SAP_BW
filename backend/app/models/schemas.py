from pydantic import BaseModel
from typing import Dict

class CompareRequest(BaseModel):
    photo_type: str
    image1_base64: str
    image2_base64: str

class CompareResponse(BaseModel):
    prompt_used: str
    similarity_score: float
    differences: Dict[str, str]