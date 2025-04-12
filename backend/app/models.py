from pydantic import BaseModel
from typing import Dict, List, Optional

class CompareRequest(BaseModel):
    image_1: str
    image_2: str
    photo_type: str

class Difference(BaseModel):
    field: str
    reference: Optional[str]
    candidate: Optional[str]
    note: Optional[str]

class CompareResponse(BaseModel):
    image_1_json: Dict
    image_2_json: Dict
    differences: List[Difference]