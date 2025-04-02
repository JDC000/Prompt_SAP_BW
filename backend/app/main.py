from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.models.schemas import CompareRequest, CompareResponse
from app.services.image_compare_qwen import compare_images

app = FastAPI(
    title="SAP BW Image Comparison",
    description="Compare SAP BW diagrams using Qwen-VL 2.5",
    version="1.0.0"
)

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/compare", response_model=CompareResponse)
def compare(req: CompareRequest):
    result = compare_images(req.photo_type, req.image1_base64, req.image2_base64)
    return result