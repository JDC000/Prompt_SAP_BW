from fastapi import FastAPI, File, Form, UploadFile

from backend.app.models import CompareRequest, CompareResponse
from backend.app.openrouter_client import generate_json_from_image, compare_jsons

app = FastAPI()

@app.post("/compare")
async def compare_images(
    image_1: UploadFile = File(...),
    image_2: UploadFile = File(...),
    photo_type: str = Form(...)
):
    try:
        json1 = generate_json_from_image(await image_1.read(), photo_type)
        json2 = generate_json_from_image(await image_2.read(), photo_type)
        diffs = compare_jsons(json1, json2)

        return {
            "image_1_json": json1,
            "image_2_json": json2,
            "differences": diffs
        }

    except Exception as e:
        return {"error": f"Lỗi khi so sánh ảnh: {str(e)}"}
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # hoặc ["http://localhost:5173"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
