from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import base64
from backend.app.openrouter_client import client
from dotenv import load_dotenv
load_dotenv()




app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # hoặc ['http://localhost:5173']
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/compare")
async def compare_images(
    image_1: UploadFile = File(...),
    image_2: UploadFile = File(...),
    photo_type: str = Form(...)
):
    try:
        img1_bytes = await image_1.read()
        img2_bytes = await image_2.read()

        img1_b64 = f"data:image/png;base64,{base64.b64encode(img1_bytes).decode()}"
        img2_b64 = f"data:image/png;base64,{base64.b64encode(img2_bytes).decode()}"


        json1 = client.generate_json_from_image(img1_b64, photo_type)
        json2 = client.generate_json_from_image(img2_b64, photo_type)
        diffs = client.compare_jsons(json1, json2)

        return {
            "image_1_json": json1,
            "image_2_json": json2,
            "differences": diffs
        }

    except Exception as e:
        return {"error": f"Lỗi khi so sánh ảnh: {str(e)}"}