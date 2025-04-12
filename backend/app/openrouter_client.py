import requests
import base64
import json
from .prompts import get_prompt
from .utils import base64_to_image_bytes

OPENROUTER_API_KEY = "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"  # Thay API Key thật
API_URL = "https://openrouter.ai/chat?models=qwen/qwen2.5-vl-3b-instruct:free"
MODEL = "qwen/qwen2.5-vl-3b-instruct:free"

def generate_json_from_image(base64_img: str, photo_type: str) -> dict:
    prompt = get_prompt(photo_type) + "\nChỉ trả về JSON."

    image_bytes = base64_to_image_bytes(base64_img)
    image_b64 = base64.b64encode(image_bytes).decode()

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost"
    }

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_b64}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 1024
    }

    response = requests.post(API_URL, headers=headers, json=payload)
    result = response.json()

    try:
        text = result["choices"][0]["message"]["content"]
        start = text.find("{")
        return json.loads(text[start:])
    except Exception:
        return {"error": "Không thể phân tích JSON", "raw_output": result}

def compare_jsons(json1: dict, json2: dict):
    diffs = []
    for key in set(json1.keys()).union(json2.keys()):
        if json1.get(key) != json2.get(key):
            diffs.append({
                "field": key,
                "reference": json1.get(key),
                "candidate": json2.get(key),
                "note": f"Khác ở '{{key}}'"
            })
    return diffs
