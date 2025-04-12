import base64
import json
import logging
import os
from typing import Dict, List, Union
from openai import OpenAI
from .prompts import get_prompt
from .utils import base64_to_image_bytes

# Logger setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OpenRouterClient:
    def __init__(self):
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY", "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")
        )
        self.headers = {
            "HTTP-Referer": os.getenv("HTTP_REFERER", "http://localhost"),
            "X-Title": os.getenv("APP_TITLE", "SAP BW Image Comparator")
        }
        self.model = "qwen/qwen2.5-vl-3b-instruct:free"
        self.max_tokens = 1024

    def generate_json_from_image(self, base64_img: str, photo_type: str) -> Union[Dict, List]:
        if not base64_img.startswith("data:image"):
            return {"error": "Invalid base64 image format", "received": base64_img[:50]}  # debug
        prompt = f"{get_prompt(photo_type)}\nChỉ trả về JSON."

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                extra_headers=self.headers,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": base64_img}},
                        ],
                    }
                ],
                max_tokens=self.max_tokens,
            )

            # Trích xuất nội dung chuỗi JSON
            raw_output = response.choices[0].message.content.strip()
            json_start = raw_output.find("{") if "{" in raw_output else raw_output.find("[")
            if json_start == -1:
                return {"error": "No JSON content returned", "raw_output": raw_output}

            cleaned = raw_output[json_start:]
            return json.loads(cleaned)

        except json.JSONDecodeError as e:
            logger.warning("JSON decode error: %s", str(e))
            return {
                "error": "Phân tích JSON thất bại",
                "raw_output": raw_output,
                "exception": str(e)
            }

        except Exception as e:
            logger.error("API call failed: %s", str(e))
            return {
                "error": "Lỗi xử lý ảnh",
                "exception": str(e)
            }

    def compare_jsons(
        self,
        reference: Union[Dict, List],
        candidate: Union[Dict, List],
        ignore_fields: List[str] = None
    ) -> List[Dict]:
        if ignore_fields is None:
            ignore_fields = []

        diffs = []

        # Support comparing lists of fields (common with table JSONs)
        if isinstance(reference, list) and isinstance(candidate, list):
            for i, (ref, cand) in enumerate(zip(reference, candidate)):
                for key in set(ref.keys()).union(cand.keys()):
                    if key in ignore_fields:
                        continue
                    if ref.get(key) != cand.get(key):
                        diffs.append({
                            "index": i,
                            "field": key,
                            "reference": ref.get(key),
                            "candidate": cand.get(key),
                            "type": "value_mismatch"
                        })
            return diffs

        # Fallback: flat dictionary comparison
        for key in set(reference.keys()).union(candidate.keys()):
            if key in ignore_fields:
                continue
            if reference.get(key) != candidate.get(key):
                diffs.append({
                    "field": key,
                    "reference": reference.get(key),
                    "candidate": candidate.get(key),
                    "type": "value_mismatch"
                })
        return diffs


# Singleton
client = OpenRouterClient()
