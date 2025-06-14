import os
import json
import base64
import logging
import re
from io import BytesIO
from typing import Dict, List, Union, Optional
import binascii
import numpy as np
from PIL import Image
from openai import OpenAI
from dotenv import load_dotenv
import cv2
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, before_sleep_log

from .prompts import get_prompt

# Load environment variables
load_dotenv()

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OpenRouterClient:
    """Refactored OpenRouter client with improved structure and error handling"""

    def __init__(self):
        # Configuration
        self.model = "qwen/qwen2.5-vl-72b-instruct"
        self.max_tokens = 2048
        self.max_retries = 3
        self.timeout = 30

        # Initialize client
        self._validate_environment()
        self.client = self._initialize_client()
        self.headers = {
            "HTTP-Referer": os.getenv("HTTP_REFERER", "http://localhost"),
            "X-Title": os.getenv("APP_TITLE", "SAP BW Image Comparator"),
            "Content-Type": "application/json"
        }

    def _validate_environment(self):
        """Validate required environment variables"""
        if not os.getenv("OPENROUTER_API_KEY"):
            raise EnvironmentError("OPENROUTER_API_KEY missing in environment variables")

    def _initialize_client(self):
        """Initialize OpenAI client"""
        return OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            timeout=self.timeout
        )

    def _prepare_image_data(self, base64_img: str) -> Optional[str]:
        """Prepare and standardize image data from base64"""
        if not base64_img or not isinstance(base64_img, str):
            logger.warning("Invalid base64_img input")
            return None

        # Remove data URI prefix if present
        if base64_img.startswith(("data:image/jpeg;base64,", "data:image/png;base64,", "data:image/gif;base64,")):
            base64_img = base64_img.split(",", 1)[-1]

        try:
            # Validate base64 string
            if len(base64_img) % 4 != 0 or not re.fullmatch(r'^[A-Za-z0-9+/]+={0,2}$', base64_img):
                raise ValueError("Invalid Base64 data")

            # Decode base64 to image
            decoded = base64.b64decode(base64_img, validate=True)
            np_arr = np.frombuffer(decoded, np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if image is None:
                raise ValueError("Could not decode image")

            # Resize while maintaining aspect ratio
            height, width = image.shape[:2]
            scale = 1024 / max(height, width)
            new_dim = (int(width * scale), int(height * scale))
            resized = cv2.resize(image, new_dim)

            # Add padding to make it 1024x1024
            padded = np.ones((1024, 1024, 3), dtype=np.uint8) * 255
            pad_x = (1024 - new_dim[0]) // 2
            pad_y = (1024 - new_dim[1]) // 2
            padded[pad_y:pad_y + new_dim[1], pad_x:pad_x + new_dim[0]] = resized

            # Convert to PNG and back to base64
            pil_img = Image.fromarray(padded)
            buffered = BytesIO()
            pil_img.save(buffered, format="PNG")
            processed_base64 = base64.b64encode(buffered.getvalue()).decode()

            return f"data:image/png;base64,{processed_base64}"

        except (ValueError, binascii.Error) as e:
            logger.error(f"Image validation failed: {str(e)}")
            return None
        except Exception as e:
            logger.exception("Unexpected error during image preparation")
            return None

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(Exception),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    def _call_api(self, messages: List[Dict]) -> Optional[str]:
        """Make API call with retry logic"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=0,
                top_p=0.1,
                response_format={"type": "json_object"},
                seed=42,
                timeout=self.timeout,
                extra_headers=self.headers
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"API call failed: {str(e)}")
            raise

    def _extract_json(self, raw_text: str) -> Optional[Union[Dict, List]]:
        """Extract JSON object from text"""
        if not raw_text or not isinstance(raw_text, str):
            return None

        try:
            return json.loads(raw_text)
        except json.JSONDecodeError:
            try:
                json_match = re.search(r'\{[\s\S]*\}|\[[\s\S]*\]', raw_text)
                if json_match:
                    extracted = json_match.group()
                    if not extracted.endswith('}') and not extracted.endswith(']'):
                        extracted = extracted.rsplit(',', 1)[0] + ('}' if '{' in extracted else ']')
                    return json.loads(extracted)
            except Exception as e:
                logger.warning(f"JSON extraction failed: {str(e)}", extra={"input_sample": raw_text[:100] + "..."})
        return None

    def generate_json_from_image(self, base64_img: str, photo_type: str) -> Dict:
        """Generate JSON from image using AI model"""
        prepared_image = self._prepare_image_data(base64_img)
        if not prepared_image:
            return {
                "error": "Invalid image format",
                "details": "Base64 image validation failed"
            }

        prompt = f"{get_prompt(photo_type)}\nReturn only JSON according to the specified structure, without additional text."

        try:
            response = self._call_api([
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": prepared_image}}
                    ]
                }
            ])

            if not response:
                return {"error": "Empty API response"}

            parsed = self._extract_json(response)
            if parsed:
                return parsed

            return {
                "error": "Invalid JSON response",
                "raw_output": response,
                "suggestion": "The model returned malformed JSON. Please try again."
            }

        except Exception as e:
            logger.error(f"API request failed after retries: {str(e)}")
            return {
                "error": "API request failed",
                "exception": str(e)
            }

    def compare_jsons(
            self,
            reference: Union[Dict, List],
            candidate: Union[Dict, List],
            ignore_fields: List[str] = None,
            word_match_fields: List[str] = None
    ) -> List[Dict]:
        """Compare two JSON objects with various options"""
        ignore_fields = ignore_fields or []
        word_match_fields = word_match_fields or []
        diffs = []

        def compare_words(ref_str: str, cand_str: str) -> bool:
            if not isinstance(ref_str, str) or not isinstance(cand_str, str):
                return False
            ref_words = set(ref_str.lower().split())
            cand_words = set(cand_str.lower().split())
            return ref_words.issubset(cand_words)

        def compare_items(ref, cand, path=""):
            if isinstance(ref, dict) and isinstance(cand, dict):
                common_keys = set(ref.keys()).intersection(cand.keys())
                for key in common_keys:
                    if key in ignore_fields:
                        continue
                    new_path = f"{path}.{key}" if path else key
                    if key in word_match_fields and isinstance(ref[key], str) and isinstance(cand[key], str):
                        if not compare_words(ref[key], cand[key]):
                            diffs.append({
                                "path": new_path,
                                "reference": ref[key],
                                "candidate": cand[key],
                                "type": "word_mismatch"
                            })
                    else:
                        compare_items(ref[key], cand[key], new_path)

                ref_only_keys = set(ref.keys()) - set(cand.keys())
                for key in ref_only_keys:
                    if key in ignore_fields:
                        continue
                    new_path = f"{path}.{key}" if path else key
                    diffs.append({
                        "path": new_path,
                        "reference": ref[key],
                        "candidate": None,
                        "type": "value is missing in candidate"
                    })

                cand_only_keys = set(cand.keys()) - set(ref.keys())
                for key in cand_only_keys:
                    if key in ignore_fields:
                        continue
                    new_path = f"{path}.{key}" if path else key
                    diffs.append({
                        "path": new_path,
                        "reference": None,
                        "candidate": cand[key],
                        "type": "value is missing in reference"
                    })

            elif isinstance(ref, list) and isinstance(cand, list):
                for i, (r, c) in enumerate(zip(ref, cand)):
                    compare_items(r, c, f"{path}[{i}]")

                if len(ref) != len(cand):
                    min_len = min(len(ref), len(cand))
                    max_len = max(len(ref), len(cand))
                    for i in range(min_len, max_len):
                        if i < len(ref):
                            diffs.append({
                                "path": f"{path}[{i}]",
                                "reference": ref[i],
                                "candidate": None,
                                "type": "value is missing in candidate"
                            })
                        else:
                            diffs.append({
                                "path": f"{path}[{i}]",
                                "reference": None,
                                "candidate": cand[i],
                                "type": "value is missing in reference"
                            })

            elif ref != cand:
                diffs.append({
                    "path": path,
                    "reference": ref,
                    "candidate": cand,
                    "type": "value_mismatch"
                })

        compare_items(reference, candidate)
        return diffs


# Create singleton client instance
client = OpenRouterClient()