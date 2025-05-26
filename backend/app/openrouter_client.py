import os
import json
import base64
import logging
import re
from typing import Dict, List, Union, Optional
import binascii
from openai import OpenAI
from dotenv import load_dotenv
from .prompts import get_prompt

# Load environment variables
load_dotenv()

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OpenRouterClient:
    def __init__(self):
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENROUTER_API_KEY fehlt in den Umgebungsvariablen.")

        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )

        self.headers = {
            "HTTP-Referer": os.getenv("HTTP_REFERER", "http://localhost"),
            "X-Title": os.getenv("APP_TITLE", "SAP BW Image Comparator"),
            "Content-Type": "application/json"
        }
        self.model = "google/gemini-2.5-flash-preview-05-20"
        self.max_tokens = 2048  # Increased for complex responses
        self.max_retries = 3
        self.timeout = 30  # seconds

    def _prepare_image_data(self, base64_img: str) -> Optional[str]:

        if not base64_img or not isinstance(base64_img, str):
            return None

        # Check if already has valid data URI prefix
        if base64_img.startswith(("data:image/jpeg;base64,",
                                  "data:image/png;base64,",
                                  "data:image/gif;base64,")):
            return base64_img

        # Remove existing prefix if invalid
        if base64_img.startswith("data:image"):
            base64_img = base64_img.split(",", 1)[-1]

        # Validate base64 format
        try:
            # Check length is multiple of 4
            if len(base64_img) % 4 != 0:
                raise ValueError("Invalid base64 length")

            # Check character set
            if not re.fullmatch(r'^[A-Za-z0-9+/]+={0,2}$', base64_img):
                raise ValueError("Invalid base64 characters")

            # Verify decodable
            decoded = base64.b64decode(base64_img, validate=True)

            # Try to determine image type (simplified check)
            image_type = "jpeg" if base64_img[0] == "/" else "png"

            return f"data:image/{image_type};base64,{base64_img}"

        except (ValueError, binascii.Error) as e:
            logger.error(f"Image validation failed: {str(e)}")
            return None
        except Exception as e:
            logger.exception("Unexpected error during image validation")
            return None

    def _extract_json(self, raw_text: str) -> Optional[Union[Dict, List]]:

        if not raw_text or not isinstance(raw_text, str):
            return None

        try:
            # First try to parse directly
            return json.loads(raw_text)
        except json.JSONDecodeError:
            # Fallback: find JSON-like substrings
            try:
                # Improved regex pattern to match both {} and [] JSON structures
                json_match = re.search(r'\{[\s\S]*\}|\[[\s\S]*\]', raw_text)
                if json_match:
                    extracted = json_match.group()
                    # Clean common truncations
                    if not extracted.endswith('}') and not extracted.endswith(']'):
                        extracted = extracted.rsplit(',', 1)[0] + ('}' if '{' in extracted else ']')
                    return json.loads(extracted)
            except json.JSONDecodeError as je:
                logger.warning(f"JSON extraction failed: {str(je)}",
                               extra={"input_sample": raw_text[:100] + "..."})
            except Exception as e:
                logger.error(f"Unexpected error during JSON extraction: {str(e)}")
        return None

    def generate_json_from_image(self, base64_img: str, photo_type: str) -> Dict:
        """Generate JSON from image with enhanced error handling"""
        prepared_image = self._prepare_image_data(base64_img)
        if not prepared_image:
            return {
                "error": "Ungültiges Bildformat",
                "details": "Base64-Bildvalidierung fehlgeschlagen"
            }

        prompt = f"{get_prompt(photo_type)}\nGeben Sie nur JSON gemäß der festgelegten Struktur zurück, ohne zusätzlichen Text."

        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    extra_headers=self.headers,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": prepared_image}}
                            ]
                        }
                    ],
                    max_tokens=self.max_tokens,
                    response_format={"type": "json_object"},  # Force JSON output
                    timeout=self.timeout
                )

                if response and response.choices and len(response.choices) > 0 and response.choices[0].message and response.choices[0].message.content:
                    raw_output = response.choices[0].message.content.strip()
                    logger.debug(f"Roh-API-Antwort: {raw_output[:200]}...")  # Log first part

                    parsed = self._extract_json(raw_output)
                    if parsed:
                        return parsed

                    return {
                        "error": "Ungültige JSON-Antwort",
                        "raw_output": raw_output,
                        "suggestion": "Das Modell hat ein fehlerhaftes JSON zurückgegeben. Bitte versuchen Sie es erneut."
                    }
                else:
                    logger.error(f"Leere oder fehlerhafte Antwort von OpenRouter (Versuch {attempt + 1}): {response}")
                    if attempt == self.max_retries - 1:
                        return {
                            "error": "Leere Antwort von der API nach mehreren Versuchen",
                            "raw_response": str(response)
                        }

            except json.JSONDecodeError as je:
                logger.error(f"JSON-Decodierungsfehler (Versuch {attempt + 1}): {str(je)}")
                if attempt == self.max_retries - 1:
                    return {
                        "error": "JSON-Parsing fehlgeschlagen nach mehreren Versuchen",
                        "exception": str(je),
                        "raw_output": raw_output if 'raw_output' in locals() else None
                    }

            except Exception as e:

                logger.error(f"API-Aufruf fehlgeschlagen (Versuch {attempt + 1}): {str(e)}")

                if attempt == self.max_retries - 1:
                    return {

                        "error": "API-Anfrage fehlgeschlagen",
                        "exception": str(e)
                    }
        return {"error": "Maximale Versuche überschritten"}

    def compare_jsons(
            self,
            reference: Union[Dict, List],
            candidate: Union[Dict, List],
            ignore_fields: List[str] = None
    ) -> List[Dict]:
        """Enhanced JSON comparison with nested structure support"""
        if ignore_fields is None:
            ignore_fields = []

        diffs = []

        def compare_items(ref, cand, path=""):
            if isinstance(ref, dict) and isinstance(cand, dict):
                for key in set(ref.keys()).union(cand.keys()):
                    if key in ignore_fields:
                        continue
                    new_path = f"{path}.{key}" if path else key
                    if key not in ref:
                        diffs.append({
                            "path": new_path,
                            "reference": None,
                            "candidate": cand[key],
                            "type": "missing_in_reference"
                        })
                    elif key not in cand:
                        diffs.append({
                            "path": new_path,
                            "reference": ref[key],
                            "candidate": None,
                            "type": "missing_in_candidate"
                        })
                    else:
                        compare_items(ref[key], cand[key], new_path)
            elif isinstance(ref, list) and isinstance(cand, list):
                for i, (r, c) in enumerate(zip(ref, cand)):
                    compare_items(r, c, f"{path}[{i}]")
            elif ref != cand:
                diffs.append({
                    "path": path,
                    "reference": ref,
                    "candidate": cand,
                    "type": "value_mismatch"
                })

        compare_items(reference, candidate)
        return diffs


# Singleton instance
client = OpenRouterClient()