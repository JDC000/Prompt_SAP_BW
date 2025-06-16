import os
import json
import base64
import logging
import re
from io import BytesIO
from typing import Dict, List, Union, Optional
import binascii
import datetime
import numpy as np
from PIL import Image
from openai import OpenAI
from dotenv import load_dotenv
import cv2

from .prompts import get_prompt

# Lade Umgebungsvariablen aus der .env-Datei
load_dotenv()

# Initialisiere Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OpenRouterClient:
    def __init__(self):
        # API-Key aus Umgebungsvariablen lesen
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENROUTER_API_KEY fehlt in den Umgebungsvariablen.")

        # Initialisiere OpenAI-Client mit OpenRouter-API
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )

        # Zusätzliche Header für den API-Aufruf
        self.headers = {
            "HTTP-Referer": os.getenv("HTTP_REFERER", "http://localhost"),
            "X-Title": os.getenv("APP_TITLE", "SAP BW Image Comparator"),
            "Content-Type": "application/json"
        }

        # Konfiguration des Modells und der Anfragelogik
        self.model = "qwen/qwen2.5-vl-72b-instruct"
        self.max_tokens = 2048
        self.max_retries = 3
        self.timeout = 30

        self.debug_save_path = "/Users/jennycao/Desktop/Thesis/Debug_save_path"
        os.makedirs(self.debug_save_path, exist_ok=True)

    def save_debug_image(self, image: np.ndarray, prefix: str = "debug") -> Optional[str]:
        """Save debug image to specified directory with timestamp prefix

        Args:
            image: numpy array representing the image
            prefix: filename prefix for the saved image

        Returns:
            str: Path to saved image if successful, None otherwise
        """
        try:
            # Validate input
            if not isinstance(image, np.ndarray):
                logger.error("Invalid image type, expected numpy array")
                return None

            # Check if debug directory exists and is writable
            if not os.path.exists(self.debug_save_path):
                os.makedirs(self.debug_save_path, exist_ok=True)

            if not os.access(self.debug_save_path, os.W_OK):
                logger.error(f"No write permission for debug directory: {self.debug_save_path}")
                return None

            # Generate timestamp and filename
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"{prefix}_{timestamp}.png"
            filepath = os.path.join(self.debug_save_path, filename)

            # Convert color space if needed (OpenCV uses BGR by default)
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Save image
            success = cv2.imwrite(filepath, image)
            if success:
                logger.info(f"Saved debug image to: {filepath}")
                return filepath
            else:
                logger.error(f"Failed to save debug image to: {filepath}")
                return None

        except Exception as e:
            logger.error(f"Error saving debug image: {str(e)}", exc_info=True)
            return None


    def _prepare_image_data(self, base64_img: str) -> Optional[str]:
        # Validierung und Konvertierung von Base64-Bilddaten in ein standardisiertes PNG
        if not base64_img or not isinstance(base64_img, str):
            return None

        # Entferne Data-URI-Prefix falls vorhanden
        if base64_img.startswith(("data:image/jpeg;base64,", "data:image/png;base64,", "data:image/gif;base64,")):
            base64_img = base64_img.split(",", 1)[-1]

        try:
            # Validierung der Base64-Länge und erlaubten Zeichen
            if len(base64_img) % 4 != 0:
                raise ValueError("Invalid Base64 length")
            if not re.fullmatch(r'^[A-Za-z0-9+/]+={0,2}$', base64_img):
                raise ValueError("Invalid Base64 characters")

            # Dekodieren des Base64-Strings zu einem Bild
            decoded = base64.b64decode(base64_img, validate=True)
            np_arr = np.frombuffer(decoded, np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if image is None:
                raise ValueError("Could not decode image")

            # Resize des Bildes unter Beibehaltung des Seitenverhältnisses
            height, width = image.shape[:2]
            scale = 2048 / max(height, width)
            new_dim = (int(width * scale), int(height * scale))
            resized = cv2.resize(image, new_dim)

            # Padding hinzufügen, um das Bild auf 1024x1024 zu bringen
            padded = np.ones((2048, 2048, 3), dtype=np.uint8) * 255
            pad_x = (2048 - new_dim[0]) // 2
            pad_y = (2048 - new_dim[1]) // 2
            padded[pad_y:pad_y + new_dim[1], pad_x:pad_x + new_dim[0]] = resized

            self.save_debug_image(padded, "padded")
            # Konvertierung zu PNG und zurück zu Base64
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

    def _extract_json(self, raw_text: str) -> Optional[Union[Dict, List]]:
        # Versucht, ein JSON-Objekt aus einem String zu extrahieren
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
        # Hauptfunktion: Bild senden und JSON von Modell extrahieren
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
                    temperature=0.01,
                    response_format={"type": "json_object"},
                    timeout=self.timeout
                )

                if response and response.choices and len(response.choices) > 0 and response.choices[0].message and \
                        response.choices[0].message.content:
                    raw_output = response.choices[0].message.content.strip()
                    logger.debug(f"Roh-API-Antwort: {raw_output[:200]}...")
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
                        "raw_output": raw_output if 'padded' in locals() else None
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
            ignore_fields: List[str] = None,
            word_match_fields: List[str] = None
    ) -> List[Dict]:
        # Vergleicht zwei JSON-Objekte mit verschiedenen Optionen
        if ignore_fields is None:
            ignore_fields = []
        if word_match_fields is None:
            word_match_fields = []

        diffs = []

        def compare_words(ref_str: str, cand_str: str) -> bool:
            # Vergleicht Wörter in Strings (case-insensitiv)
            if not isinstance(ref_str, str) or not isinstance(cand_str, str):
                return False
            ref_words = set(ref_str.lower().split())
            cand_words = set(cand_str.lower().split())
            return ref_words.issubset(cand_words)

        def compare_items(ref, cand, path=""):
            # Rekursiver Vergleich von JSON-Elementen
            if isinstance(ref, dict) and isinstance(cand, dict):
                gemeinsame_keys = set(ref.keys()).intersection(cand.keys())
                for key in gemeinsame_keys:
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

# Erstelle eine Singleton-Instanz des Clients
client = OpenRouterClient()