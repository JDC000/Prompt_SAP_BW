# Standard- und Drittanbieter-Imports für Bildverarbeitung, Logging, Umgebungsvariablen und OpenAI-API
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

# Lokaler Import eines Prompt-Generators
from .prompts import get_prompt


# Lade die Umgebungsvariablen aus einer .env-Datei (z. B. API-Keys)
load_dotenv()

# Konfiguriere das Logging-Modul zur Ausgabe von Informationen auf Konsole
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



class OpenRouterClient:
    def __init__(self):
        # Hole den OpenRouter API-Key aus den Umgebungsvariablen
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENROUTER_API_KEY fehlt in den Umgebungsvariablen.")

        # Initialisiere den OpenAI-Client zur Kommunikation mit der OpenRouter-Schnittstelle
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )

        # Definiere zusätzliche HTTP-Header für die API-Anfrage
        self.headers = {
            "HTTP-Referer": os.getenv("HTTP_REFERER", "http://localhost"),
            "X-Title": os.getenv("APP_TITLE", "SAP BW Image Comparator"),
            "Content-Type": "application/json"
        }

        # Modellkonfiguration
        self.model = "qwen/qwen2.5-vl-72b-instruct"  # Spezifiziert das zu verwendende Sprach-/Bildmodell
        self.max_tokens = 2048                      # Maximale Anzahl an Token in der Antwort
        self.max_retries = 3                        # Anzahl der Wiederholungsversuche bei Fehler
        self.timeout = 30                           # Timeout in Sekunden

        # Pfad zum Speichern von Debug-Bildern
        self.debug_save_path = "/Users/jennycao/Desktop/Thesis/Debug_save_path"
        os.makedirs(self.debug_save_path, exist_ok=True)
        self.reference_json = {...}

    def save_debug_image(self, image: np.ndarray, prefix: str = "debug") -> Optional[str]:
        """
        Speichert ein Debug-Bild mit Zeitstempel zur späteren Analyse.
        """
        try:
            if not isinstance(image, np.ndarray):
                logger.error("Ungültiger Bildtyp, numpy-Array erwartet.")
                return None

            if not os.access(self.debug_save_path, os.W_OK):
                logger.error(f"Kein Schreibzugriff auf das Debug-Verzeichnis: {self.debug_save_path}")
                return None

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"{prefix}_{timestamp}.png"
            filepath = os.path.join(self.debug_save_path, filename)

            # Konvertiere BGR zu RGB für korrekte Farbdarstellung
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            success = cv2.imwrite(filepath, image)
            if success:
                logger.info(f"Debug-Bild gespeichert: {filepath}")
                return filepath
            else:
                logger.error(f"Speichern fehlgeschlagen: {filepath}")
                return None

        except Exception as e:
            logger.error(f"Fehler beim Speichern des Debug-Bildes: {str(e)}", exc_info=True)
            return None

    def _prepare_image_data(self, base64_img: str) -> Optional[str]:
        """
        Validiert, skaliert und konvertiert ein Base64-kodiertes Bild zu einem standardisierten
        PNG-Bild im Format 2048x2048, das für die API-Eingabe geeignet ist.
        """

        # Überprüfe, ob ein gültiger Base64-String übergeben wurde
        if not base64_img or not isinstance(base64_img, str):
            return None  # Ungültiger Eingabewert

        # Entfernt den optionalen Data-URI-Präfix (z. B. "data:image/png;base64,") vom String
        if base64_img.startswith(("data:image/jpeg;base64,", "data:image/png;base64,", "data:image/gif;base64,")):
            base64_img = base64_img.split(",", 1)[-1]

        try:
            # Validierung: Länge des Base64-Strings muss durch 4 teilbar sein
            if len(base64_img) % 4 != 0:
                raise ValueError("Ungültige Base64-Länge")

            # Validierung: String darf nur Base64-konforme Zeichen enthalten
            if not re.fullmatch(r'^[A-Za-z0-9+/]+={0,2}$', base64_img):
                raise ValueError("Ungültige Base64-Zeichen gefunden")

            # Dekodiert den Base64-String zu Bytes
            decoded = base64.b64decode(base64_img, validate=True)

            # Wandelt Byte-Daten in ein NumPy-Array (OpenCV benötigt das)
            np_arr = np.frombuffer(decoded, np.uint8)

            # Dekodiert das NumPy-Array als Farbbild (BGR) mit OpenCV
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            # Falls das Bild nicht korrekt geladen wurde
            if image is None:
                raise ValueError("Bild konnte nicht dekodiert werden")

            # Ermittelt Höhe und Breite des Originalbilds
            height, width = image.shape[:2]

            # Berechnet Skalierungsfaktor: das größere Maß soll 2048px sein
            scale = 2048 / max(height, width)
            new_dim = (int(width * scale), int(height * scale))

            # Skaliert das Bild proportional auf die neue Größe
            resized = cv2.resize(image, new_dim)

            # Erstellt ein weißes Quadrat von 2048x2048 Pixeln als Hintergrund
            padded = np.ones((2048, 2048, 3), dtype=np.uint8) * 255  # RGB: (255, 255, 255) = Weiß

            # Berechnet den Abstand, um das Bild mittig einzufügen (Padding)
            pad_x = (2048 - new_dim[0]) // 2
            pad_y = (2048 - new_dim[1]) // 2

            # Fügt das skalierte Bild zentriert in das weiße Quadrat ein
            padded[pad_y:pad_y + new_dim[1], pad_x:pad_x + new_dim[0]] = resized

            # Speichert das Bild optional zu Debug-Zwecken
            self.save_debug_image(padded, "padded")

            # Konvertiert das gepaddete Bild in ein PIL-Image (für PNG-Speicherung)
            pil_img = Image.fromarray(padded)

            # Speichert das PIL-Image in einen Zwischenspeicher (BytesIO)
            buffered = BytesIO()
            pil_img.save(buffered, format="PNG")

            # Kodiert den PNG-Byteinhalt wieder zu Base64
            processed_base64 = base64.b64encode(buffered.getvalue()).decode()

            # Gibt den finalen Base64-String mit PNG-Data-URI zurück
            return f"data:image/png;base64,{processed_base64}"

        # Fehlerbehandlung: ungültige Base64-Formate oder Zeichen
        except (ValueError, binascii.Error) as e:
            logger.error(f"Bildvalidierung fehlgeschlagen: {str(e)}")
            return None

        # Fehlerbehandlung: alle anderen unerwarteten Fehler
        except Exception as e:
            logger.exception("Unerwarteter Fehler während der Bildvorbereitung")
            return None

    def _extract_json(self, raw_text: str) -> Optional[Union[Dict, List]]:
        """
        Extrahiert ein JSON-Objekt aus einem beliebigen Text – hilfreich bei "verrauschten" Antworten.
        """
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
                logger.warning(f"JSON-Extraktion fehlgeschlagen: {str(e)}",
                               extra={"input_sample": raw_text[:100] + "..."})
        return None

    def generate_json_from_image(self, base64_img: str, photo_type: str) -> Dict:
        """
        Wandelt ein Bild in JSON um, basierend auf einem bestimmten Prompt-Typ.
        Führt mehrere Schritte aus: Vorverarbeitung, Vorbereitung, Modellaufruf, Fehlerbehandlung.
        """
        prepared_image = self._prepare_image_data(base64_img)
        if not prepared_image:
            return {"error": "Ungültiges Bildformat", "details": "Base64-Bildvalidierung fehlgeschlagen"}

        prompt = f"{get_prompt(photo_type)}\nGeben Sie nur JSON gemäß der festgelegten Struktur zurück, ohne zusätzlichen Text."

        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    extra_headers=self.headers,
                    messages=[{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url",
                                                                                              "image_url": {
                                                                                                  "url": prepared_image}}]}],
                    max_tokens=self.max_tokens,
                    temperature=0.01,
                    response_format={"type": "json_object"},
                    timeout=self.timeout,
                    seed = 42
                )

                if response and response.choices and len(response.choices) > 0 and response.choices[0].message and \
                        response.choices[0].message.content:
                    raw_output = response.choices[0].message.content.strip()
                    logger.debug(f"Roh-API-Antwort: {raw_output[:200]}...")
                    parsed = self._extract_json(raw_output)

                    if parsed:

                        return parsed

                    return {"error": "Ungültige JSON-Antwort", "raw_output": raw_output}

            except Exception as e:
                logger.error(f"API-Aufruf fehlgeschlagen (Versuch {attempt + 1}): {str(e)}")
                if attempt == self.max_retries - 1:
                    return {"error": "API-Anfrage fehlgeschlagen", "exception": str(e)}

        return {"error": "Maximale Versuche überschritten"}

    def compare_jsons(self, reference: Union[Dict, List], candidate: Union[Dict, List], ignore_fields: List[str] = None,
                      word_match_fields: List[str] = None) -> List[Dict]:
        """
        Vergleicht zwei JSON-Strukturen, mit Unterstützung für Feld-Ausschlüsse und unscharfe Wortvergleiche.
        Gibt eine Liste mit Unterschieden zurück.
        """
        if ignore_fields is None:
            ignore_fields = []
        if word_match_fields is None:
            word_match_fields = []

        diffs = []

        def compare_words(ref_str: str, cand_str: str) -> bool:
            if not isinstance(ref_str, str) or not isinstance(cand_str, str):
                return False
            ref_words = set(ref_str.lower().split())
            cand_words = set(cand_str.lower().split())
            return ref_words.issubset(cand_words)

        def compare_items(ref, cand, path=""):
            if isinstance(ref, dict) and isinstance(cand, dict):
                gemeinsame_keys = set(ref.keys()).intersection(cand.keys())
                for key in gemeinsame_keys:
                    if key in ignore_fields:
                        continue
                    new_path = f"{path}.{key}" if path else key
                    if key in word_match_fields and isinstance(ref[key], str) and isinstance(cand[key], str):
                        if not compare_words(ref[key], cand[key]):
                            diffs.append({"path": new_path, "reference": ref[key], "candidate": cand[key],
                                          "type": "word_mismatch"})
                    else:
                        compare_items(ref[key], cand[key], new_path)
                for key in set(ref.keys()) - set(cand.keys()):
                    if key in ignore_fields:
                        continue
                    diffs.append({"path": f"{path}.{key}" if path else key, "reference": ref[key], "candidate": None,
                                  "type": "value is missing in candidate"})
                for key in set(cand.keys()) - set(ref.keys()):
                    if key in ignore_fields:
                        continue
                    diffs.append({"path": f"{path}.{key}" if path else key, "reference": None, "candidate": cand[key],
                                  "type": "value is missing in reference"})
            elif isinstance(ref, list) and isinstance(cand, list):
                for i, (r, c) in enumerate(zip(ref, cand)):
                    compare_items(r, c, f"{path}[{i}]")
                if len(ref) != len(cand):
                    for i in range(min(len(ref), len(cand)), max(len(ref), len(cand))):
                        if i < len(ref):
                            diffs.append({"path": f"{path}[{i}]", "reference": ref[i], "candidate": None,
                                          "type": "value is missing in candidate"})
                        else:
                            diffs.append({"path": f"{path}[{i}]", "reference": None, "candidate": cand[i],
                                          "type": "value is missing in reference"})
            elif ref != cand:
                diffs.append({"path": path, "reference": ref, "candidate": cand, "type": "value_mismatch"})

        compare_items(reference, candidate)
        return diffs


# Erstelle eine Singleton-Instanz des Clients
client = OpenRouterClient()