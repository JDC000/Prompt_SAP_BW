import os
import json
import base64
import logging
import re
from io import BytesIO
from typing import Dict, List, Union, Optional
import binascii
import image
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from openai.types import Image

from .prompts import get_prompt
import cv2


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
        self.model = "qwen/qwen2.5-vl-72b-instruct"
        self.max_tokens = 2048  # Increased for complex responses
        self.max_retries = 3
        self.timeout = 30  # seconds

    def sharpen_image(image):
        """
        Wendet einen Schärfefilter auf das Eingabebild an, um Kanten hervorzuheben.
        Besonders nützlich bei verschwommenen Screenshots oder UI-Aufnahmen,
        bei denen Text oder Linien besser erkennbar gemacht werden sollen.

        """
        # Definiert einen Kernel, der den Mittelpunkt betont und die Nachbarn subtrahiert.
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])

        # Wendet den Schärfekernel per Faltung auf das Bild an (2D-Filter)
        return cv2.filter2D(image, -1, kernel)

    def apply_clahe(gray):
        """
        Wendet CLAHE (Contrast Limited Adaptive Histogram Equalization) auf ein Graustufenbild an.
        CLAHE verbessert den lokalen Kontrast, insbesondere in schwach beleuchteten oder kontrastarmen Bereichen,
        sodass Text und Details besser sichtbar werden.
        """
        # Erstellt ein CLAHE-Objekt mit sinnvollen Standardwerten:
        # - clipLimit: Begrenzung der Kontrastverstärkung zur Rauschvermeidung
        # - tileGridSize: Unterteilung des Bildes in 8x8-Kacheln für lokale Histogramm-Ausgleichung
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        # Wendet CLAHE an und gibt das verbesserte Bild zurück
        return clahe.apply(gray)

    def _prepare_image_data(self, base64_img: str) -> Optional[str]:
        if not base64_img or not isinstance(base64_img, str):
            return None  # Ungültiger oder kein String

        # Überprüfen, ob die Eingabe bereits ein gültiges data-URI ist (Base64-kodiertes Bild)
        if base64_img.startswith(("data:image/jpeg;base64,", "data:image/png;base64,", "data:image/gif;base64,")):
            return base64_img

        # Entfernt unvollständige oder fehlerhafte Prefixes
        if base64_img.startswith("data:image"):
            base64_img = base64_img.split(",", 1)[-1]

        try:
            # Validiert die Base64-Länge (muss durch 4 teilbar sein)
            if len(base64_img) % 4 != 0:
                raise ValueError("Ungültige Base64-Länge")

            # Überprüft erlaubte Base64-Zeichen (inkl. '=' Padding)
            if not re.fullmatch(r'^[A-Za-z0-9+/]+={0,2}$', base64_img):
                raise ValueError("Ungültige Base64-Zeichen")

            # Base64-Dekodierung und Umwandlung in Bilddaten
            decoded = base64.b64decode(base64_img, validate=True)
            np_arr = np.frombuffer(decoded, np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            # Erweiterte Bildvorverarbeitung für verschwommene Screenshots
            # Schritt 1: Bild schärfen zur Verbesserung der Textkanten
            sharpened = self.sharpen_image(image)

            # Schritt 2: In Graustufenbild umwandeln
            gray = cv2.cvtColor(sharpened, cv2.COLOR_BGR2GRAY)

            # Schritt 3: CLAHE anwenden, um lokalen Kontrast zu verbessern
            clahe_img = self.apply_clahe(gray)

            # Schritt 4: Bild auf 1024x1024 skalieren zur Standardisierung
            resized = cv2.resize(clahe_img, (1024, 1024))

            # Schritt 5: Rauschunterdrückung anwenden, Text bleibt erhalten
            denoised = cv2.fastNlMeansDenoising(resized, h=10)

            # --- Umwandlung des Bildes in Base64 (PNG) ---

            pil_img = Image.fromarray(denoised)  # NumPy → PIL
            buffered = BytesIO()  # Speicherpuffer
            pil_img.save(buffered, format="PNG")
            processed_base64 = base64.b64encode(buffered.getvalue()).decode()

            return f"data:image/png;base64,{processed_base64}"

        except (ValueError, binascii.Error) as e:
            logger.error(f"Bildvalidierung fehlgeschlagen: {str(e)}")
            return None  # Bei Validierungsfehlern

        except Exception as e:
            logger.exception("Unerwarteter Fehler bei der Bildvalidierung")
            return None  # Allgemeiner Fehler

    def _extract_json(self, raw_text: str) -> Optional[Union[Dict, List]]:
        """
        Extrahiert JSON aus einem gegebenen Text.
        Wenn der Text kein direktes valides JSON enthält, versucht die Methode, JSON-ähnliche Strukturen zu erkennen und zu parsen.
        """
        # Überprüfung: Eingabe muss ein String sein und darf nicht leer sein
        if not raw_text or not isinstance(raw_text, str):
            return None

        try:
            # 1. Versuch: Versuche, den gesamten Text direkt als JSON zu parsen
            return json.loads(raw_text)

        except json.JSONDecodeError:
            # Falls das Parsen fehlschlägt, z. B. bei zusätzlichem Text vor/nach dem JSON:
            try:
                # Verwende einen regulären Ausdruck, um entweder ein JSON-Objekt ({...}) oder eine JSON-Liste ([...]) zu finden
                json_match = re.search(r'\{[\s\S]*\}|\[[\s\S]*\]', raw_text)

                if json_match:
                    # Extrahiere den gefundenen JSON-ähnlichen String
                    extracted = json_match.group()

                    # Behebt das häufige Problem, dass ein JSON-Ausdruck nicht vollständig zurückgegeben wurde
                    # Beispiel: {"a": 1, "b": 2,   → wird abgeschnitten und muss bereinigt werden
                    if not extracted.endswith('}') and not extracted.endswith(']'):
                        extracted = extracted.rsplit(',', 1)[0] + ('}' if '{' in extracted else ']')

                    # Versuche, das bereinigte Fragment als JSON zu laden
                    return json.loads(extracted)

            except json.JSONDecodeError as je:
                # Wenn selbst das Parsen des extrahierten Fragments fehlschlägt, protokolliere eine Warnung
                logger.warning(
                    f"JSON-Extraktion fehlgeschlagen: {str(je)}",
                    extra={"input_sample": raw_text[:100] + "..."}  # Loggt die ersten 100 Zeichen zur Analyse
                )

            except Exception as e:
                # Unerwartete Fehler (z. B. bei Encoding-Problemen) werden hier protokolliert
                logger.error(f"Unerwarteter Fehler bei der JSON-Extraktion: {str(e)}")

        # Rückgabe None, wenn alle Versuche fehlschlagen
        return None

    def generate_json_from_image(self, base64_img: str, photo_type: str) -> Dict:
        """
        Sendet ein Bild an die API, um daraus strukturiertes JSON zu generieren.
        Enthält Fehlerbehandlung und Wiederholungslogik
        """
        # Vorbereitung und Validierung des Bildes
        prepared_image = self._prepare_image_data(base64_img)
        if not prepared_image:
            return {
                "error": "Ungültiges Bildformat",
                "details": "Base64-Bildvalidierung fehlgeschlagen"
            }

        # Prompt generieren, z. B. „Extrahiere alle Tabelleneinträge als JSON...“
        prompt = f"{get_prompt(photo_type)}\nGeben Sie nur JSON gemäß der festgelegten Struktur zurück, ohne zusätzlichen Text."

        # Mehrfachversuch bei API-Fehlern (z. B. Netzprobleme, Timeouts)
        for attempt in range(self.max_retries):
            try:
                # API-Aufruf mit Bild + Prompt im Multi-Modal-Modus
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
                    temperature=0.01,  # Niedrig, um deterministische Ausgabe zu fördern
                    response_format={"type": "json_object"},  # Erzwingt JSON-Format
                    timeout=self.timeout
                )

                # Validierung der Antwort: ist überhaupt ein Ergebnis vorhanden?
                if response and response.choices and len(response.choices) > 0 and response.choices[0].message and \
                        response.choices[0].message.content:
                    raw_output = response.choices[0].message.content.strip()
                    logger.debug(f"Roh-API-Antwort: {raw_output[:200]}...")  # Nur die ersten Zeichen loggen

                    # Versuch, die Antwort in ein echtes JSON zu parsen
                    parsed = self._extract_json(raw_output)
                    if parsed:
                        return parsed  # Erfolgreich extrahiertes JSON zurückgeben

                    # Falls keine gültige JSON-Struktur extrahiert werden konnte
                    return {
                        "error": "Ungültige JSON-Antwort",
                        "raw_output": raw_output,
                        "suggestion": "Das Modell hat ein fehlerhaftes JSON zurückgegeben. Bitte versuchen Sie es erneut."
                    }
                else:
                    # Kein Inhalt in der Antwort
                    logger.error(f"Leere oder fehlerhafte Antwort von OpenRouter (Versuch {attempt + 1}): {response}")
                    if attempt == self.max_retries - 1:
                        return {
                            "error": "Leere Antwort von der API nach mehreren Versuchen",
                            "raw_response": str(response)
                        }

            except json.JSONDecodeError as je:
                # Fehler beim Parsen des JSON-Antworttexts
                logger.error(f"JSON-Decodierungsfehler (Versuch {attempt + 1}): {str(je)}")
                if attempt == self.max_retries - 1:
                    return {
                        "error": "JSON-Parsing fehlgeschlagen nach mehreren Versuchen",
                        "exception": str(je),
                        "raw_output": raw_output if 'raw_output' in locals() else None
                    }

            except Exception as e:
                # Allgemeiner API-Fehler, z. B. Timeout, Netzwerkproblem etc.
                logger.error(f"API-Aufruf fehlgeschlagen (Versuch {attempt + 1}): {str(e)}")
                if attempt == self.max_retries - 1:
                    return {
                        "error": "API-Anfrage fehlgeschlagen",
                        "exception": str(e)
                    }

        # Alle Wiederholungsversuche sind fehlgeschlagen
        return {"error": "Maximale Versuche überschritten"}

    def compare_jsons(
            self,
            reference: Union[Dict, List],
            candidate: Union[Dict, List],
            ignore_fields: List[str] = None,
            word_match_fields: List[str] = None
    ) -> List[Dict]:
        """
        Erweiterter JSON-Vergleich mit Unterstützung für:
        - Verschachtelte Strukturen
        - Wort-basierter Vergleich für spezifische Felder
        - Ignorieren von definierten Feldern
        Gibt eine Liste von Unterschieden mit Pfadangabe zurück.
        """
        # Initialisiere Ignorier-Listen falls nicht angegeben
        if ignore_fields is None:
            ignore_fields = []
        if word_match_fields is None:
            word_match_fields = []

        diffs = []  # Liste zur Sammlung aller Unterschiede

        def compare_words(ref_str: str, cand_str: str) -> bool:
            """
            Hilfsfunktion für wortweisen String-Vergleich
            Vergleicht ob alle Wörter des Referenz-Strings im Kandidaten-String enthalten sind
            (case-insensitive)
            """
            if not isinstance(ref_str, str) or not isinstance(cand_str, str):
                return False

            # Normalisiere Strings: Kleinbuchstaben, entferne überflüssige Leerzeichen
            ref_words = set(ref_str.lower().split())
            cand_words = set(cand_str.lower().split())

            # Prüfe ob alle Referenzwörter im Kandidaten vorhanden sind
            return ref_words.issubset(cand_words)

        def compare_items(ref, cand, path=""):
            """
            Rekursive Vergleichsfunktion für Dictionary- und Listenelemente.
            Args:
                ref: Referenzwert (aus dem Original-JSON)
                cand: Kandidatenwert (aus dem zu prüfenden JSON)
                path: Aktueller Pfad für Fehlermeldungen (wird rekursiv aufgebaut)
            """
            # Fall 1: Beide Werte sind Dictionaries
            if isinstance(ref, dict) and isinstance(cand, dict):
                # 1. Schritt: Gemeinsame Keys vergleichen
                gemeinsame_keys = set(ref.keys()).intersection(cand.keys())
                for key in gemeinsame_keys:
                    if key in ignore_fields:
                        continue  # Ignoriere Felder aus der Ignore-Liste

                    new_path = f"{path}.{key}" if path else key

                    # Wort-basierten Vergleich für spezielle Felder
                    if key in word_match_fields:
                        if isinstance(ref[key], str) and isinstance(cand[key], str):
                            if not compare_words(ref[key], cand[key]):
                                diffs.append({
                                    "path": new_path,
                                    "reference": ref[key],
                                    "candidate": cand[key],
                                    "type": "word_mismatch"
                                })
                        else:
                            # Fallback für nicht-String Felder
                            compare_items(ref[key], cand[key], new_path)
                    else:
                        # Normaler Vergleich für andere Felder
                        compare_items(ref[key], cand[key], new_path)

                # 2. Schritt: Fehlende Keys in einem der Dictionaries prüfen
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

            # Fall 2: Beide Werte sind Listen
            elif isinstance(ref, list) and isinstance(cand, list):
                # Elementweise Vergleich für gemeinsame Indizes
                for i, (r, c) in enumerate(zip(ref, cand)):
                    compare_items(r, c, f"{path}[{i}]")

                # Behandlung unterschiedlicher Listenlängen
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

            # Fall 3: Einfache Werte (keine weiteren Unterstrukturen)
            elif ref != cand:
                diffs.append({
                    "path": path,
                    "reference": ref,
                    "candidate": cand,
                    "type": "value_mismatch"
                })

        # Starte den Vergleich auf oberster Ebene
        compare_items(reference, candidate)

        # Rückgabe aller gefundenen Unterschiede
        return diffs
# Singleton-Instanz: Diese einzige Instanz der Klasse wird im gesamten Projekt wiederverwendet,
# um API-Aufrufe zentral zu verwalten und mehrfaches Initialisieren zu vermeiden.
client = OpenRouterClient()