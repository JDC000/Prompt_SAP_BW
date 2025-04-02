import json
import os

PROMPT_FILE_PATH = os.path.join(os.path.dirname(__file__), "../../prompts/prompts.json")

def load_prompts():
    with open(PROMPT_FILE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def get_prompt(photo_type: str) -> str:
    prompts = load_prompts()
    return prompts.get(photo_type, "Please compare the two SAP BW diagrams and describe the differences.")
