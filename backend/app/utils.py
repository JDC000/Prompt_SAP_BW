
import base64

def base64_to_image_bytes(base64_str: str) -> bytes:
    if "," in base64_str:
        base64_str = base64_str.split(",")[1]
    return base64.b64decode(base64_str)

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

