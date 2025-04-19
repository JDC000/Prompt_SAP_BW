PROMPTS = {
  "ABAP Dictionary": "Extrahieren Sie aus der folgenden ABAP-Datentabelle nur die technisch relevanten Felder (ohne benutzerspezifische Codes) und geben Sie das Ergebnis im JSON-Format zurück. Anforderungen: Ignorieren Sie alle Felder mit: Präfixen wie /BIC/ oder /BI0/ (z. B. /BIC/TE9_056FB).  Extrahieren Sie für jedes Feld: Feldname (field_name), Primärschlüssel-Kennzeichen (is_key: true/false, basierend auf '✔'), Data-Element (data_element), Datentyp (data_type), Länge (length), Dezimalstellen (decimals). Format der Ausgabe: JSON-Array mit allen extrahierten Feldern.",
  "Data Source": "Compare two SAP BW data loading process flows. Note changes in data sources, targets, or steps.",
  "Datenvorschau": "Analyze SAP BW process chains. List steps and highlight missing or changed tasks.",
  "DTP": "Compare two SAP BW data loading process flows. Note changes in data sources, targets, or steps.",
  "Bewegungsdaten": "Compare two SAP BW data loading process flows. Note changes in data sources, targets, or steps.",
  "Composite Provider": "Compare two SAP BW data loading process flows. Note changes in data sources, targets, or steps.",
  "Data Flow Object": "Compare two SAP BW data loading process flows. Note changes in data sources, targets, or steps.",
  "Excel": "Compare two SAP BW data loading process flows. Note changes in data sources, targets, or steps.",
  "Query": "Compare two SAP BW data loading process flows. Note changes in data sources, targets, or steps.",
  "Transformationen": "Compare two SAP BW data loading process flows. Note changes in data sources, targets, or steps."

}

def get_prompt(photo_type: str) -> str:
  print("Vom Benutzer ausgewählter Bildtyp:", photo_type)
  return PROMPTS.get(photo_type, "Bildbeschreibung und Rückgabe von JSON.")

