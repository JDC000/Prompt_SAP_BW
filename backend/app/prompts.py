PROMPTS = {
  "ABAP Dictionary": "Extract only the core technical structure of a SAP Dictionary Display Table screen from the image. Return the result as JSON in the following format: { \"Fields\": [ { \"FieldName\": \"name\", \"Key\": true/false, \"Initial\": true/false, \"DataElement\": \"data_element_name\", \"DataType\": \"CHAR/DATS/...\", \"Length\": number, \"DecimalPlaces\": number }, ... ] } Only include technical metadata fields: FieldName, Key, Initial, DataElement, DataType, Length, DecimalPlaces. Do not include any fields that contain personal identifiers (Frields contain /BIC/) or user-specific codes. Do not include any language-dependent elements such as descriptions or labels (e.g., Short Description, Kurzbeschreibung, gültig ab). Output must be anonymized and language-independent, reusable across SAP systems.",
  "Data Source": "Compare two SAP BW data loading process flows. Note changes in data sources, targets, or steps.",
  "Datenvorschau": "Analyze SAP BW process chains. List steps and highlight missing or changed tasks.",
  "DTP": "Compare two SAP BW data loading process flows. Note changes in data sources, targets, or steps.",
  "Bewegungsdaten": "From the provided image of a structured text data file, extract the following metadata in JSON format:\n\n{\n  \"ColumnDelimiter\": \"Detected character used to separate columns (e.g., ';' or '#')\",\n  \"FirstRow\": [\"First\", \"row\", \"values\", \"...\"],\n  \"ColumnCount\": number,\n  \"Columns\": [\"Header1\", \"Header2\", \"...\"] or null if not available\n}\n\nInstructions:\n- Identify the delimiter character used to separate columns.\n- Extract the first complete data row, split it using the delimiter, and output as an array under \"FirstRow\".\n- Set \"ColumnCount\" equal to the length of this array.\n- If a header row exists, extract column names; otherwise, set \"Columns\": null.\n- Only output the above metadata, no additional data rows are required.",
  "Composite Provider": "Compare two SAP BW data loading process flows. Note changes in data sources, targets, or steps.",
  "Data Flow Object": "Compare two SAP BW data loading process flows. Note changes in data sources, targets, or steps.",
  "Excel": "Compare two SAP BW data loading process flows. Note changes in data sources, targets, or steps.",
  "Query": "Compare two SAP BW data loading process flows. Note changes in data sources, targets, or steps.",
  "Transformationen": "Compare two SAP BW data loading process flows. Note changes in data sources, targets, or steps."

}

def get_prompt(photo_type: str) -> str:
  print("Vom Benutzer ausgewählter Bildtyp:", photo_type)
  return PROMPTS.get(photo_type, "Bildbeschreibung und Rückgabe von JSON.")

