type Props = {
  onSelect: (value: string) => void;
};

function PhotoTypeSelector({ onSelect }: Props) {
  return (
    <div style={{ marginBottom: '1rem' }}>
      <label>Phototyp auswählen:</label>
      <select onChange={(e) => onSelect(e.target.value)}>
        <option value="">-- auswählen --</option>
        <option value="ABAP Dictionary">ABAP Dictionary</option>
        <option value="BW4Cockpit (Stammdaten)">BW4Cockpit (Stammdaten)</option>
        <option value="Datenvorschau">Datenvorschau</option>
        <option value="DTP">DTP</option>
        <option value="Bewegungsdaten">Bewegungsdaten</option>
        <option value="Composite Provider">Composite Provider</option>
        <option value="Data Flow Object">Data Flow Object</option>
        <option value="Excel">Excel</option>
        <option value="Query">Query</option>
        <option value="Transformationen">Transformationen</option>
      </select>
    </div>
  );
}

export default PhotoTypeSelector;
