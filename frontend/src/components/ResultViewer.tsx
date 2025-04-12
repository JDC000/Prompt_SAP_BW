import React from 'react';

type Props = {
  result: any;
};

function ResultViewer({ result }: Props) {
  return (
    <div style={{ marginTop: '1rem' }}>
      <h3>Ergebnis:</h3>
      <pre style={{ background: '#eee', padding: '1rem' }}>
        {JSON.stringify(result, null, 2)}
      </pre>
    </div>
  );
}

export default ResultViewer;
