import React from 'react';

type Props = {
  result: {
    image_1_json?: any;
    image_2_json?: any;
    differences?: any[];
    error?: string;
  };
};

function ResultViewer({ result }: Props) {
  if (result?.error) {
    return (
      <div style={{ marginTop: '1rem' }}>
        <h3>Ergebnis:</h3>
        <pre style={{ background: '#ffe0e0', padding: '1rem', border: '1px solid red' }}>
          {JSON.stringify(result, null, 2)}
        </pre>
      </div>
    );
  }

  if (result?.image_1_json && result?.image_2_json && result?.differences) {
    return (
      <div style={{ marginTop: '1rem' }}>
        <h3>Ergebnis:</h3>
        <div style={{ display: 'flex', width: '100%', gap: '1rem' }}>
          <div style={{ flex: 1, background: '#f0f8ff', padding: '1rem', border: '1px solid #add8e6' }}>
            <h4>JSON 1:</h4>
            <pre style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
              {JSON.stringify(result.image_1_json, null, 2)}
            </pre>
          </div>
          <div style={{ flex: 1, background: '#f0f8ff', padding: '1rem', border: '1px solid #add8e6' }}>
            <h4>JSON 2:</h4>
            <pre style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
              {JSON.stringify(result.image_2_json, null, 2)}
            </pre>
          </div>
          <div style={{ flex: 1, background: '#f0f8ff', padding: '1rem', border: '1px solid #add8e6' }}>
            <h4>So sánh:</h4>
            <pre style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
              {JSON.stringify(result.differences, null, 2)}
            </pre>
          </div>
        </div>
      </div>
    );
  }

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
