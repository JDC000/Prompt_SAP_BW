import React from 'react';

const ResultViewer = ({ result }) => {
  return (
    <div className="mt-6 p-4 border rounded bg-gray-100">
      <h2 className="text-lg font-semibold mb-2">Result:</h2>
      <pre className="whitespace-pre-wrap text-sm text-gray-800">
        {JSON.stringify(result, null, 2)}
      </pre>
    </div>
  );
};

export default ResultViewer;