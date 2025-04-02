import React, { useState } from 'react';
import ImageUploader from './components/ImageUploader';
import PhotoTypeSelector from './components/PhotoTypeSelector';
import ResultViewer from './components/ResultViewer';
import { compareImages } from './utils/api';

function App() {
  const [image1, setImage1] = useState(null);
  const [image2, setImage2] = useState(null);
  const [photoType, setPhotoType] = useState('transformation');
  const [result, setResult] = useState(null);

  const handleSubmit = async () => {
    if (!image1 || !image2 || !photoType) return alert("Please select all inputs.");
    const payload = {
      photo_type: photoType,
      image1_base64: image1,
      image2_base64: image2
    };
    const res = await compareImages(payload);
    setResult(res);
  };

  return (
    <div className="p-8 max-w-3xl mx-auto space-y-6">
      <h1 className="text-2xl font-bold">SAP BW Image Comparator</h1>
      <ImageUploader setImage1={setImage1} setImage2={setImage2} />
      <PhotoTypeSelector setPhotoType={setPhotoType} />
      <button
        onClick={handleSubmit}
        className="bg-blue-600 text-white px-6 py-2 rounded hover:bg-blue-700"
      >
        Compare
      </button>
      {result && <ResultViewer result={result} />}
    </div>
  );
}

export default App;
