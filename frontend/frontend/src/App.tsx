import { useState } from 'react';
import ImageUploader from './components/ImageUploader';
import PhotoTypeSelector from './components/PhotoTypeSelector';
import ResultViewer from './components/ResultViewer';
import { compareImages } from './utils/api';

function App() {
  const [image1, setImage1] = useState(null);
  const [image2, setImage2] = useState(null);
  const [photoType, setPhotoType] = useState('');
  const [result, setResult] = useState(null);

  const handleCompare = async () => {
    if (!image1 || !image2 || !photoType) {
      alert('Fotos selektieren!');
      return;
    }
    console.log("Photo type gửi lên:", photoType);
    const res = await compareImages(image1, image2, photoType);
    setResult(res);
  };

  return (
    <div style={{ maxWidth: 800, margin: '0 auto', padding: 20 }}>
      <h1>SAP BW Image Comparator</h1>
      <ImageUploader label="Abgabe" onChange={setImage1} />
      <ImageUploader label="Musterlösung" onChange={setImage2} />
      <PhotoTypeSelector onSelect={setPhotoType} />
      <button onClick={handleCompare}>Vergleich</button>
      {result && <ResultViewer result={result} />}
    </div>
  );
}

export default App;
