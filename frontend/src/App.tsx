import { useState } from 'react';
import ImageUploader from './components/ImageUploader';
import PhotoTypeSelector from './components/PhotoTypeSelector';
import ResultViewer from './components/ResultViewer';
import { compareImages } from './utils/api';
import './AppStyle.css';
import './index.css';

function App() {
  const [image1, setImage1] = useState<File | null>(null);
  const [image2, setImage2] = useState<File | null>(null);
  const [photoType, setPhotoType] = useState<string>('');
  const [result, setResult] = useState<any>(null);
  const handleCompare = async () => {
    if (!image1 || !image2 || !photoType) {
      alert('Fotos selektieren!');
      return;
    }
    console.log('Photo type gửi lên:', photoType);
    const res = await compareImages(image1, image2, photoType);
    setResult(res);
  };
  return (
    <div style={{ maxWidth: 800, margin: '0 auto', padding: 20 }}>
      <h1>SAP BW Image Comparator</h1>
      <ImageUploader
          label="Abgabe"
          onChange={(file) => setImage1(file)}/>
      <ImageUploader
          label="Abgabe"
          onChange={(file) => setImage2(file)}
/>
      <PhotoTypeSelector onSelect={setPhotoType} />
      <button onClick={handleCompare}>Vergleich</button>
      {result && <ResultViewer result={result} />}
    </div>
  );
}

export default App;