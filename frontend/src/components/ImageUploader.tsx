import React, { useState } from 'react';
import ImageModal from "./ImageModal";

interface ImageUploaderProps {
  label: string;
  onChange: (file: File) => void;
}

const ImageUploader: React.FC<ImageUploaderProps> = ({ label, onChange }) => {
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [showModal, setShowModal] = useState(false);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      onChange(file);
      // Create preview URL
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreviewUrl(reader.result as string);
      };
      reader.readAsDataURL(file);
    }
  };

  return (
    <div style={{ marginBottom: '1rem' }}>
      <label>{label}</label>
      <input type="file" accept="image/*" onChange={handleChange} />

      {previewUrl && (
        <div style={{ marginTop: '10px' }}>
          <img
            src={previewUrl}
            alt="Preview"
            style={{ maxWidth: '200px', maxHeight: '200px', cursor: 'pointer' }}
            onClick={() => setShowModal(true)}
          />
        </div>
      )}

      {showModal && previewUrl && (
        <ImageModal imageUrl={previewUrl} onClose={() => setShowModal(false)} />
      )}
    </div>
  );
};

export default ImageUploader;