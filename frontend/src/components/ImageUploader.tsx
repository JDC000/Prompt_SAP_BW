import React from 'react';

type Props = {
  label: string;
  onChange: (file: File) => void;
};

function ImageUploader({ label, onChange }: Props) {
  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      onChange(file);
    }
  };

  return (
    <div style={{ marginBottom: '1rem' }}>
      <label>{label}</label>
      <input type="file" accept="image/*" onChange={handleChange} />
    </div>
  );
}

export default ImageUploader;

