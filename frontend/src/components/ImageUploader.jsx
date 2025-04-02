import React from 'react';

const ImageUploader = ({ setImage1, setImage2 }) => {
  const handleUpload = (e, setter) => {
    const file = e.target.files[0];
    const reader = new FileReader();
    reader.onloadend = () => setter(reader.result.split(',')[1]);
    if (file) reader.readAsDataURL(file);
  };

  return (
    <div className="flex space-x-6">
      <div>
        <label>Image 1:</label>
        <input type="file" accept="image/*" onChange={(e) => handleUpload(e, setImage1)} />
      </div>
      <div>
        <label>Image 2:</label>
        <input type="file" accept="image/*" onChange={(e) => handleUpload(e, setImage2)} />
      </div>
    </div>
  );
};

export default ImageUploader;
