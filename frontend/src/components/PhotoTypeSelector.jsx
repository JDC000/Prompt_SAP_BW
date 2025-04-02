import React from 'react';

const PhotoTypeSelector = ({ setPhotoType }) => {
  return (
    <div>
      <label className="block mb-1">Select Photo Type:</label>
      <select
        onChange={(e) => setPhotoType(e.target.value)}
        className="border p-2 rounded w-full"
      >
        <option value="transformation">Transformation</option>
        <option value="loading">Loading</option>
        <option value="process_chain">Process Chain</option>
      </select>
    </div>
  );
};
export default PhotoTypeSelector;