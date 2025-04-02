import axios from 'axios';

export const compareImages = async (data) => {
  const response = await axios.post('http://localhost:8000/compare', data);
  return response.data;
};