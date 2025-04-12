import axios from 'axios';

export async function compareImages(
  image1: File,
  image2: File,
  photoType: string
): Promise<any> {
  const formData = new FormData();
  formData.append('image_1', image1);
  formData.append('image_2', image2);
  formData.append('photo_type', photoType);

  try {
    const response = await axios.post("http://localhost:8000/compare", formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  } catch (error) {
    return { error: 'Lỗi khi so sánh ảnh!' };
  }
}
