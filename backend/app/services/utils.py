import base64
from io import BytesIO
from PIL import Image

def decode_base64_image(image_base64: str) -> Image.Image:
    image_data = base64.b64decode(image_base64)
    image = Image.open(BytesIO(image_data)).convert("RGB")
    return image
