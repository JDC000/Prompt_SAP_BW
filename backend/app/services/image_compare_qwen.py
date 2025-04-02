from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from app.services.prompt_loader import get_prompt
from app.services.utils import decode_base64_image
from difflib import SequenceMatcher
# Load Qwen-VL model
qwen_pipe = pipeline(Tasks.multi_modal_conversation, model='qwen/Qwen-VL-Chat-Int4', device='cpu')

def compare_texts(text1: str, text2: str) -> float:
    return round(SequenceMatcher(None, text1, text2).ratio(), 2)

def compare_images(photo_type: str, img1_base64: str, img2_base64: str):
    prompt = get_prompt(photo_type)

    image1 = decode_base64_image(img1_base64)
    image2 = decode_base64_image(img2_base64)

    output1 = qwen_pipe({'image': image1, 'text': prompt})
    output2 = qwen_pipe({'image': image2, 'text': prompt})

    answer1 = output1["text"]
    answer2 = output2["text"]

    similarity = compare_texts(answer1, answer2)

    return {
        "prompt_used": prompt,
        "similarity_score": similarity,
        "differences": {
            "image_1_analysis": answer1,
            "image_2_analysis": answer2
        }
    }

