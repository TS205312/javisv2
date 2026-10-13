import pytesseract
from PIL import Image

def ocr(params):
    path = params.get("path")
    try:
        img = Image.open(path)
        text = pytesseract.image_to_string(img)
        return text
    except Exception as e:
        return str(e)

def analyze(params):
    # Dùng LLM Vision ở đây nếu cần, hiện tại mock
    return "Phân tích ảnh (mock): Bức ảnh chứa thông tin..."
