import pyautogui
import time
import mss

def click(params):
    x = params.get("x")
    y = params.get("y")
    pyautogui.click(x, y)
    return f"Đã click tại ({x}, {y})"

def type_text(params):
    text = params.get("text")
    time.sleep(1)
    pyautogui.write(text, interval=0.05)
    return f"Đã gõ: {text}"

def screenshot(params):
    path = params.get("path", "screenshot.png")
    with mss.mss() as sct:
        sct.shot(output=path)
    return f"Đã lưu ảnh màn hình vào {path}"
