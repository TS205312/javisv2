import subprocess
from core.safety import is_safe

def run(params, yolo=False):
    if not isinstance(params, dict):
        return "Thiếu hoặc không hợp lệ param cmd"
    cmd = params.get("cmd") or params.get("command", "")
    if not cmd: return "Thiếu param cmd"
    
    safe, msg = is_safe(cmd, yolo)
    if not safe: return f"LỖI SAFETY: {msg}"
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
        return f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    except Exception as e:
        return f"Exception: {str(e)}"
