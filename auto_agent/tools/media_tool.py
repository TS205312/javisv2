import subprocess
from core.config import FFMPEG_PATH

def ffmpeg_run(params):
    args = params.get("args")
    cmd = f'"{FFMPEG_PATH}" {args}'
    try:
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return f"STDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
    except Exception as e:
        return str(e)
