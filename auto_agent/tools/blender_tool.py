import subprocess
from core.config import BLENDER_PATH

def run_script(params):
    script_path = params.get("script_path", "temp_blender.py")
    script_content = params.get("script_content")
    
    if script_content:
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(script_content)
            
    cmd = f'"{BLENDER_PATH}" --background --python {script_path}'
    try:
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return f"STDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
    except Exception as e:
        return str(e)
