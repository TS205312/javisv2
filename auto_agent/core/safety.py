import re

BLACKLIST = [
    r"rm -rf", r"format\s+", r"dd\s+", r"shutdown", r"sudo rm", r"drop\s+db", r"git push -f"
]

def is_safe(cmd, yolo=False):
    if yolo:
        return True
    cmd_lower = cmd.lower()
    for pattern in BLACKLIST:
        if re.search(pattern, cmd_lower):
            return False, f"Lệnh bị chặn bởi safety (chứa '{pattern}')"
    return True, "Safe"
