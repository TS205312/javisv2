from tools.write_safety import (
    UnsafeWritePath,
    backup_existing_file,
    prepare_write_path,
)


def read(params):
    try:
        with open(params.get("path"), "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return str(e)

def write(params):
    try:
        path = prepare_write_path(params.get("path"))
        path.parent.mkdir(parents=True, exist_ok=True)
        backup_existing_file(path)
        with open(path, "w", encoding="utf-8") as f:
            f.write(params.get("content", ""))
        return "Ghi file thành công"
    except UnsafeWritePath as e:
        return f"[SAFETY] {e}"
    except Exception as e:
        return str(e)
