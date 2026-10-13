import sys
sys.path.insert(0, 'src')

# Force UTF-8 encoding for stdout/stderr
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from src.pipeline.commands import build_default_registry

registry = build_default_registry()

# Test all command triggers
test_inputs = [
    'thoát', 'exit', 'quit',
    'trợ giúp', 'help',
    'trạng thái', 'status',
    'nạp test', 'ingest test',
    'hỏi tài liệu test', 'rag test',
    'chụp ảnh test', 'see test',
    'tts online', 'tts offline',
    'quên', 'clear',
]

print('Testing command registry matching:')
for text in test_inputs:
    result = registry.match(text)
    print(f'  Input: "{text}" -> handled={result.handled}, command={result.command}, action={result.action}')