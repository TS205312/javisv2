import sys
sys.path.insert(0, 'src')

# Force UTF-8 encoding for stdout/stderr
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from src.pipeline.commands import build_default_registry
registry = build_default_registry()

# Check the actual patterns
for cmd in registry._commands:
    print(f'{cmd.name}: pattern={cmd.pattern.pattern}')

# Test with exact strings from file
test_strings = [
    'hỏi tài liệu test',
    'hỏi tài liệu',
    'chụp ảnh test',
    'chụp ảnh',
    'quên',
    'quằn',
]
for text in test_strings:
    result = registry.match(text)
    print(f'  Input: "{text}" -> handled={result.handled}, command={result.command}, action={result.action}')