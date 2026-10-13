import sys
sys.path.insert(0, 'src')

# Force UTF-8 encoding for stdout/stderr
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Check the actual triggers in commands.py
with open('src/pipeline/commands.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find the clear command triggers
import re
match = re.search(r'triggers=\[(.*?)\]', content[content.find('Quên'):content.find('Quên')+200])
if match:
    print('Clear triggers in file:', match.group(1))

# Check actual bytes of "quên" in different forms
test_strings = ['quên', 'qu\u1eb1n', 'qu\u00ean', 'qu\u1eb1\u0300n']
for s in test_strings:
    print(f'  "{s}" -> bytes: {s.encode("utf-8").hex()}, codepoints: {[hex(ord(c)) for c in s]}')

# Test with exact string from file
from src.pipeline.commands import build_default_registry
registry = build_default_registry()

# Get the actual command def for clear
for cmd in registry._commands:
    if cmd.name == 'Quên':
        print(f'Clear command pattern: {cmd.pattern.pattern}')
        print(f'Clear command triggers: {cmd.pattern.pattern}')
        # Test matching
        test = 'quên'
        m = cmd.pattern.fullmatch(test)
        print(f'  Match "{test}": {m}')
        if m:
            print(f'  Groups: {m.groups()}')