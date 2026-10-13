import sys
sys.path.insert(0, 'src')

# Force UTF-8 encoding for stdout/stderr
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from src.cli.main import main

# Test CLI chat clear
sys.argv = ['jarvis-local', '--no-rich', 'chat', 'clear']
result = main()
print('Exit code:', result)