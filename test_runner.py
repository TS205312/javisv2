import sys
sys.path.insert(0, 'src')

# Force UTF-8 encoding for stdout
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Write output to file instead of console
output_lines = []

def log(msg):
    output_lines.append(msg)

log('=== AssistantOrchestrator Chat & Command Tests (with proper Vietnamese) ===')

try:
    from src.core.config import Config
    from src.core.paths import PathManager
    from src.pipeline.orchestrator import AssistantOrchestrator
    
    cfg = Config()
    pm = PathManager()
    
    orch = AssistantOrchestrator(config=cfg, paths=pm)
    
    log('Engines health status:')
    for k, v in orch._engines.items():
        if k == 'vision':
            log('  ' + k + ': ' + str({k2: getattr(v2, 'health', 'N/A') for k2, v2 in v.items()}))
        else:
            h = getattr(v, 'health', None)
            if callable(h):
                log('  ' + k + ': ' + str(h()))
            else:
                log('  ' + k + ': ' + str(h))
    
    # Test 1: Normal chat (falls back to LLM fallback)
    log('')
    log('Test 1: Normal chat')
    result = orch.chat('Xin chao')
    log('Response: ' + result)
    
    # Test 2: Help command (with diacritics)
    log('')
    log('Test 2: Help command')
    result = orch.chat('trợ giúp')
    log('Response: ' + (result[:200] + '...' if len(result) > 200 else result))
    
    # Test 3: Status command (with diacritics)
    log('')
    log('Test 3: Status command')
    result = orch.chat('trạng thái')
    log('Response: ' + result)
    
    # Test 4: Clear command (with diacritics)
    log('')
    log('Test 4: Clear command')
    result = orch.chat('quên')
    log('Response: ' + result)
    
    # Test 5: Voice stub behavior
    log('')
    log('Test 5: Voice turn (stub STT)')
    result = orch.voice_turn(duration=1.0, speak=False)
    log('Response: ' + result)
    
    # Test 6: Vision stub behavior (no camera available)
    log('')
    log('Test 6: Vision (no image, stub)')
    result = orch.vision('test prompt', image=None, camera_index=0, speak=False)
    log('Response: ' + result)
    
    # Test 7: Ingest stub
    log('')
    log('Test 7: Ingest (stub RAG)')
    result = orch.ingest([__file__])
    log('Response: ' + result)
    
    # Test 8: Close twice
    log('')
    log('Test 8: Close twice')
    orch.close()
    log('First close: OK')
    orch.close()
    log('Second close: OK')
    
    # Test 9: Session path check - ensure no C: data created
    log('')
    log('Test 9: Session path check')
    log('Session path: ' + str(orch.session.session_path))
    log('Path is on C: ' + str(str(orch.session.session_path).startswith('C:')))
    
    # Test 10: Test chat after close
    log('')
    log('Test 10: Chat after close')
    result = orch.chat('test after close')
    log('Response: ' + result)
    
except Exception as e:
    import traceback
    log('[FAIL] ' + str(e))
    log(traceback.format_exc())

log('')
log('AssistantOrchestrator chat & command tests complete!')

# Write output to file
with open('test_output.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(output_lines))

print('Done. Output written to test_output.txt')