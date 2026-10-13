# JAVIS Runtime Read-Only/No-Edits Test Results
# Date: 2026-09-13
# Project: C:\javis

## 1. py_compile All Source Files
**Result: PASS** - All 36 Python files compiled successfully with no syntax errors.

Files tested:
- src/core/*.py (6 files)
- src/llm/*.py (4 files)
- src/rag/*.py (4 files)
- src/tts/*.py (5 files)
- src/stt/*.py (3 files)
- src/vision/*.py (4 files)
- src/pipeline/*.py (4 files)
- src/cli/*.py (3 files)
- src/__init__.py

---

## 2. Import All Packages
**Result: PASS** - All packages import successfully.

| Package | Status |
|---------|--------|
| src.core | OK |
| src.llm | OK |
| src.rag | OK |
| src.tts | OK |
| src.stt | OK |
| src.vision | OK |
| src.pipeline | OK |
| src.cli | OK |
| src (root) | OK |

---

## 3. Load Config & PathManager
**Result: PASS**

- **Config**: Loaded with defaults (ollama_host=http://localhost:11434, llm_model=qwen2.5:0.5b)
- **PathManager**: Created with D: drive paths (base_dir=D:\AI_System, brain_dir=D:\AI_Brain)
- No C: drive paths used - all paths correctly point to D:

---

## 4. AssistantOrchestrator Instantiation
**Result: PASS** - Instantiates successfully with stub/fallback engines when dependencies unavailable.

### Engine Status (with Ollama running, no Vosk model, no FAISS, no camera):
| Engine | Health | Notes |
|--------|--------|-------|
| STT | False | Vosk model missing → stub |
| LLM | True | Ollama reachable at localhost:11434 |
| TTS | True | edge-tts/pyttsx3 available |
| RAG | False | FAISS not installed → fallback |
| Vision (OpenCV) | False | Camera unavailable |
| Vision (Moondream) | True | Ollama reachable |
| Event Bus | True | Created successfully |

---

## 5. AssistantOrchestrator Chat & Command Tests

| Test | Input | Result | Status |
|------|-------|--------|--------|
| Normal Chat | "Xin chao" | Falls back to deterministic response | **EXPECTED FALLBACK** (Ollama model not loaded) |
| Help Command | "trợ giúp" | Returns help text | **PASS** |
| Status Command | "trạng thái" | Returns formatted status | **PASS** |
| Clear Command | "quên" | Falls to LLM instead of matching | **FAIL** - Trigger typo: "quằn" vs "quên" |
| Voice Turn | duration=1.0 | "STT khong kha dung" | **EXPECTED FALLBACK** (no Vosk model) |
| Vision | prompt, no image | "may anh khong kha dung" | **EXPECTED FALLBACK** (no camera) |
| Ingest | [file] | "FAISS not available" | **EXPECTED FALLBACK** (missing dependency) |
| Close (1st) | - | Success | **PASS** |
| Close (2nd) | - | Idempotent, no error | **PASS** |
| Session Path | - | D:\AI_System\session.json | **PASS** (not on C:) |
| Chat After Close | "test" | "he thong da dong" | **PASS** |

### Known Issues in Command Registry (src/pipeline/commands.py):
1. **"quên" trigger typo**: Uses "quằn" (U+1EB1) instead of "quên" (U+EA)
2. **"hỏi tài liệu" trigger typo**: Uses "hệi tài liệu" (U+1EC7) instead of "hỏi" (U+1ECF)
3. **"chụp ảnh" trigger typo**: Uses "chấp ảnh" (U+1EA5) instead of "chụp" (U+1EE3)

These cause Vietnamese commands to not match, falling through to LLM.

---

## 6. CLI Commands (--no-rich)

| Command | Result | Status |
|---------|--------|--------|
| `--help` | Shows usage correctly | **PASS** |
| `config` | Shows resolved config table | **PASS** |
| `status` | Returns JSON status | **PASS** |
| `doctor` | Runs health checks table | **PASS** |
| `chat help` | Returns help text | **PASS** |
| `chat status` | Returns formatted status | **PASS** |
| `chat clear` | Returns "Đã xóa lịch sử..." | **PASS** |

### Doctor Check Results:
| Check | Status | Detail |
|-------|--------|--------|
| Base directory | OK | D:\AI_System |
| Brain directory | OK | D:\AI_Brain |
| Models directory | OK | D:\AI_Brain\models |
| Cache directory | WARN | Missing: D:\AI_System\cache |
| Logs directory | WARN | Missing: D:\AI_System\logs |
| FAISS directory | WARN | Missing: D:\AI_Brain\faiss |
| Audio directory | OK | D:\AI_System\audio |
| Vision directory | WARN | Missing: D:\AI_System\vision |
| Hugging Face cache | OK | D:\AI_Brain\huggingface |
| Ollama health | OK | /api/version returned 200 |
| STT model | FAIL | Missing model parts: am, conf, graph |
| TTS health | OK | At least one provider available |
| RAG health | WARN | Dependencies or index unavailable |
| Camera health | WARN | Camera 0 unavailable |

---

## 7. Session Path Data Isolation
**Result: PASS** - Session file path is `D:\AI_System\session.json`, **NOT on C: drive**.

---

## 8. Expected Fallbacks vs Failures Summary

### EXPECTED FALLBACKS (by design - missing optional dependencies/services):
- LLM chat → deterministic fallback (Ollama server not serving model)
- STT/Voice → "STT khong kha dung" (no Vosk model installed)
- Vision → "may anh khong kha dung" (no camera available)
- RAG/Ingest → "FAISS not available" (faiss-cpu not installed)
- Missing directories → WARN in doctor (dirs created on first use)

### ACTUAL FAILURES (bugs to fix):
1. **Command registry typos** (3 Vietnamese triggers incorrect):
   - "quằn" → should be "quên"
   - "hệi tài liệu" → should be "hỏi tài liệu"
   - "chấp ảnh" → should be "chụp ảnh"
2. **CLI chat encoding**: Vietnamese output fails in PowerShell cp1252 (works in UTF-8 terminals)

---

## 9. Dependencies Not Installed (as requested)
- faiss-cpu → RAG unavailable
- vosk → STT unavailable
- sentence-transformers → embeddings unavailable
- opencv-python-headless → camera unavailable (installed but no camera)
- ollama server → running but model not pulled

---

## Conclusion
**Core functionality works correctly with proper fallbacks.** The system gracefully degrades when optional dependencies/services are missing. Main issues are Vietnamese diacritic typos in command triggers and PowerShell encoding limitations for Vietnamese output.