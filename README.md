# Jarvis — Standalone Local AI Assistant

Wave 3/4 adds the runnable CLI, package entrypoint, Windows setup scripts, and
CPU-oriented dependency metadata. The project root is **`C:\javis`**. Heavy
runtime data stays on **`D:\AI_System`** and **`D:\AI_Brain`**.

## Quick start

Requirements:

- Python 3.10 or newer
- Ollama installed and running at `http://localhost:11434`
- A microphone for voice mode
- A camera for camera vision (optional for image-file vision)

From `C:\javis`, create the virtual environment, install dependencies, create
the D: directories, copy an existing legacy model tree when present, and
download the Vietnamese Vosk model when it is absent:

```powershell
Set-Location C:\javis
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup.ps1
```

The setup script prints the Ollama model commands but does not pull models
automatically:

```powershell
ollama pull qwen2.5:0.5b
ollama pull moondream:latest
```

Start the voice assistant:

```powershell
.\scripts\run_assistant.ps1
# equivalent:
.\.venv\Scripts\python.exe -m src.cli.main voice
```

The installed console entrypoint is also available after activating the
environment:

```powershell
.\.venv\Scripts\Activate.ps1
jarvis-local --help
```

## CLI commands

Run the module form from `C:\javis`:

```powershell
python -m src.cli.main --help
python -m src.cli.main chat "Xin chao Jarvis"
python -m src.cli.main chat "Hom nay the nao?" --speak
python -m src.cli.main voice --duration 5 --max-turns 10
python -m src.cli.main vision "Mo ta anh nay" --image "C:\path\image.jpg"
python -m src.cli.main vision "Ban thay gi?" --camera-index 0
python -m src.cli.main ingest "D:\AI_System\documents" "D:\notes.txt"
python -m src.cli.main status
python -m src.cli.main doctor
python -m src.cli.main config
python -m src.cli.main config llm_model --json
```

`voice` runs until Ctrl-C unless `--once` or `--max-turns` is supplied. Its
duration and inter-turn pause are configurable. If a voice turn returns no
text, the CLI sends a short text fallback through `chat`; ordinary runtime
errors are reported without a traceback. `--no-speak` disables response audio.

`doctor` checks managed paths, Ollama health, the Vosk model structure, TTS
availability, RAG/FAISS readiness, and camera 0. It performs health probes but
does not load speech, embedding, or vision models.

## Offline and model behavior

- Ollama chat and vision use the local Ollama service and the configured local
  model tags.
- Vosk speech recognition is offline after `vosk-model-small-vn-0.4` is
  present under `D:\AI_Brain\models`.
- RAG uses `faiss-cpu` and `all-MiniLM-L6-v2`. The embedding model is cached
  under `D:\AI_Brain\huggingface`; the first ingest needs model access unless
  the cache is already populated.
- TTS prefers Edge TTS when online and falls back to the local `pyttsx3`
  provider. Offline operation therefore depends on the local provider and
  installed Windows voice.
- `setup.ps1` only downloads the Vosk Vietnamese model when needed. Ollama
  models and the RAG embedding model are not downloaded by setup.

## Configuration

`src.core.config.Config` merges defaults, `config.yaml`, and `JARVIS_*`
environment variables in that order. Common overrides include:

```text
JARVIS_OLLAMA_HOST=http://localhost:11434
JARVIS_LLM_MODEL=qwen2.5:0.5b
JARVIS_STT_MODEL_PATH=D:\AI_Brain\models\vosk-model-small-vn-0.4
JARVIS_TTS_PREFER_ONLINE=true
JARVIS_RAG_MODEL=all-MiniLM-L6-v2
JARVIS_VISION_MODEL=moondream:latest
```

Use `--config path\to\config.yaml` before a subcommand to select another
configuration file. `--base-dir` and `--brain-dir` override the default storage
roots for diagnostics or tests.

## Project layout

```text
C:\javis\
├── config.yaml
├── requirements.txt
├── pyproject.toml
├── scripts\
│   ├── setup.ps1
│   ├── download_models.ps1
│   └── run_assistant.ps1
└── src\
    ├── cli\
    │   ├── __init__.py
    │   ├── console.py
    │   └── main.py
    ├── core\
    ├── llm\
    ├── rag\
    ├── stt\
    ├── tts\
    └── vision\
```

The CLI intentionally imports `AssistantOrchestrator` lazily from
`src.pipeline.orchestrator` (or `JARVIS_ORCHESTRATOR_MODULE`) when a command
needs it. This keeps `--help`, `config`, and `doctor` usable without loading
the pipeline or its models.
